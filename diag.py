#!/usr/bin/env python
# Localize the corruption: (1) where in the GEMM output, (2) which (m,k,n), (3) ResNet backbone (cuDNN) vs FC (cuBLAS).
import os, sys, json
import numpy as np
import torch, torch.nn.functional as F, torchvision

dev = torch.device("cuda")
sm = torch.cuda.get_device_properties(0).multi_processor_count
print("SM", sm, "pct", os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"), "torch", torch.__version__, flush=True)
DT = [("fp16", torch.float16), ("bf16", torch.bfloat16)]
TOL = dict(fp16=2e-2, bf16=1e-1)


def gemm(m, k, n, dt, seed=0):
    rs = np.random.RandomState(seed + m * 7 + k * 3 + n)
    a = torch.from_numpy(rs.standard_normal((m, k)).astype(np.float32)).to(dt)
    b = torch.from_numpy((rs.standard_normal((k, n)) / np.sqrt(k)).astype(np.float32)).to(dt)
    ref = (a.double() @ b.double())
    g = (a.to(dev) @ b.to(dev)).double().cpu()
    return g, ref


# (1) pattern of bad elements for the ResNet FC shape
for dn, dt in DT:
    for m in (1, 16, 64):
        g, ref = gemm(m, 2048, 1000, dt)
        bad = (~torch.isfinite(g)) | ((g - ref).abs() > TOL[dn] * (1 + ref.abs()))
        idx = bad.nonzero()
        if len(idx) == 0:
            print("PATTERN", dn, m, "no bad elements")
            continue
        rows, cols = idx[:, 0], idx[:, 1]
        vals = g[bad]
        print("PATTERN %s m=%d bad=%d/%d nonfinite=%d rows[%d..%d] nrows=%d cols[%d..%d] ncols=%d sample_vals=%s sample_cols=%s" % (
            dn, m, len(idx), g.numel(), int((~torch.isfinite(g)).sum()), rows.min(), rows.max(), len(rows.unique()),
            cols.min(), cols.max(), len(cols.unique()),
            ["%.3g" % v for v in vals[:6].tolist()], cols.unique()[:24].tolist()), flush=True)

# (2) shape scan, fp16 + bf16: print only corrupted shapes, then a count
for dn, dt in DT:
    badshapes, total = [], 0
    for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        for k in (64, 256, 512, 1024, 2048, 4096):
            for n in (64, 250, 500, 1000, 2000, 4000):
                g, ref = gemm(m, k, n, dt)
                total += 1
                rel = float(torch.nan_to_num(g - ref, nan=1e30, posinf=1e30, neginf=1e30).norm() / ref.norm())
                if not rel < TOL[dn]:
                    badshapes.append((m, k, n))
    print("SCAN", dn, "bad %d/%d" % (len(badshapes), total), json.dumps(badshapes), flush=True)

# (3) ResNet50: backbone features (cuDNN) saved for offline comparison; FC on GPU vs FC recomputed on CPU
m0 = torchvision.models.resnet50(weights="IMAGENET1K_V1").eval()
x0 = torch.from_numpy(np.random.RandomState(1234 + 16).standard_normal((16, 3, 224, 224)).astype(np.float32))
feats = {}
for dn, dt in DT:
    mod = m0.to(dev).to(dt)
    fc = mod.fc
    with torch.inference_mode():
        x = x0.to(dev).to(dt)
        h = x
        for name, layer in mod.named_children():
            if name == "fc":
                break
            h = layer(h)
        f = torch.flatten(h, 1)                       # backbone output (conv/bn/pool only)
        y_gpu = fc(f).float().cpu()                   # cuBLAS GEMM on GPU
        y_cpu = F.linear(f.float().cpu(), fc.weight.float().cpu(), fc.bias.float().cpu())
        y_full = mod(x).float().cpu()
    feats[dn] = f.float().cpu().numpy()
    rel_fc = float(torch.nan_to_num(y_gpu - y_cpu, nan=1e30).norm() / y_cpu.norm())
    rel_full = float(torch.nan_to_num(y_full - y_cpu, nan=1e30).norm() / y_cpu.norm())
    print("RESNET50 %s feat_nonfinite=%d feat_norm=%.4f  fc_gpu_vs_cpu_rel=%.3e  full_vs_cpufc_rel=%.3e  top1_match(gpu fc vs cpu fc)=%.2f" % (
        dn, int((~torch.isfinite(f)).sum()), float(f.float().norm()), rel_fc, rel_full,
        float((y_gpu.argmax(1) == y_cpu.argmax(1)).float().mean())), flush=True)
    m0 = m0.float().cpu()
out = sys.argv[1] if len(sys.argv) > 1 else None
if out:
    np.savez(out, **feats)
    ref = os.path.join(os.path.dirname(out), "diag_nomps.npz")
    if os.path.exists(ref) and os.path.abspath(ref) != os.path.abspath(out):
        r = np.load(ref)
        for dn in feats:
            print("BACKBONE %s rel_vs_fullgpu=%.3e" % (dn, np.linalg.norm(feats[dn] - r[dn]) / np.linalg.norm(r[dn])))
print("DONE")
