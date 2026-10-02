#!/usr/bin/env python
# Per-layer check of half-precision cuDNN convolutions under an MPS share.
# Every Conv2d of ResNet-50 is re-computed in FP32 from the same (FP16/BF16) input and weights and compared.
#   python convdiag.py [fp16|bf16] [cl0|cl1] [bs]            -> lists the convolutions whose output is wrong
#   python convdiag.py one <dt> <cl> <N> <Cin> <H> <W> <Cout> <k> <stride> <pad>   -> one standalone convolution
import os, sys
import numpy as np, torch, torch.nn.functional as F, torchvision

dev = torch.device("cuda")
sm = torch.cuda.get_device_properties(0).multi_processor_count
pct = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
DT = {"fp16": torch.float16, "bf16": torch.bfloat16}


def check(x, w, y, stride, pad, groups, tol):
    ref = F.conv2d(x.float(), w.float(), None, stride, pad, 1, groups)
    d = torch.nan_to_num(y.float() - ref, nan=1e30, posinf=1e30, neginf=-1e30)
    rel = float(d.norm() / ref.norm())
    badfrac = float(((~torch.isfinite(y)) | (d.abs() > tol * (1 + ref.abs()))).float().mean())
    return rel, badfrac


if sys.argv[1] == "one":
    dn, cl = sys.argv[2], int(sys.argv[3][-1])
    N, Cin, H, W, Cout, k, st, pd = [int(v) for v in sys.argv[4:12]]
    dt = DT[dn]
    g = torch.Generator().manual_seed(0)
    x = torch.randn(N, Cin, H, W, generator=g).to(dev).to(dt)
    w = (torch.randn(Cout, Cin, k, k, generator=g) / (Cin * k * k) ** 0.5).to(dev).to(dt)
    if cl:
        x = x.contiguous(memory_format=torch.channels_last); w = w.contiguous(memory_format=torch.channels_last)
    for bench in (0, 1):
        torch.backends.cudnn.benchmark = bool(bench)
        y = F.conv2d(x, w, None, st, pd)
        rel, bf = check(x, w, y, st, pd, 1, 2e-2 if dn == "fp16" else 1e-1)
        print("CONV1 sm=%d pct=%s %s cl%d N=%d Cin=%d HxW=%dx%d Cout=%d k=%d s=%d p=%d bench=%d rel=%.2e badfrac=%.3f %s" % (
            sm, pct, dn, cl, N, Cin, H, W, Cout, k, st, pd, bench, rel, bf, "CORRUPT" if bf > 0 else "ok"), flush=True)
    sys.exit(0)

dn = sys.argv[1] if len(sys.argv) > 1 else "fp16"
cl = int(sys.argv[2][-1]) if len(sys.argv) > 2 else 0
bs = int(sys.argv[3]) if len(sys.argv) > 3 else 16
dt = DT[dn]
m = torchvision.models.resnet50(weights="IMAGENET1K_V1").eval().to(dev).to(dt)
if cl:
    m = m.to(memory_format=torch.channels_last)
recs = []


def hook(name):
    def f(mod, inp, out):
        rel, bf = check(inp[0], mod.weight, out, mod.stride, mod.padding, mod.groups, 2e-2 if dn == "fp16" else 1e-1)
        recs.append((name, tuple(inp[0].shape), tuple(mod.weight.shape), mod.stride[0], mod.padding[0], rel, bf))
    return f


for n_, mod in m.named_modules():
    if isinstance(mod, torch.nn.Conv2d):
        mod.register_forward_hook(hook(n_))
x = torch.from_numpy(np.random.RandomState(1234 + bs).standard_normal((bs, 3, 224, 224)).astype(np.float32)).to(dev).to(dt)
if cl:
    x = x.contiguous(memory_format=torch.channels_last)
with torch.inference_mode():
    m(x)
bad = [r for r in recs if r[6] > 0]
print("CONVDIAG sm=%d pct=%s %s cl%d bs=%d  convs=%d corrupt=%d" % (sm, pct, dn, cl, bs, len(recs), len(bad)))
seen = set()
for name, ish, wsh, st, pd, rel, bf in bad:
    key = (ish, wsh, st, pd)
    if key in seen:
        continue
    seen.add(key)
    print("  first=%-22s in=%s w=%s stride=%d pad=%d rel=%.2e badfrac=%.3f   args: %d %d %d %d %d %d %d %d" % (
        name, ish, wsh, st, pd, rel, bf, ish[0], ish[1], ish[2], ish[3], wsh[0], wsh[2], st, pd))
