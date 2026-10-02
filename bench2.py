# Cost of the cuBLASLt cluster-shape guard: GEMM by size, and end-to-end FP16 inference (ResNet-50, ViT-B/16).
# Run each configuration alone on the GPU:   TAG=off|guard python bench2.py
import os, time, copy, ctypes
import numpy as np, torch, torchvision

sm = torch.cuda.get_device_properties(0).multi_processor_count
pct = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
tag = os.environ.get("TAG", "-")
dev = torch.device("cuda")
try:
    _g = ctypes.CDLL(None); _g.ltguard_rewritten.restype = ctypes.c_ulong; _g.ltguard_calls.restype = ctypes.c_ulong
    stats = lambda: (_g.ltguard_calls(), _g.ltguard_rewritten())
except Exception:
    stats = lambda: (0, 0)


def timed(fn, min_s=1.5, chunk=10):
    for _ in range(chunk): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter(); it = 0
    while time.perf_counter() - t0 < min_s:
        for _ in range(chunk): fn()
        torch.cuda.synchronize(); it += chunk
    return (time.perf_counter() - t0) / it


def line(kind, desc, value, unit, ok, s0):
    c, r = stats()
    dc, dr = c - s0[0], r - s0[1]
    print("BENCH2 pct=%s sm=%d cfg=%s %s %s value=%.2f unit=%s %s rewritten=%.0f%%" % (
        pct, sm, tag, kind, desc, value, unit, ok, 100.0 * dr / dc if dc else 0.0), flush=True)


# ---- GEMM by size (FP16): square, "tokens x hidden" (m small, k=n=4096), and classifier-like
shapes = [(n, n, n) for n in (128, 256, 512, 1024, 2048, 4096, 8192)]
shapes += [(m, 4096, 4096) for m in (1, 16, 64, 256, 1024)]
shapes += [(16, 2048, 1000), (64, 2048, 1000), (1024, 4096, 128), (32, 4096, 32000)]
for dn, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
    for (m, k, n) in (shapes if dn == "fp16" else shapes[2:6]):
        g = torch.Generator().manual_seed(0)
        a = torch.randn(m, k, generator=g).to(dev).to(dt); b = (torch.randn(k, n, generator=g) / k ** 0.5).to(dev).to(dt)
        ref = a.float() @ b.float()
        s0 = stats()
        out = (a @ b).float()
        bad = int(((~torch.isfinite(out)) | ((out - ref).abs() > 0.1 * (1 + ref.abs()))).sum())
        us = timed(lambda: a @ b) * 1e6
        line("gemm", "%s_%dx%dx%d" % (dn, m, k, n), us, "us", "CORRUPT" if bad else "ok", s0)
        del a, b, ref, out

# ---- end-to-end FP16 inference; correctness = logits vs the FP32 run of the same model and input
models = [("resnet50", lambda: torchvision.models.resnet50(weights="IMAGENET1K_V1")),
          ("vit_b_16", lambda: torchvision.models.vit_b_16(weights="IMAGENET1K_V1"))]
for name, mk in models:
    m0 = mk().eval().to(dev)
    for cl in ((0, 1) if name.startswith("resnet") else (0,)):
        for bs in (1, 16, 64):
            x = torch.from_numpy(np.random.RandomState(1234 + bs).standard_normal((bs, 3, 224, 224)).astype(np.float32)).to(dev)
            mh = copy.deepcopy(m0).half()
            xh = x.half()
            if cl:
                mh = mh.to(memory_format=torch.channels_last); xh = xh.contiguous(memory_format=torch.channels_last)
            with torch.inference_mode():
                r32 = m0(x)
                s0 = stats()
                y = mh(xh).float()
                rel = float(torch.nan_to_num(y - r32, nan=1e30, posinf=1e30, neginf=-1e30).norm() / r32.norm())
                agree = float((y.argmax(1) == r32.argmax(1)).float().mean())
                s = timed(lambda: mh(xh), min_s=2.0, chunk=5)
            line("e2e", "%s_cl%d_bs%d" % (name, cl, bs), bs / s, "img/s", "ok" if rel < 0.05 else "CORRUPT(rel=%.1e,top1=%.2f)" % (rel, agree), s0)
            del mh
    del m0
