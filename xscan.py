#!/usr/bin/env python
# Extended correctness scan for one MPS share (used to decide which SM counts are safe).
#   GEMM : 12 x 4 x 6 (m,k,n) grid + large and model-like shapes, matmul and linear(+bias), FP16 and BF16, plus bmm
#   CONV : 1x1 / 3x3 convolutions over a grid of batch / channels / spatial size, NCHW and channels-last, FP16 and BF16
# Reference = the same operation in FP32 on the same (already quantized) operands; FP32 was never wrong in any
# CPU-referenced run (scan.py).  usage: python xscan.py <out.json>
import os, sys, json
import torch, torch.nn.functional as F

out = sys.argv[1]
dev = torch.device("cuda")
p = torch.cuda.get_device_properties(0)
DT = [("fp16", torch.float16, 2e-2), ("bf16", torch.bfloat16, 1e-1)]
res = dict(meta=dict(gpu=p.name, sm=p.multi_processor_count, pct=os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"),
                     torch=torch.__version__, preload=os.environ.get("LD_PRELOAD")), n={}, bad={}, examples=[])


def judge(kind, desc, y, ref, tol):
    d = torch.nan_to_num(y.float() - ref, nan=1e30, posinf=1e30, neginf=-1e30)
    badfrac = float(((~torch.isfinite(y)) | (d.abs() > tol * (1 + ref.abs()))).float().mean())
    res["n"][kind] = res["n"].get(kind, 0) + 1
    if badfrac > 0:
        res["bad"][kind] = res["bad"].get(kind, 0) + 1
        if len(res["examples"]) < 400:
            res["examples"].append([kind, desc, round(badfrac, 4)])


gen = torch.Generator(device="cuda").manual_seed(0)
shapes = [(m, k, n) for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
          for k in (64, 256, 1024, 2048) for n in (16, 64, 128, 512, 1000, 2048)]
shapes += [(4096, 4096, 4096), (4096, 1024, 4096), (64, 4096, 4096), (1, 4096, 4096), (16, 4096, 4096), (4096, 4096, 16),
           (1024, 4096, 128), (256, 4096, 4096), (2048, 4096, 2048), (3000, 3000, 3000), (33, 777, 1291), (100, 300, 500), (197, 768, 768), (197, 768, 3072), (197, 3072, 768),
           (512, 768, 3072), (512, 3072, 768), (8, 4096, 32000), (1, 4096, 32000), (64, 4096, 11008), (64, 11008, 4096),
           (128, 1024, 4096), (128, 4096, 1024), (16, 2048, 1000), (256, 8192, 8192), (1024, 1280, 5120)]
for (m, k, n) in shapes:
    A = torch.randn(m, k, device=dev, generator=gen)
    B = torch.randn(k, n, device=dev, generator=gen) / k ** 0.5
    bias = torch.randn(n, device=dev, generator=gen)
    for dn, dt, tol in DT:
        a, b, c = A.to(dt), B.to(dt), bias.to(dt)
        ref = a.float() @ b.float()
        judge("matmul_" + dn, "%dx%dx%d" % (m, k, n), a @ b, ref, tol)
        judge("linear_" + dn, "%dx%dx%d" % (m, k, n), F.linear(a, b.t().contiguous(), c), ref + c.float(), tol)
    del A, B
for (bt, m, k, n) in [(64, 256, 256, 256), (8, 1024, 512, 1024), (512, 64, 64, 64), (96, 197, 64, 197), (96, 197, 197, 64), (32, 128, 128, 128), (12, 1024, 64, 1024)]:
    A = torch.randn(bt, m, k, device=dev, generator=gen)
    B = torch.randn(bt, k, n, device=dev, generator=gen) / k ** 0.5
    for dn, dt, tol in DT:
        a, b = A.to(dt), B.to(dt)
        judge("bmm_" + dn, "%dx%dx%dx%d" % (bt, m, k, n), torch.bmm(a, b), torch.bmm(a.float(), b.float()), tol)
json.dump(res, open(out, "w"))

pairs = [(64, 64), (64, 256), (256, 64), (128, 512), (512, 128), (256, 1024), (1024, 256), (512, 2048), (2048, 512),
         (1024, 2048), (256, 256), (512, 512)]
for (ci, co) in pairs:
    for hw in (7, 14, 28, 56):
        for N in (1, 8, 32):
            X = torch.randn(N, ci, hw, hw, device=dev, generator=gen)
            for ks in ((1, 3) if ci == co else (1,)):
                W = torch.randn(co, ci, ks, ks, device=dev, generator=gen) / (ci * ks * ks) ** 0.5
                ref = None
                for dn, dt, tol in DT:
                    x, w = X.to(dt), W.to(dt)
                    ref = F.conv2d(x.float(), w.float(), None, 1, ks // 2)
                    for cl in (0, 1):
                        xx, ww = (x.contiguous(memory_format=torch.channels_last), w.contiguous(memory_format=torch.channels_last)) if cl else (x, w)
                        judge("conv%d_cl%d_%s" % (ks, cl, dn), "N%d_C%d->%d_%dx%d" % (N, ci, co, hw, hw), F.conv2d(xx, ww, None, 1, ks // 2), ref, tol)
            del X
json.dump(res, open(out, "w"))
tot = sum(res["n"].values()); bad = sum(res["bad"].values())
print("XSCAN sm=%d pct=%s cases=%d bad=%d %s" % (res["meta"]["sm"], res["meta"]["pct"], tot, bad,
                                               " ".join("%s=%d" % kv for kv in sorted(res["bad"].items()))))
