# Minimal repro: half-precision GEMM returns wrong output under a small MPS share on Hopper.
# Run as an MPS client with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=14 (H100/H200: 18 SMs visible).
import os, torch

M, K, N = [int(v) for v in os.environ.get("MKN", "16,2048,1000").split(",")]
sm = torch.cuda.get_device_properties(0).multi_processor_count
for dt in (torch.float32, torch.float16, torch.bfloat16):
    g = torch.Generator().manual_seed(0)
    a = torch.randn(M, K, generator=g).to(dt)
    b = (torch.randn(K, N, generator=g) / K ** 0.5).to(dt)
    ref = a.double() @ b.double()                      # CPU float64 reference on the same quantized inputs
    out = (a.cuda() @ b.cuda()).double().cpu()         # cuBLAS GEMM
    bad = ~torch.isclose(out, ref, rtol=5e-2, atol=5e-2)
    cols = bad.any(0).nonzero().flatten()
    print("SM=%d pct=%s %-14s MxKxN=%dx%dx%d bad=%d/%d bad_cols=%s nonfinite=%d zeros=%d max_abs_err=%.3g" % (
        sm, os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"), str(dt), M, K, N, int(bad.sum()), bad.numel(),
        ("[%d..%d]" % (cols[0], cols[-1])) if len(cols) else "-", int((~torch.isfinite(out)).sum()),
        int((out == 0).sum()), float(torch.nan_to_num(out - ref, nan=1e30, posinf=1e30, neginf=-1e30).abs().max())), flush=True)
