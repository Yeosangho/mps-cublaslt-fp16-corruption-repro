#!/usr/bin/env python
# GEMM correctness scan for one MPS share: many (m,k,n) x {fp32,fp16,bf16}, each checked against a CPU float64
# reference on the same quantized inputs, with the cuBLASLt algorithm chosen for every call (from CUBLASLT_LOG).
#   python scan.py run   <out-prefix>      -> <out>.json  (+ <out>.ltlog written by cuBLASLt at process exit)
#   python scan.py parse <out-prefix>      -> adds the "algo" field to every case in <out>.json
import os, sys, json, re

mode, out = sys.argv[1], sys.argv[2]
logf = out + ".ltlog"

if mode == "run":
    os.environ["CUBLASLT_LOG_LEVEL"] = "5"
    os.environ["CUBLASLT_LOG_FILE"] = logf
    if os.path.exists(logf):
        os.remove(logf)
    import numpy as np, torch
    dev = torch.device("cuda")
    p = torch.cuda.get_device_properties(0)
    meta = dict(gpu=p.name, sm=p.multi_processor_count, pct=os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"),
                torch=torch.__version__, cuda=torch.version.cuda, preload=os.environ.get("LD_PRELOAD"),
                spoof_sm=os.environ.get("SPOOF_SM"), cc="%d.%d" % torch.cuda.get_device_capability(0))
    DT = [("fp32", torch.float32, 1e-3), ("fp16", torch.float16, 2e-2), ("bf16", torch.bfloat16, 1e-1)]
    cases = []
    for m in (1, 4, 16, 64, 256, 1024, 2048):
        for k in (64, 512, 2048, 4096):
            for n in (16, 128, 1000, 2048):
                rs = np.random.RandomState(m * 7 + k * 3 + n)
                A = torch.from_numpy(rs.standard_normal((m, k)).astype(np.float32))
                B = torch.from_numpy((rs.standard_normal((k, n)) / np.sqrt(k)).astype(np.float32))
                for dn, dt, tol in DT:
                    a, b = A.to(dt), B.to(dt)
                    ref = a.double() @ b.double()
                    c = dict(dt=dn, m=m, k=k, n=n)
                    try:
                        g = (a.to(dev) @ b.to(dev)).double().cpu()
                        d = torch.nan_to_num(g - ref, nan=1e30, posinf=1e30, neginf=1e30)
                        badel = (~torch.isfinite(g)) | (d.abs() > tol * (1 + ref.abs()))
                        c.update(rel=float(d.norm() / ref.norm()), badfrac=float(badel.float().mean()))
                        c["bad"] = bool(c["badfrac"] > 0)
                    except Exception as e:
                        c.update(error=repr(e)[:200], bad=True, rel=-1.0, badfrac=1.0)
                    cases.append(c)
    json.dump(dict(meta=meta, cases=cases), open(out + ".json", "w"))
    nb = {dn: sum(1 for c in cases if c["dt"] == dn and c["bad"]) for dn, _, _ in DT}
    print("SCAN sm=%s pct=%s bad fp32=%d fp16=%d bf16=%d of %d shapes" % (meta["sm"], meta["pct"], nb["fp32"], nb["fp16"], nb["bf16"], len(cases) // 3))

elif mode == "parse":
    r = json.load(open(out + ".json"))
    algo = {}
    TY = {"R_32F": "fp32", "R_16F": "fp16", "R_16BF": "bf16"}
    pat = re.compile(r"\[Trace\]\[cublasLt\w*Matmul\].*?Adesc=\[type=(\w+) rows=(\d+) cols=(\d+)[^\]]*\].*?Bdesc=\[type=\w+ rows=(\d+) cols=(\d+)[^\]]*\].*?algo=\[([^\]]*)\]")
    if os.path.exists(logf):
        for l in open(logf, errors="replace"):
            mm = pat.search(l)
            if mm:
                ty, ar, ac, br, bc, al = mm.groups()
                al = re.sub(r"MATMUL_TILE_|MATMUL_STAGES_|CLUSTER_SHAPE_|REDUCTION_SCHEME_", "", al)
                algo[(TY.get(ty, ty), int(ar), int(ac), int(bc))] = al
        os.remove(logf)
    for c in r["cases"]:
        # torch (row-major) a[m,k] @ b[k,n] reaches cuBLAS as column-major C^T[n,m] = b^T[n,k] * a^T[k,m]
        c["algo"] = algo.get((c["dt"], c["n"], c["k"], c["m"])) or algo.get((c["dt"], c["m"], c["k"], c["n"])) or "?"
    json.dump(r, open(out + ".json", "w"))
