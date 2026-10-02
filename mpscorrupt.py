#!/usr/bin/env python
# MPS small-share corruption probe: ResNet inference (fp32/fp16/bf16) + direct GEMM checks.
# Outputs: <out>.npz (resnet logits per config), <out>.json (meta + per-config stats + gemm errors).
import os, sys, json, time, copy, argparse, ctypes, re
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--models", default="resnet50,resnet152")
ap.add_argument("--bs", default="1,16,64")
ap.add_argument("--iters", type=int, default=3)
ap.add_argument("--skip-gemm", action="store_true")
ap.add_argument("--skip-resnet", action="store_true")
a = ap.parse_args()
BS = [int(x) for x in a.bs.split(",")]
dev = torch.device("cuda")
DT = [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]


def P(*s):
    print(*s, flush=True)


def libver():
    v = {}
    torch.zeros(4, 4, device=dev) @ torch.zeros(4, 4, device=dev)  # force cublas load
    maps = open("/proc/self/maps").read()
    for name in ("libcublas.so", "libcublasLt.so", "libcudnn.so", "libcuda.so"):
        m = re.search(r"(/\S*%s\S*)" % re.escape(name), maps)
        v[name + "_path"] = m.group(1) if m else None
    try:
        lib = ctypes.CDLL(v["libcublas.so_path"])
        x = ctypes.c_int()
        t = []
        for k in (0, 1, 2):
            lib.cublasGetProperty(k, ctypes.byref(x))
            t.append(x.value)
        v["cublas"] = ".".join(map(str, t))
    except Exception as e:
        v["cublas"] = "ERR %r" % (e,)
    try:
        lt = ctypes.CDLL(v["libcublasLt.so_path"])
        lt.cublasLtGetVersion.restype = ctypes.c_size_t
        v["cublaslt"] = int(lt.cublasLtGetVersion())
    except Exception as e:
        v["cublaslt"] = "ERR %r" % (e,)
    return v


props = torch.cuda.get_device_properties(0)
meta = dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
            tv=torchvision.__version__, gpu=props.name, sm=props.multi_processor_count,
            pct=os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"),
            pipe=os.environ.get("CUDA_MPS_PIPE_DIRECTORY"))
meta.update(libver())
P("META", json.dumps(meta))
res = dict(meta=meta, gemm=[], resnet=[], errors=[])


def dump():
    json.dump(res, open(a.out + ".json", "w"))


# ---------------------------------------------------------------- GEMM (self-referenced vs CPU float64)
def gemm_tests():
    shapes = [(1, 2048, 1000), (16, 2048, 1000), (64, 2048, 1000), (256, 2048, 1000),
              (33, 777, 1291), (1024, 1024, 1024), (2048, 2048, 2048), (3000, 3000, 3000),
              (4096, 4096, 4096), (8192, 512, 8192), (512, 8192, 512), (4096, 64, 4096),
              (64, 4096, 64), (16384, 256, 1024)]
    for (m, k, n) in shapes:
        rs = np.random.RandomState(m * 7 + k * 3 + n)
        A = rs.standard_normal((m, k)).astype(np.float32)
        B = (rs.standard_normal((k, n)) / np.sqrt(k)).astype(np.float32)
        bias = rs.standard_normal((n,)).astype(np.float32)
        for dn, dt in DT:
            ac = torch.from_numpy(A).to(dt)
            bc = torch.from_numpy(B).to(dt)
            cc = torch.from_numpy(bias).to(dt)
            ref = ac.double() @ bc.double()
            refl = ref + cc.double()
            ag, bg, cg = ac.to(dev), bc.to(dev), cc.to(dev)
            wg = bg.t().contiguous()
            for op in ("matmul", "linear"):
                for it in range(a.iters):
                    rec = dict(op=op, m=m, k=k, n=n, dt=dn, it=it)
                    try:
                        g = (ag @ bg) if op == "matmul" else F.linear(ag, wg, cg)
                        g = g.double().cpu()
                        r = ref if op == "matmul" else refl
                        bad = int((~torch.isfinite(g)).sum())
                        d = torch.nan_to_num(g - r, nan=1e30, posinf=1e30, neginf=1e30)
                        rec.update(nonfinite=bad, rel=float(d.norm() / r.norm()), maxabs=float(d.abs().max()))
                    except Exception as e:
                        rec.update(error=repr(e)[:300])
                        res["errors"].append(rec)
                    res["gemm"].append(rec)
                    P("GEMM", json.dumps(rec))
    # batched matmul (bmm path)
    for (b, m, k, n) in [(64, 256, 256, 256), (8, 1024, 512, 1024), (512, 64, 64, 64)]:
        rs = np.random.RandomState(b + m + k + n)
        A = rs.standard_normal((b, m, k)).astype(np.float32)
        B = (rs.standard_normal((b, k, n)) / np.sqrt(k)).astype(np.float32)
        for dn, dt in DT:
            ac, bc = torch.from_numpy(A).to(dt), torch.from_numpy(B).to(dt)
            ref = ac.double() @ bc.double()
            for it in range(a.iters):
                rec = dict(op="bmm", m=m, k=k, n=n, b=b, dt=dn, it=it)
                try:
                    g = torch.bmm(ac.to(dev), bc.to(dev)).double().cpu()
                    bad = int((~torch.isfinite(g)).sum())
                    d = torch.nan_to_num(g - ref, nan=1e30, posinf=1e30, neginf=1e30)
                    rec.update(nonfinite=bad, rel=float(d.norm() / ref.norm()), maxabs=float(d.abs().max()))
                except Exception as e:
                    rec.update(error=repr(e)[:300])
                    res["errors"].append(rec)
                res["gemm"].append(rec)
                P("GEMM", json.dumps(rec))


# ---------------------------------------------------------------- ResNet (compared offline vs full-GPU run)
def inp(bs):
    return torch.from_numpy(np.random.RandomState(1234 + bs).standard_normal((bs, 3, 224, 224)).astype(np.float32))


outs = {}


def resnet_tests(name):
    m0 = getattr(torchvision.models, name)(weights="IMAGENET1K_V1").eval()
    for dn, dt in DT:
        for mode in (["cast"] if dn == "fp32" else ["cast", "autocast"]):
            for cl in (0, 1):
                try:
                    m = copy.deepcopy(m0).to(dev)
                    if mode == "cast":
                        m = m.to(dt)
                    if cl:
                        m = m.to(memory_format=torch.channels_last)
                except Exception as e:
                    res["errors"].append(dict(stage="model", name=name, dt=dn, mode=mode, cl=cl, error=repr(e)[:300]))
                    continue
                for bench in (0, 1):
                    torch.backends.cudnn.benchmark = bool(bench)
                    for bs in BS:
                        key = "%s|%s|%s|cl%d|bench%d|bs%d" % (name, dn, mode, cl, bench, bs)
                        rec = dict(key=key)
                        P("START", key)
                        try:
                            x = inp(bs).to(dev)
                            if mode == "cast":
                                x = x.to(dt)
                            if cl:
                                x = x.contiguous(memory_format=torch.channels_last)
                            ys = []
                            t0 = time.time()
                            with torch.inference_mode():
                                for it in range(a.iters):
                                    if mode == "autocast":
                                        with torch.autocast("cuda", dtype=dt):
                                            y = m(x)
                                    else:
                                        y = m(x)
                                    ys.append(y.float().cpu().numpy())
                            y0 = ys[0]
                            rec.update(nonfinite=int(sum((~np.isfinite(y)).sum() for y in ys)),
                                       norm=float(np.linalg.norm(np.nan_to_num(y0))),
                                       iter_maxdiff=float(max(np.abs(np.nan_to_num(y - y0)).max() for y in ys)),
                                       sec=round(time.time() - t0, 3))
                            outs[key] = y0
                            # keep the worst later iteration too, in case corruption is intermittent
                            outs[key + "|last"] = ys[-1]
                        except Exception as e:
                            rec.update(error=repr(e)[:300])
                            res["errors"].append(rec)
                        res["resnet"].append(rec)
                        P("RESNET", json.dumps(rec))
                del m
        np.savez(a.out + ".npz", **outs)
        dump()


if not a.skip_gemm:
    gemm_tests()
    dump()
if not a.skip_resnet:
    for nm in a.models.split(","):
        resnet_tests(nm)
dump()
P("DONE")
