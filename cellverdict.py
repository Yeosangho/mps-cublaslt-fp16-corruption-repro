#!/usr/bin/env python3
# usage: cellverdict.py <label> <pct|nomps> [-v]
# One line: VOID <reason> / CRASH <where> / CORRUPT n/m / CLEAN 0/m  (+ max relative errors per dtype)
import sys, os, json
import numpy as np

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
lab, pct = sys.argv[1], sys.argv[2]
verbose = "-v" in sys.argv
tag = "nomps" if pct == "nomps" else "p" + pct
d = os.path.join(W, lab)
GT = dict(fp32=1e-4, fp16=5e-3, bf16=3e-2)   # GEMM vs CPU float64 of the same quantized inputs
RT = dict(fp32=1e-3, fp16=5e-2, bf16=2e-1)   # ResNet logits vs full-GPU (no MPS) run, same dtype/config


def out(s):
    print("%s %s %s" % (lab, tag, s))
    sys.exit(0)


log = open(os.path.join(d, tag + ".log"), errors="replace").read() if os.path.exists(os.path.join(d, tag + ".log")) else ""
rc = open(os.path.join(d, tag + ".rc")).read().strip() if os.path.exists(os.path.join(d, tag + ".rc")) else "?"
if "META" not in log:
    out("VOID no-META rc=%s tail=%r" % (rc, log[-300:]))
meta = json.loads([l for l in log.splitlines() if l.startswith("META ")][0][5:])
ref_meta = None
if os.path.exists(os.path.join(d, "nomps.json")):
    ref_meta = json.load(open(os.path.join(d, "nomps.json")))["meta"]
if tag != "nomps":
    if not meta.get("pipe"):
        out("VOID no-mps-pipe")
    if ref_meta and int(pct) < 95 and meta["sm"] >= ref_meta["sm"]:
        out("VOID sm=%d not reduced (MPS share not applied)" % meta["sm"])

# parse records from the log (survives a crash, unlike the json)
gem, rsn = [], []
last_start = None
for l in log.splitlines():
    if l.startswith("GEMM "):
        gem.append(json.loads(l[5:]))
    elif l.startswith("RESNET "):
        rsn.append(json.loads(l[7:]))
    elif l.startswith("START "):
        last_start = l[6:]
done = "DONE" in log.splitlines()[-3:] if log else False

bad = []
gmax = dict(fp32=0.0, fp16=0.0, bf16=0.0)
errs = 0
for r in gem:
    if "error" in r:
        errs += 1
        bad.append(("gemm-error", r))
        continue
    gmax[r["dt"]] = max(gmax[r["dt"]], r["rel"])
    if r["nonfinite"] or not (r["rel"] <= GT[r["dt"]]):
        bad.append(("gemm", r))

rmax = dict(fp32=0.0, fp16=0.0, bf16=0.0)
ncmp = 0
ref = None
if tag != "nomps" and os.path.exists(os.path.join(d, "nomps.npz")):
    ref = np.load(os.path.join(d, "nomps.npz"))
cur = np.load(os.path.join(d, tag + ".npz")) if os.path.exists(os.path.join(d, tag + ".npz")) else None
for r in rsn:
    if "error" in r:
        errs += 1
        bad.append(("resnet-error", r))
        continue
    dt = r["key"].split("|")[1]
    if r["nonfinite"]:
        bad.append(("resnet-nonfinite", r))
    if ref is not None and cur is not None and r["key"] in cur.files and r["key"] in ref.files:
        for k in (r["key"], r["key"] + "|last"):
            y, y0 = cur[k].astype(np.float64), ref[r["key"]].astype(np.float64)
            rel = float(np.linalg.norm(np.nan_to_num(y - y0, nan=1e30)) / np.linalg.norm(y0))
            top1 = float((y.argmax(1) == y0.argmax(1)).mean())
            ncmp += 1
            rmax[dt] = max(rmax[dt], rel)
            if not (rel <= RT[dt]):
                bad.append(("resnet", dict(key=k, rel=rel, top1=top1)))

stat = "sm=%s gemm_rel[32/16/bf]=%.1e/%.1e/%.1e resnet_rel[32/16/bf]=%.1e/%.1e/%.1e n_gemm=%d n_resnet=%d n_cmp=%d err=%d rc=%s" % (
    meta["sm"], gmax["fp32"], gmax["fp16"], gmax["bf16"], rmax["fp32"], rmax["fp16"], rmax["bf16"], len(gem), len(rsn), ncmp, errs, rc)
if verbose:
    for b in bad[:60]:
        print("  BAD", b[0], json.dumps(b[1]))
total = len(gem) + len(rsn)
if not done:
    out("CRASH after=%r bad=%d/%d %s" % (last_start or ("gemm#%d" % len(gem)), len(bad), total, stat))
if bad:
    out("CORRUPT %d/%d %s" % (len(bad), total, stat))
out("CLEAN 0/%d %s" % (total, stat))
