#!/usr/bin/env python3
# usage: benchsum.py <log> [<log> ...] [--skip rep:pct ...]
# Markdown table: guard-on speed as % of guard-off, min–max over cases and repetitions, per share.
# Only pairs whose guard-off result is correct are used; a guard-on result that is not "ok" is flagged with "!".
import re, sys
from collections import defaultdict

logs = [a for a in sys.argv[1:] if not a.startswith("--") and ":" not in a]
skip = set(tuple(int(x) for x in a.split(":")) for a in sys.argv[1:] if re.fullmatch(r"\d+:\d+", a))
d = defaultdict(dict)
for f in logs:
    for l in open(f):
        m = re.match(r"rep=(\d) BENCH2 pct=(\d+) sm=(\d+) cfg=(\w+) (gemm|e2e) (\S+) value=([\d.]+) unit=(\S+) (\S+) rewritten=(\d+)%", l)
        if not m:
            continue
        rep, pct, sm, cfg, kind, desc, val, unit, st, rw = m.groups()
        if (int(rep), int(pct)) in skip:
            continue
        d[(kind, desc, unit)][(int(pct), cfg, int(rep))] = (float(val), st)


def group(kind, desc):
    if kind == "gemm":
        m, k, n = [int(x) for x in desc.split("_")[1].split("x")]
        if m == k == n:
            return "square GEMM N×N×N, N = 128 … 8192"
        if k == n == 4096:
            return "GEMM m×4096×4096, m = 1, 16, 64" if m <= 64 else "GEMM m×4096×4096, m = 256, 1024"
        return "GEMM 16×2048×1000, 64×2048×1000, 1024×4096×128, 32×4096×32000"
    name, bs = desc.rsplit("_bs", 1)
    model = "ResNet-50" if name.startswith("resnet50") else "ViT-B/16"
    return "%s FP16 inference, batch %s" % (model, "1" if bs == "1" else "16 / 64")


order = ["square GEMM N×N×N, N = 128 … 8192", "GEMM m×4096×4096, m = 256, 1024", "GEMM m×4096×4096, m = 1, 16, 64",
         "GEMM 16×2048×1000, 64×2048×1000, 1024×4096×128, 32×4096×32000", "ResNet-50 FP16 inference, batch 16 / 64",
         "ResNet-50 FP16 inference, batch 1", "ViT-B/16 FP16 inference, batch 16 / 64", "ViT-B/16 FP16 inference, batch 1"]
pcts = sorted(set(k[0] for v in d.values() for k in v), reverse=True)
vals = defaultdict(list); flag = set(); nwrong = defaultdict(int)
for (kind, desc, unit), v in d.items():
    for (p, cfg, rep) in list(v):
        if cfg != "off" or (p, "guard", rep) not in v:
            continue
        o, g = v[(p, "off", rep)], v[(p, "guard", rep)]
        if g[1] != "ok":
            flag.add((group(kind, desc), p))
        if o[1] != "ok":
            nwrong[(group(kind, desc), p)] += 1
            continue
        vals[(group(kind, desc), p)].append(100.0 * ((o[0] / g[0]) if unit == "us" else (g[0] / o[0])))
print("| Guard-on speed, % of guard-off | " + " | ".join("%d %%" % p for p in pcts) + " |")
print("|---|" + "---|" * len(pcts))
for gname in order:
    cells = []
    for p in pcts:
        x = vals.get((gname, p))
        c = "–" if not x else ("%.0f" % x[0] if len(x) == 1 or round(min(x)) == round(max(x)) else "%.0f–%.0f" % (min(x), max(x)))
        if nwrong.get((gname, p)):
            c += " (%d excl.)" % nwrong[(gname, p)]
        cells.append(c + ("!" if (gname, p) in flag else ""))
    print("| %s | %s |" % (gname, " | ".join(cells)))
