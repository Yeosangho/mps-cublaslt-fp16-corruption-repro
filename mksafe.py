#!/usr/bin/env python3
# usage: mksafe.py <out/xscan/label> [backbone-log] [out/scan/label]
# Per-share verdict of the extended scan, merged with the ResNet backbone sweep and the 112-shape GEMM scan,
# then the mapping "requested share -> largest share <= requested with no wrong result in any harness".
import sys, os, json, glob, re
from collections import defaultdict


def ranges(vals, step=1):
    vals = sorted(vals); out = []; i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] - vals[j] <= step:
            j += 1
        out.append(str(vals[i]) if i == j else "%d-%d" % (vals[i], vals[j])); i = j + 1
    return ",".join(out) or "-"


sm_of, bad = {}, defaultdict(dict)          # bad[pct][harness] = count
kinds = defaultdict(lambda: [0, 0])
for f in glob.glob(os.path.join(sys.argv[1], "p*.json")):
    p = int(re.search(r"p(\d+)\.json", f).group(1))
    r = json.load(open(f)); sm_of[p] = r["meta"]["sm"]
    bad[p]["xscan"] = sum(r["bad"].values())
    for k, n in r["n"].items():
        g = re.sub(r"_(fp16|bf16)$", "", k)
        kinds[g][0] += n; kinds[g][1] += r["bad"].get(k, 0)
if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
    for l in open(sys.argv[2]):
        m = re.search(r"BACKBONE sm=(\d+) pct=(\d+) configs=\d+ +backbone\(cuDNN\) max_rel=\S+ bad=(\d+) +fc\(cuBLAS\) max_rel=\S+ bad=(\d+)", l)
        if m and "guard" not in l.split("BACKBONE")[0]:
            bad[int(m.group(2))]["backbone"] = int(m.group(3)) + int(m.group(4)); sm_of.setdefault(int(m.group(2)), int(m.group(1)))
if len(sys.argv) > 3:
    for f in glob.glob(os.path.join(sys.argv[3], "p*.json")):
        p = int(re.search(r"p(\d+)\.json", f).group(1)); r = json.load(open(f))
        bad[p]["scan"] = sum(c["bad"] for c in r["cases"]); sm_of.setdefault(p, r["meta"]["sm"])

pcts = sorted(sm_of)
print("extended scan, cases per kind (all shares): " + "  ".join("%s %d/%d wrong" % (k, v[1], v[0]) for k, v in sorted(kinds.items())))
for h in ("xscan", "backbone", "scan"):
    b = [p for p in pcts if bad[p].get(h)]
    if any(h in bad[p] for p in pcts):
        print("%-8s shares with wrong results: %3d  %s" % (h, len(b), ranges(b)))
unsafe = [p for p in pcts if any(bad[p].values())]
safe = [p for p in pcts if p not in unsafe]
print("union    shares with wrong results: %3d  %s   | SM counts: %s" % (len(unsafe), ranges(unsafe), ranges(set(sm_of[p] for p in unsafe), 2)))
print("clean in every harness            : %3d  %s   | SM counts: %s" % (len(safe), ranges(safe), ranges(set(sm_of[p] for p in safe), 2)))
print("\n| requested share | SMs | wrong results | largest clean share <= requested | SMs | SMs given up |")
print("|---|---|---|---|---|---|")
i = 0
while i < len(pcts):
    p = pcts[i]; st = p in unsafe; j = i
    while j + 1 < len(pcts) and (pcts[j + 1] in unsafe) == st:
        j += 1
    lo, hi = pcts[i], pcts[j]
    if st:
        below = [q for q in safe if q < lo]
        t = max(below) if below else None
        print("| %d–%d %% | %d–%d | yes | %s | %s | %s |" % (lo, hi, sm_of[lo], sm_of[hi], ("%d %%" % t) if t else "none", sm_of[t] if t else "-",
              ("%d–%d (%.0f–%.0f %%)" % (sm_of[lo] - sm_of[t], sm_of[hi] - sm_of[t], 100.0 * (sm_of[lo] - sm_of[t]) / sm_of[lo], 100.0 * (sm_of[hi] - sm_of[t]) / sm_of[hi])) if t else "-"))
    else:
        print("| %d–%d %% | %d–%d | no | as requested | %d–%d | 0 |" % (lo, hi, sm_of[lo], sm_of[hi], sm_of[lo], sm_of[hi]))
    i = j + 1
