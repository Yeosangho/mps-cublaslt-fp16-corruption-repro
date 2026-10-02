#!/usr/bin/env python3
# usage: agg.py <out/scan/label> [-a]   Per-share table and per-algorithm summary of a scan campaign.
import sys, os, json, re, glob
from collections import defaultdict

d = sys.argv[1]
rows = []
for f in glob.glob(os.path.join(d, "p*.json")):
    r = json.load(open(f))
    rows.append((int(re.search(r"p(\d+)\.json", f).group(1)), r))
rows.sort(key=lambda x: x[0])


def short(al):
    g = lambda k: (re.search(k + r"=(\S+)", al) or [None, "-"])[1]
    return "id%s/%s/%s/c%s/%s" % (g("algoId"), g("tile"), g("stages"), g("customOption"), g("clusterShape"))


full = max(r["meta"]["sm"] for _, r in rows) if rows else 0
by_sm = {}
alg = defaultdict(lambda: defaultdict(lambda: [0, 0]))      # config -> sm -> [ok, bad]
print("gpu=%s cc=%s torch=%s  shapes per dtype=%d" % (rows[0][1]["meta"]["gpu"], rows[0][1]["meta"].get("cc"), rows[0][1]["meta"]["torch"], len(rows[0][1]["cases"]) // 3))
print("%4s %4s %6s %6s %6s  %s" % ("pct", "SM", "fp32", "fp16", "bf16", "corrupted algorithm configs (algoId/tile/stages/customOption/clusterShape)"))
for p, r in rows:
    sm = r["meta"]["sm"]
    nb = defaultdict(int)
    badalg = set()
    for c in r["cases"]:
        s = short(c.get("algo", "?"))
        if c["dt"] != "fp32":
            alg[s][sm][1 if c["bad"] else 0] += 1
        if c["bad"]:
            nb[c["dt"]] += 1
            badalg.add(s)
    by_sm.setdefault(sm, []).append(p)
    void = " VOID(SM not reduced)" if (p < 95 and sm >= full and len(rows) > 3) else ""
    print("%4d %4d %6d %6d %6d  %s%s" % (p, sm, nb["fp32"], nb["fp16"], nb["bf16"], " ".join(sorted(badalg)), void))
bad_sm = sorted(sm for sm in by_sm if any(v[sm][1] for v in alg.values() if sm in v))
print("\nSM counts with corruption:", bad_sm)
print("SM counts clean          :", sorted(set(by_sm) - set(bad_sm)))
if "-a" in sys.argv:
    print("\nper algorithm config (fp16+bf16):  config  ->  SM counts where it was used: ok / CORRUPT")
    for s in sorted(alg, key=lambda s: -sum(v[1] for v in alg[s].values())):
        oks = sorted(sm for sm, v in alg[s].items() if v[1] == 0)
        bads = sorted(sm for sm, v in alg[s].items() if v[1] > 0)
        mixed = sorted(sm for sm, v in alg[s].items() if v[1] > 0 and v[0] > 0)
        print("  %-44s ok@SM=%s  CORRUPT@SM=%s%s" % (s, oks, bads, ("  (mixed ok/bad within SM: %s)" % mixed) if mixed else ""))
