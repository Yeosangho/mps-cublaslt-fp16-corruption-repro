#!/usr/bin/env python3
# usage: benchtab.py bench2.log   -> guard-on speed relative to guard-off, per share
import re, sys
from collections import defaultdict
d = defaultdict(dict)
for l in open(sys.argv[1]):
    m = re.match(r"rep=(\d) BENCH2 pct=(\d+) sm=(\d+) cfg=(\w+) (gemm|e2e) (\S+) value=([\d.]+) unit=(\S+) (\S+) rewritten=(\d+)%", l)
    if not m: continue
    rep, pct, sm, cfg, kind, desc, val, unit, st, rw = m.groups()
    d[(kind, desc, unit)][(int(pct), cfg, int(rep))] = (float(val), st, int(rw))
pcts = sorted(set(k[0] for v in d.values() for k in v), reverse=True)
reps = sorted(set(k[2] for v in d.values() for k in v))
for rep in reps:
    ps = [p for p in pcts if any((p, "guard", rep) in v for v in d.values())]
    print("rep %d: guard-on speed as %% of guard-off (>100 = faster).  x = baseline output CORRUPT, r = guard rewrote the cluster shape" % rep)
    print("%-28s" % "case" + "".join("%11s" % ("%d%%" % p) for p in ps))
    for (kind, desc, unit), v in d.items():
        row = "%-28s" % (kind + " " + desc)
        for p in ps:
            o, g = v.get((p, "off", rep)), v.get((p, "guard", rep))
            if not o or not g: row += "%11s" % "-"; continue
            r = (o[0] / g[0]) if unit == "us" else (g[0] / o[0])
            row += "%11s" % ("%.0f%s%s%s" % (100 * r, "x" if o[1] != "ok" else "", "r" if g[2] else "", "!" if g[1] != "ok" else ""))
        print(row)
    print()
