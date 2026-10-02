#!/usr/bin/env python3
# usage: nptab.py <lt_np log>    per input/output type: shares with wrong results, algorithm ids, cluster shapes
import re, sys
from collections import defaultdict

CL = {0: "auto", 2: "1x1x1", 3: "2x1x1", 4: "4x1x1", 5: "1x2x1", 6: "2x2x1", 7: "4x2x1", 8: "1x4x1", 9: "2x4x1", 10: "4x4x1", 11: "8x1x1", 12: "1x8x1"}


def ranges(v):
    v = sorted(v); o = []; i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[j + 1] - v[j] <= 1:
            j += 1
        o.append(str(v[i]) if i == j else "%d-%d" % (v[i], v[j])); i = j + 1
    return ",".join(o) or "-"


d = defaultdict(lambda: dict(n=0, bad=0, uns=0, fail=0, pb=set(), ids=set(), cl=defaultdict(lambda: [0, 0]), pcts=set()))
for l in open(sys.argv[1]):
    m = re.match(r"NP (\w+) sm=(\d+) pct=(\S+) (\S+) (UNSUPPORTED.*|algo\[id=(\d+) tile=\d+ custom=\d+ cluster=(\d+)\] status=(\d+) sync=(\d+) (\w+) bad=(\d+)/(\d+) unwritten=(\d+))", l)
    if not m:
        continue
    c = d[m.group(1)]; pct = int(m.group(3)); c["pcts"].add(pct)
    if m.group(5).startswith("UNSUPPORTED"):
        c["uns"] += 1; continue
    c["n"] += 1; c["ids"].add(int(m.group(6))); cs = CL.get(int(m.group(7)), m.group(7)); v = m.group(10)
    c["cl"][cs][0] += 1
    if v == "CORRUPT":
        c["bad"] += 1; c["pb"].add(pct); c["cl"][cs][1] += 1
    elif v != "ok":
        c["fail"] += 1
print("| type | shares | products run | wrong | failed | unsupported | shares with wrong results | algo ids | cluster shape: wrong/run |")
print("|---|---|---|---|---|---|---|---|---|")
for k, c in d.items():
    print("| %s | %d | %d | %d | %d | %d | %s | %s | %s |" % (k, len(c["pcts"]), c["n"], c["bad"], c["fail"], c["uns"], ranges(c["pb"]), sorted(c["ids"]),
          ", ".join("%s %d/%d" % (s, v[1], v[0]) for s, v in sorted(c["cl"].items()))))
