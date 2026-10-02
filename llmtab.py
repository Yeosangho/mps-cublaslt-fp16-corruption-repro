#!/usr/bin/env python3
# usage: llmtab.py <log>   guard on/off table for llm_bench.py output
import re, sys
from collections import defaultdict
d = defaultdict(dict)
for l in open(sys.argv[1]):
    m = re.match(r"LLM pct=(\d+) sm=(\d+) cfg=(\w+) (\w+) bs=(\d+) prefill=(\d+) tok/s decode=([\d.]+) tok/s rewritten=(\d+)% (\S+)", l)
    if m:
        p, sm, cfg, dt, bs, pre, dec, rw, v = m.groups()
        d[(dt, int(p), int(sm), int(bs))][cfg] = (float(pre), float(dec), int(rw), v)
print("| dtype | share | SMs | batch | prefill tok/s off | guard | guard/off | generate tok/s off | guard | guard/off | calls rewritten | guard-off result | guard-on result |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for k in sorted(d, key=lambda k: (k[0], -k[1], k[3])):
    o, g = d[k].get("off"), d[k].get("guard")
    if not o or not g:
        continue
    gm = lambda v: float(re.search(r"gen_match=([0-9.]+)", v).group(1))
    good = lambda v: v.startswith("ok") and gm(v) >= 0.9      # wrong = non-finite / far-off logits, or the generated tokens do not match the 100 % run
    ok = good(o[3])
    f = lambda a, b: ("%.0f %%" % (100 * b / a)) if ok else "n/a"
    print("| %s | %d %% | %d | %d | %.0f | %.0f | %s | %.1f | %.1f | %s | %d %% | %s | %s |" % (
        k[0], k[1], k[2], k[3], o[0], g[0], f(o[0], g[0]), o[1], g[1], f(o[1], g[1]), g[2],
        "correct" if ok else "**wrong** " + o[3][o[3].index("(") + 1:-1].replace(",", ", "), "correct" if good(g[3]) else "WRONG " + g[3]))
