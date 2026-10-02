#!/usr/bin/env python3
# usage: summarize.py <out/scan> [label ...]   One markdown row per scan campaign.
import sys, os, json, glob, re


def ranges(vals, step=1):
    vals = sorted(vals); out = []; i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] - vals[j] <= step:
            j += 1
        out.append(str(vals[i]) if i == j else "%d-%d" % (vals[i], vals[j])); i = j + 1
    return ",".join(out) or "-"


root = sys.argv[1]
labels = sys.argv[2:] or sorted(os.listdir(root))
print("| campaign | GPU | torch | shares run | shares crashed | shares with corruption | corrupted shares (%) | corrupted SM counts | corrupted fp16+bf16 GEMMs | fp32 corrupted |")
print("|---|---|---|---|---|---|---|---|---|---|")
for lab in labels:
    fs = glob.glob(os.path.join(root, lab, "p*.json"))
    if not fs:
        continue
    badp, bads, nb, nt, nb32, done = [], set(), 0, 0, 0, set()
    for f in fs:
        p = int(re.search(r"p(\d+)\.json", f).group(1)); done.add(p)
        r = json.load(open(f)); meta = r["meta"]
        b = [c for c in r["cases"] if c["dt"] != "fp32"]
        k = sum(c["bad"] for c in b); nb += k; nt += len(b)
        nb32 += sum(c["bad"] for c in r["cases"] if c["dt"] == "fp32")
        if k:
            badp.append(p); bads.add(meta["sm"])
    crashed = sorted(set(range(1, 101)) - done)
    print("| %s | %s | %s | %d | %s | %d | %s | %s | %d / %d (%.1f%%) | %d |" % (
        lab, meta["gpu"].replace("NVIDIA ", ""), meta["torch"], len(done), ranges(crashed), len(badp), ranges(badp),
        ranges(bads, 2), nb, nt, 100.0 * nb / max(nt, 1), nb32))
