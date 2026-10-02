#!/bin/bash
# usage: ctxcmp.sh <label>   (environment selects the context)  occupancy table + algorithm + launch for a few FP16 products
cd /root/mpscorrupt; export LD_LIBRARY_PATH=/usr/local/cuda/lib64
rm -f /tmp/clog.txt
LD_PRELOAD=./clog.so CLOG_OUT=/tmp/clog.txt ./lt_test -- 0 > /dev/null 2>&1
python3 - "$1" <<PY
import re, sys
d = {}
for l in open("/tmp/clog.txt"):
    m = re.search(r"occupancy rc=0 numClusters=(\d+) block=128x1x1 smem=232448 cluster=(\d+)x1x1", l)
    if m: d[int(m.group(2))] = int(m.group(1))
mx = max([c for c in d if d[c] > 0] or [0])
print("CTX %s occupancy by cluster size 3..%d: %s" % (sys.argv[1], mx, ",".join(str(d.get(c, 0)) for c in range(3, mx + 1))))
PY
for shp in "1000 16 2048" "2048 64 512" "4096 256 64" "256 1024 1024" "1024 1024 1024"; do
  rm -f /tmp/clog.txt
  r=$(LD_PRELOAD=./clog.so CLOG_OUT=/tmp/clog.txt ./lt_test $shp -- 0 2>&1 | tail -1 | sed -E "s/target=0 +#0 +//; s/ status=0 sync=0 +/ /; s/ bad_rows.*//")
  h=$(head -1 /tmp/x 2>/dev/null); l=$(grep launch /tmp/clog.txt | head -1 | sed -E "s/.*launch rc=0 //; s/ \?$//")
  echo "CTX $1 [$shp] $r | $l"
done
