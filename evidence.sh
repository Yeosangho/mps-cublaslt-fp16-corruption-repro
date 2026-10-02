#!/bin/bash
# Regenerates the C-level evidence quoted in the README (sections 3 and 5) on the host (no container).
# Needs: MPS control daemon running (./mps.sh start), lt_test, cluster_probe, clog.so, smspoof.so built.
cd "$(dirname "$0")"; export LD_LIBRARY_PATH=/usr/local/cuda/lib64
MPS=/tmp/nvidia-mps; NOMPS=/nonexistent
T="0 8 10 12 14 16 18 20 22 24 26 28 30 32 36 40 48 64 132"
run() { echo; echo "### $1"; shift; "$@" 2>&1 | cut -c1-200; }

run "section 3: MPS 12 % / 14 %, default heuristic" true
for p in 12 14; do CUDA_MPS_PIPE_DIRECTORY=$MPS CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$p ./lt_test -- 0; done
run "section 3: MPS 40 %, 2048 x 2048 x 2048" env CUDA_MPS_PIPE_DIRECTORY=$MPS CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40 ./lt_test 2048 2048 2048 -- 0

run "5.2 SM_COUNT_TARGET sweep, full GPU without MPS" env CUDA_MPS_PIPE_DIRECTORY=$NOMPS ./lt_test -- $T
run "5.2 SM_COUNT_TARGET sweep, MPS 14 %" env CUDA_MPS_PIPE_DIRECTORY=$MPS CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=14 ./lt_test -- $T
run "5.2 SM_COUNT_TARGET sweep, MPS 12 %" env CUDA_MPS_PIPE_DIRECTORY=$MPS CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=12 ./lt_test -- $T

run "5.2 kernel launches (clog.so): full GPU target 18 / MPS 14 % / MPS 12 % / MPS 14 % target 12" true
for c in "$NOMPS - 18" "$MPS 14 0" "$MPS 12 0" "$MPS 14 12"; do
  set -- $c; rm -f /tmp/clog.txt
  CUDA_MPS_PIPE_DIRECTORY=$1 CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$2 LD_PRELOAD=./clog.so CLOG_OUT=/tmp/clog.txt ./lt_test -- $3 | tail -1 | cut -c1-160
  grep launch /tmp/clog.txt | cut -c1-160
done

occtab() {   # cluster occupancy answers cuBLASLt receives in a given context
  rm -f /tmp/clog.txt
  CUDA_MPS_PIPE_DIRECTORY=$1 CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$2 LD_PRELOAD=./clog.so CLOG_OUT=/tmp/clog.txt ./lt_test -- 0 >/dev/null
  python3 - <<'EOF'
import re
d = {}
for l in open("/tmp/clog.txt"):
    m = re.search(r"occupancy rc=0 numClusters=(\d+) block=128x1x1 smem=232448 cluster=(\d+)x1x1", l)
    if m: d[int(m.group(2))] = int(m.group(1))
mx = max(c for c in d if d[c] > 0)
print(",".join(str(d.get(c, 0)) for c in range(1, mx + 1)))
EOF
}
FULL=$(occtab $NOMPS 100); M18=$(occtab $MPS 14); M14=$(occtab $MPS 12)
echo; echo "### 5.2 cuOccupancyMaxActiveClusters answers by cluster size (1,2,3,...; sizes not queried = 0)"
echo "full GPU : $FULL"; echo "MPS 14 % : $M18"; echo "MPS 12 % : $M14"
L() { ./lt_test -- 0 2>&1 | tail -1 | cut -c1-150; }
echo; echo "### 5.2 swapping the reported SM count / cluster occupancy (smspoof.so)"
echo "full GPU, SM=18 + occupancy(MPS 14 %):"; CUDA_MPS_PIPE_DIRECTORY=$NOMPS LD_PRELOAD=./smspoof.so SPOOF_SM=18 SPOOF_OCC=$M18 L
echo "full GPU, SM=18 only:";                  CUDA_MPS_PIPE_DIRECTORY=$NOMPS LD_PRELOAD=./smspoof.so SPOOF_SM=18 L
echo "full GPU, occupancy(MPS 14 %) only:";    CUDA_MPS_PIPE_DIRECTORY=$NOMPS LD_PRELOAD=./smspoof.so SPOOF_OCC=$M18 L
export CUDA_MPS_PIPE_DIRECTORY=$MPS CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=14
echo "MPS 14 %, SM=14 only:";                  LD_PRELOAD=./smspoof.so SPOOF_SM=14 L
echo "MPS 14 %, SM=132 only:";                 LD_PRELOAD=./smspoof.so SPOOF_SM=132 L
echo "MPS 14 %, occupancy(full GPU) only:";    LD_PRELOAD=./smspoof.so SPOOF_OCC=$FULL L
unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE

echo; echo "### 5.3 cluster_probe (grid gx gy, cluster cx cy, threads, dynamic shared memory; 20 repetitions)"
probe() { for a in "4 4 4 1 384 180692" "4 3 4 1 384 221460" "4 4 2 2 384 180692" "8 2 8 1 128 180692" "16 4 4 1 128 1024"; do ./cluster_probe $a 20; done; }
CUDA_MPS_PIPE_DIRECTORY=$NOMPS probe 2>&1 | cut -c1-200
for p in 12 14 16 20 40; do CUDA_MPS_PIPE_DIRECTORY=$MPS CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$p probe 2>&1 | cut -c1-200; done

echo; echo "### 5.4 cluster shape override through cublasLtMatmulAlgoConfigSetAttribute (LT_CLUSTER: 2=1x1x1 3=2x1x1 5=1x2x1)"
for p in 14 40; do for cs in "" 2 3 5; do
  echo "MPS $p % LT_CLUSTER=${cs:-as selected}:"; CUDA_MPS_PIPE_DIRECTORY=$MPS CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$p LT_CLUSTER=$cs ./lt_test -- 0 2>&1 | tail -n +2 | cut -c1-160
done; done
echo; echo "### candidates returned by the heuristic at MPS 14 % (LT_ENUM=24)"
CUDA_MPS_PIPE_DIRECTORY=$MPS CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=14 LT_ENUM=24 ./lt_test -- 0 2>&1 | cut -c1-160
