#!/bin/bash
# usage: scan_campaign.sh <image> <label> "<pct list>"
# env: W (work dir, default dir of this script), GPUS (docker --gpus value, default all),
#      PIPE (MPS pipe dir, default /tmp/nvidia-mps), EXTRA (extra docker args, e.g. "-e LD_PRELOAD=/w/smspoof.so"),
#      J (parallel MPS clients, default 4)
IMG=$1; LAB=$2; PCTS=$3
W=${W:-$(cd "$(dirname "$0")" && pwd)}; GPUS=${GPUS:-all}; PIPE=${PIPE:-/tmp/nvidia-mps}; J=${J:-4}
mkdir -p $W/out/scan/$LAB
cell() {
  p=$1
  docker run --rm --gpus "$GPUS" --ipc=host -v $W:/w -v $PIPE:$PIPE -e CUDA_MPS_PIPE_DIRECTORY=$PIPE \
    -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$p -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 $EXTRA $IMG \
    sh -c "python /w/scan.py run /w/out/scan/$LAB/p$p && python /w/scan.py parse /w/out/scan/$LAB/p$p" 2>&1 | grep -E "^SCAN|Error|error" | sed "s/^/$LAB p$p /"
}
export -f cell; export IMG LAB W GPUS PIPE EXTRA
echo $PCTS | tr ' ' '\n' | xargs -P $J -I{} bash -c 'cell {}'
echo "SCAN_DONE $LAB"
