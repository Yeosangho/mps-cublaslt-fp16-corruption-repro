#!/bin/bash
# usage: run_cell.sh <image> <label> <pct|nomps> [extra args to mpscorrupt.py]
# nomps = plain full-GPU context (reference). pct = MPS client with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=pct.
IMG=$1; LAB=$2; PCT=$3; shift 3
W=/root/mpscorrupt; OUT=$W/out/$LAB; mkdir -p $OUT
if [ "$PCT" = nomps ]; then
  TAG=nomps
  docker run --rm --name mc_${LAB}_$TAG --gpus all --ipc=host -v $W:/w -e TORCH_HOME=/w/hub \
    $IMG python /w/mpscorrupt.py --out /w/out/$LAB/$TAG "$@" > $OUT/$TAG.log 2>&1
else
  TAG=p$PCT
  docker run --rm --name mc_${LAB}_$TAG --gpus all --ipc=host -v $W:/w -e TORCH_HOME=/w/hub \
    -v /tmp/nvidia-mps:/tmp/nvidia-mps -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
    -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$PCT \
    $IMG python /w/mpscorrupt.py --out /w/out/$LAB/$TAG "$@" > $OUT/$TAG.log 2>&1
fi
echo $? > $OUT/$TAG.rc
