#!/bin/bash
# usage: np_sweep.sh <out> <cfg list> [guard]     FP8 / FP4 cublasLtMatmul over every MPS share (host binary, no container)
cd /root/mpscorrupt; export LD_LIBRARY_PATH=/usr/local/cuda/lib64 CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
OUT=$1; CFGS=$2; G=$3
S="1024 16 2048 2048 64 512 256 1024 1024 16 4096 4096 64 2048 1024 2048 2048 512 4096 256 64 128 128 4096"
: > $OUT
for p in $(seq 1 100); do for c in $CFGS; do
  if [ -n "$G" ]; then CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$p LD_PRELOAD=./ltguard_v2.so ./lt_np $c $S >> $OUT 2>&1
  else CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$p ./lt_np $c $S >> $OUT 2>&1; fi
done; done
echo NP_SWEEP_DONE >> $OUT
