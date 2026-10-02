#!/bin/bash
# Full GPU (no MPS). Only the SM count reported by cuDeviceGetAttribute is overridden (smspoof.so); for each
# reported SM count run FP16 cublasLtMatmul for a few shapes and record the chosen algorithm + correctness.
cd /root/mpscorrupt; export LD_LIBRARY_PATH=/usr/local/cuda/lib64 CUDA_MPS_PIPE_DIRECTORY=/nonexistent
for sm in $(seq 1 132); do
  for shp in "1000 16 2048" "2048 64 512" "128 4 4096" "4096 256 64"; do
    r=$(LD_PRELOAD=./smspoof.so SPOOF_SM=$sm ./lt_test $shp -- 0 2>&1 | tail -1)
    echo "sm=$sm shape=[$shp] $r"
  done
done
echo SPOOF_SWEEP_DONE
