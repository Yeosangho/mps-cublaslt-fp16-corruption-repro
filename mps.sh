#!/bin/bash
# usage: mps.sh start|stop|status
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps
case "$1" in
  start)
    mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
    pgrep -x nvidia-cuda-mps >/dev/null || pgrep -f '^nvidia-cuda-mps-control' >/dev/null || nvidia-cuda-mps-control -d
    sleep 1; pgrep -af '^nvidia-cuda-mps' ;;
  stop)
    pgrep -f '^nvidia-cuda-mps-control' >/dev/null && echo quit | nvidia-cuda-mps-control
    sleep 2; pgrep -af '^nvidia-cuda-mps' || echo "mps stopped" ;;
  status)
    pgrep -af '^nvidia-cuda-mps' || echo "mps not running" ;;
esac
