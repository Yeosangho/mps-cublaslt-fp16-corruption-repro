#!/bin/bash
# For each MIG profile: create one instance, run the reproductions inside it WITHOUT MPS, destroy it.
cd /root/mpscorrupt; IMG=pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime; mkdir -p out/xscan/mig
for prof in "9 3g.71gb" "15 1g.35gb" "19 1g.18gb" "14 2g.35gb" "5 4g.71gb" "0 7g.141gb"; do
  set -- $prof; id=$1; name=$2
  nvidia-smi mig -cgi $id -C > /dev/null 2>&1 || { echo "MIG $name create failed"; continue; }
  dev=$(nvidia-smi -L | grep -o "MIG-[0-9a-f-]*" | head -1)
  R="docker run --rm --gpus \"device=$dev\" --ipc=host -v /root/mpscorrupt:/w -e TORCH_HOME=/w/hub -e OMP_NUM_THREADS=8"
  eval $R $IMG python /w/repro_min.py 2>&1 | grep "^SM=" | sed "s/^/MIG $name /"
  eval $R $IMG python /w/xscan.py /w/out/xscan/mig/$name.json 2>&1 | grep -E "XSCAN|Error" | sed "s/^/MIG $name /"
  eval $R $IMG python /w/backbone.py check /w/out/backbone/ref.npz 2>&1 | grep -E "BACKBONE|Error" | sed "s/^/MIG $name /"
  eval $R -e LD_PRELOAD=/w/ltguard_v2.so -e LTGUARD_LOG=1 $IMG python /w/xscan.py /w/out/xscan/mig/${name}_guard.json 2>&1 | grep -E "XSCAN|Error|LTGUARD" | sed "s/^/MIG $name guard /"
  nvidia-smi mig -dci > /dev/null 2>&1; nvidia-smi mig -dgi > /dev/null 2>&1
done
echo MIG_TEST_DONE
