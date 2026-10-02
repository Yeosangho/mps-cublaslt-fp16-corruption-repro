#!/bin/bash
# usage: argdump.sh <label> "<m n k>" <target>   prints result + launch + kernel argument words (context from the environment)
cd /root/mpscorrupt; export LD_LIBRARY_PATH=/usr/local/cuda/lib64; rm -f /tmp/clog.txt
r=$(LD_PRELOAD=./clog.so CLOG_OUT=/tmp/clog.txt CLOG_PARAMS=1 ./lt_test $2 -- $3 2>&1 | tail -1 | sed -E "s/target=([0-9]+) +#0 +//; s/ status=0 sync=0 +/ /; s/ bad_rows.*//")
echo "$1 | $r | $(grep launch /tmp/clog.txt | sed -E "s/.*launch rc=0 //; s/ \?$//")"
grep -E "argbuf|params" /tmp/clog.txt | sed -E "s/\[CLOG\] /    /" | cut -c1-700
