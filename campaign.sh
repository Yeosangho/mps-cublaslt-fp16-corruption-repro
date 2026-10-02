#!/bin/bash
# usage: campaign.sh <image> <label> "<pct list>"
# Full-GPU reference first (only if missing), then one MPS cell per percentage, verdict after each cell.
IMG=$1; LAB=$2; PCTS=$3
W=/root/mpscorrupt; V=$W/out/${LAB}_verdicts.txt; MAX_VOID=${MAX_VOID:-2}
cd $W
if [ ! -f out/$LAB/nomps.json ] || ! grep -q '^DONE' out/$LAB/nomps.log; then
  ./mps.sh stop >/dev/null 2>&1
  ./run_cell.sh $IMG $LAB nomps
  docker run --rm -v $W:/w $IMG python /w/cellverdict.py $LAB nomps | tee -a $V
fi
./mps.sh start >/dev/null 2>&1
for p in $PCTS; do
  void=0
  while :; do
    ./run_cell.sh $IMG $LAB $p
    line=$(docker run --rm -v $W:/w $IMG python /w/cellverdict.py $LAB $p)
    echo "$(date +%H:%M:%S) $line" | tee -a $V
    case "$line" in
      *" VOID "*) void=$((void+1)); [ $void -gt $MAX_VOID ] && { echo "$LAB p$p UNRESOLVED" | tee -a $V; break; } ;;
      *) break ;;
    esac
  done
done
echo "CAMPAIGN_DONE $LAB" | tee -a $V
