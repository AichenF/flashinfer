#!/bin/bash
# usage: JOB=<hold> final_cells.sh TP MODEL:TAG [MODEL:TAG..] -> per cell: b2b correctness gate (my harness, every M) then suite 3 reps (1f, 2f, 3f; trt reused)
TP=$1; shift
cd /home/scratch.aichenf_wwfo/suite_b200
G=$([ $TP = 4 ] && echo 0,1,2,3 || echo 0,1,2,3,4,5,6,7)
for mt in "$@"; do
  M=${mt%%:*}; TAG=${mt##*:}
  T=real-trace:/home/scratch.aichenf_wwfo/routing_caps/$([ $M = dsv41 ] && echo dsv41_real || echo dsv4f_real)
  GPUS=$G ./sctr.sh gate_$TAG python sb.py --model $M --tp $TP --ms 8 16 32 64 128 256 384 512 --arms ours --b2b 12 --b2b-reps 4 --check 1 --routing trace --proto b --cold-iters 5
  A2= ./suite.sh $M $TP $TAG $T
  ./rep3.sh $M $TP $TAG $T
done
echo FINAL_DONE
