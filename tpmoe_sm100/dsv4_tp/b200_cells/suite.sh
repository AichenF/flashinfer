#!/bin/bash
# usage: JOB=<hold> suite.sh MODEL TP TAG ROUTING [ARMSETS]  -> suite harness (tpmoe_suite harness/bench.py) with suite_b200 adapters
# 2 reps: rep1 fwd (ours,gc,mega,fimega then fitrt), rep2 rev (fitrt then ours,gc,mega,fimega); logs runs/s_TAG_*.log, results results/b200/<cell>/TAG_*
M=$1; TP=$2; TAG=$3; ROUT=$4
S=/home/scratch.aichenf_wwfo/tpmoe_suite; W=/home/scratch.aichenf_wwfo/suite_b200
MM=$([ $M = dsv41 ] && echo dsv41f || echo $M)
CELL=b200_${MM}_tp${TP}
G=$([ $TP = 4 ] && echo 0,1,2,3 || echo 0,1,2,3,4,5,6,7)
MS=${MS:-8,16,32,64,128,256,384,512}
A1=${A1:-ours,gc,mega,fimega}; A2=${A2-fitrt}
O=$W/results/b200/${MM}_tp${TP}; mkdir -p $O
B="python3 harness/bench.py --machine b200 --model $MM --tp $TP --routing $ROUT --ms $MS --ours-adapter $W/kernels/$CELL --iters ${IT:-50} $XARGS"
cd $S
LOGDIR=$W/runs MACH=sm100 GPUS=$G env/ctr.sh s_${TAG}_1f "$B --arms $A1 --protocol b,a --check 1 --order fwd --out $O/${TAG}_1f"
[ -n "$A2" ] && LOGDIR=$W/runs MACH=sm100 GPUS=$G env/ctr.sh s_${TAG}_1t "$B --arms $A2 --protocol b,a --check 1 --order fwd --out $O/${TAG}_1t"
[ -n "$A2" ] && LOGDIR=$W/runs MACH=sm100 GPUS=$G env/ctr.sh s_${TAG}_2t "$B --arms $A2 --protocol a,b --check 0 --order rev --out $O/${TAG}_2t"
LOGDIR=$W/runs MACH=sm100 GPUS=$G env/ctr.sh s_${TAG}_2f "$B --arms $A1 --protocol a,b --check 0 --order rev --out $O/${TAG}_2f"
echo SUITE_DONE
