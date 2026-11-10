#!/bin/bash
# usage: JOB=<hold> rep3.sh MODEL TP TAG ROUTING -> one more suite rep (ours,gc,mega,fimega; fwd order, check 0) -> TAG_3f
M=$1; TP=$2; TAG=$3; ROUT=$4
S=/home/scratch.aichenf_wwfo/tpmoe_suite; W=/home/scratch.aichenf_wwfo/suite_b200
MM=$([ $M = dsv41 ] && echo dsv41f || echo $M)
CELL=b200_${MM}_tp${TP}
G=$([ $TP = 4 ] && echo 0,1,2,3 || echo 0,1,2,3,4,5,6,7)
O=$W/results/b200/${MM}_tp${TP}
B="python3 harness/bench.py --machine b200 --model $MM --tp $TP --routing $ROUT --ms ${MS:-8,16,32,64,128,256,384,512} --ours-adapter $W/kernels/$CELL --iters ${IT:-50}"
cd $S
LOGDIR=$W/runs MACH=sm100 GPUS=$G env/ctr.sh s_${TAG}_${SUF:-3f} "$B --arms ${A1:-ours,gc,mega,fimega} --protocol b,a --check 0 --order fwd --out $O/${TAG}_${SUF:-3f}"
echo REP3_DONE
