#!/bin/bash
# tp8v41.cu = suite_b200 tail patches (tail/patch_tail.py, copied from /home/scratch.aichenf_wwfo/suite_b200 @ 4c9634b) over port_v41.py's output
set -e
cd /home/scratch.aichenf_wwfo/tp8_v41_b300
python3 port_v41.py /home/scratch.aichenf_wwfo/k3_b300/L/k3h.cu tp8v41_core.cu
python3 tail/patch_tail.py tp8v41_core.cu tp8v41.cu
