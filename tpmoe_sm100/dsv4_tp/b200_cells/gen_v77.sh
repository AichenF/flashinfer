#!/bin/bash
# regenerate v77/tpmoe_b2.cu from the pristine v77 copy (commit ddbfaf9 = tpmoe_b200/src 502e1d2) + the suite_b200 patches
set -e
cd /home/scratch.aichenf_wwfo/suite_b200
git show ddbfaf9:v77/tpmoe_b2.cu > v77/tpmoe_b2.cu
python3 patch_ro.py v77/tpmoe_b2.cu && python3 patch_ndev.py v77/tpmoe_b2.cu && python3 patch_v77large.py v77/tpmoe_b2.cu
