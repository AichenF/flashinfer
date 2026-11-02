#!/bin/bash
# ptxas resource report of a source on the node (inside the container): b2_ptxas.sh SRC
cd $(dirname $0); /usr/local/cuda/bin/nvcc -std=c++17 -O3 -gencode=arch=compute_100a,code=sm_100a -cubin -o /tmp/p.cubin $1 -DB2_NO_TORCH -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK -Xptxas -v 2>&1 | grep -E "registers|stack|spill|error" 
