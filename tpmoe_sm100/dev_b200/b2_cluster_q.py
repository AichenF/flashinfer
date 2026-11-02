"""max active clusters (sizes 1, 2, 4, 8, 16) for a 1-CTA/SM kernel with the tpmoe_b2 smem.  usage: python3 b2_cluster_q.py --model x --cap x --out x"""
import argparse, os
from pathlib import Path
import torch
SRV = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument('--model'); ap.add_argument('--cap'); ap.add_argument('--out'); ap.parse_args()
from torch.utils.cpp_extension import load
os.environ['TORCH_CUDA_ARCH_LIST'] = '10.0a'
(SRV / 'build_cq').mkdir(exist_ok=True)
X = load(name='b2_cluster_q', sources=[str(SRV / 'b2_cluster_q.cu')], build_directory=str(SRV / 'build_cq'), extra_cuda_cflags=['-O3', '-gencode=arch=compute_100a,code=sm_100a'], verbose=False)
for smem in (200 * 1024, 227 * 1024):
    print(f'smem {smem // 1024} KB: max active clusters for sizes 1/2/4/8/16:', X.query(smem), flush=True)
print('CQ_DONE')
