"""i-cache retention across back-to-back launches: per-CTA cycles of a straight-line block, launch 1 vs 2 vs 3 (b2_icache.cu).
usage: python3 b2_icache.py --model x --cap x --out x"""
import argparse, os
from pathlib import Path
import torch
SRV = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument('--model'); ap.add_argument('--cap'); ap.add_argument('--out'); ap.parse_args()
from torch.utils.cpp_extension import load
os.environ['TORCH_CUDA_ARCH_LIST'] = '10.0a'
(SRV / 'build_ic').mkdir(exist_ok=True)
X = load(name='b2_icache', sources=[str(SRV / 'b2_icache.cu')], build_directory=str(SRV / 'build_ic'), extra_cuda_cflags=['-O3', '-gencode=arch=compute_100a,code=sm_100a'], verbose=False)
for nu in (256, 1024, 3072, 6144, 12288):
    # cold: a different kernel (another size) ran in between; then 3 back-to-back launches
    other = 256 if nu != 256 else 1024
    X.run(other, 1); torch.cuda.synchronize()
    r = []
    for i in range(3):
        o = X.run(nu, i); torch.cuda.synchronize()
        r.append(float(o.double().median()))
    kb = nu * 8 * 16 / 1024
    print(f'block ~{kb:6.0f} KB of code: cycles launch1 {r[0]:9.0f}  launch2 {r[1]:9.0f}  launch3 {r[2]:9.0f}  ratio 1/2 {r[0] / r[1]:.2f}', flush=True)
print('IC_DONE')
