"""b2_mma_bench7.cu: clean templated M128 block-scaled MMA rate, current FC1 shape vs swapped (tokens as A, weights as B N=128/256).
usage: python3 b2_mma_bench7.py --model x --cap x --out x"""
import argparse
import os
import subprocess
from pathlib import Path

import torch

SRV = Path(__file__).resolve().parent
SHAPES = {0: ('cur  A=w N8   per8', 128), 1: ('cur  A=w N32  per8', 128), 2: ('cur  A=w N8   per4', 128),
          3: ('swap B=w N128 per8', 128), 4: ('swap B=w N256 per8', 256), 5: ('swap B=w N256 per4', 256), 6: ('cur  A=w N128 per8', 128)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model'); ap.add_argument('--cap'); ap.add_argument('--out')
    ap.parse_args()
    from torch.utils.cpp_extension import load
    os.environ['TORCH_CUDA_ARCH_LIST'] = '10.0a'
    bd = SRV / 'build_mma_bench7'
    bd.mkdir(exist_ok=True)
    X = load(name='b2_mma_bench7', sources=[str(SRV / 'b2_mma_bench7.cu')], build_directory=str(bd),
             extra_cuda_cflags=['-O3', '-gencode=arch=compute_100a,code=sm_100a', '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK'], verbose=False)
    so = next(bd.glob('*.so'))
    sass = subprocess.run(['cuobjdump', '-sass', str(so)], capture_output=True, text=True).stdout
    fn = None
    for line in sass.splitlines():
        if 'Function :' in line: fn = line.split('Function :')[1].strip(); nq = nr = 0
        if fn and 'UTCQMMA' in line: nq += 1
        if fn and 'R2UR' in line: nr += 1
        if fn and 'EXIT' in line and 'BRA' not in line: print(f'SASS {fn[:60]}: UTCQMMA {nq} R2UR {nr}', flush=True); fn = None
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    rounds = 4000
    for rep in range(2):
        for depth in (1, 2, 4):
            for s, (tag, rows) in SHAPES.items():
                per = 4 if 'per4' in tag else 8
                cyc = X.run(s, rounds, depth, sms).double()
                torch.cuda.synchronize()
                c = float(cyc.median()) / (rounds * per)
                print(f'{tag} depth{depth} grid{sms}: {c:6.1f} clk/MMA  {c * 128 / rows:6.1f} clk per 128 weight rows x K32  ({c * per:6.0f} clk/round)', flush=True)
    print('MMA_BENCH_DONE')


if __name__ == '__main__':
    main()
