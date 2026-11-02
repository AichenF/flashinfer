"""MMA rate under kernel-like conditions (b2_mma_bench.cu mma_bench5): +3 tcgen05.cp per round, + a concurrent TMA weight stream into smem.
usage: python3 b2_mma_bench5.py --model x --cap x --out x"""
import argparse, os
from pathlib import Path
import torch
SRV = Path(__file__).resolve().parent
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model'); ap.add_argument('--cap'); ap.add_argument('--out'); ap.parse_args()
    from torch.utils.cpp_extension import load
    os.environ['TORCH_CUDA_ARCH_LIST'] = '10.0a'
    (SRV / 'build_mma_bench').mkdir(exist_ok=True)
    X = load(name='b2_mma_bench', sources=[str(SRV / 'b2_mma_bench.cu')], build_directory=str(SRV / 'build_mma_bench'),
             extra_cuda_cflags=['-O3', '-gencode=arch=compute_100a,code=sm_100a', '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK'], extra_ldflags=['-lcuda'], verbose=False)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    w = torch.randint(0, 255, (256 * 1024 * 2, 2048), dtype=torch.uint8, device='cuda')
    clk, rounds = 1.965, 4096
    for n in (8, 32):
        for accm in (0, 1, 2):
            for tma in (0, 1):
                cyc = X.run5(w, n, rounds, 1, tma, sms, accm).double(); torch.cuda.synchronize()
                ns = float(cyc[cyc > 0].median()) / clk / (rounds * 8)
                print(f'N{n:3d} cp1 tma{tma} accm{accm} : {ns:6.1f} ns/MMA  {ns * 8:6.0f} ns/stage', flush=True)
    print('MMA_BENCH5_DONE')
if __name__ == '__main__':
    main()
