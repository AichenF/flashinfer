"""Weight-stream ceiling on B200 without MMA (b2_tma_bench.cu): GB/s by ring depth NS, chunks per stage KC, 16U4 (unpacked FP4) vs UINT8,
L2 cache hint and promotion.  usage: python3 b2_tma_bench.py --model x --cap x --out x"""
import argparse
import os
from pathlib import Path

import torch

SRV = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model'); ap.add_argument('--cap'); ap.add_argument('--out')
    ap.parse_args()
    from torch.utils.cpp_extension import load
    os.environ['TORCH_CUDA_ARCH_LIST'] = '10.0a'
    (SRV / 'build_tma_bench').mkdir(exist_ok=True)
    X = load(name='b2_tma_bench', sources=[str(SRV / 'b2_tma_bench.cu')], build_directory=str(SRV / 'build_tma_bench'),
             extra_cuda_cflags=['-O3', '-gencode=arch=compute_100a,code=sm_100a', '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK'], extra_ldflags=['-lcuda'],
             verbose=False)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    w = torch.randint(0, 255, (256 * 1024 * 2, 2048), dtype=torch.uint8, device='cuda')   # 1 GB (> L2)
    print('kind KC NS grid hint promo : GB/s', flush=True)
    for kind in (0, 1):
        for kc in (1, 2, 4):
            for ns in (2, 3, 4, 6, 8, 12):
                sbytes = kc * (16384 if kind == 0 else 8192)
                if ns * sbytes + 2048 > 232448:
                    continue
                gbs = X.run(w, kind, kc, ns, sms, 1, 2)
                print(f'{"16U4" if kind == 0 else "UINT8"} KC{kc} NS{ns:2d} smem/stage {sbytes // 1024:3d} KB inflight/SM {ns * kc * 8:4d} KB : {gbs:7.0f} GB/s', flush=True)
    for hint in (0, 1):
        for promo in (0, 1, 2):
            gbs = X.run(w, 0, 2, 4, sms, hint, promo)
            print(f'16U4 KC2 NS4 hint {"evict_first" if hint else "evict_normal"} promo {promo} : {gbs:7.0f} GB/s', flush=True)
    print('TMA_BENCH_DONE')


if __name__ == '__main__':
    main()
