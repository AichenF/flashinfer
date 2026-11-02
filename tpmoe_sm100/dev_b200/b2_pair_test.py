"""B200: cta_group::2 block-scaled MMA operand placement (b2_pair_test.cu).  FC1 tiles T, T + 1 of an expert (256 w13 rows) x N tokens,
B rows split between the pair; SFB per CTA = its own N/2 tokens (mode 0) or all N (mode 1).  Each CTA's D rows are matched exactly
against the fp32 reference (b2_umma_test.lane_match): correct placement = every lane matches a row of its CTA's tile.
usage: python3 b2_pair_test.py --model MODEL [--cap x --out x]"""
import argparse
import sys
from pathlib import Path

import torch

SRV = Path(__file__).resolve().parent
sys.path.insert(0, str(SRV))
import b2_common as C  # noqa: E402
import b2_ref as R  # noqa: E402
import b2_umma_test as U  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--cap'); ap.add_argument('--out')
    args = ap.parse_args()
    dev = torch.device('cuda', 0)
    torch.manual_seed(0)
    from torch.utils.cpp_extension import load
    import os
    os.environ['TORCH_CUDA_ARCH_LIST'] = '10.0a'
    (SRV / 'build_pair_test').mkdir(exist_ok=True)
    X = load(name='b2_pair_test', sources=[str(SRV / 'b2_pair_test.cu')], build_directory=str(SRV / 'build_pair_test'),
             extra_cuda_cflags=['-O3', '-gencode=arch=compute_100a,code=sm_100a', '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK'],
             extra_ldflags=['-lcuda'], verbose=False)
    wm = C.weight_map(args.model)
    raw = U.load_some_experts(args.model, wm, 20, 4, 0, [7], dev)
    lay = C.sglang_trtllm_layer(dict(raw, inter=raw['inter']), dev)
    w13p = lay.w13_weight.data.view(torch.uint8).contiguous()      # [1, 1024, 2048]
    s13p = lay.w13_weight_scale_inv.data.view(torch.uint8).contiguous()
    rows13 = w13p.shape[1]
    W13 = R.dequant_mxfp4(raw['w13'][0], raw['s13'][0])
    chunks = 32
    for N in (16, 32):
        nb = N // 2
        x = (torch.randn(N, 4096, device=dev) * 3).bfloat16()
        xd, img, sfb = U.mx_operand(x, N)          # img [32][N][128]; sfb [16][32][16] lane = token
        ref = W13 @ xd.t()                          # [1024, N]
        for T in (0, 2, 6):
            sfa = torch.stack([s13p[0].reshape(-1)[(T + r) * 32 * 512:(T + r + 1) * 32 * 512] for r in (0, 1)]).contiguous()
            # CTA r's B rows: tokens r nb .. r nb + nb - 1, re-swizzled for their own row index (SW128 granule ^ (row % 8))
            _, img0, sfb0 = U.mx_operand(x[:nb], nb)
            _, img1, sfb1 = U.mx_operand(x[nb:], nb)
            bimg = torch.stack([img0, img1]).contiguous()
            for mode in (0, 1):
                sfbs = torch.stack([sfb0, sfb1]) if mode == 0 else torch.stack([sfb, sfb])
                out = torch.zeros(2, 128, N, dtype=torch.float32, device=dev)
                X.run(w13p.reshape(rows13, 2048), rows13, 2048, sfa, bimg, sfbs.contiguous(), out, T * 128, chunks, N, nb)
                torch.cuda.synchronize()
                res = []
                for r in (0, 1):
                    lanes, merr = U.lane_match(out[r], ref)
                    bad = lanes.count(-1)
                    tile_ok = all(l < 0 or (64 * (T + r) <= l < 64 * (T + r + 1)) or (raw['inter'] + 64 * (T + r) <= l < raw['inter'] + 64 * (T + r + 1))
                                  for l in lanes)
                    res.append(f'CTA{r}: unmatched {bad}/128 tile_rows_ok {tile_ok} best_err {merr:.1e}')
                print(f'N {N} T {T} sfb_mode {mode}: ' + ' | '.join(res), flush=True)
    print('PAIR_TEST_DONE')


if __name__ == '__main__':
    main()
