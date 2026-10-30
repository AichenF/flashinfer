"""B200 unit test: tcgen05 block-scaled MXFP4 x MXFP8 GEMM (b2_umma_test.cu) on SGLang's trtllm-gen expert layout read in place,
for N = 8..32 token columns, FC1 (w13 tiles, K 4096) and FC2 (w2 tiles, K 512).  Also derives the layout's lane tables: which
checkpoint row (w1 gate / w3 up / w2 output row) each TMEM lane of a 128-row tile holds -- by matching every lane's exact dot
products against all rows of the fp32 reference (unique match required).
usage (B200, serving image): python3 b2_umma_test.py --model MODEL [--experts 0 7 200] [--ns 8 16 24 32]"""
import argparse
import json
import sys
from pathlib import Path

import torch

SRV = Path(__file__).resolve().parent
sys.path.insert(0, str(SRV))
import b2_common as C  # noqa: E402
import b2_ref as R  # noqa: E402


def ext():
    import os
    from torch.utils.cpp_extension import load
    os.environ['TORCH_CUDA_ARCH_LIST'] = '10.0a'
    build = SRV / 'build_umma_test'
    build.mkdir(exist_ok=True)
    return load(name='b2_umma_test', sources=[str(SRV / 'b2_umma_test.cu')], build_directory=str(build),
                extra_cuda_cflags=['-O3', '-gencode=arch=compute_100a,code=sm_100a', '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK'],
                extra_ldflags=['-lcuda'], verbose=False)


def load_some_experts(model, wm, index, ndev, rank, experts, dev):
    """load_experts_raw restricted to a few experts (same slicing)"""
    from safetensors import safe_open
    model = Path(model)
    inter = C.INTER_ALL // ndev
    r0, r1 = inter * rank, inter * (rank + 1)
    out = dict(w13=[], s13=[], w2=[], s2=[], inter=inter)
    p = f'layers.{index}.ffn.experts'

    def get(k, sl):
        with safe_open(str(model / wm[k]), framework='pt', device='cpu') as f:
            s = f.get_slice(k)
            return s[sl] if sl is not None else s[:]
    for e in experts:
        w1, w3 = get(f'{p}.{e}.w1.weight', slice(r0, r1)), get(f'{p}.{e}.w3.weight', slice(r0, r1))
        s1, s3 = get(f'{p}.{e}.w1.scale', slice(r0, r1)), get(f'{p}.{e}.w3.scale', slice(r0, r1))
        w2 = get(f'{p}.{e}.w2.weight', None)[:, (inter // 2) * rank:(inter // 2) * (rank + 1)]
        s2 = get(f'{p}.{e}.w2.scale', None)[:, (inter // 32) * rank:(inter // 32) * (rank + 1)]
        out['w13'].append(torch.cat([w1, w3])); out['s13'].append(torch.cat([s1, s3]))
        out['w2'].append(w2); out['s2'].append(s2)
    for k in ('w13', 's13', 'w2', 's2'):
        t = torch.stack(out[k]).contiguous().to(dev)
        out[k] = t.view(torch.int8) if k.startswith('w') else t.float()
    return out


def mx_operand(x, npad):
    """x [N, K] -> (dequantized fp32 [N, K], SW128 image [K/128][npad][128] uint8, SFB images [K/256][32][16] uint8)"""
    N, K = x.shape
    xd, ex = R.quant_mxfp8(x)
    s = torch.exp2(ex.float()).repeat_interleave(32, dim=-1)
    codes = (xd / s).to(torch.float8_e4m3fn).view(torch.uint8)   # exact: xd / 2^e is an e4m3 value
    nch = K // 128
    img = torch.zeros(nch, npad, 128, dtype=torch.uint8, device=x.device)
    rows = torch.arange(N, device=x.device)
    for c in range(nch):
        blk = codes[:, 128 * c:128 * (c + 1)].reshape(N, 8, 16)   # 16-B granules
        g = torch.arange(8, device=x.device)
        dst = (g.unsqueeze(0) ^ (rows % 8).unsqueeze(1))           # swizzled granule slot
        tile = torch.zeros(N, 8, 16, dtype=torch.uint8, device=x.device)
        tile.scatter_(1, dst.unsqueeze(-1).expand(N, 8, 16), blk)
        img[c, :N] = tile.reshape(N, 128)
    sfb = torch.zeros(nch // 2 if nch > 1 else 1, 32, 16, dtype=torch.uint8, device=x.device)
    e8 = (ex + 127).to(torch.uint8)   # [N, K/32]
    for c in range(nch):
        for b in range(4):
            sfb[c >> 1, :N, 8 * (c & 1) + b] = e8[:, 4 * c + b]
    return xd, img.contiguous(), sfb.contiguous()


def lane_match(D, ref, tol=2e-5):
    """D [128, N], ref [rows, N]: per lane the unique reference row with a relative match within tol (else -1)"""
    d = D.double()
    r = ref.double()
    err = (d.unsqueeze(1) - r.unsqueeze(0)).abs().amax(-1) / r.abs().amax(-1).clamp_min(1e-30).unsqueeze(0)   # [128, rows]
    best = err.argmin(1)
    berr = err.gather(1, best.unsqueeze(1)).squeeze(1)
    nmatch = (err <= tol).sum(1)
    ok = (berr <= tol) & (nmatch == 1)
    return [int(b) if o else -1 for b, o in zip(best.tolist(), ok.tolist())], float(berr.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--layer', type=int, default=20)
    ap.add_argument('--rank', type=int, default=0)
    ap.add_argument('--experts', type=int, nargs='+', default=[0, 77, 255])
    ap.add_argument('--ns', type=int, nargs='+', default=[8, 16, 24, 32])
    ap.add_argument('--cap', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    dev = torch.device('cuda', 0)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    X = ext()
    wm = C.weight_map(args.model)
    raw = load_some_experts(args.model, wm, args.layer, 4, args.rank, args.experts, dev)
    E = len(args.experts)
    lay = C.sglang_trtllm_layer(dict(raw, inter=raw['inter']), dev)
    w13p = lay.w13_weight.data.view(torch.uint8).contiguous()      # [E, 1024, 2048]
    s13p = lay.w13_weight_scale_inv.data.view(torch.uint8).contiguous()   # [E, 1024, 128]
    w2p = lay.w2_weight.data.view(torch.uint8).contiguous()        # [E, 4096, 256]
    s2p = lay.w2_weight_scale_inv.data.view(torch.uint8).contiguous()     # [E, 4096, 16]
    print('processed', tuple(w13p.shape), tuple(s13p.shape), tuple(w2p.shape), tuple(s2p.shape), flush=True)
    rows13, inter = w13p.shape[1], raw['inter']
    report = dict(fc1={}, fc2={})
    gate_tab, up_tab, fc2_tab = None, None, None
    for ei in range(E):
        W13 = R.dequant_mxfp4(raw['w13'][ei], raw['s13'][ei])   # [1024, 4096] rows [w1; w3]
        W2 = R.dequant_mxfp4(raw['w2'][ei], raw['s2'][ei])      # [4096, 512]
        for N in args.ns:
            npad = (N + 7) // 8 * 8
            x = (torch.randn(N, 4096, device=dev) * 3).bfloat16()
            xd, img, sfb = mx_operand(x, npad)
            ref = W13 @ xd.t()   # [1024, N]
            for T in range(rows13 // 128):
                out = torch.zeros(128, npad, dtype=torch.float32, device=dev)
                sfa = s13p[ei].reshape(-1)[T * 32 * 512:(T + 1) * 32 * 512].contiguous()
                X.run(w13p.reshape(E * rows13, 2048), E * rows13, 2048, 5, sfa, img, sfb, out, ei * rows13 + T * 128, 32, npad)
                torch.cuda.synchronize()
                lanes, merr = lane_match(out[:, :N], ref)
                if -1 in lanes:
                    report['fc1'][f'e{args.experts[ei]} N{N} T{T}'] = dict(bad_lanes=lanes.count(-1), max_err=merr)
                    print(f'FC1 e{args.experts[ei]} N{N} T{T}: {lanes.count(-1)} unmatched lanes (max best err {merr:.2e})', flush=True)
                    continue
                g = [None] * 64
                u = [None] * 64
                for lane, row in enumerate(lanes):
                    if row < inter:
                        j = row - 64 * T
                        assert 0 <= j < 64, (T, lane, row)
                        g[j] = lane
                    else:
                        j = row - inter - 64 * T
                        assert 0 <= j < 64, (T, lane, row)
                        u[j] = lane
                assert None not in g and None not in u, (T, g, u)
                if gate_tab is None:
                    gate_tab, up_tab = g, u
                assert (g, u) == (gate_tab, up_tab), f'lane tables differ at e{args.experts[ei]} T{T}'
            report['fc1'][f'e{args.experts[ei]} N{N}'] = 'ok'
            print(f'FC1 e{args.experts[ei]} N{N}: all {rows13 // 128} tiles exact, same lane tables', flush=True)
            # FC2: tokens of h [N, 512]
            h = (torch.randn(N, inter, device=dev)).bfloat16()
            hd, himg, hsfb = mx_operand(h, npad)
            ref2 = W2 @ hd.t()   # [4096, N]
            for T in range(4096 // 128):
                out = torch.zeros(128, npad, dtype=torch.float32, device=dev)
                sfa = s2p[ei].reshape(-1)[T * 4 * 512:(T + 1) * 4 * 512].contiguous()
                X.run(w2p.reshape(E * 4096, 256), E * 4096, 256, 5, sfa, himg, hsfb, out, ei * 4096 + T * 128, 4, npad)
                torch.cuda.synchronize()
                lanes, merr = lane_match(out[:, :N], ref2)
                lanes = [l - 128 * T if l >= 0 else -1 for l in lanes]
                if -1 in lanes:
                    report['fc2'][f'e{args.experts[ei]} N{N} T{T}'] = dict(bad_lanes=lanes.count(-1), max_err=merr)
                    print(f'FC2 e{args.experts[ei]} N{N} T{T}: {lanes.count(-1)} unmatched lanes (max best err {merr:.2e})', flush=True)
                    continue
                if fc2_tab is None:
                    fc2_tab = lanes
                assert lanes == fc2_tab, f'FC2 lane table differs at e{args.experts[ei]} T{T}'
            report['fc2'][f'e{args.experts[ei]} N{N}'] = 'ok'
            print(f'FC2 e{args.experts[ei]} N{N}: all 32 tiles exact, same lane table', flush=True)
    report['gate_lane'], report['up_lane'], report['fc2_row_of_lane'] = gate_tab, up_tab, fc2_tab
    print('GATE_LANE', gate_tab)
    print('UP_LANE', up_tab)
    print('FC2_ROW_OF_LANE', fc2_tab)
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / 'umma_test.json').write_text(json.dumps(report, indent=1) + '\n')
    print('UMMA_TEST_DONE')


if __name__ == '__main__':
    main()
