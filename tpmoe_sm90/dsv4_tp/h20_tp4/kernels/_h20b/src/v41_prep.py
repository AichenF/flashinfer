"""DSV4.1-Flash TP8 rank shards in the kernel's Humming fused-e8m0 layouts (synthetic MXFP4 weights, seeded per global expert).

  main  w13 stacks [gate rows 0..255 | up rows 0..255] of the rank's 288-column slice, K 5120  (items q = 0, 1; = the DSV4-Flash TP8 layout)
  tail  w13 stacks [gate rows 256..287 + 32 zero rows | up rows 256..287 + 32 zero rows]          (item q = 2; one 64-row block per stack)
  w2    [5120 rows] x K 288 padded to 320 (10 k32 slices; Humming needs K % 64 == 0)
Humming's converter is vendored (SGLang prepare_humming_moe_layer's per-sublayer steps, bitwise-checked against the serving image's
own call for the DSV4-Flash shapes); only shape_n_stacks / shape_k / the pad multiples differ.  Needs humming-kernels 0.1.10.
usage: python3 v41_prep.py --out DIR [--ranks 0,1,..] [--model v41|flash] [--check-flash-sglang]"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from humming.layer import HummingInputSchema, HummingMethod
from humming.schema import BaseWeightSchema

MODELS = {'v41': (5120, 2304, 384), 'flash': (4096, 2048, 256)}
SCALE_LO, SCALE_N = 119, 6   # E8M0 2^-8 .. 2^-3 (checkpoint-like)


def expert(e, H, I, dev):
    """global expert e, seeded: W13 packed [2I, H/2] (gate rows then up), E8M0 [2I, H/32], W2 packed [H, I/2], E8M0 [H, I/32]"""
    g = torch.Generator(device=dev).manual_seed(100003 + e)
    p1 = torch.randint(0, 256, (2 * I, H // 2), device=dev, dtype=torch.uint8, generator=g)
    s1 = (SCALE_LO + torch.randint(0, SCALE_N, (2 * I, H // 32), device=dev, generator=g)).to(torch.uint8)
    p2 = torch.randint(0, 256, (H, I // 2), device=dev, dtype=torch.uint8, generator=g)
    s2 = (SCALE_LO + torch.randint(0, SCALE_N, (H, I // 32), device=dev, generator=g)).to(torch.uint8)
    return p1, s1, p2, s2


def tp_raw(H, I, E, ndev, rank, dev):
    """rank's raw slices of every expert: gate [E, i, H/2] + sf, up, w2 [E, H, i/2] + sf (i = I / ndev)"""
    i = I // ndev
    r0, r1 = i * rank, i * (rank + 1)
    out = {k: [] for k in ('g', 'gs', 'u', 'us', 'w2', 'w2s')}
    for e in range(E):
        p1, a1, p2, a2 = expert(e, H, I, dev)
        out['g'].append(p1[r0:r1]); out['gs'].append(a1[r0:r1])
        out['u'].append(p1[I + r0:I + r1]); out['us'].append(a1[I + r0:I + r1])
        out['w2'].append(p2[:, i // 2 * rank:i // 2 * (rank + 1)]); out['w2s'].append(a2[:, i // 32 * rank:i // 32 * (rank + 1)])
    return {k: torch.stack(v).contiguous() for k, v in out.items()}


def humming(weight, scale, stacks, shape_k, E, pad_n, pad_k):
    """one Humming sublayer: weight int8 [E, N, K/2], scale e8m0 [E, N, K/32] (N = sum(stacks)) -> (weight, weight_scale u8, global_scale)"""
    layer = torch.nn.Module()
    layer.register_parameter('x_weight', torch.nn.Parameter(weight.view(torch.int8).contiguous(), requires_grad=False))
    layer.register_parameter('x_weight_scale', torch.nn.Parameter(scale.view(torch.float8_e8m0fnu).contiguous(), requires_grad=False))
    weight_schema = BaseWeightSchema.from_config({'quant_method': 'mxfp4'})
    input_schema = HummingInputSchema.from_config(json.loads(os.environ.get('SGLANG_HUMMING_INPUT_QUANT_CONFIG', '{"dtype":"float8e4m3"}')))
    tensors = {k.removeprefix('x_'): v for k, v in layer.state_dict().items() if k.startswith('x_')}
    shape_n = sum(stacks)
    ws_new, tensors = weight_schema.convert_humming(tensors=tensors, shape_n_stacks=list(stacks), shape_k_stacks=[shape_k], num_experts=E,
                                                    param_dtype=torch.bfloat16)
    for name, _ in list(layer.named_parameters()):
        delattr(layer, name)
    for name, t in tensors.items():
        setattr(layer, f'x_{name}', torch.nn.Parameter(t, requires_grad=False))
    HummingMethod.prepare_layer_meta(layer=layer, shape_n=shape_n, shape_k=shape_k, pad_n_to_multiple=pad_n, pad_k_to_multiple=pad_k,
                                     input_schema=input_schema, weight_schema=ws_new, has_bias=False, num_experts=E, torch_dtype=torch.bfloat16,
                                     sublayer_name='x')
    HummingMethod.transform_humming_layer(layer, sublayer_name='x')
    return (layer.x_weight.data.contiguous(), layer.x_weight_scale.data.view(torch.uint8).contiguous(),
            layer.x_global_scale.data.reshape(-1).float().contiguous())


def raw_from_canonical(tw):
    """tpmoe_suite ctx.tp_w() (w13 [E, 2i, H/2] gate rows then up rows, s13, w2 [E, H, i/2], s2) -> this module's raw slices R"""
    i = tw['w13'].shape[1] // 2
    return dict(g=tw['w13'][:, :i], gs=tw['s13'][:, :i], u=tw['w13'][:, i:], us=tw['s13'][:, i:], w2=tw['w2'], w2s=tw['s2'])


def prep_rank(model, ndev, rank, dev, R=None):
    H, I, E = MODELS[model]
    i = I // ndev
    if R is None:
        R = tp_raw(H, I, E, ndev, rank, dev)
    out = {}
    im = (i // 128) * 128   # main rows per stack (256)
    w13 = torch.cat([R['g'][:, :im], R['u'][:, :im]], 1)
    s13 = torch.cat([R['gs'][:, :im], R['us'][:, :im]], 1)
    out['w1'], out['s1'], out['f1'] = humming(w13, s13, [im, im], H, E, 256, 128)
    if im < i:   # tail item: gate rows im.. + zero rows to 64 | up rows im.. + zero rows (zero codes, scale bytes of the real neighbour)
        nt = i - im
        assert nt <= 64
        z = torch.zeros(E, 64 - nt, H // 2, dtype=torch.uint8, device=dev)
        zg = R['gs'][:, im:im + 1].expand(E, 64 - nt, H // 32)   # (scale rows of a real neighbour: no effect on Humming's exponent window)
        zu = R['us'][:, im:im + 1].expand(E, 64 - nt, H // 32)
        w13t = torch.cat([R['g'][:, im:], z, R['u'][:, im:], z], 1)
        s13t = torch.cat([R['gs'][:, im:], zg, R['us'][:, im:], zu], 1)
        out['w1t'], out['s1t'], out['f1t'] = humming(w13t, s13t, [64, 64], H, E, 64, 128)
        # T4 split tail: ONE 64-row block (gate rows im.. then up rows im.., 32 + 32 for v41) split 4 ways over K: block w (stack w >> 1,
        # row block w & 1) = its k-tiles 4 j + w  ->  a [256, H / 4] matrix, stacks [128, 128]   (TP8 only: nt == 32)
    if im < i and i - im == 32:
        nt = i - im
        b64 = torch.cat([R['g'][:, im:], R['u'][:, im:]], 1)                    # [E, 64, H/2]
        s64 = torch.cat([R['gs'][:, im:], R['us'][:, im:]], 1)                  # [E, 64, H/32]
        nj = H // 512
        w4 = b64.view(E, 64, nj, 4, 64).permute(0, 3, 1, 2, 4).reshape(E, 256, H // 8).contiguous()
        s4 = s64.view(E, 64, nj, 4, 4).permute(0, 3, 1, 2, 4).reshape(E, 256, H // 128).contiguous()
        out['w1t4'], out['s1t4'], out['f1t4'] = humming(w4, s4, [128, 128], H // 4, E, 128, 128)
    if im < i and i - im == 64 and os.environ.get('V41_T2', '1') == '1':
        # T2 split tail (TP4, port_t2.py): stack s (gate | up) = its 64 tail rows split 2 ways over K: rows 0..63 = even k-tiles, 64..127 = odd
        # k-tiles  ->  a [256, H / 2] matrix, stacks [128, 128] (passed as w1t4 / s1t4 / f1t4)
        nj = H // 256
        parts_w, parts_s = [], []
        for k in ('g', 'u'):
            w64, s64 = R[k][:, im:], R[k + 's'][:, im:]                      # [E, 64, H/2], [E, 64, H/32]
            parts_w.append(w64.reshape(E, 64, nj, 2, 64).permute(0, 3, 1, 2, 4).reshape(E, 128, H // 4))
            parts_s.append(s64.reshape(E, 64, nj, 2, 4).permute(0, 3, 1, 2, 4).reshape(E, 128, H // 64))
        w2t = torch.cat(parts_w, 1).contiguous()
        s2t = torch.cat(parts_s, 1).contiguous()
        out['w1t4'], out['s1t4'], out['f1t4'] = humming(w2t, s2t, [128, 128], H // 2, E, 128, 128)
    out['w2'], out['s2'], out['f2'] = humming(R['w2'], R['w2s'], [H], i, E, 256, 64)   # (Humming: K multiple of 64 -> 288 + 32 zero columns = 10 slices)
    return out, R


def main():
    os.environ.setdefault('SGLANG_HUMMING_INPUT_QUANT_CONFIG', '{"dtype":"float8e4m3"}')
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model', default='v41', choices=list(MODELS))
    ap.add_argument('--ndev', type=int, default=8)
    ap.add_argument('--ranks', default='')
    ap.add_argument('--save-raw', action='store_true', help='also save the raw MXFP4 slices (reference)')
    args = ap.parse_args()
    dev = torch.device('cuda', 0)
    torch.cuda.set_device(dev)
    args.out.mkdir(parents=True, exist_ok=True)
    ranks = [int(r) for r in args.ranks.split(',')] if args.ranks else list(range(args.ndev))
    man = {}
    for rank in ranks:
        w, R = prep_rank(args.model, args.ndev, rank, dev)
        w = {k: v.cpu() for k, v in w.items()}
        torch.save(w, args.out / f'rank{rank}.pt')
        if args.save_raw:
            torch.save({k: v.cpu() for k, v in R.items()}, args.out / f'raw{rank}.pt')
        h = hashlib.sha256()
        for k in sorted(w):
            h.update(w[k].contiguous().view(torch.uint8).numpy().tobytes())
        man[rank] = dict(shapes={k: [list(v.shape), str(v.dtype)] for k, v in w.items()}, content_sha256=h.hexdigest())
        print(rank, json.dumps(man[rank]), flush=True)
        torch.cuda.empty_cache()
    (args.out / f'manifest_{args.model}_{"_".join(map(str, ranks))}.json').write_text(json.dumps(man, indent=1))
    print('PREP_DONE')


if __name__ == '__main__':
    main()
