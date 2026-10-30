"""Which reference numerics does SGLang's B200 routed MoE (trtllm-gen MXFP4 x MXFP8) follow?  One GPU, rank-0 shard of layer 20,
captured tokens: the production routed output vs b2_ref variants (routing weight before / after the h quantization, bf16 gate/up,
the h scale rule).  usage: python3 b2_refcheck.py --model MODEL --cap CAP --out OUT"""
import argparse
import sys
from pathlib import Path

import torch

SRV = Path(__file__).resolve().parent
sys.path.insert(0, str(SRV))
import b2_common as C  # noqa: E402
import b2_ref as R  # noqa: E402


def variant(x, ids, w, raw, xq, w_pre=False, w_bf16=True, gu_bf16=False, act_bf16=False, floor_scale=False, out_bf16_slot=False):
    M = x.shape[0]
    inter = raw['inter']
    wb = w.to(torch.bfloat16).float() if w_bf16 else w.float()
    out = torch.zeros((M, x.shape[1]), dtype=torch.float32, device=x.device)

    def q8(a):
        if not floor_scale:
            return R.quant_mxfp8(a)[0]
        sh = a.shape
        ab = a.float().reshape(*sh[:-1], sh[-1] // 32, 32)
        amax = ab.abs().amax(-1, keepdim=True)
        e = (torch.floor(torch.log2(amax.clamp_min(1e-38))) - 8).clamp(-127, 127)   # OCP MX: 2^(floor(log2 amax) - emax)
        s = torch.exp2(e)
        return ((ab / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s).reshape(sh)
    for e in torch.unique(ids).tolist():
        tok, slot = (ids == e).nonzero(as_tuple=True)
        y = xq[tok] @ R.dequant_mxfp4(raw['w13'][e], raw['s13'][e]).t()
        g, u = y[:, :inter], y[:, inter:]
        if gu_bf16:
            g, u = g.bfloat16().float(), u.bfloat16().float()
        act = R.swiglu_clamp(g, u)
        if w_pre:
            act = act * wb[tok, slot].unsqueeze(1)
        if act_bf16:
            act = act.bfloat16().float()
        h = q8(act)
        o = h @ R.dequant_mxfp4(raw['w2'][e], raw['s2'][e]).t()
        if out_bf16_slot:
            o = o.bfloat16().float()
        out.index_add_(0, tok, o if w_pre else o * wb[tok, slot].unsqueeze(1))
    return out.bfloat16()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--cap', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ms', type=int, nargs='+', default=[16, 128])
    args = ap.parse_args()
    dev = torch.device('cuda', 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    wm = C.weight_map(args.model)
    raw = C.load_experts_raw(args.model, wm, 20, 4, 0, dev)
    layer = C.sglang_trtllm_layer(raw, dev)
    router, bias = C.load_router(args.model, wm, 20, dev)
    P = C.Prod(layer, dict(inter=raw['inter']), router, bias, None, None)
    xs = C.capture_pool(args.cap, args.ms, dev)
    from flashinfer import mxfp8_quantize
    for m in args.ms:
        x = xs[m]
        lg = P.logits(x)
        w, i = P.topk(lg)
        r = P.routed(x, w, i).clone()
        xq, xsf = mxfp8_quantize(x, False, alignment=C.DIM, backend='cute-dsl')
        se = xsf.view(torch.uint8).reshape(m, -1)[:, :C.DIM // 32].float()
        xd = (xq.view(torch.float8_e4m3fn).float().reshape(m, C.DIM // 32, 32) * torch.exp2(se - 127).unsqueeze(-1)).reshape(m, C.DIM)
        rows = {}
        for name, kw in [('post', {}), ('post_wfp32', dict(w_bf16=False)), ('pre', dict(w_pre=True)), ('pre_wfp32', dict(w_pre=True, w_bf16=False)),
                         ('post_gu16', dict(gu_bf16=True)), ('pre_gu16', dict(w_pre=True, gu_bf16=True)), ('post_act16', dict(act_bf16=True)),
                         ('pre_act16', dict(w_pre=True, act_bf16=True)), ('post_floor', dict(floor_scale=True)), ('pre_floor', dict(w_pre=True, floor_scale=True)),
                         ('post_slot16', dict(out_bf16_slot=True)), ('pre_gu16_act16', dict(w_pre=True, gu_bf16=True, act_bf16=True))]:
            ref = variant(x, i, w, raw, xd, **kw)
            rows[name] = (R.rel(r, ref), float((r.float() - ref.float()).abs().max()), float(r.float().abs().max()))
        print(f'M {m}: ' + '  '.join(f'{k} {v[0]:.5f}' for k, v in sorted(rows.items(), key=lambda kv: kv[1][0])), flush=True)
    print('REFCHECK_DONE')


if __name__ == '__main__':
    main()
