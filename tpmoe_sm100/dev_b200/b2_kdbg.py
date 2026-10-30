"""Single-GPU debug of tpmoe_b2 (ndev 1: T1 writes the output, no all-reduce): rank-0 shard of layer 20, captured tokens.
Checks every stage against the reference: x MXFP8 (xq / xs), routing (ids / rw vs production), per-slot FC2 outputs (outp),
the output (shared + 1.5 routed).  usage: python3 b2_kdbg.py --model MODEL --cap CAP --out OUT [--ms 1 6 16 64]"""
import argparse
import sys
from pathlib import Path

import torch

SRV = Path(__file__).resolve().parent
sys.path.insert(0, str(SRV))
import b2_common as C  # noqa: E402
import b2_ref as R  # noqa: E402
import tpb2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--cap', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ms', type=int, nargs='+', default=[1, 6, 16, 64])
    ap.add_argument('--grid', type=int, default=0)
    ap.add_argument('--shared', action='store_true', help='fused shared expert (production DeepGEMM slice) vs the production shared output')
    ap.add_argument('--force', action='store_true', help='every token routed to experts 3 50 100 150 200 250 (wide panels), given top-k')
    args = ap.parse_args()
    dev = torch.device('cuda', 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    KE = tpb2.load_ext()
    st = tpb2.RankState(KE, dev, ndev=1, rank=0)
    if args.grid:
        st.grid = args.grid
    wm = C.weight_map(args.model)
    raw = C.load_experts_raw(args.model, wm, 20, 4, 0, dev)
    layer = C.sglang_trtllm_layer(raw, dev)
    router, bias = C.load_router(args.model, wm, 20, dev)
    shw = C.load_shared(args.model, wm, 20, 4, 0, dev) if args.shared else dict(inter=raw['inter'])
    P = C.Prod(layer, shw, router, bias, None, None)
    SW = tpb2.SharedWeights(KE, shw['gu_w'], shw['gs_f'], shw['dn_w'], shw['ds_f']) if args.shared else None
    W = tpb2.Weights(KE, layer.w13_weight, layer.w13_weight_scale_inv, layer.w2_weight, layer.w2_weight_scale_inv)
    xs = C.capture_pool(args.cap, args.ms, dev)
    st.dbg = torch.zeros(st.grid * 32, dtype=torch.int64, device=dev)

    def stamps():   # per-CTA phase times (us from the CTA's start): medians and maxima
        d = st.dbg.view(st.grid, 32).double().cpu()
        t0 = d[:, 0:1]
        names = {1: 'tables', 2: 'quant', 8: 'x_ready', 9: 'mma_first', 3: 'fc1_issued', 10: 'mma_fc1_last', 11: 'epi_first_fc1', 12: 'epi_last_fc1_acc',
                 13: 'epi_last_merge', 14: 'swiglu', 15: 'am_bar', 4: 'fc1_last_epi', 5: 'fc2_end', 6: 't1_end', 7: 'end'}
        rel = (d - d[:, 0].min()) / 1e3
        out = []
        for k, nm in names.items():
            v = rel[:, k][d[:, k] > 0]
            if len(v):
                out.append(f'{nm} {float(v.median()):.1f}/{float(v.max()):.1f}')
        sums = {16: 'mma_wait_full', 17: 'mma_wait_acce', 18: 'epi_wait_accf', 19: 'epi_busy', 20: 'prod_wait_empty', 21: 'prod_wait_hflag'}
        out.append('| FC2 sums us (median/max): ' + '  '.join(f'{nm} {float(d[:, k].median()) / 1e3:.1f}/{float(d[:, k].max()) / 1e3:.1f}' for k, nm in sums.items()))
        return '  '.join(out)
    for m in args.ms:
        x = xs[m]
        lg = P.logits(x)
        w, i = P.topk(lg)
        if args.force:
            i = torch.tensor([3, 50, 100, 150, 200, 250], dtype=torch.int32, device=dev).repeat(m, 1)
            w = torch.full((m, 6), 1.0 / 6, dtype=torch.float32, device=dev)
        sh = P.shared(x) if args.shared else (torch.randn(m, 4096, device=dev) * 0.1).bfloat16()
        out = torch.full((m, 4096), float('nan'), dtype=torch.bfloat16, device=dev)
        st.outp.fill_(float('nan'))
        kw = dict(ids=i.to(torch.int32).contiguous(), rw=w.float().contiguous()) if args.force else dict(glog=lg, gbias=bias)
        if args.shared:
            kw.update(SW=SW)
        tpb2.run(st, W, x, out, shared=None if args.shared else sh, **kw)
        if args.shared:
            torch.cuda.synchronize()
            so = st.shout[:m * 4096].view(m, 4096)
            print(f'   shared: bitwise {bool(torch.equal(so, sh))}  mismatches {int((so != sh).sum())}  rel {R.rel(so, sh):.6f}', flush=True)
            # stage checks: shared x image (per-128 codes) and h image vs production's own intermediates
            from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
            from sglang.jit_kernel.dsv4.moe import silu_and_mul_clamp
            from sglang.srt.layers.quantization.fp8_utils import deepgemm_w8a8_block_fp8_linear_with_fallback as lin

            def decode(img, rows, nch):   # [nch][32 rows][128] SW128 (a panel) -> [rows][nch * 128] bytes
                img = img.view(nch, 32 * 128)
                out_ = torch.empty(rows, nch * 128, dtype=torch.uint8, device=dev)
                for r in range(rows):
                    base = (r >> 3) * 1024 + (r & 7) * 128
                    for g in range(8):
                        src = base + ((g ^ (r & 7)) << 4)
                        out_[r, torch.arange(nch, device=dev).repeat_interleave(16) * 128 + g * 16 + torch.arange(16, device=dev).repeat(nch)] = img[:, src:src + 16].reshape(-1)
                return out_
            mm = min(m, 32)
            xq_p = torch.empty(0)
            xq8, _ = sglang_per_token_group_quant_fp8(x, 128, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True)
            xg_codes = decode(st.xgs[:32 * 32 * 128], mm, 32)
            print('   shared x image rows == production codes:', [bool(torch.equal(xg_codes[r], xq8[r].view(torch.uint8))) for r in range(mm)], flush=True)
            gu = lin(x, shw['gu_w'], [128, 128], shw['gu_s'])
            act = torch.empty((m, 512), dtype=torch.bfloat16, device=dev)
            silu_and_mul_clamp(gu, act, 10.0)
            aq8, _ = sglang_per_token_group_quant_fp8(act, 128, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True)
            h_codes = decode(st.hsimg[:4 * 32 * 128], mm, 4)
            print('   shared h image rows == production act codes:', [bool(torch.equal(h_codes[r], aq8[r].view(torch.uint8))) for r in range(mm)], flush=True)
            print('   shared h image per item k (all rows):', [bool(torch.equal(h_codes[:, 128 * k:128 * (k + 1)], aq8[:mm, 128 * k:128 * (k + 1)].view(torch.uint8))) for k in range(4)], flush=True)
            hs_ = st.hssf[:2 * 512].view(2, 32, 16)
            print('   shared h scale bytes token 0 (k = 0..3):', [int(hs_[k >> 1, 0, 8 * (k & 1)]) for k in range(4)], flush=True)
            G_ = st.grid
            cnt_ = torch.bincount(i.flatten().long(), minlength=256)
            R1 = int(((cnt_ + 31) // 32).sum()) * 8 * 16
            S1 = ((m + 31) // 32) * 4 * 32
            totc = R1 + 2 * S1
            def us(c):
                x_ = (totc * c) // G_
                return x_ if x_ < R1 else R1 + (x_ - R1) // 2
            own = []
            for k in range(4):
                st0 = R1 + k * 32
                owner = max(c for c in range(G_) if us(c) <= st0)
                contrib = [c for c in range(owner + 1, G_) if us(c) < st0 + 32]
                own.append((owner, us(owner), contrib[:6]))
            print('   shared item owners / contributors:', own, ' R1', R1, 'totc', totc, flush=True)
            if not torch.equal(so, sh):
                print('   shared per-token rel:', [round(R.rel(so[t], sh[t]), 4) for t in range(min(m, 40))], flush=True)
                print('   shared per-128-col rel (token 0..1):', [round(R.rel(so[:2, 128 * c:128 * (c + 1)], sh[:2, 128 * c:128 * (c + 1)]), 3) for c in range(32)], flush=True)
        torch.cuda.synchronize()
        ids_k, rw_k = (i.to(torch.int32), w.float()) if args.force else (st.ids[:m * 6].view(m, 6), st.rw[:m * 6].view(m, 6))
        gate_eq = bool(torch.equal(ids_k, i.to(torch.int32))) and bool(torch.equal(rw_k, w.float()))
        xq_ref, xe_ref = R.quant_mxfp8(x)
        xq_k = st.xq[:m * 4096].view(m, 4096).view(torch.float8_e4m3fn).float()
        xe_k = st.xs[:m * 128].view(m, 128).to(torch.int32) - 127
        xq_deq = (xq_k.view(m, 128, 32) * torch.exp2(xe_k.float()).unsqueeze(-1)).view(m, 4096)
        x_eq = bool(torch.equal(xq_deq, xq_ref)) and bool(torch.equal(xe_k, xe_ref))
        # per-slot reference outputs (no routing weight): slot k of token t = FC2_e(q(act)) for e = ids[t, k]
        inter = raw['inter']
        slot_ref = torch.zeros(m, 6, 4096, device=dev)
        for e in torch.unique(i).tolist():
            tok, slot = (i == e).nonzero(as_tuple=True)
            y = xq_ref[tok] @ R.dequant_mxfp4(raw['w13'][e], raw['s13'][e]).t()
            h = R.quant_mxfp8(R.swiglu_clamp(y[:, :inter], y[:, inter:]), rule='floor')[0]
            slot_ref[tok, slot] = h @ R.dequant_mxfp4(raw['w2'][e], raw['s2'][e]).t()
        outp = st.outp[:m * 8 * 4096].view(m, 8, 4096)[:, :6].float()
        slot_rel = R.rel(outp, slot_ref)
        bad_slots = int((~torch.isfinite(outp)).any(-1).sum())
        ref_routed = R.routed_ref(x, i, w, raw)
        ref = (sh.float() + 1.5 * ref_routed.float()).bfloat16()
        prod = (sh.float() + 1.5 * P.routed(x, w, i).float()).bfloat16()
        if args.force:   # per-slot error by token; panel 0 (expert 3) x image and h image rows vs the reference
            per_tok = [round(R.rel(outp[t], slot_ref[t]), 4) for t in range(m)]
            print('   per-token slot rel:', per_tok, flush=True)

            def unswz(img, rows):   # [chunks][32 rows][128] SW128 -> [rows][chunks * 128] codes
                ch = img.shape[0]
                out_ = torch.empty(rows, ch * 128, dtype=torch.uint8, device=dev)
                for r in range(rows):
                    base = (r >> 3) * 1024 + (r & 7) * 128
                    for g in range(8):
                        src = base + ((g ^ (r & 7)) << 4)
                        out_[r, g * 16:(g + 1) * 16] = img.reshape(ch, 4096)[:, src:src + 16].reshape(ch, 16)[0] if ch == 1 else 0
                return out_
            xg0 = st.xg[:32 * 32 * 128].view(32, 32 * 128)   # panel 0, chunks 0..31
            xrows = []
            for r in range(m):
                row = []
                base = (r >> 3) * 1024 + (r & 7) * 128
                for c in range(32):
                    for g in range(8):
                        src = base + ((g ^ (r & 7)) << 4)
                        row.append(xg0[c, src:src + 16])
                xrows.append(torch.cat(row))
            xcodes = torch.stack(xrows)
            xref_codes = (xq_ref / torch.exp2(xe_ref.float()).repeat_interleave(32, -1)).to(torch.float8_e4m3fn).view(torch.uint8)
            print('   x image rows == ref codes:', [bool(torch.equal(xcodes[r], xref_codes[r])) for r in range(m)], flush=True)
            h0 = st.himg[:4 * 32 * 128].view(4, 32 * 128)   # panel 0, h chunks 0..3
            hrows = []
            for r in range(m):
                row = []
                base = (r >> 3) * 1024 + (r & 7) * 128
                for c in range(4):
                    for g in range(8):
                        src = base + ((g ^ (r & 7)) << 4)
                        row.append(h0[c, src:src + 16])
                hrows.append(torch.cat(row))
            hcodes = torch.stack(hrows).view(torch.float8_e4m3fn).float()
            y = xq_ref @ R.dequant_mxfp4(raw['w13'][3], raw['s13'][3]).t()
            href, hex_ = R.quant_mxfp8(R.swiglu_clamp(y[:, :inter], y[:, inter:]), rule='floor')
            hrc = (href / torch.exp2(hex_.float()).repeat_interleave(32, -1))
            print('   h image rows nonzero:', [int((hcodes[r] != 0).sum()) for r in range(m)], flush=True)
            print('   h image rows == ref codes:', [bool(torch.equal(hcodes[r], hrc[r])) for r in range(m)], flush=True)
        print(f'M {m:4d}: gate==prod {gate_eq}  x==ref {x_eq}  slots rel {slot_rel:.5f} nonfinite slots {bad_slots}  '
              f'out vs ref {R.rel(out, ref):.5f}  out vs prod {R.rel(out, prod):.5f}  prod vs ref {R.rel(prod, ref):.5f}  '
              f'out finite {bool(torch.isfinite(out.float()).all())}', flush=True)
        for _ in range(3):
            st.dbg.zero_()
            tpb2.run(st, W, x, out, shared=None if args.shared else sh, **kw)
        torch.cuda.synchronize()
        cnt = torch.bincount(i.flatten().long(), minlength=256)
        ptot = int(((cnt + 31) // 32).sum())
        print(f'   panels {ptot} (FC1 {ptot * 2.125:.0f} MB, FC2 {ptot * 1.0625:.0f} MB)  stamps (median/max us from grid start): {stamps()}', flush=True)
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.dbg = None
        for _ in range(3):
            tpb2.run(st, W, x, out, shared=None if args.shared else sh, **kw)
        ev0.record()
        for _ in range(20):
            tpb2.run(st, W, x, out, shared=None if args.shared else sh, **kw)
        ev1.record()
        torch.cuda.synchronize()
        print(f'   warm (hot L2) kernel us: {ev0.elapsed_time(ev1) * 1e3 / 20:.1f}', flush=True)
        st.dbg = torch.zeros(st.grid * 32, dtype=torch.int64, device=dev)
    print('KDBG_DONE')


if __name__ == '__main__':
    main()
