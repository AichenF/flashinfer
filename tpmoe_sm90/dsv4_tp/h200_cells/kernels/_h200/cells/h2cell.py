"""Shared suite-harness adapter body for the H200 cells (suite_h200/cells/h200_<model>_tp<N>/adapter.py).
Kernels: tpmoe family sources in suite_h200/src built with tphost.build (rank 0 builds into $SUITE_BUILD, then every rank loads),
or prebuilt .so's (ctx.so, M-dispatch SO[,SO@LO-HI]).  Host: suite_h200/tphost.py (RankState / run; ABI auto: tail-weight builds
export NSL2).  Weights: Humming layout from the canonical MXFP4 TP slices ctx.tp_w() -- DSV4-Flash via the suite's humming_prep
(= prep_humming_synth), DSV4.1 via v41_prep's sublayer recipe (main items, 64-row tail item, T4 split tail at TP8, w2).
LL all-reduce buffers: ctx.ipc_alloc.  Routed-only: run(E, st, w, ids, rw, 1.0, out, rank, N, xb) (in-kernel x quant + all-reduce).
Options (--opt): src=<file in src/>, flags=A=1+B=2 (extra -D flags), dispatch=src@lo-hi;src2@lo-hi (several builds by M)."""
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

H2 = Path(__file__).resolve().parents[1]   # suite_h200
sys.path.insert(0, str(H2))
import tphost as TH  # noqa: E402


def git_head():
    return '56f1450 (imported into tpmoe_suite kernels/_h200)'   # (tpmoe_suite import: not a suite_h200 checkout)
    try:
        return subprocess.run(['git', '-C', str(H2), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return '?'


def v41_weights(raw, H, i, dev):
    """canonical TP slices (w13 [E, 2i, H/2] gate rows then up rows, s13, w2 [E, H, i/2], s2) -> tpv41 tensors (v41_prep recipe)"""
    os.environ.setdefault('SGLANG_HUMMING_INPUT_QUANT_CONFIG', '{"dtype":"float8e4m3"}')
    import v41_prep as VP
    E = raw['w13'].shape[0]
    w13, s13 = raw['w13'].to(dev), raw['s13'].to(dev)
    R = dict(g=w13[:, :i], gs=s13[:, :i], u=w13[:, i:], us=s13[:, i:], w2=raw['w2'].to(dev), w2s=raw['s2'].to(dev))
    out = {}
    im = (i // 128) * 128
    out['w1'], out['s1'], out['f1'] = VP.humming(torch.cat([R['g'][:, :im], R['u'][:, :im]], 1).contiguous(),
                                                 torch.cat([R['gs'][:, :im], R['us'][:, :im]], 1).contiguous(), [im, im], H, E, 256, 128)
    if im < i:
        nt = i - im
        z = torch.zeros(E, 64 - nt, H // 2, dtype=torch.uint8, device=dev)
        zg = R['gs'][:, im:im + 1].expand(E, 64 - nt, H // 32)
        zu = R['us'][:, im:im + 1].expand(E, 64 - nt, H // 32)
        out['w1t'], out['s1t'], out['f1t'] = VP.humming(torch.cat([R['g'][:, im:], z, R['u'][:, im:], z], 1).contiguous(),
                                                        torch.cat([R['gs'][:, im:], zg, R['us'][:, im:], zu], 1).contiguous(), [64, 64], H, E, 64, 128)
        if nt == 32:   # T4 split tail (TP8)
            b64 = torch.cat([R['g'][:, im:], R['u'][:, im:]], 1)
            s64 = torch.cat([R['gs'][:, im:], R['us'][:, im:]], 1)
            nj = H // 512
            w4 = b64.reshape(E, 64, nj, 4, 64).permute(0, 3, 1, 2, 4).reshape(E, 256, H // 8).contiguous()
            s4 = s64.reshape(E, 64, nj, 4, 4).permute(0, 3, 1, 2, 4).reshape(E, 256, H // 128).contiguous()
            out['w1t4'], out['s1t4'], out['f1t4'] = VP.humming(w4, s4, [128, 128], H // 4, E, 128, 128)
    out['w2'], out['s2'], out['f2'] = VP.humming(R['w2'].contiguous(), R['w2s'].contiguous(), [H], i, E, 256, 64)
    return {k: v.contiguous() for k, v in out.items()}


def parse_flags(s):
    return tuple('-D' + f for f in s.split('+') if f) if s else ()


class Cell:
    xq_mode = 'tok'

    def __init__(self, ctx, name, model, builds, provenance):
        """builds: [(lo, hi, src, extra_flags)] (first = default)"""
        self.ctx, self.name, self.model = ctx, name, model
        H, I, E, K = ctx.shape
        N = ctx.ndev
        self.prov = dict(provenance, suite_h200_commit=git_head())
        t0 = time.time()
        exts = []
        if ctx.so:
            for s in ctx.so:
                p, _, rng = s.partition('@')
                lo, hi = (int(v) for v in rng.split('-')) if rng else (0, 1 << 30)
                exts.append((lo, hi, TH.load_so(p), p, ()))
        else:
            os.environ['TP_BUILDS'] = os.environ.get('SUITE_BUILD', os.environ.get('TP_BUILDS', '/tmp/tpsuite_build'))
            TH.BUILDS = Path(os.environ['TP_BUILDS'])
            mdl = 1 if model == 'dsv41f' else None
            if ctx.rank == 0:   # rank 0 builds (node-local cache), then every rank loads the cached builds
                for lo, hi, src, fl in builds:
                    TH.build(src, N, model=mdl, shared=0, extra=fl)
            dist.barrier(group=ctx.gloo)
            for lo, hi, src, fl in builds:
                e = TH.build(src, N, model=mdl, shared=0, extra=fl)
                exts.append((lo, hi, e, e.__file__, fl))
        self.exts = []
        wcache = {}

        def weights(isplit):   # Humming tensors of this rank: TP-N slices of all experts, or (isplit < N, port_isplit hybrid) the
            if isplit in wcache:   # isplit-way slice r % isplit of expert group r / isplit
                return wcache[isplit]
            if isplit == N:
                raw = ctx.tp_w()
            else:
                import tps_common as TC
                g, ng = ctx.rank // isplit, N // isplit
                full = TC.tp_raw(ctx.shape, isplit, ctx.rank % isplit, ctx.dev)
                Eg = E // ng
                raw = {k: v[g * Eg:(g + 1) * Eg].contiguous() for k, v in full.items()}
                del full
            if model == 'dsv41f':
                w = v41_weights(raw, H, I // isplit, ctx.dev)
            else:
                from humming_prep import humming_tp_weights
                w = humming_tp_weights(raw, H, I // isplit, ctx.dev)
            wcache[isplit] = w
            return w
        self.max_m = 1 << 30
        for lo, hi, e, path, fl in exts:
            assert int(e.NDEV) == N, (path, int(e.NDEV), N)
            isplit = int(getattr(e, 'ISPLIT', N))
            st = TH.RankState(e, ctx.dev, alloc_comm=False)
            st.link_ptrs(ctx.ipc_alloc(st.rs_bytes), ctx.ipc_alloc(st.ag_bytes))
            self.exts.append((lo, hi, e, st, TH.adapt_weights(e, weights(isplit))))
            self.max_m = min(self.max_m, int(e.MAXM))
        wcache.clear()
        self.prov['so'] = {Path(p).name: dict(sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16], lo=lo, hi=hi, flags=list(fl))
                           for lo, hi, _, p, fl in exts}
        ctx.log(f'  {name}: {len(self.exts)} build(s) + weights {time.time() - t0:.0f} s, MAXM {self.max_m}')

    def pick(self, m):
        for x in self.exts[1:]:
            if x[0] <= m <= x[1]:
                return x
        return self.exts[0]

    def bind(self, M, x, ids, w, out):
        if M > self.max_m:
            return None
        _, _, e, st, wts = self.pick(M)
        rank, N = self.ctx.rank, self.ctx.ndev

        def fn():
            TH.run(e, st, wts, ids, w, 1.0, out, rank, N, x)
        return fn

    def pre_replay(self):   # the host launch tag (seq) is frozen inside a one-call graph: clear every build's per-token quant flags
        for x in self.exts:
            x[3].xflag.zero_()

    def info(self):
        return self.prov


def make(ctx, name, model, default_builds, provenance):
    """default_builds [(lo, hi, src, flags)]; --opt src=.. / flags=.. / dispatch=src@lo-hi;.. override"""
    o = ctx.opts or {}
    builds = default_builds
    if 'dispatch' in o:
        builds = []
        for i, spec in enumerate(o['dispatch'].split(';')):
            src, _, rng = spec.partition('@')
            src, _, fl = src.partition('#')
            lo, hi = (int(v) for v in rng.split('-')) if rng else (0, 1 << 30)
            builds.append((lo, hi, src, parse_flags(fl)))
    elif 'src' in o or 'flags' in o:
        builds = [(0, 1 << 30, o.get('src', default_builds[0][2]), parse_flags(o.get('flags', '')))]
    return Cell(ctx, name, model, builds, dict(provenance, builds=[(lo, hi, s, list(f)) for lo, hi, s, f in builds]))
