"""Suite-harness adapter (tpmoe_suite harness/bench.py --ours-adapter suite_b200/kernels/<cell>) for the B200 cells' kernels in suite_b200:
  b200_dsv4f_tp4  v77 routed-only (TPB2_SPEC=2, RS=0; per-M v77 flag table + exact builds up to M 512)       -- arm_v77.py
  b200_dsv4f_tp8  v77 with TPB2_NDEV=8 (INTER 256)                                                          -- arm_v77.py
                  (11-10, both dsv4f cells) dispatch: v77 for M < 384, v4ft (port_v4f over tp8_v41_b300 e066ec9, the B300b cell
                  flags ACTX + T1PW2) for M >= 384 -- arm_disp.py; SB_DSV4F_DISP=0: v77 at every M; SB_V4FT_MINM moves the switch
  b200_dsv41f_tp8 tp8v41 (B300 lineage) built for sm_100a                                                   -- arm_tp8v41.py
Builds go to $SUITE_BUILD/suite_b200.  --opt flags="-DX=.." adds build flags, oneshot=N (tp8v41) changes ONESHOT_MAXM.
ref_partial = the kernels' numerics emulation (sbcommon.ref_rank: MXFP8 x ceil rule, h floor rule, bf16 slot outputs, ordered slot sum;
bf16 routing weights for v77 as trtllm-gen / PackTopkIds)."""
import os
import subprocess
import sys
import time

import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, '/home/scratch.aichenf_wwfo/tpmoe_suite/harness')
os.environ['SB_BUILD'] = os.path.join(os.environ.get('SUITE_BUILD', '/tmp/tpsuite_build'), 'suite_b200')
os.environ['TP8_BUILD'] = os.environ['SB_BUILD']
import sbcommon as C   # noqa: E402

ADAPTER_API = 1


def _commit():
    try:
        return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'], text=True).strip()
    except Exception:
        return '?'


class Cell:
    xq_mode = 'mx32b'
    max_m = 512

    def __init__(self, ctx):
        model = {'dsv41f': 'dsv41'}.get(ctx.model, ctx.model)
        self.ctx, self.sh = ctx, C.Shape(model, ctx.ndev)
        self.kind = {('dsv4f', 4): 'v77', ('dsv4f', 8): 'v77', ('dsv41', 8): 'tp8v41', ('dsv41', 4): 'tp8v41'}[(model, ctx.ndev)]
        if self.kind == 'v77' and int(os.environ.get('SB_DSV4F_DISP', '1')):
            self.kind = 'disp'
        self.extra = tuple(ctx.opts.get('flags', '').split())
        base = os.environ.get('SB_TP8V41_DIR', 'tp8v41h2') if (self.kind == 'tp8v41' and ctx.ndev == 8) else self.kind
        self.name = f'{base}{"-ndev8" if self.kind in ("v77", "disp") and ctx.ndev == 8 else ""}@suite_b200:{_commit()}'
        if self.kind == 'disp':
            import arm_disp
            self.name = f'v77{"-ndev8" if ctx.ndev == 8 else ""}<{arm_disp.MINM}<=v4ft(e066ec9)@suite_b200:{_commit()}'
        if base == 'tp8v41h2':
            self.name += f"(F2H_MINM={os.environ.get('TP8_F2H_MINM', 'host default: 256 at 2625d29')})"
        if int(os.environ.get('SB_PROBE', '0')):   # (diagnostics) SM-availability probe before every 25th replay of ours (outside the timed region)
            self.pre_replay = self._probe_pre
            self._np, self._M = 0, None

    def _probe_pre(self):
        import torch.distributed as dist
        import smprobe
        ctx = self.ctx
        if torch.cuda.is_current_stream_capturing():   # (the suite's b2b graph captures pre_replay: nothing to do there)
            return
        k = self._np
        self._np += 1
        if k % 25:
            return
        if smprobe._EXT is None:
            if ctx.rank == 0:
                smprobe._ext()
            dist.barrier(group=ctx.gloo)
            smprobe._ext()
        d = smprobe.probe(ctx.dev)
        nv, hb = smprobe.bw(ctx.dev, int(self._nvl[(ctx.rank + 1) % ctx.ndev])) if self._nvl else (0.0, 0.0)
        lw = smprobe.lat(ctx.dev, self._wlat[0], self._wlat[1]); lf = smprobe.lat(ctx.dev, self._fresh.data_ptr(), self._wlat[1])
        l2a = smprobe.lat_l2(ctx.dev, self._l2a); l2b = smprobe.lat_l2(ctx.dev, self._l2b)
        try:   # (L2 set-aside for persisting accesses: a library call could shrink the normal L2 for everyone)
            import ctypes
            _l = ctypes.CDLL('libcudart.so.13') if not hasattr(self, '_crt') else self._crt
            self._crt = _l
            v = ctypes.c_size_t(0); _l.cudaDeviceGetLimit(ctypes.byref(v), 0x06)   # cudaLimitPersistingL2CacheSize
            lw = (lw, int(v.value) >> 20)
        except Exception:  # noqa: BLE001
            lw = (lw, -1)
        try:
            clk = (torch.cuda.clock_rate(ctx.dev), torch.cuda.power_draw(ctx.dev) // 1000)
        except Exception:  # noqa: BLE001
            clk = (0, 0)
        allp = [None] * ctx.ndev
        dist.all_gather_object(allp, (d, clk, nv, hb, lw, lf, l2a, l2b), group=ctx.gloo)
        w = max(range(ctx.ndev), key=lambda r: (allp[r][0]['cyc_ratio'], allp[r][0]['start_max']))
        nvs = [round(v[2]) for v in allp]; hbs = [round(v[3]) for v in allp]
        ctx.log(f'  PROBE M={self._M} call {k}: r{w} ' + smprobe.fmt(allp[w][0]) + f' | clk min {min(c[1][0] for c in allp)} MHz P max {max(c[1][1] for c in allp)} W | nvl GB/s min {min(nvs)} max {max(nvs)} | hbm GB/s min {min(hbs)} max {max(hbs)}' + ' | lat ns weights/fresh per rank ' + ' '.join(f'{v[4][0]:.0f}/{v[5]:.0f}' for v in allp) + ' | L2-hit lat ns 8MB/64MB per rank ' + ' '.join(f'{v[6]:.0f}/{v[7]:.0f}' for v in allp) + ' | L2 persisting MB ' + ' '.join(str(v[4][1]) for v in allp))
        if self.dbg_arm is not None and self._M:
            x, ids, wt = self._io
            G = self.dbg_arm.grid
            dbgt = torch.zeros(G * 32, dtype=torch.int64, device=ctx.dev)
            outd = torch.empty(self._M, self.sh.DIM, dtype=torch.bfloat16, device=ctx.dev)
            torch.cuda.synchronize(); dist.barrier(group=ctx.gloo)
            self.dbg_arm.run_dbg(self._M, x, ids, wt, outd, dbgt)
            torch.cuda.synchronize()
            d = dbgt.view(G, 32).double()
            t0 = d[:, 0].min()
            keys = (1, 3, 5, 6, 7) if self.kind == 'v77' else (1, 3, 27, 5, 6, 7)
            st = {kk: float(((d[:, kk][d[:, kk] > 0] - t0) / 1e3).max()) for kk in keys if (d[:, kk] > 0).any()}
            alls = [None] * ctx.ndev
            dist.all_gather_object(alls, st, group=ctx.gloo)
            ctx.log(f'  DBGP M={self._M} call {k}: max over CTAs, per rank: ' + ' ; '.join('r%d ' % r + ' '.join(f'{kk}:{v:.0f}' for kk, v in a.items()) for r, a in enumerate(alls)))

    def prepare(self, Ms):
        ctx, sh = self.ctx, self.sh
        t0 = time.time()
        Ms = [m for m in Ms if m <= self.max_m]
        self.raw = ctx.tp_w()
        if self.kind == 'disp':   # (v77 below arm_disp.MINM, v4ft from it: separate workspaces, one stream)
            import arm_disp
            import arm_v77
            import sbtrt
            trtW = sbtrt.prep_weights(sh, self.raw)
            self.arm = arm_disp.Arm(sh, ctx.rank, ctx.dev, ctx.gloo, ctx.ipc_alloc, self.raw, trtW, Ms, extra=self.extra)
        elif self.kind == 'v77':
            import arm_v77
            import sbtrt
            trtW = sbtrt.prep_weights(sh, self.raw)
            self.arm = arm_v77.Arm(sh, ctx.rank, ctx.dev, ctx.gloo, ctx.ipc_alloc, trtW, Ms, extra=self.extra)
        else:
            import arm_tp8v41
            self.arm = arm_tp8v41.Arm(sh, ctx.rank, ctx.dev, ctx.gloo, ctx.ipc_alloc, self.raw, Ms, extra=self.extra,
                                      oneshot_maxm=int(ctx.opts.get('oneshot', 32)))
        self.dbg_arm = None
        self._nvl = ctx.ipc_alloc(32 << 20) if int(os.environ.get('SB_PROBE', '0')) else None   # (NVLink probe target buffers)
        if int(os.environ.get('SB_PROBE', '0')):   # (latency probe targets: the kernel's FC1 weights, and a fresh buffer of the same size)
            wt = trtW['w13'] if self.kind in ('v77', 'disp') else self.arm.W.w13
            self._wlat = (wt.data_ptr(), wt.numel() * wt.element_size())
            self._fresh = torch.empty(self._wlat[1], dtype=torch.uint8, device=ctx.dev)
            self._l2a = torch.zeros(8 << 20, dtype=torch.uint8, device=ctx.dev); self._l2b = torch.zeros(64 << 20, dtype=torch.uint8, device=ctx.dev)
        if int(os.environ.get('SB_PROBE', '0')) >= 2 and self.kind != 'disp':   # (diagnostics) a -DTPB2_DBG=1 twin (own workspace): phase stamps of one call per probe
            if self.kind == 'v77':
                self.dbg_arm = arm_v77.Arm(sh, ctx.rank, ctx.dev, ctx.gloo, ctx.ipc_alloc, trtW, Ms, extra=self.extra + ('-DTPB2_DBG=1',))
            else:
                self.dbg_arm = arm_tp8v41.Arm(sh, ctx.rank, ctx.dev, ctx.gloo, ctx.ipc_alloc, self.raw, Ms, extra=self.extra + ('-DTPB2_DBG=1',),
                                              oneshot_maxm=int(ctx.opts.get('oneshot', 32)))
        ctx.log(f'  {self.name}: builds + weights {time.time() - t0:.0f} s')

    def bind(self, M, x, ids, w, out):
        if M > self.max_m:
            return None
        self._M = M
        self._io = (x, ids, w)
        return self.arm.bind(M, x, ids, w, out)[0]

    def ref_partial(self, x, ids, w):
        if self.kind == 'disp' and self.arm.kind_of(int(x.shape[0])) == 'v4ft':   # (v4ft numerics: fp32 routing weights; F2H off)
            return self.arm.ref_partial(x, ids, w)
        return C.ref_rank(self.sh, self.raw, x, ids, w, bf16_weights=(self.kind in ('v77', 'disp')))

    def info(self):
        a = self.arm
        fl = getattr(a, 'flags', None) or getattr(a, 'fl', {})
        if self.kind == 'disp':
            fl = {M: (a.v4ft.fl[M] if a.kind_of(M) == 'v4ft' else a.v77.flags[M]) for M in sorted(a.ext) if M in a.v77.flags or (a.v4ft and M in a.v4ft.fl)}
        return dict(kernel=self.name, src='/home/scratch.aichenf_wwfo/suite_b200', extra=list(self.extra), flags={M: list(f) for M, f in fl.items()})


def make_cell(ctx):
    return Cell(ctx)
