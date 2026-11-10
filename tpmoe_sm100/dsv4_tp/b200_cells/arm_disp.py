"""DSV4-Flash TP4 / TP8 dispatch arm (11-10): v77 (arm_v77: routed-only builds) for M < MINM, v4ft (arm_v4ft: port_v4f over
tp8_v41_b300 e066ec9, cell flags ACTX + T1PW2 at M384 / 512) for M >= MINM (SB_V4FT_MINM, default 384).  Two kernel families, two
separate workspaces (v77: per-build RS / AG LLS buffers; v4ft: one RS buffer for its builds), one stream: consecutive calls may switch
families (b2b-mix and stress --mix2 test exactly that).  Numerics per M = the selected family's (v77: trtllm-gen routed, bf16 routing
weights; v4ft: tp8v41 numerics, fp32 routing weights)."""
import os

import arm_v4ft
import arm_v77

MINM = int(os.environ.get('SB_V4FT_MINM', '384'))
V4FT_PER_M = {384: ('-DTPB2_ACTX=1', '-DTPB2_T1PW=2'), 512: ('-DTPB2_ACTX=1', '-DTPB2_T1PW=2')}   # (the B300b cell's kept flags; B200 A/B 11-10)
V77_EXTRA_MS = (64, 128)   # (v77 builds always present: the b2b-mix / stress --mix2 partner Ms)


class Arm:
    kind = 'disp'

    def __init__(self, sh, rank, dev, gloo, ipc, raw, trtW, Ms, extra=(), extra_v4ft=(), oneshot_maxm=32, minm=None, per_m=None):
        self.minm = MINM if minm is None else minm
        self.sh, self.rank, self.dev = sh, rank, dev
        m77 = sorted({M for M in Ms if M < self.minm} | set(V77_EXTRA_MS))
        m4 = sorted(M for M in Ms if M >= self.minm)
        self.v77 = arm_v77.Arm(sh, rank, dev, gloo, ipc, trtW, m77, extra=tuple(extra))
        self.v4ft = arm_v4ft.Arm(sh, rank, dev, gloo, ipc, raw, m4, extra=tuple(extra_v4ft), oneshot_maxm=oneshot_maxm,
                                 per_m=V4FT_PER_M if per_m is None else per_m) if m4 else None
        self.name = f'v77<{self.minm}<=v4ft'

    def kind_of(self, M):
        return 'v4ft' if M >= self.minm else 'v77'

    def arm(self, M):
        return self.v4ft if M >= self.minm else self.v77

    @property
    def ext(self):   # (built Ms, for the b2b-mix partner choice)
        return {M: None for M in list(self.v77.flags) + list(self.v4ft.ext if self.v4ft else [])}

    def ref_partial(self, x, ids, rw):
        return self.v4ft.ref_partial(x, ids, rw)

    def bind(self, M, x, ids, rw, out=None):
        return self.arm(M).bind(M, x, ids, rw, out=out)

    def run_dbg(self, M, x, ids, rw, out, dbg):
        return self.arm(M).run_dbg(M, x, ids, rw, out, dbg)

    @property
    def grid(self):
        return self.v77.grid
