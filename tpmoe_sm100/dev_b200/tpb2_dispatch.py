"""Per-M build table of tpmoe_b2 (v77; v71 / v74 / v76 kept for reference): which compile-time specialized build serves a token count M, and which router feeds it.

Every build is the same tpmoe_b2.cu compiled with extra -D flags (tpb2.load_ext(flags=...)).  A build checks its M range on the host
(TORCH_CHECK), so a wrong pick fails loudly instead of computing garbage.  Exact-M builds (MMIN == MMAX) exist for the batch sizes a
CUDA-graph decode server captures most often; the bucket builds cover every other M <= 128.

Router: for 5 <= M <= 64 with a probed cuBLAS split (router_splits.router_parts(M): K parts <= 2048) the kernel runs its own router
(tpb2.run(..., router=(W, parts))), bitwise == the production split-K logits; otherwise (M <= 4, M 49..63 = one 4096 part, M > 64)
production's router logits (dsv3_router_gemm / torch.mm) are passed as glog.
"""

COMMON = ('-DTPB2_SPEC=1', '-DTPB2_GRID=148', '-DTPB2_WHOLE=1', '-DTPB2_T1CNT=1', '-DTPB2_PUT1=1', '-DTPB2_QX1=1')

# (M min, M max, extra flags) -- disp_v71.sh
V71 = [
    (1, 1, ('-DTPB2_GUNR=1', '-DTPB2_RS=0', '-DTPB2_TABW=1', '-DTPB2_FC2PAIR_OFFM=0', '-DTPB2_ACTU=1')),
    (2, 2, ('-DTPB2_GUNR=1', '-DTPB2_RS=0')),
    (3, 4, ('-DTPB2_GUNR=1', '-DTPB2_RS=0')),
    (5, 7, ('-DTPB2_GUNR=1', '-DTPB2_SHGU_MAXM=8')),
    (8, 8, ('-DTPB2_GUNR=1', '-DTPB2_SHGU_MAXM=8')),
    (9, 15, ('-DTPB2_FC2S_HI=16',)),
    (16, 16, ('-DTPB2_FC2S_HI=16',)),
    (17, 31, ('-DTPB2_SHGU_LO2=17', '-DTPB2_SHGU_HI2=31')),
    (32, 32, ('-DTPB2_ACTX=1',)),
    (33, 48, ('-DTPB2_ACTX=1',)),
    (49, 63, ('-DTPB2_ACTX=1',)),
    (64, 64, ('-DTPB2_ACTX=1',)),
    (65, 127, ('-DTPB2_ACTX=1',)),
    (128, 128, ('-DTPB2_ACTX=1',)),
]

# v74 = v71 + SFBI (routed FC1 SFB staged per item) in every build but M1; M1: FC2CAP=1 + PUBRED; PUBRED at M8 / M9-16;
# exact builds at M5 / 6 / 7 / 20 / 24 / 96 (narrower buckets around them) -- disp_v74.sh
_SFBI = ('-DTPB2_SFBI=1',)
_PUB = ('-DTPB2_PUBRED=1',)
V74 = (
    [(1, 1, ('-DTPB2_GUNR=1', '-DTPB2_RS=0', '-DTPB2_TABW=1', '-DTPB2_FC2PAIR_OFFM=0', '-DTPB2_ACTU=1', '-DTPB2_FC2CAP=1') + _PUB),
     (2, 2, ('-DTPB2_GUNR=1', '-DTPB2_RS=0') + _SFBI),
     (3, 4, ('-DTPB2_GUNR=1', '-DTPB2_RS=0') + _SFBI)]
    + [(m, m, ('-DTPB2_GUNR=1',) + _SFBI + ('-DTPB2_SHGU_MAXM=8',)) for m in (5, 6, 7)]
    + [(8, 8, ('-DTPB2_GUNR=1',) + _SFBI + ('-DTPB2_SHGU_MAXM=8',) + _PUB),
       (9, 15, _SFBI + ('-DTPB2_FC2S_HI=16',) + _PUB),
       (16, 16, _SFBI + ('-DTPB2_FC2S_HI=16',) + _PUB)]
    + [(lo, hi, _SFBI + ('-DTPB2_SHGU_LO2=17', '-DTPB2_SHGU_HI2=31')) for lo, hi in ((17, 19), (20, 20), (21, 23), (24, 24), (25, 31))]
    + [(lo, hi, ('-DTPB2_ACTX=1',) + _SFBI) for lo, hi in ((32, 32), (33, 48), (49, 63), (64, 64), (65, 95), (96, 96), (97, 127), (128, 128))]
)

# v76 = v74 + ACTU (FC1 activation unrolled over the live tokens) at M2-8 + FC2CAP=1 at M2 -- disp_v76.sh.  Rank-aligned paired
# timing (b2_layer_ab --gpu-align): M2 -3.7..-4.0%, M4 -0.9..-1.5, M6 -1.1..-1.6, M8 -0.5..-0.8%, every other M unchanged
_ACTU = ('-DTPB2_ACTU=1',)
V76 = ([V74[0], (2, 2, V74[1][2] + _ACTU + ('-DTPB2_FC2CAP=1',))]
       + [(lo, hi, extra + _ACTU) for lo, hi, extra in V74[2:7]]
       + V74[7:])

# v77 = v76 + the all-reduce rework (same arithmetic: bitwise == v76 == production) -- disp_v77.sh.  One-shot builds (M <= 32):
# TPB2_ARV=40 (8: a message's pending source polls issued together instead of one dependent load after another; 32: CTA exit with a
# fire-and-forget u64 count instead of fence + returning atomic).  M33-48 moved to RS + AG (ONESHOT_MAXM=32: paired M48 -0.8%).  RS + AG
# builds: TPB2_ARV=34 (2: T2 sources and T3 messages of a thread polled together in one loop + 32)
V77 = ([(lo, hi, extra + ('-DTPB2_ARV=40',)) for lo, hi, extra in V76 if hi <= 32]
       + [(33, 48, V76[15][2] + ('-DTPB2_ARV=34', '-DTPB2_ONESHOT_MAXM=32'))]
       + [(lo, hi, extra + ('-DTPB2_ARV=34',)) for lo, hi, extra in V76 if lo > 48])
assert [(lo, hi) for lo, hi, _ in V77] == [(lo, hi) for lo, hi, _ in V76]


def build_flags(m, table=V77):
    """-D flags of the build that serves M (1 <= M <= 128)."""
    for lo, hi, extra in table:
        if lo <= m <= hi:
            return (f'-DTPB2_MMIN={lo}', f'-DTPB2_MMAX={hi}') + COMMON + extra
    raise ValueError(f'tpmoe_b2: no build for M = {m} (1..128)')


def in_kernel_router(m):
    """The K parts for the kernel's own router at M, or None (pass production's router logits instead)."""
    from router_splits import router_parts
    sp = router_parts(m)
    return sp if isinstance(sp, tuple) and 5 <= m <= 64 and max(sp) <= 2048 else None


def all_builds(table=V77):
    """Every distinct flag tuple (prebuild them once, e.g. in parallel with b2_sass.py --sets=...)."""
    return [build_flags(lo, table) for lo, _, _ in table]
