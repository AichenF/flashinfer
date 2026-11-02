"""Per-M build table of tpmoe_b2 (v71): which compile-time specialized build serves a token count M, and which router feeds it.

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


def build_flags(m, table=V71):
    """-D flags of the build that serves M (1 <= M <= 128)."""
    for lo, hi, extra in table:
        if lo <= m <= hi:
            return (f'-DTPB2_MMIN={lo}', f'-DTPB2_MMAX={hi}') + COMMON + extra
    raise ValueError(f'tpmoe_b2 v71: no build for M = {m} (1..128)')


def in_kernel_router(m):
    """The K parts for the kernel's own router at M, or None (pass production's router logits instead)."""
    from router_splits import router_parts
    sp = router_parts(m)
    return sp if isinstance(sp, tuple) and 5 <= m <= 64 and max(sp) <= 2048 else None


def all_builds(table=V71):
    """Every distinct flag tuple (prebuild them once, e.g. in parallel with b2_sass.py --sets=...)."""
    return [build_flags(lo, table) for lo, _, _ in table]
