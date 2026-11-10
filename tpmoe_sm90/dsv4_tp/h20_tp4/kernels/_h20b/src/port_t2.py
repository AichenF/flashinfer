"""port_t2.py: tp4v41.cu -> tp4v41b.cu: generalize TP8's T4 split tail (one 64-row block = 32 gate + 32 up rows, split 4 ways over K,
N8 panels) to a split factor TS, adding T2 for DSV4.1 TP4: the tail item's two real 64-row blocks (gate rows 512..575 | up rows
512..575) are split 2 ways over K -- warpgroup w computes stack w >> 1 over the k-tiles of parity w & 1 (instead of 2 of 4
warpgroups multiplying out-of-bounds zero blocks), for panels of width <= 16 (two x tiles of <= 2 KB per ring stage).
Weights (v41_prep.py, nt == 64): [E, 256, H/2] Humming stacks [128, 128], stack s rows 0..63 = even k-tiles, 64..127 = odd k-tiles,
passed as (w1t4, s1t4, f1t4) (ABI unchanged).  Epilogue: gate = WG0 + WG1 partials, up = WG2 + WG3 (fp32), WG1/WG3 slots zero.
  python3 port_t2.py [tp4v41.cu] [tp4v41b.cu]"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / 'tp4v41.cu')
DST = Path(sys.argv[2] if len(sys.argv) > 2 else HERE / 'tp4v41b.cu')
s = SRC.read_text()
n = 0


def rep(old, new, count=1):
    global s, n
    c = s.count(old)
    assert c == count, f'patch expects {count} occurrence(s), found {c}:\n{old[:300]}'
    s = s.replace(old, new)
    n += 1


rep('''constexpr bool T4 = TAIL && TP2_TAIL4;
static_assert(!T4 || INTER_R - INTER_M == 32, "T4 split tail: 32 + 32 rows in one block (TP8 only)");''',
    '''#ifndef TP2_TAIL2
#define TP2_TAIL2 (TP2_MODEL == 1 && TP2_NDEV == 4)   // v41 TP4: N <= 16 tail items' 2 real 64-row blocks split 2 ways over K (maps 6, 7)
#endif
constexpr bool T4 = TAIL && (TP2_TAIL4 || TP2_TAIL2);   // split tail active (T4: TS 4, TP8 | T2: TS 2, TP4)
constexpr int TS = !T4 ? 1 : (TP2_TAIL4 ? 4 : 2);       // K split factor of the split tail
constexpr int TS_MAXN = 32 / TS;                        // panel widths (classes) on the split tail: T4 8 | T2 16
static_assert(!T4 || (TS == 4 ? INTER_R - INTER_M == 32 : INTER_R - INTER_M == 64), "split tail: T4 32 + 32 rows in one block (TP8), T2 2 x 64 rows (TP4)");''')
rep('constexpr int BALIGN = T4 ? (KPS > 4 ? KPS : 4) : KPS;   // stream-K bound alignment (T4: a tail ring stage = 4 k-tiles)',
    'constexpr int BALIGN = T4 ? (KPS > TS ? KPS : TS) : KPS;   // stream-K bound alignment (split tail: a tail ring stage = TS k-tiles)')
rep('    if (T4 && cl == 0 && !sh) return (uint32_t)NQG * c_epi(cl, dense) + (uint32_t)NKT1 * c_cost_x(cl, sh) * (4u * (NQG - 1) + 1u) / 4u;   // (T4: the tail item k-tiles cost 1/4)',
    '    if (T4 && cl < TS_MAXN / 8 && !sh) return (uint32_t)NQG * c_epi(cl, dense) + (uint32_t)NKT1 * c_cost_x(cl, sh) * ((uint32_t)TS * (NQG - 1) + 1u) / (uint32_t)TS;   // (split tail: the tail item k-tiles cost 1/TS)')
rep('    if (T4 && cl == 0 && !sh && q == NQG - 1) return (int)(q * (uint32_t)NKT1 + (y < e ? 0u : min((uint32_t)NKT1 - 1u, (y - e) * 4u / cc))) & ~(BALIGN - 1);',
    '    if (T4 && cl < TS_MAXN / 8 && !sh && q == NQG - 1) return (int)(q * (uint32_t)NKT1 + (y < e ? 0u : min((uint32_t)NKT1 - 1u, (y - e) * (uint32_t)TS / cc))) & ~(BALIGN - 1);')
rep('static_assert(!T4 || (T4_SCR + 4 * 4 * 128 * 4 <= STG_BYTES && KPS == 1), "T4 epilogue scratch / one k-tile stages");',
    'static_assert(!T4 || (T4_SCR + (TS == 4 ? 4 * 4 : 2 * (TS_MAXN / 2)) * 128 * 4 <= STG_BYTES && KPS == 1), "split-tail epilogue scratch / one k-tile stages");')
rep('__device__ __forceinline__ bool is_t4(const Params& P, const Item& I) { return T4 && I.q == NQG - 1 && (P.dense ? P.md == 8 : I.n <= 8); }',
    '__device__ __forceinline__ bool is_t4(const Params& P, const Item& I) { return T4 && I.q == NQG - 1 && (P.dense ? P.md <= TS_MAXN : I.n <= TS_MAXN); }')
rep('    if (T4 && t4) { for (int w = 0; w < 4; ++w) tma_load_2d(st + STAGE_WS + w * 1024, &P.xmap, (kt + w) * 128, r0, bar); }',
    '    if (T4 && t4) { for (int w = 0; w < TS; ++w) tma_load_2d(st + STAGE_WS + w * P.md * 128, &P.xmap, (kt + w) * 128, r0, bar); }')
rep('''        tma_load_4d(st, P.tmaps + 6, 0, 0, 0, e * NKT1 + kt, bar);
        tma_load_3d(st + STAGE_WS - STAGE_S, P.tmaps + 7, 0, 0, e * NKT1 + kt, bar);''',
    '''        const int sl = e * (4 * NKT1 / TS) + 4 * kt / TS;   // (split-tail matrix: 4 NKT1 / TS k32 slices per expert, k-tile kt / TS)
        tma_load_4d(st, P.tmaps + 6, 0, 0, 0, sl, bar);
        tma_load_3d(st + STAGE_WS - STAGE_S, P.tmaps + 7, 0, 0, sl, bar);''')
rep('        for (int kt = kt0; kt < kt1 && nA < (P.dense ? NST : min(NST, TP2_PASSA)); kt += (t4 ? 4 : 1), ++nA) {',
    '        for (int kt = kt0; kt < kt1 && nA < (P.dense ? NST : min(NST, TP2_PASSA)); kt += (t4 ? TS : 1), ++nA) {')
rep('            if ((nA & (KPS - 1)) == 0) mbar_arrive_expect_tx(&full[nA / KPS], (unsigned)(KPS * (STAGE_WS + (t4 ? 4 : 1) * np * 128)));   // (a stage = KPS k-tiles of one item)',
    '            if ((nA & (KPS - 1)) == 0) mbar_arrive_expect_tx(&full[nA / KPS], (unsigned)(KPS * (STAGE_WS + (t4 ? TS : 1) * np * 128)));   // (a stage = KPS k-tiles of one item)')
rep('        for (int kt = kt0; kt < kt1; kt += (t4 ? 4 : 1), ++n) {',
    '        for (int kt = kt0; kt < kt1; kt += (t4 ? TS : 1), ++n) {')
rep('                    mbar_arrive_expect_tx(fb, (unsigned)(KPS * (STAGE_WS + (t4 ? 4 : 1) * np * 128)));',
    '                    mbar_arrive_expect_tx(fb, (unsigned)(KPS * (STAGE_WS + (t4 ? TS : 1) * np * 128)));')
rep('            else if (T4 && t4) { for (int w = 0; w < 4; ++w) bulk_g2s(st + STAGE_WS + w * 1024, xsrc + (size_t)(kt + w) * np * 128, (unsigned)(np * 128), fb); }',
    '            else if (T4 && t4) { for (int w = 0; w < TS; ++w) bulk_g2s(st + STAGE_WS + w * np * 128, xsrc + (size_t)(kt + w) * np * 128, (unsigned)(np * 128), fb); }')
rep('        fs_s2 = __ldg((TAIL && I.q == NQG - 1 ? (T4 && N == 8 ? P.fc1t4_s2 : P.fc1t_s2) : P.fc1_s2) + I.e);',
    '        fs_s2 = __ldg((TAIL && I.q == NQG - 1 ? (T4 && N <= TS_MAXN ? P.fc1t4_s2 : P.fc1t_s2) : P.fc1_s2) + I.e);')
rep('''    const bool t4 = T4 && N == 8 && I.q == NQG - 1;   // (N8 <=> panel width <= 8 / dense md 8)
    const uint64_t dsc = desc0 + (t4 ? (uint64_t)(L.w * (1024 >> 4)) : 0ull);   // T4: warpgroup w reads x tile w of the stage
    int nk = t4 ? (kt1 - kt0) >> 2 : kt1 - kt0;''',
    '''    const bool t4 = T4 && N <= TS_MAXN && I.q == NQG - 1;   // (T4: N8 <=> panel width <= 8 / dense md 8; T2: N <= 16)
    const uint64_t dsc = desc0 + (t4 ? (uint64_t)((TS == 4 ? L.w : (L.w & 1)) * (N * 128 >> 4)) : 0ull);   // split tail: warpgroup w reads x tile w % TS of the stage
    int nk = t4 ? (kt1 - kt0) / TS : kt1 - kt0;''')
rep('''    if (T4 && t4) {   // the 4 warpgroups' K-quarter partials of the one tail block (rows 0..31 gate, 32..63 up) -> gate / up of WG 0 / 2 rows 0..31''',
    '''    if (T4 && t4 && TS == 2) {   // T2: K-half partials of stack w >> 1: gate = WG0 + WG1, up = WG2 + WG3 (fp32, even + odd k-tiles); WG1 / WG3 rows (OOB) -> 0
        float* S = reinterpret_cast<float*>(smem + OFF_STG + T4_SCR);
        if (L.w & 1) {
#pragma unroll
            for (int i = 0; i < R; ++i) S[((L.w >> 1) * R + i) * 128 + (tid & 127)] = D[i];
        }
        named_bar_sync(1, NCONS);
#pragma unroll
        for (int i = 0; i < R; ++i) {
            const int tk = 8 * (i >> 2) + 2 * L.tig + (i & 1);
            float v = 0.f;
            if (!(L.w & 1)) {
                v = bf16r((D[i] + S[((L.w >> 1) * R + i) * 128 + (tid & 127)]) * T.fs[tk]);
                v = L.w == 0 ? fminf(v, 10.f) : fminf(fmaxf(v, -10.f), 10.f);
            }
            gx[(L.w * (NMAX / 2) + i) * 128 + (tid & 127)] = __float2bfloat16_rn(v);
        }
    } else if (T4 && t4) {   // the 4 warpgroups' K-quarter partials of the one tail block (rows 0..31 gate, 32..63 up) -> gate / up of WG 0 / 2 rows 0..31''')
rep('constexpr int64_t FC1_W_EXPERT_T4 = (int64_t)NKT1 * 4 * HLB, FC1_S_EXPERT_T4 = (int64_t)NKT1 * 256;   // (T4: [slice NKT1][stack 2][blk 2][1 KB] + [slice][2][128])',
    'constexpr int64_t FC1_W_EXPERT_T4 = (int64_t)(4 * NKT1 / (TS > 1 ? TS : 4)) * 4 * HLB, FC1_S_EXPERT_T4 = (int64_t)(4 * NKT1 / (TS > 1 ? TS : 4)) * 256;   // (split tail: [slice 4 NKT1 / TS][stack 2][blk 2][1 KB] + [slice][2][128])')
rep('''                const cuuint64_t d_w[4] = {256, 2, 2, (cuuint64_t)n * NKT1}, s_w[3] = {HLB, 2 * HLB, 4 * HLB};
                const cuuint32_t b_w[4] = {256, 2, 2, 4};
                const cuuint64_t d_s[3] = {128, 2, (cuuint64_t)n * NKT1}, s_s[2] = {128, 256};''',
    '''                const cuuint64_t d_w[4] = {256, 2, 2, (cuuint64_t)n * (4 * NKT1 / TS)}, s_w[3] = {HLB, 2 * HLB, 4 * HLB};
                const cuuint32_t b_w[4] = {256, 2, 2, 4};
                const cuuint64_t d_s[3] = {128, 2, (cuuint64_t)n * (4 * NKT1 / TS)}, s_s[2] = {128, 256};''')
rep('    m.attr("T4") = (int)TP2_NS::T4;', '    m.attr("TS") = (int)TP2_NS::TS;\n    m.attr("T4") = (int)TP2_NS::T4;')

DST.write_text(s)
print(f'{n} patches -> {DST} ({len(s.splitlines())} lines)')
