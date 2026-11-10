"""patch_tail.py: tail (T1 + RS / AG all-reduce) variants for the tp8v41 lineage (tp8v41.cu, and v41tp4.cu generated from it).
Shape- and TP-size-generic, every variant behind a knob that defaults to OFF (same code path / bitwise as the input when off).
    python3 patch_tail.py SRC.cu DST.cu

Knobs (all keep the arithmetic: per-slot T1 sums in slot order, rank-ordered fp32 owner sums -> bitwise == knob off):
  TPB2_TAILOV (0)      RS + AG mode: warps >= TPB2_TAILOV_W1 reduce this rank's owned (row tile, token) pairs and push the all-gather
                       messages WHILE warps < W1 run T1 (instead of after T1 on every warp); then all warps poll the all-gathered rows.
  TPB2_TAILOV_W1 (8)   T1 warps under TAILOV (of 12).
  TPB2_TAILOV_SLEEP (0) ns of __nanosleep between owner polls under TAILOV (0: spin).
  TPB2_TAILB (0)       RS + AG mode: raw bf16 rows (512 B per (256-row tile, token)) + per-(source / owner, row tile) arrival counters
                       instead of LL messages (12 B payload per 16 B): -25% NVLink bytes.  T1 rows -> the owner's RS area; CTA barrier +
                       fence.sys + red.release.sys on the owner's counter [src][rtile]; owner: a warp per owned pair waits for every source's
                       counter of the row tile, sums the sources in rank order (fp32, as the LL path), writes out + every peer's AG area,
                       fence + red.release on the peer's counter [owner][rtile]; receivers copy the AG rows into out.  Areas after the LL
                       parity halves (+1 MB); counters in 4 epoch slots, slot e + 2 zeroed at every kernel start (every build of this
                       source), so builds of both all-reduce modes share one workspace.  Host: peer buffer size + ext.TB_BYTES.
  TPB2_LLS (0)         all-reduce messages carry 8 bf16 per 16 B (no tag): arrival = word w != 0xFFFFFFFF, every slot reset to the
                       sentinel by its reader (parity halves keep a slot's next writer two calls away); T1 lane l = message l (no smem
                       staging): -25% NVLink bytes vs LL (6 bf16 + 32-bit tag).  Host: the LL region memset to 0xFF once; every build
                       sharing the workspace must be LLS.  A computed NaN pair 0xFFFF 0xFFFF in word w is sent as canonical NaNs.
Debug stamps (TPB2_DBG): 14 = this CTA's owner reduce done, 15 = its all-gather receive done.
"""
import sys
from pathlib import Path

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
s = SRC.read_text()
n = 0


def rep(old, new, count=1):
    global s, n
    c = s.count(old)
    assert c == count, f'expected {count}, found {c}: {old[:300]}'
    s = s.replace(old, new)
    n += 1


# ---- all suite_b200 tail knobs: defaults early in the file (before any use)
rep('#ifndef TPB2_F1F\n',
    '#ifndef TPB2_TAILOV\n#define TPB2_TAILOV 0   // (suite_b200 tail) RS + AG: owner reduce on warps >= TAILOV_W1 concurrently with T1 on warps < W1\n#endif\n'
    '#ifndef TPB2_TAILOV_W1\n#define TPB2_TAILOV_W1 8\n#endif\n'
    '#ifndef TPB2_TAILOV_SLEEP\n#define TPB2_TAILOV_SLEEP 0\n#endif\n'
    '#ifndef TPB2_TAILB\n#define TPB2_TAILB 0   // (suite_b200 tail) RS + AG with raw bf16 rows + arrival counters instead of LL messages\n#endif\n'
    '#ifndef TPB2_TB_SLEEP\n#define TPB2_TB_SLEEP 200   // (TAILB) ns between counter polls\n#endif\n'
    '#ifndef TPB2_TB_FSC\n#define TPB2_TB_FSC 0   // (TAILB) 1: fence.sc.sys, 0: fence.acq_rel.sys before the counter updates\n#endif\n'
    '#ifndef TPB2_TB_NOF\n#define TPB2_TB_NOF 0   // (TAILB diagnostic) 1: no fences (UNSAFE)\n#endif\n'
    '#ifndef TPB2_LLS\n#define TPB2_LLS 0   // (suite_b200 tail) all-reduce messages = 8 bf16 per 16 B; arrival = word w != 0xFFFFFFFF (slots reset by their reader)\n#endif\n'
    '#ifndef TPB2_ARB\n#define TPB2_ARB 8   // (v41 RS + AG) all-gathered messages polled in flight per thread\n#endif\n'
    '#ifndef TPB2_F1F\n')

# ---- tail(): the overlap decision + T1 warp count
rep('''    const int G = TPB2_GRID ? TPB2_GRID : static_cast<int>(gridDim.x), b = blockIdx.x, M = pM(P);
    const unsigned need = static_cast<unsigned>(T.misc[2] + T.misc[4]) * 2u;''',
    '''    const int G = TPB2_GRID ? TPB2_GRID : static_cast<int>(gridDim.x), b = blockIdx.x, M = pM(P);
    // (TAILOV) RS + AG with the plain T1PW T1: owner reduce on warps >= NWT while warps < NWT run T1
    const bool tov = TPB2_TAILOV && TPB2_T1PW > 0 && !TPB2_FC2ACC && !(TPB2_T1W && TPB2_SFCP_MMA) && !TPB2_PROBE_T1LOCAL && pND(P) > 1 && !pOS(P) &&
                     !t1l_on(M) && !fc2l_on(P) && !t1ll_on(P);
    const int NWT = tov ? TPB2_TAILOV_W1 : THREADS / 32, wtail = tid >> 5;
    static_assert(!TPB2_TAILOV || (TPB2_TAILOV_W1 >= 1 && TPB2_TAILOV_W1 < THREADS / 32), "TAILOV: T1 warps");
    const unsigned need = static_cast<unsigned>(T.misc[2] + T.misc[4]) * 2u;''')
rep('''    for (;;) {
        if (TPB2_T1W && TPB2_SFCP_MMA) break;   // (T1: warp 7 during FC2 + the loop above)''',
    '''    if (!tov || wtail < NWT) for (;;) {
        if (TPB2_T1W && TPB2_SFCP_MMA) break;   // (T1: warp 7 during FC2 + the loop above)''')
rep('''            *s_item = it;
        }
        __syncthreads();
        const int it = *s_item;
        __syncthreads();
        if (it >= nitem) break;''',
    '''            *s_item = it;
        }
        if (tov) named_bar(15, 32 * NWT); else __syncthreads();
        const int it = *s_item;
        if (tov) named_bar(15, 32 * NWT); else __syncthreads();
        if (it >= nitem) break;''')
rep('''        constexpr int NW = THREADS / 32;
        if (TPB2_FC2ACC) {''', '''        const int NW = NWT;   // (TAILOV: the T1 warps only)
        if (TPB2_FC2ACC) {''')
# ---- RS + AG: the owner loop on the reduce warps (TAILOV) or every thread; stamps; join before the all-gather receive
rep('''        const int npo = NRT * mown, mineo = b < npo ? (npo - b + G - 1) / G : 0;
        for (int x = tid; x < mineo * NMSG; x += THREADS) {
            const int pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G, rtile = j / mown, t = rk + NDEV * (j - rtile * mown);
            uint4 v[NDEV];
            for (;;) {
#pragma unroll
                for (int d = 0; d < NDEV; ++d) v[d] = ld_ll(rs_me + os_idx(d, rtile, t, m));
                bool ok = true;
#pragma unroll
                for (int d = 0; d < NDEV; ++d) ok &= v[d].w == epoch;
                if (ok) break;
            }''',
    '''        const int npo = NRT * mown, mineo = b < npo ? (npo - b + G - 1) / G : 0;
        const int xo = tov ? tid - 32 * NWT : tid, xst = tov ? THREADS - 32 * NWT : THREADS;
        for (int x = (!tov || wtail >= NWT) ? xo : mineo * NMSG; x < mineo * NMSG; x += xst) {
            const int pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G, rtile = j / mown, t = rk + NDEV * (j - rtile * mown);
            uint4 v[NDEV];
            for (;;) {
#pragma unroll
                for (int d = 0; d < NDEV; ++d) v[d] = ld_ll(rs_me + os_idx(d, rtile, t, m));
                bool ok = true;
#pragma unroll
                for (int d = 0; d < NDEV; ++d) ok &= v[d].w == epoch;
                if (ok) break;
                if (TPB2_TAILOV_SLEEP && tov) __nanosleep(TPB2_TAILOV_SLEEP);
            }''')
rep('''        // the other ranks' tokens (i < M - mown: t = 8 (i / 7) + the (i % 7)-th residue != rk): their owner's reduced messages, TPB2_ARB
        // polls in flight per thread (one round trip per batch, not per message)''',
    '''        if (DBGP && tid == (tov ? 32 * NWT : 0)) DBGP[b * 32 + 14] = gtime();
        if (tov) __syncthreads();   // (TAILOV: T1 warps + reduce warps; every owned pair sent before any all-gathered one is polled)
        // the other ranks' tokens (i < M - mown: t = 8 (i / 7) + the (i % 7)-th residue != rk): their owner's reduced messages, TPB2_ARB
        // polls in flight per thread (one round trip per batch, not per message)''')
rep('''                while (v[q].w != epoch) v[q] = ld_ll(pm[q]);
                const int m = (x0 + q * THREADS) % NMSG;
                po[q][0] = v[q].x; po[q][1] = v[q].y;
                if (m < NMSG - 1) po[q][2] = v[q].z;
            }
        }
    }
}''',
    '''                while (v[q].w != epoch) v[q] = ld_ll(pm[q]);
                const int m = (x0 + q * THREADS) % NMSG;
                po[q][0] = v[q].x; po[q][1] = v[q].y;
                if (m < NMSG - 1) po[q][2] = v[q].z;
            }
        }
        if (DBGP && tid == 0) DBGP[b * 32 + 15] = gtime();
    }
}''')

# ================================================================ TAILB (raw bf16 rows + arrival counters)
rep('constexpr size_t AG_PAR = static_cast<size_t>(NRT) * MAXM * NMSG;\n',
    'constexpr size_t AG_PAR = static_cast<size_t>(NRT) * MAXM * NMSG;\n'
    '// (TAILB) after the LL parity halves + 1 MB of a peer buffer: per parity [RS rows: src x NRT x TB_MO x 512 B][AG rows: NRT x MAXM x 512 B],\n'
    '// then the arrival counters u32 [4 epoch slots][2 kinds (0 RS by source, 1 AG by owner)][NDEV][NRT]\n'
    'constexpr int TB_MO = (MAXM + NDEV - 1) / NDEV;\n'
    'constexpr size_t TB_OFF = 2 * RS_PAR * 16 + (1u << 20);\n'
    'constexpr size_t TB_RSZ = static_cast<size_t>(NDEV) * NRT * TB_MO * 512, TB_AGZ = static_cast<size_t>(NRT) * MAXM * 512;\n'
    'constexpr size_t TB_PAR = TB_RSZ + TB_AGZ, TB_CNT = TB_OFF + 2 * TB_PAR, TB_NCNT = 2 * NDEV * NRT;\n'
    'constexpr size_t TB_BYTES = 2 * TB_PAR + 4 * TB_NCNT * 4;\n'
    'static_assert(TB_NCNT <= THREADS, "TAILB: one thread per counter of a slot");\n'
    '__device__ __forceinline__ uint8_t* tb_area(const Params& P, int r, unsigned epoch) { return reinterpret_cast<uint8_t*>(P.rs[r]) + TB_OFF + (epoch & 1u) * TB_PAR; }\n'
    '__device__ __forceinline__ unsigned* tb_cnt(const Params& P, int r, unsigned epoch, int kind, int src, int rt) {\n'
    '    return reinterpret_cast<unsigned*>(reinterpret_cast<uint8_t*>(P.rs[r]) + TB_CNT) + (((epoch & 3u) * 2 + kind) * NDEV + src) * NRT + rt;\n'
    '}\n'
    '__device__ __forceinline__ unsigned tb_ld_acq(const unsigned* p) { unsigned v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }\n'
    '__device__ __forceinline__ void tb_red_rel(unsigned* p, unsigned v) { if (TPB2_TB_NOF) asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); else asm volatile("red.release.sys.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }\n'
    '__device__ __forceinline__ void tb_fence() { if (TPB2_TB_NOF) return; if (TPB2_TB_FSC) __threadfence_system(); else asm volatile("fence.acq_rel.sys;" ::: "memory"); }\n'
    '__device__ __forceinline__ void tb_st(void* p, uint4 v) { asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }\n')
# T1 result of (rtile, t) as a raw row slice to the token's owner (the same values t1_fin packs into messages)
rep('__device__ __forceinline__ void t1_pair(const Params& P, const Tabs& T, bool small_rw, int rtile, int t, unsigned epoch, __nv_bfloat16* stg, int lane,',
    '__device__ __forceinline__ void t1_fin_tb(const Params& P, int rtile, int t, unsigned epoch, int lane, const float* acc) {   // (TAILB)\n'
    '    uint32_t o4[4];\n'
    '#pragma unroll\n'
    '    for (int j = 0; j < 4; ++j) o4[j] = pack_bf16x2(__fmul_rn(P.factor, bf16r(acc[2 * j])), __fmul_rn(P.factor, bf16r(acc[2 * j + 1])));\n'
    '    uint8_t* dst = tb_area(P, t & (NDEV - 1), epoch) + ((static_cast<size_t>(pRK(P)) * NRT + rtile) * TB_MO + t / NDEV) * 512 + 16 * lane;\n'
    '    tb_st(dst, make_uint4(o4[0], o4[1], o4[2], o4[3]));\n'
    '}\n'
    '__device__ __forceinline__ void t1_pair(const Params& P, const Tabs& T, bool small_rw, int rtile, int t, unsigned epoch, __nv_bfloat16* stg, int lane,')
# kernel start: zero the counters of epoch slot e + 2 (every build of this source does it, so builds of both modes share a workspace)
rep('    const unsigned epoch = __ldcg(P.work + (TPB2_CEPOCH ? 64 + b : 0)) + 1u, par = epoch & 1u;\n',
    '    const unsigned epoch = __ldcg(P.work + (TPB2_CEPOCH ? 64 + b : 0)) + 1u, par = epoch & 1u;\n'
    '    if (TPB2_TAILB && b == G - 1 && tid < static_cast<int>(TB_NCNT) && pND(P) > 1) {   // (TAILB) arrival counters of epoch slot e + 2 (calls e - 2, e + 2): zero\n'
    '        unsigned* c = reinterpret_cast<unsigned*>(reinterpret_cast<uint8_t*>(P.rs[pRK(P)]) + TB_CNT) + ((epoch + 2u) & 3u) * TB_NCNT + tid;\n'
    '        asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(c), "r"(0u) : "memory");\n'
    '        tb_fence();\n'
    '    }\n')
# tail: the TAILB mode flag
rep('    const int NWT = tov ? TPB2_TAILOV_W1 : THREADS / 32, wtail = tid >> 5;\n',
    '    const int NWT = tov ? TPB2_TAILOV_W1 : THREADS / 32, wtail = tid >> 5;\n'
    '    const bool tb = TPB2_TAILB && TPB2_T1PW > 0 && !TPB2_FC2ACC && !(TPB2_T1W && TPB2_SFCP_MMA) && !TPB2_PROBE_T1LOCAL && pND(P) > 1 && !pOS(P) &&\n'
    '                    !t1l_on(M) && !fc2l_on(P) && !t1ll_on(P);\n')
# T1PW: raw rows, then the item's arrival counts on every owner
rep('                    t1_fin(P, rtile, t0 + tt + j * NW, epoch, stg1, lane, acc);\n'
    '                }\n'
    '            }\n'
    '            if (DBGP && tid == 0 && done_it == 1u) DBGP[b * 32 + 13] = gtime();\n'
    '            continue;\n'
    '        }\n'
    '        if (TPB2_T1BK) {',
    '                    if (TPB2_TAILB && tb) t1_fin_tb(P, rtile, t0 + tt + j * NW, epoch, lane, acc);\n'
    '                    else t1_fin(P, rtile, t0 + tt + j * NW, epoch, stg1, lane, acc);\n'
    '                }\n'
    '            }\n'
    '            if (TPB2_TAILB && tb) {   // (TAILB) the item rows of owner o have landed: + their count on o\'s counter [this rank][rtile]\n'
    '                tb_fence();\n'
    '                if (tov) named_bar(15, 32 * NWT); else __syncthreads();\n'
    '                if (tid < NDEV) {\n'
    '                    const int o = tid, f = t0 + ((o - t0) % NDEV + NDEV) % NDEV, no = f < t0 + nt ? (t0 + nt - 1 - f) / NDEV + 1 : 0;\n'
    '                    if (no) tb_red_rel(tb_cnt(P, o, epoch, 0, pRK(P), rtile), static_cast<unsigned>(no));\n'
    '                    if (DBGP && tid == 0) DBGP[b * 32 + 21] = gtime();\n'
    '                }\n'
    '            }\n'
    '            if (DBGP && tid == 0 && done_it == 1u) DBGP[b * 32 + 13] = gtime();\n'
    '            continue;\n'
    '        }\n'
    '        if (TPB2_T1BK) {')
# owner reduce + AG send: a warp per owned pair
rep('        const int xo = tov ? tid - 32 * NWT : tid, xst = tov ? THREADS - 32 * NWT : THREADS;\n'
    '        for (int x = (!tov || wtail >= NWT) ? xo : mineo * NMSG; x < mineo * NMSG; x += xst) {',
    '        const int xo = tov ? tid - 32 * NWT : tid, xst = tov ? THREADS - 32 * NWT : THREADS;\n'
    '        if (TPB2_TAILB && tb) {   // (TAILB) warp per owned pair (rtile, token t = rk + NDEV i)\n'
    '            const int lane = tid & 31, nwo = tov ? THREADS / 32 - NWT : THREADS / 32, w0 = tov ? wtail - NWT : wtail;\n'
    '            const uint8_t* myrs = tb_area(P, rk, epoch);\n'
    '            for (int pi = (!tov || wtail >= NWT) ? w0 : mineo; pi < mineo; pi += nwo) {\n'
    '                const int j = b + pi * G, rtile = j / mown, i = j - rtile * mown, t = rk + NDEV * i;\n'
    '                if (lane < NDEV) { const unsigned* c = tb_cnt(P, rk, epoch, 0, lane, rtile); while (tb_ld_acq(c) < static_cast<unsigned>(mown)) { if (TPB2_TB_SLEEP) __nanosleep(TPB2_TB_SLEEP); } }\n'
    '                __syncwarp();\n'
    '                if (DBGP && lane == 0 && pi == w0) DBGP[b * 32 + 20] = gtime();\n'
    '                uint4 v[NDEV];\n'
    '#pragma unroll\n'
    '                for (int d = 0; d < NDEV; ++d) v[d] = __ldcv(reinterpret_cast<const uint4*>(myrs + ((static_cast<size_t>(d) * NRT + rtile) * TB_MO + i) * 512) + lane);\n'
    '                float a[8] = {bf16lo(v[0].x), bf16hi(v[0].x), bf16lo(v[0].y), bf16hi(v[0].y), bf16lo(v[0].z), bf16hi(v[0].z), bf16lo(v[0].w), bf16hi(v[0].w)};\n'
    '#pragma unroll\n'
    '                for (int d = 1; d < NDEV; ++d) {\n'
    '                    a[0] = __fadd_rn(a[0], bf16lo(v[d].x)); a[1] = __fadd_rn(a[1], bf16hi(v[d].x));\n'
    '                    a[2] = __fadd_rn(a[2], bf16lo(v[d].y)); a[3] = __fadd_rn(a[3], bf16hi(v[d].y));\n'
    '                    a[4] = __fadd_rn(a[4], bf16lo(v[d].z)); a[5] = __fadd_rn(a[5], bf16hi(v[d].z));\n'
    '                    a[6] = __fadd_rn(a[6], bf16lo(v[d].w)); a[7] = __fadd_rn(a[7], bf16hi(v[d].w));\n'
    '                }\n'
    '                const uint4 o4 = make_uint4(pack_bf16x2(a[0], a[1]), pack_bf16x2(a[2], a[3]), pack_bf16x2(a[4], a[5]), pack_bf16x2(a[6], a[7]));\n'
    '                *reinterpret_cast<uint4*>(P.out + static_cast<size_t>(t) * DIM + rtile * 256 + 8 * lane) = o4;\n'
    '#pragma unroll\n'
    '                for (int d = 1; d < NDEV; ++d)\n'
    '                    tb_st(tb_area(P, (rk + d) & (NDEV - 1), epoch) + TB_RSZ + (static_cast<size_t>(rtile) * MAXM + t) * 512 + 16 * lane, o4);\n'
    '                tb_fence();\n'
    '                __syncwarp();\n'
    '                if (lane >= 1 && lane < NDEV) tb_red_rel(tb_cnt(P, (rk + lane) & (NDEV - 1), epoch, 1, rk, rtile), 1u);\n'
    '                if (DBGP && lane == 0 && pi == w0) DBGP[b * 32 + 23] = gtime();\n'
    '            }\n'
    '        }\n'
    '        for (int x = (TPB2_TAILB && tb) ? mineo * NMSG : ((!tov || wtail >= NWT) ? xo : mineo * NMSG); x < mineo * NMSG; x += xst) {')
# AG receive: a warp per pair of another owner's token
rep('        const int mo = M - mown, npr = NRT * mo, miner = b < npr ? (npr - b + G - 1) / G : 0, nrx = miner * NMSG;\n'
    '        constexpr int RB = TPB2_ARB;\n'
    '        for (int x0 = tid; x0 < nrx; x0 += RB * THREADS) {',
    '        const int mo = M - mown, npr = NRT * mo, miner = b < npr ? (npr - b + G - 1) / G : 0, nrx = miner * NMSG;\n'
    '        if (TPB2_TAILB && tb) {   // (TAILB) warp per pair of another owner\'s token: wait for that owner\'s counter of the row tile, copy its row\n'
    '            const int lane = tid & 31;\n'
    '            const uint8_t* myag = tb_area(P, rk, epoch) + TB_RSZ;\n'
    '            for (int pi = wtail; pi < miner; pi += THREADS / 32) {\n'
    '                const int j = b + pi * G, rtile = j / mo, i = j - rtile * mo, rr = i % (NDEV - 1);\n'
    '                const int t = NDEV * (i / (NDEV - 1)) + (rr < rk ? rr : rr + 1), o = t & (NDEV - 1), mown_o = (M - o + NDEV - 1) / NDEV;\n'
    '                if (lane == 0) { const unsigned* c = tb_cnt(P, rk, epoch, 1, o, rtile); while (tb_ld_acq(c) < static_cast<unsigned>(mown_o)) { if (TPB2_TB_SLEEP) __nanosleep(TPB2_TB_SLEEP); } }\n'
    '                __syncwarp();\n'
    '                const uint4 v = __ldcv(reinterpret_cast<const uint4*>(myag + (static_cast<size_t>(rtile) * MAXM + t) * 512) + lane);\n'
    '                *reinterpret_cast<uint4*>(P.out + static_cast<size_t>(t) * DIM + rtile * 256 + 8 * lane) = v;\n'
    '            }\n'
    '        }\n'
    '        constexpr int RB = TPB2_ARB;\n'
    '        for (int x0 = (TPB2_TAILB && tb) ? nrx : tid; x0 < nrx; x0 += RB * THREADS) {')
rep('    m.attr("MAXM") = TPB2_NS::MAXM;',
    '    m.attr("TB_BYTES") = static_cast<int64_t>(TPB2_NS::TB_BYTES); m.attr("TAILB") = static_cast<int64_t>(TPB2_TAILB);\n    m.attr("MAXM") = TPB2_NS::MAXM;')

# ================================================================ LLS: sentinel LL messages (8 bf16 per 16 B, no tag)
rep('__device__ __forceinline__ void t1_emit(const Params& P, int rtile, int t, unsigned epoch, __nv_bfloat16* stg, int lane, const T1In& in, Xt* xt) {',
    '// (LLS) message = 8 bf16 (a lane\'s 16 B of a 256-row tile); the w word never equals the sentinel (a NaN pair 0xFFFF 0xFFFF -> canonical NaNs)\n'
    'constexpr uint32_t LLS_S = 0xFFFFFFFFu;\n'
    '__device__ __forceinline__ uint4 lls_msg(uint32_t x, uint32_t y, uint32_t z, uint32_t w) { return make_uint4(x, y, z, w == LLS_S ? 0x7FC07FC0u : w); }\n'
    '__device__ __forceinline__ void lls_send(const Params& P, int rtile, int t, unsigned epoch, int lane, const uint32_t* o4) {   // (LLS) T1 row slice\n'
    '    const uint4 msg = lls_msg(o4[0], o4[1], o4[2], o4[3]);\n'
    '    if (pOS(P)) {\n'
    '#pragma unroll\n'
    '        for (int d = 0; d < NDEV; ++d) st_ll(P.rs[(pRK(P) + d) & (NDEV - 1)] + (epoch & 1u) * RS_PAR + os_idx(pRK(P), rtile, t, lane), msg);\n'
    '    } else {\n'
    '        st_ll(P.rs[TPB2_PROBE_T1LOCAL ? pRK(P) : (t & (NDEV - 1))] + (epoch & 1u) * RS_PAR + os_idx(pRK(P), rtile, t, lane), msg);\n'
    '    }\n'
    '}\n'
    '__device__ __forceinline__ void lls_acc(const uint4& v, float* a, bool first) {\n'
    '    const uint32_t w4[4] = {v.x, v.y, v.z, v.w};\n'
    '#pragma unroll\n'
    '    for (int j = 0; j < 4; ++j) {\n'
    '        a[2 * j] = first ? bf16lo(w4[j]) : __fadd_rn(a[2 * j], bf16lo(w4[j]));\n'
    '        a[2 * j + 1] = first ? bf16hi(w4[j]) : __fadd_rn(a[2 * j + 1], bf16hi(w4[j]));\n'
    '    }\n'
    '}\n'
    '// (LLS) the all-reduce of the T1 rows: one-shot (every source -> every rank) or RS + AG by token owner t % NDEV; sums in rank order\n'
    '__device__ __noinline__ void lls_allreduce(const Params& P, int b, int G, int M, int tid, unsigned epoch) {\n'
    '    const uint4 S4 = make_uint4(LLS_S, LLS_S, LLS_S, LLS_S);\n'
    '    const int rk = pRK(P);\n'
    '    uint4* rs_me = P.rs[rk] + (epoch & 1u) * RS_PAR;\n'
    '    if (pOS(P)) {\n'
    '        const int npair = NRT * M, mine = b < npair ? (npair - b + G - 1) / G : 0;\n'
    '        for (int x = tid; x < mine * 32; x += THREADS) {\n'
    '            const int pi = x >> 5, m = x & 31, j = b + pi * G, rtile = j / M, t = j - rtile * M;\n'
    '            uint4 v[NDEV];\n'
    '            for (;;) {\n'
    '#pragma unroll\n'
    '                for (int d = 0; d < NDEV; ++d) v[d] = ld_ll(rs_me + os_idx(d, rtile, t, m));\n'
    '                bool ok = true;\n'
    '#pragma unroll\n'
    '                for (int d = 0; d < NDEV; ++d) ok &= v[d].w != LLS_S;\n'
    '                if (ok) break;\n'
    '            }\n'
    '            float a[8];\n'
    '#pragma unroll\n'
    '            for (int d = 0; d < NDEV; ++d) lls_acc(v[d], a, d == 0);\n'
    '            *reinterpret_cast<uint4*>(P.out + static_cast<size_t>(t) * DIM + rtile * 256 + 8 * m) =\n'
    '                make_uint4(pack_bf16x2(a[0], a[1]), pack_bf16x2(a[2], a[3]), pack_bf16x2(a[4], a[5]), pack_bf16x2(a[6], a[7]));\n'
    '#pragma unroll\n'
    '            for (int d = 0; d < NDEV; ++d) *(rs_me + os_idx(d, rtile, t, m)) = S4;   // (slots free for the call after next)\n'
    '        }\n'
    '        return;\n'
    '    }\n'
    '    const int mown = M > rk ? (M - rk + NDEV - 1) / NDEV : 0;\n'
    '    const int npo = NRT * mown, mineo = b < npo ? (npo - b + G - 1) / G : 0;\n'
    '    for (int x = tid; x < mineo * 32; x += THREADS) {   // owner: reduce, write out, push to every other rank (this rank\'s source slot there)\n'
    '        const int pi = x >> 5, m = x & 31, j = b + pi * G, rtile = j / mown, t = rk + NDEV * (j - rtile * mown);\n'
    '        uint4 v[NDEV];\n'
    '        for (;;) {\n'
    '#pragma unroll\n'
    '            for (int d = 0; d < NDEV; ++d) v[d] = ld_ll(rs_me + os_idx(d, rtile, t, m));\n'
    '            bool ok = true;\n'
    '#pragma unroll\n'
    '            for (int d = 0; d < NDEV; ++d) ok &= v[d].w != LLS_S;\n'
    '            if (ok) break;\n'
    '        }\n'
    '        float a[8];\n'
    '#pragma unroll\n'
    '        for (int d = 0; d < NDEV; ++d) lls_acc(v[d], a, d == 0);\n'
    '        const uint4 mo = lls_msg(pack_bf16x2(a[0], a[1]), pack_bf16x2(a[2], a[3]), pack_bf16x2(a[4], a[5]), pack_bf16x2(a[6], a[7]));\n'
    '        *reinterpret_cast<uint4*>(P.out + static_cast<size_t>(t) * DIM + rtile * 256 + 8 * m) =\n'
    '            make_uint4(pack_bf16x2(a[0], a[1]), pack_bf16x2(a[2], a[3]), pack_bf16x2(a[4], a[5]), pack_bf16x2(a[6], a[7]));\n'
    '#pragma unroll\n'
    '        for (int d = 1; d < NDEV; ++d) st_ll(P.rs[(rk + d) & (NDEV - 1)] + (epoch & 1u) * RS_PAR + os_idx(rk, rtile, t, m), mo);\n'
    '#pragma unroll\n'
    '        for (int d = 0; d < NDEV; ++d) *(rs_me + os_idx(d, rtile, t, m)) = S4;\n'
    '    }\n'
    '    const int mo = M - mown, npr = NRT * mo, miner = b < npr ? (npr - b + G - 1) / G : 0, nrx = miner * 32;\n'
    '    constexpr int RB = TPB2_ARB;\n'
    '    for (int x0 = tid; x0 < nrx; x0 += RB * THREADS) {   // the other ranks\' tokens: their owner\'s pushed rows, RB polls in flight\n'
    '        uint4 v[RB];\n'
    '        uint4* pm[RB];\n'
    '        int ti[RB], ri[RB];\n'
    '#pragma unroll\n'
    '        for (int q = 0; q < RB; ++q) {\n'
    '            const int x = x0 + q * THREADS;\n'
    '            pm[q] = nullptr; ti[q] = 0; ri[q] = 0; v[q] = S4;\n'
    '            if (x >= nrx) continue;\n'
    '            const int pi = x >> 5, m = x & 31, j = b + pi * G, rtile = j / mo, i = j - rtile * mo, rr = i % (NDEV - 1);\n'
    '            const int t = NDEV * (i / (NDEV - 1)) + (rr < rk ? rr : rr + 1);\n'
    '            pm[q] = rs_me + os_idx(t & (NDEV - 1), rtile, t, m);\n'
    '            ti[q] = t; ri[q] = rtile * 256 + 8 * m;\n'
    '            v[q] = ld_ll(pm[q]);\n'
    '        }\n'
    '#pragma unroll\n'
    '        for (int q = 0; q < RB; ++q) {\n'
    '            if (!pm[q]) continue;\n'
    '            while (v[q].w == LLS_S) v[q] = ld_ll(pm[q]);\n'
    '            *reinterpret_cast<uint4*>(P.out + static_cast<size_t>(ti[q]) * DIM + ri[q]) = v[q];\n'
    '            *pm[q] = S4;\n'
    '        }\n'
    '    }\n'
    '}\n'
    '__device__ __forceinline__ void t1_emit(const Params& P, int rtile, int t, unsigned epoch, __nv_bfloat16* stg, int lane, const T1In& in, Xt* xt) {')
rep('''        o4[j] = pack_bf16x2(y0, y1);
    }
    *reinterpret_cast<uint4*>(stg + 8 * lane) = make_uint4(o4[0], o4[1], o4[2], o4[3]);
    __syncwarp();
    if (xt && lane == 0) { xt->rec(XT_END(56)); xt->rec(XT_BEG(57)); }''',
    '''        o4[j] = pack_bf16x2(y0, y1);
    }
    if (TPB2_LLS && pND(P) > 1) { lls_send(P, rtile, t, epoch, lane, o4); return; }   // (LLS) lane l = message l
    *reinterpret_cast<uint4*>(stg + 8 * lane) = make_uint4(o4[0], o4[1], o4[2], o4[3]);
    __syncwarp();
    if (xt && lane == 0) { xt->rec(XT_END(56)); xt->rec(XT_BEG(57)); }''')
rep('''    for (int j = 0; j < 4; ++j) o4[j] = pack_bf16x2(__fmul_rn(P.factor, bf16r(acc[2 * j])), __fmul_rn(P.factor, bf16r(acc[2 * j + 1])));
    *reinterpret_cast<uint4*>(stg + 8 * lane) = make_uint4(o4[0], o4[1], o4[2], o4[3]);''',
    '''    for (int j = 0; j < 4; ++j) o4[j] = pack_bf16x2(__fmul_rn(P.factor, bf16r(acc[2 * j])), __fmul_rn(P.factor, bf16r(acc[2 * j + 1])));
    if (TPB2_LLS && pND(P) > 1) { lls_send(P, rtile, t, epoch, lane, o4); return; }   // (LLS) lane l = message l
    *reinterpret_cast<uint4*>(stg + 8 * lane) = make_uint4(o4[0], o4[1], o4[2], o4[3]);''')
rep('''    if (pND(P) == 1) return;
    if (TPB2_PROBE_T1LOCAL) return;   // (probe: no all-reduce -- wrong output)''',
    '''    if (pND(P) == 1) return;
    if (TPB2_PROBE_T1LOCAL) return;   // (probe: no all-reduce -- wrong output)
    if (TPB2_LLS) {   // (LLS) sentinel messages: their own receive / reduce / push loops
        static_assert(!TPB2_LLS || (!TPB2_K3F && !TPB2_UPF && !TPB2_FC2L && !TPB2_T1LL_MAXM && !TPB2_TAILB && !TPB2_TAILOV), "LLS: the plain routed tail");
        lls_allreduce(P, b, G, M, tid, epoch);
        return;
    }''')
rep('    m.attr("MAXM") = TPB2_NS::MAXM;', '    m.attr("LLS") = static_cast<int64_t>(TPB2_LLS); m.attr("RS_LL_BYTES") = static_cast<int64_t>(2 * TPB2_NS::RS_PAR * 16);\n    m.attr("MAXM") = TPB2_NS::MAXM;')
DST.write_text(s)
print(f'{DST}: {n} patches')
