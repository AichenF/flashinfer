"""patch_v77large.py: v77 tpmoe_b2.cu (after patch_ro.py / patch_ndev.py) -> large-M knobs (default OFF: unchanged when off).
    python3 patch_v77large.py [v77/tpmoe_b2.cu]
  TPB2_IDV (0)   1 / 2: build_tables without a fused gate: every routing id loaded as int4 vectors, all of a thread's loads in flight together before
                 the clear phase (M * 6 / 4 <= MAXV x THREADS), instead of one dependent global load per loop iteration (M512: 8 cold
                 round trips ~ 12 us of the 14 us table phase).  2: the loads issued before the x quantization (land under it).
                 Index structures only: outputs bitwise unchanged.
  TPB2_LLS (0)   sentinel all-reduce messages (8 bf16 / 16 B, no tag, reader re-arms the slot): -25% NVLink bytes, bitwise.
  (refuted, removed: IDV 3 = __match_any_sync-merged table atomics (+1..4 us); PFD = L2 prefetch of the weight stream PFD stages ahead
   (+7..13 us at M128..512).)
"""
import sys
from pathlib import Path

P = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'v77' / 'tpmoe_b2.cu')
s = P.read_text()
n = 0


def rep(old, new, count=1):
    global s, n
    c = s.count(old)
    assert c == count, f'expected {count}, found {c}: {old[:200]}'
    s = s.replace(old, new)
    n += 1


rep('#define TPB2_IDPRE 0   // build_tables (M > SMALLM): the global ids loaded into registers before the clear phase (one L2 round trip hidden)\n#endif\n',
    '#define TPB2_IDPRE 0   // build_tables (M > SMALLM): the global ids loaded into registers before the clear phase (one L2 round trip hidden)\n#endif\n'
    '#ifndef TPB2_IDV\n#define TPB2_IDV 0   // (suite_b200) build_tables, ids given: int4 vector loads of all ids, every load in flight before the clear phase\n#endif\n')
rep('''    int ide[2] = {-1, -1};
    const bool idpre = TPB2_IDPRE && !gids && MT <= 2 * THREADS;''',
    '''    int ide[2] = {-1, -1};
    const bool idpre = TPB2_IDPRE && !gids && MT <= 2 * THREADS;
    constexpr int MAXV = ((TPB2_MMAX ? TPB2_MMAX : MAXM) * 6 / 4 + THREADS - 1) / THREADS;   // (IDV) int4 loads per thread
    const bool idv = TPB2_IDV && !gids && topk == 6 && (MT & 3) == 0 && (reinterpret_cast<uintptr_t>(P.ids) & 15) == 0;
    int4 iv[MAXV];
    if (idv) {
#pragma unroll
        for (int j = 0; j < MAXV; ++j) {
            const int v = tid + j * THREADS;
            iv[j] = v < MT / 4 ? __ldcg(reinterpret_cast<const int4*>(P.ids) + v) : make_int4(-1, -1, -1, -1);
        }
    }''')
rep('''    for (int p = tid; p < MT; p += THREADS) {
        const int e = idpre ? ide[(p - tid) / THREADS] : (gids ? gids[p] : __ldcg(P.ids + p));
        const int t = p / topk, k = p - t * topk;
        if (static_cast<unsigned>(e) < static_cast<unsigned>(NEXP)) {
            T.ids[t * 8 + k] = static_cast<uint16_t>(e);
            atomicOr(&bits[(t >> 5) * NEXP + e], 1u << (t & 31));
        }
    }''',
    '''    if (idv) {   // (IDV) the vector-loaded ids: entries p = 4 v .. 4 v + 3
#pragma unroll
        for (int j = 0; j < MAXV; ++j) {
            const int v = tid + j * THREADS;
            if (v >= MT / 4) continue;
            const int e4[4] = {iv[j].x, iv[j].y, iv[j].z, iv[j].w};
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int p = 4 * v + q, e = e4[q], t = p / 6, k = p - t * 6;
                if (static_cast<unsigned>(e) < static_cast<unsigned>(NEXP)) {
                    T.ids[t * 8 + k] = static_cast<uint16_t>(e);
                    atomicOr(&bits[(t >> 5) * NEXP + e], 1u << (t & 31));
                }
            }
        }
    } else
    for (int p = tid; p < MT; p += THREADS) {
        const int e = idpre ? ide[(p - tid) / THREADS] : (gids ? gids[p] : __ldcg(P.ids + p));
        const int t = p / topk, k = p - t * topk;
        if (static_cast<unsigned>(e) < static_cast<unsigned>(NEXP)) {
            T.ids[t * 8 + k] = static_cast<uint16_t>(e);
            atomicOr(&bits[(t >> 5) * NEXP + e], 1u << (t & 31));
        }
    }''')

# ---- TPB2_IDV 2: the vector id loads issued before the x quantization (they land under it, not under the tables)
rep('template <class After = NoAfter>\n__device__ void build_tables(const Params& P, uint8_t* smem, const Tabs& T, const int* gids, Xt* xt = nullptr, After after = After()) {',
    'constexpr int IDV_MAXV = ((TPB2_MMAX ? TPB2_MMAX : MAXM) * 6 / 4 + THREADS - 1) / THREADS;   // (IDV) int4 id loads per thread\n'
    'struct IdPre { int4 v[IDV_MAXV]; int on; };\n'
    '__device__ __forceinline__ void idv_load(const Params& P, IdPre& q, int tid) {   // (IDV 2) this thread\'s int4 id loads, all in flight\n'
    '    const int MT = pM(P) * pTK(P);\n'
    '    q.on = pTK(P) == 6 && (MT & 3) == 0 && (reinterpret_cast<uintptr_t>(P.ids) & 15) == 0;\n'
    '    if (!q.on) return;\n'
    '#pragma unroll\n'
    '    for (int j = 0; j < IDV_MAXV; ++j) {\n'
    '        const int v = tid + j * THREADS;\n'
    '        q.v[j] = v < MT / 4 ? __ldcg(reinterpret_cast<const int4*>(P.ids) + v) : make_int4(-1, -1, -1, -1);\n'
    '    }\n'
    '}\n'
    'template <class After = NoAfter>\n__device__ void build_tables(const Params& P, uint8_t* smem, const Tabs& T, const int* gids, Xt* xt = nullptr, After after = After(), const IdPre* ipre = nullptr) {')
rep('    int4 iv[MAXV];\n    if (idv) {\n',
    '    int4 iv[MAXV];\n'
    '    if (idv && ipre && ipre->on) {   // (IDV 2: loaded before the x quantization)\n'
    '#pragma unroll\n'
    '        for (int j = 0; j < MAXV; ++j) iv[j] = ipre->v[j];\n'
    '    } else if (idv) {\n')
rep('    // TPB2_GATHER: x -> MXFP8 (+ shared per-128) by token, independent of the routing: right after (and on other warps than) the gate\n'
    '    if (TPB2_GATHER) {\n',
    '    IdPre idp;   // (IDV 2) the routing ids: loads issued now, in flight over the x quantization\n'
    '    idp.on = 0;\n'
    '    if (TPB2_IDV == 2 && !pGL(P)) idv_load(P, idp, tid);\n'
    '    // TPB2_GATHER: x -> MXFP8 (+ shared per-128) by token, independent of the routing: right after (and on other warps than) the gate\n'
    '    if (TPB2_GATHER) {\n')
rep('            pf_bulk(P.w13_sf + ((static_cast<size_t>(e) * NT1 + tile) * NCH1 + KC * s) * 512, KC * 512);\n'
    '        }\n'
    '    });\n',
    '            pf_bulk(P.w13_sf + ((static_cast<size_t>(e) * NT1 + tile) * NCH1 + KC * s) * 512, KC * 512);\n'
    '        }\n'
    '    }, &idp);\n')

# ---- TPB2_LLS (0): sentinel all-reduce messages (8 bf16 per 16 B, no tag; arrival = word w != 0xFFFFFFFF; every slot reset to the sentinel by
#      its reader; parity halves keep a slot's next writer two calls away).  T1 lane l = message l.  Same rank-order sums -> bitwise == LL.
#      Host: rs and ag peer buffers memset to 0xFF once; every build sharing the workspace must be LLS.
rep('#ifndef TPB2_IDV\n', '#ifndef TPB2_LLS\n#define TPB2_LLS 0   // (suite_b200) all-reduce messages = 8 bf16 per 16 B, arrival = w != 0xFFFFFFFF, slots re-armed by their reader\n#endif\n#ifndef TPB2_IDV\n')
rep('__device__ __forceinline__ void t1_emit(const Params& P, int rtile, int t, unsigned epoch, __nv_bfloat16* stg, int lane, const T1In& in, Xt* xt,\n',
    'constexpr uint32_t LLS_S = 0xFFFFFFFFu;   // (LLS) sentinel of an empty message slot (word w)\n'
    '__device__ __forceinline__ uint4 lls_msg(uint32_t x, uint32_t y, uint32_t z, uint32_t w) { return make_uint4(x, y, z, w == LLS_S ? 0x7FC07FC0u : w); }\n'
    '__device__ __forceinline__ void lls_acc(const uint4& v, float* a, bool first) {\n'
    '    const uint32_t w4[4] = {v.x, v.y, v.z, v.w};\n'
    '#pragma unroll\n'
    '    for (int j = 0; j < 4; ++j) {\n'
    '        a[2 * j] = first ? bf16lo(w4[j]) : __fadd_rn(a[2 * j], bf16lo(w4[j]));\n'
    '        a[2 * j + 1] = first ? bf16hi(w4[j]) : __fadd_rn(a[2 * j + 1], bf16hi(w4[j]));\n'
    '    }\n'
    '}\n'
    '__device__ __forceinline__ uint4 lls_pack(const float* a) {\n'
    '    return make_uint4(pack_bf16x2(a[0], a[1]), pack_bf16x2(a[2], a[3]), pack_bf16x2(a[4], a[5]), pack_bf16x2(a[6], a[7]));\n'
    '}\n'
    '// (LLS) the all-reduce of the T1 rows: one-shot (every source -> every rank) or RS (row-tile owner rtile / LRT) + AG (owner -> every rank)\n'
    '__device__ __noinline__ void lls_allreduce(const Params& P, int b, int G, int M, int tid, unsigned epoch) {\n'
    '    if (TPB2_GRID && gridDim.x != static_cast<unsigned>(TPB2_GRID)) __trap();   // (LLS: one reader per slot; another grid would re-read)\n'
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
    '            *reinterpret_cast<uint4*>(P.out + static_cast<size_t>(t) * DIM + rtile * 256 + 8 * m) = lls_pack(a);\n'
    '#pragma unroll\n'
    '            for (int d = 0; d < NDEV; ++d) *(rs_me + os_idx(d, rtile, t, m)) = S4;\n'
    '        }\n'
    '        return;\n'
    '    }\n'
    '    {   // T2: owner of row tiles rk LRT .. reduces (lrt, t), pushes to every rank (self included)\n'
    '        const int npair = LRT * M, mine = b < npair ? (npair - b + G - 1) / G : 0;\n'
    '        for (int x = tid; x < mine * 32; x += THREADS) {\n'
    '            const int pi = x >> 5, m = x & 31, j = b + pi * G, lrt = j / M, t = j - lrt * M;\n'
    '            uint4 v[NDEV];\n'
    '            for (;;) {\n'
    '#pragma unroll\n'
    '                for (int d = 0; d < NDEV; ++d) v[d] = ld_ll(rs_me + rs_idx(d, lrt, t, m));\n'
    '                bool ok = true;\n'
    '#pragma unroll\n'
    '                for (int d = 0; d < NDEV; ++d) ok &= v[d].w != LLS_S;\n'
    '                if (ok) break;\n'
    '            }\n'
    '            float a[8];\n'
    '#pragma unroll\n'
    '            for (int d = 0; d < NDEV; ++d) lls_acc(v[d], a, d == 0);\n'
    '            const uint4 o = lls_pack(a), mo = lls_msg(o.x, o.y, o.z, o.w);\n'
    '            const int rtile = rk * LRT + lrt;\n'
    '#pragma unroll\n'
    '            for (int d = 0; d < NDEV; ++d) st_ll(P.ag[d] + (epoch & 1u) * AG_PAR + ag_idx(rtile, t, m), mo);\n'
    '#pragma unroll\n'
    '            for (int d = 0; d < NDEV; ++d) *(rs_me + rs_idx(d, lrt, t, m)) = S4;\n'
    '        }\n'
    '    }\n'
    '    {   // T3: every (row tile, token) from this rank\'s AG buffer, TPB2_LLS_RB polls in flight per thread\n'
    '        uint4* ag_me = P.ag[rk] + (epoch & 1u) * AG_PAR;\n'
    '        const int npair = NRT * M, mine = b < npair ? (npair - b + G - 1) / G : 0, tot = mine * 32;\n'
    '        constexpr int RB = 4;\n'
    '        for (int x0 = tid; x0 < tot; x0 += RB * THREADS) {\n'
    '            uint4 v[RB];\n'
    '            uint4* pm[RB];\n'
    '            size_t po[RB];\n'
    '#pragma unroll\n'
    '            for (int q = 0; q < RB; ++q) {\n'
    '                const int x = x0 + q * THREADS;\n'
    '                pm[q] = nullptr; po[q] = 0; v[q] = S4;\n'
    '                if (x >= tot) continue;\n'
    '                const int pi = x >> 5, m = x & 31, j = b + pi * G, rtile = j / M, t = j - rtile * M;\n'
    '                pm[q] = ag_me + ag_idx(rtile, t, m);\n'
    '                po[q] = static_cast<size_t>(t) * DIM + rtile * 256 + 8 * m;\n'
    '                v[q] = ld_ll(pm[q]);\n'
    '            }\n'
    '#pragma unroll\n'
    '            for (int q = 0; q < RB; ++q) {\n'
    '                if (!pm[q]) continue;\n'
    '                while (v[q].w == LLS_S) v[q] = ld_ll(pm[q]);\n'
    '                *reinterpret_cast<uint4*>(P.out + po[q]) = v[q];\n'
    '                *pm[q] = S4;\n'
    '            }\n'
    '        }\n'
    '    }\n'
    '}\n'
    '__device__ __forceinline__ void t1_emit(const Params& P, int rtile, int t, unsigned epoch, __nv_bfloat16* stg, int lane, const T1In& in, Xt* xt,\n')
rep('''        o4[j] = pack_bf16x2(y0, y1);
    }
    *reinterpret_cast<uint4*>(stg + 8 * lane) = make_uint4(o4[0], o4[1], o4[2], o4[3]);
    __syncwarp();
    if (xt && lane == 0) { xt->rec(XT_END(56)); xt->rec(XT_BEG(57)); }
    for (int m = lane; m < NMSG; m += 32) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(stg + 6 * m);
        const bool full6 = m < NMSG - 1;
        const uint4 msg = make_uint4(w[0], w[1], full6 ? w[2] : 0u, epoch);
        if (pND(P) == 1) {''',
    '''        o4[j] = pack_bf16x2(y0, y1);
    }
    if (TPB2_LLS && pND(P) > 1) {   // (LLS) lane l's 16 B = message l
        const uint4 msg = lls_msg(o4[0], o4[1], o4[2], o4[3]);
        if (pOS(P)) {
#pragma unroll
            for (int d = 0; d < NDEV; ++d) st_ll(P.rs[(pRK(P) + d) & (NDEV - 1)] + (epoch & 1u) * RS_PAR + os_idx(pRK(P), rtile, t, lane), msg);
        } else {
            st_ll(P.rs[TPB2_PROBE_T1LOCAL ? pRK(P) : rtile / LRT] + (epoch & 1u) * RS_PAR + rs_idx(pRK(P), rtile % LRT, t, lane), msg);
        }
        return;
    }
    *reinterpret_cast<uint4*>(stg + 8 * lane) = make_uint4(o4[0], o4[1], o4[2], o4[3]);
    __syncwarp();
    if (xt && lane == 0) { xt->rec(XT_END(56)); xt->rec(XT_BEG(57)); }
    for (int m = lane; m < NMSG; m += 32) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(stg + 6 * m);
        const bool full6 = m < NMSG - 1;
        const uint4 msg = make_uint4(w[0], w[1], full6 ? w[2] : 0u, epoch);
        if (pND(P) == 1) {''')
rep('''    if (TPB2_PROBE_T1LOCAL) return;   // (probe: no all-reduce -- wrong output)
    if (fused) return;
''', '''    if (TPB2_PROBE_T1LOCAL) return;   // (probe: no all-reduce -- wrong output)
    if (fused) return;
    if (TPB2_LLS) {   // (LLS) sentinel messages: their own receive / reduce / push loops
        static_assert(!TPB2_LLS || (!(TPB2_ARV & 16) && !TPB2_MC && !TPB2_FC2L && !TPB2_T1LL_MAXM && !TPB2_T1L_MAXM && !TPB2_T1L_MINM && !TPB2_T1W),
                      "LLS: the plain T1 + one-shot / RS + AG tail");
        lls_allreduce(P, b, G, M, tid, epoch);
        return;
    }
''')
rep('    m.attr("MAXM") = TPB2_NS::MAXM;', '    m.attr("LLS") = static_cast<int64_t>(TPB2_LLS);\n    m.attr("MAXM") = TPB2_NS::MAXM;')

# ---- TPB2_TABDBG (0): XTrace points inside build_tables' bits phase, every warp (79: its IDV id loads landed, 80: its atomics issued)
rep('#ifndef TPB2_IDV\n', '#ifndef TPB2_TABDBG\n#define TPB2_TABDBG 0   // (suite_b200, trace only) per-warp points in the build_tables bits phase\n#endif\n#ifndef TPB2_IDV\n')
rep('    if (idv) {   // (IDV) the vector-loaded ids: entries p = 4 v .. 4 v + 3\n',
    '    if (TPB2_TABDBG && idv) {   // (TABDBG) this warp\'s id loads landed\n'
    '        int s_ = 0;\n'
    '#pragma unroll\n'
    '        for (int j = 0; j < MAXV; ++j) s_ += iv[j].x + iv[j].y + iv[j].z + iv[j].w;\n'
    '        s_ = __reduce_add_sync(0xffffffffu, s_);\n'
    '        if (xt && lane == 0) xt->rec(XT_PT(79), static_cast<uint32_t>(s_) & 0xFFu);\n'
    '    }\n'
    '    if (idv) {   // (IDV) the vector-loaded ids: entries p = 4 v .. 4 v + 3\n')
rep('    __syncthreads();\n    if (xt && threadIdx.x == 0) xt->rec(XT_PT(65));\n',
    '    if (TPB2_TABDBG) { __syncwarp(); if (xt && lane == 0) xt->rec(XT_PT(80)); }   // (TABDBG) this warp\'s atomics issued\n'
    '    __syncthreads();\n    if (xt && threadIdx.x == 0) xt->rec(XT_PT(65));\n')
# ---- TPB2_TABDBG >= 2 (test only): CTA b reads the ids from replica b % TABDBG (the harness fills TABDBG copies after P.ids: SB_IDSREP)
rep('''        for (int j = 0; j < MAXV; ++j) {
            const int v = tid + j * THREADS;
            iv[j] = v < MT / 4 ? __ldcg(reinterpret_cast<const int4*>(P.ids) + v) : make_int4(-1, -1, -1, -1);''',
    '''        const int* ids_ = P.ids + (TPB2_TABDBG >= 2 ? static_cast<int>(blockIdx.x % (TPB2_TABDBG >= 2 ? TPB2_TABDBG : 1)) * MT : 0);   // (TABDBG >= 2: replicas)
        for (int j = 0; j < MAXV; ++j) {
            const int v = tid + j * THREADS;
            iv[j] = v < MT / 4 ? __ldcg(reinterpret_cast<const int4*>(ids_) + v) : make_int4(-1, -1, -1, -1);''')
# ---- TPB2_TABDBG: latency probes right after the clear phase (81 -> 82: one fresh ids int4 load, 82 -> 83: one xb int4 load), before the IDV ids are used
rep('    if (TPB2_TABDBG && idv) {   // (TABDBG) this warp\'s id loads landed\n',
    '    if (TPB2_TABDBG && idv) {   // (TABDBG) single-load latency probes\n'
    '        if (xt && lane == 0) xt->rec(XT_PT(81));\n'
    '        const int4 q1 = __ldcg(reinterpret_cast<const int4*>(P.ids) + lane + 32 * (blockIdx.x & 3));\n'
    '        int s1 = __reduce_add_sync(0xffffffffu, q1.x ^ q1.w);\n'
    '        if (xt && lane == 0) xt->rec(XT_PT(82), static_cast<uint32_t>(s1) & 0xFFu);\n'
    '        const int4 q2 = __ldcg(reinterpret_cast<const int4*>(P.xb) + blockIdx.x * 32 + lane);\n'
    '        s1 = __reduce_add_sync(0xffffffffu, q2.x ^ q2.w);\n'
    '        if (xt && lane == 0) xt->rec(XT_PT(83), static_cast<uint32_t>(s1) & 0xFFu);\n'
    '    }\n'
    '    if (TPB2_TABDBG && idv) {   // (TABDBG) this warp\'s id loads landed\n')

# ---- TPB2_ENTT (0): build_tables' entries phase per token (one ids row load, the slots' offset loads independent) instead of per
#      (token, slot) -- same entries / tke (bitwise)
rep('#ifndef TPB2_IDV\n', '#ifndef TPB2_ENTT\n#define TPB2_ENTT 0   // (suite_b200) build_tables entries per token (one thread: all slots of its token)\n#endif\n#ifndef TPB2_IDV\n')
rep('    // entries: one per (token, expert) at its first slot\n    for (int p = tid; p < MT; p += THREADS) {\n',
    '    // entries: one per (token, expert) at its first slot\n'
    '    if (TPB2_ENTT) {   // (ENTT) a thread per token: its ids row once, every slot\'s offsets loaded independently\n'
    '        for (int t = tid; t < M; t += THREADS) {\n'
    '            const uint4 row = *reinterpret_cast<const uint4*>(T.ids + t * 8);\n'
    '            const uint32_t rw[4] = {row.x, row.y, row.z, row.w};\n'
    '            uint32_t id8[8];\n'
    '#pragma unroll\n'
    '            for (int k2 = 0; k2 < 8; ++k2) id8[k2] = (rw[k2 >> 1] >> ((k2 & 1) * 16)) & 0xFFFFu;\n'
    '            const int w = t >> 5;\n'
    '            const uint32_t below = (1u << (t & 31)) - 1u;\n'
    '#pragma unroll\n'
    '            for (int k = 0; k < TOPK_MAX; ++k) {\n'
    '                if (k >= topk) break;\n'
    '                const uint32_t e = id8[k];\n'
    '                if (e == 0xFFFFu) continue;\n'
    '                unsigned km = 0;\n'
    '#pragma unroll\n'
    '                for (int k2 = 0; k2 < 8; ++k2) km |= (id8[k2] == e ? 1u : 0u) << k2;\n'
    '                if (km & ((1u << k) - 1u)) continue;\n'
    '                const int oe = offe[e];\n'
    '                const int idx = oe + pre[w * NEXP + e] + __popc(bits[w * NEXP + e] & below);\n'
    '                T.ent[idx] = static_cast<uint16_t>(t | (km << 9));\n'
    '                const int j = idx - oe;\n'
    '                T.tke[t * 8 + k] = static_cast<uint16_t>(((pide[e] + j / NMAX) << 5) | (j % NMAX));\n'
    '            }\n'
    '        }\n'
    '    } else\n'
    '    for (int p = tid; p < MT; p += THREADS) {\n')

# ---- SFBI guard (no effect on shipped builds: all have TPB2_WHOLE=1): the B warp writes a routed FC1 piece's SFB images into item buffer
#      rfi & 1 BEFORE its first EMPTY wait; the MMA warp may still be reading that buffer for piece rfi - 2 unless piece rfi - 1 has >= NS
#      stages.  Whole items (16 stages) guarantee it; split / stream-K pieces (WHOLE=0) could be shorter.
rep('static_assert(!TPB2_SFBI || (TPB2_LEANMMA == 2 && !TPB2_MPAIR && !TPB2_FC1PAIR && !TPB2_PAIR && TPB2_GATHER && TPB2_SFCP_MMA), "SFBI: the lean 1-CTA gather path");\n',
    'static_assert(!TPB2_SFBI || (TPB2_LEANMMA == 2 && !TPB2_MPAIR && !TPB2_FC1PAIR && !TPB2_PAIR && TPB2_GATHER && TPB2_SFCP_MMA), "SFBI: the lean 1-CTA gather path");\n'
    'static_assert(!TPB2_SFBI || TPB2_WHOLE, "SFBI: item images are written before the piece\'s first EMPTY wait -- needs whole FC1 items (>= NS stages per piece; '
    'split / stream-K pieces would race the MMA still reading piece rfi - 2)");\n')
P.write_text(s)
print(f'patched {P}: {n}')
