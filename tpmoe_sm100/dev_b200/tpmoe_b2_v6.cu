// tpmoe_b2: TP4 fused MoE decode kernel for B200 (sm_100a), DeepSeek-V4-Flash routed experts (hidden 4096, per-rank intermediate 512,
// 256 experts, top-k <= 8, M <= 512), SGLang's flashinfer_mxfp4 (trtllm-gen) expert tensors used in place.  The B200 port of tp4moe2
// (H20 / H100): same skeleton -- fused gate, routing tables with token panels of <= 32 entries, stream-K FC1, FC2 unit queue, T1 and the
// LL all-reduce in the same launch -- with a tcgen05 core: kind::mxf8f6f4 block-scaled MMA (MXFP4 weights x MXFP8 activations, UE8M0 per
// 32), fp32 accumulators in tensor memory, one MMA-issuing warp.  Design: design/DESIGN_tpmoe_b2.md.
//
// Numerics (the production flashinfer_mxfp4 scheme; not bitwise -- trtllm-gen sums in another order):
//   x  -> MXFP8 (e4m3, one UE8M0 scale per 32: 2^ceil(log2(amax / 448)), = flashinfer mxfp8_quantize)   FC1 fp32 (TMEM, K 4096)
//   act = up * gate * sigmoid(gate), gate <= 10, |up| <= 10 (fp32) -> h = MXFP8 per 32 with the OCP rule 2^(floor(log2 amax) - 8)
//   (trtllm-gen's FC1 output quantization, measured: b2_refcheck)                 FC2 fp32 -> bf16 per (token, slot)
//   T1: r = bf16(sum_k bf16(w_k) * o_k) (k order), y = bf16(shared + 1.5 r)     all-reduce: rank-ordered fp32 sum -> bf16
//
// One persistent CTA per SM, 8 warps:  w0 lane 0 producer (weights by TMA + SFA; work descriptors on the ring) | w1 TMEM + MMA |
// w2..w5 epilogue (TMEM lane quadrant warp & 3) | w6 B operand (FC1: x rows -> SW128 smem + SFB; FC2: h image by bulk copy) |
// w7 lane 0 scale copier (tcgen05.cp -> TMEM).  Prologue and tail use all 256 threads.
#ifndef B2_NO_TORCH
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

#ifndef TPB2_NS
#define TPB2_NS tpb2_v6
#endif
#ifndef TPB2_SMALLM
#define TPB2_SMALLM 8   // fused gate: M <= this -> every CTA gates every token itself (no grid-wide routing barrier)
#endif
#ifndef TPB2_ONESHOT_MAXM
#define TPB2_ONESHOT_MAXM 32   // one-shot LL all-reduce for M <= this, else reduce-scatter + all-gather
#endif

namespace TPB2_NS {

constexpr int NDEV = 4;
constexpr int DIM = 4096, INTER = 2048 / NDEV, NEXP = 256, TOPK_MAX = 8, MAXM = 512;
constexpr int MAXE = MAXM * 6;               // entries (token, expert) upper bound at top-6 (asserted on the host)
constexpr int NMAX = 32;                     // token panel width = MMA N bound
constexpr int MAXP = 512;                    // panels upper bound (<= NEXP + MAXE / NMAX)
constexpr int NT1 = 2 * INTER / 128;         // 8 FC1 tiles (128 w13 rows = 64 gate + 64 up) per panel
constexpr int NCH1 = DIM / 128;              // 32 FC1 k-chunks
constexpr int NRT2 = DIM / 128;              // 32 FC2 row tiles
constexpr int NRG = NRT2 / 4;                // 8 FC2 row groups (FC2G row tiles = 2 T1 row tiles of 256)
constexpr int NCH2 = INTER / 128;            // 4 FC2 k-chunks
constexpr int KC = 2;                        // k-chunks per pipeline stage
constexpr int NSTG1 = NCH1 / KC, NSTG2 = NCH2 / KC;   // 16, 2 stages per FC1 item / FC2 unit
constexpr int NS = 4;                        // ring stages
constexpr int THREADS = 384;                 // 12 warps: producer, MMA, 2 x 4 epilogue, B, SF
constexpr int FC2G = 4;                      // FC2 unit = FC2G consecutive 128-row tiles of one panel (one ticket / ring entry)
constexpr int NRT = DIM / 256;               // 16 T1 / all-reduce row tiles (256 rows)
constexpr int LRT = NRT / NDEV;              // 4 row tiles per reduce-scatter owner
constexpr int NMSG = 43;                     // LL messages per (row tile, token): 6 bf16 + tag each, the last carries 4
constexpr int MAXG = 160;
constexpr int ACC = 4, ACC_COLS = 32;        // TMEM accumulator buffers
constexpr int TMEM_COLS = 512, TSF0 = 256, TSF_STRIDE = 32, TSF_B = 16;
constexpr int WR = 16;                       // work ring depth
// shared memory (1024-aligned base): A [NS][KC][16 KB] | B [NS][KC][NMAX][128] | SFA [NS][KC][512] | SFB [NS][512] | tables | ring | bars
constexpr int A_BYTES = KC * 16384, B_BYTES = KC * NMAX * 128;
constexpr int OFF_A = 0, OFF_B = OFF_A + NS * A_BYTES, OFF_SFA = OFF_B + NS * B_BYTES, OFF_SFB = OFF_SFA + NS * KC * 512;
constexpr int OFF_TAB = OFF_SFB + NS * 512;
constexpr int TAB_IDS = 0;                             // uint16 [MAXM][8] expert of (t, k), 0xFFFF = none
constexpr int TAB_ENT = TAB_IDS + MAXM * 8 * 2;        // uint16 [MAXE] token of entry
constexpr int TAB_KM = TAB_ENT + MAXE * 2;             // uint8 [MAXE] top-k slot mask of entry
constexpr int TAB_PINF = TAB_KM + MAXE;                // uint32 [MAXP] panel: e | n << 8 | ebase << 14
constexpr int TAB_RW = TAB_PINF + MAXP * 4;            // float [TPB2_SMALLM * 8] small fused gate: routing weights of (t, k)
constexpr int TAB_AM = TAB_RW + TPB2_SMALLM * 8 * 4;   // float [4][NMAX] FC1 epilogue: per-warp amax exchange
constexpr int TAB_MISC = TAB_AM + 4 * NMAX * 4;        // int [32]: 0 U, 1 entries, 2 panels, 8 T1 item, 9 epoch
constexpr int TAB_TKE = TAB_MISC + 32 * 4;             // uint16 [MAXM][8] (t, k) -> panel << 5 | row (first slot of each (t, expert)), else 0xFFFF
constexpr int TAB_PRDY = TAB_TKE + MAXM * 8 * 2;       // uint32 [MAXP / 32] FC2 B producer: panels whose h is known ready
constexpr int TAB_LANES = TAB_PRDY + MAXP / 8;          // uint8 [128] FC2 row of TMEM lane
constexpr int TAB_STG = TAB_LANES + 128;               // bf16 [2][16][128] FC2 epilogue transpose staging
constexpr int TAB_RING = TAB_STG + 2 * 16 * 128 * 2;   // int4 [WR] work ring
constexpr int TAB_BAR = TAB_RING + WR * 16;            // uint64 barriers
constexpr int NBAR = 3 * NS + 2 * ACC + 2 * WR;
constexpr int TAB_BYTES = TAB_BAR + NBAR * 8;
constexpr int SMEM_TOTAL = OFF_TAB + TAB_BYTES, SMEM_ALLOC = SMEM_TOTAL + 1024;
static_assert(SMEM_ALLOC <= 232448, "smem budget");
static_assert(OFF_B % 1024 == 0 && B_BYTES % 1024 == 0 && OFF_TAB % 16 == 0 && TAB_BAR % 8 == 0, "alignment");
// prologue scratch inside the A ring (dead before the first TMA)
constexpr int PRO_BITS = 0;                            // uint32 [16 words][NEXP] token bits per expert
constexpr int PRO_PRE = PRO_BITS + 16 * NEXP * 4;      // uint16 [16][NEXP] exclusive word prefix
constexpr int PRO_OFFE = PRO_PRE + 16 * NEXP * 2;      // int [NEXP] entry offset of expert
constexpr int PRO_PIDE = PRO_OFFE + NEXP * 4;          // int [NEXP] first panel of expert
constexpr int PRO_WS = PRO_PIDE + NEXP * 4;            // int [32] warp sums
constexpr int PRO_GIDS = PRO_WS + 32 * 4;              // int [TPB2_SMALLM * 8] small fused gate: ids
static_assert(PRO_GIDS + TPB2_SMALLM * 8 * 4 <= NS * A_BYTES, "prologue scratch");

struct Params {
    CUtensorMap map1, map2;                  // w13 [E * 1024 rows][2048 B], w2 [E * 4096 rows][256 B]: 16U4_ALIGN16B {128, rows, chunks}
    const uint8_t* w13_sf;                   // [E][1024][128] block_scale_interleave bytes: 512 B per (tile, chunk)
    const uint8_t* w2_sf;                    // [E][4096][16]
    const uint8_t* lanes;                    // [128] FC2 output row of TMEM lane (per 128-row tile)
    const __nv_bfloat16* xb;                 // [M][4096] MoE input
    const float* glog;                       // fused gate: router logits fp32 [M][256], else null (ids / rw given)
    const float* gbias;                      // [256] score correction bias
    int* ids;                                // [M][topk] (fused gate: written here)
    float* rw;                               // [M][topk]
    int topk;
    const __nv_bfloat16* shared;             // [M][4096] shared expert output (alt stream) or null
    float factor;                            // routed scaling factor (1.5)
    uint8_t* xq;                             // [MAXM][4096] e4m3
    uint8_t* xs;                             // [MAXM][128] ue8m0
    uint8_t* xg;                             // [MAXP][NCH1][NMAX][128] x per panel row, K-major SW128 (FC1 B operand images)
    uint8_t* xgsf;                           // [MAXP][NSTG1][512] their SFB images (lane = row, byte 8 (c & 1) + kb)
    unsigned long long* dbg;                 // [grid][16] globaltimer phase stamps, or null
    uint8_t* himg;                           // [MAXP][NCH2][NMAX][128] h, K-major SW128
    uint8_t* hsf;                            // [MAXP][NSTG2][512] h SFB images (lane = token, byte 8 (c & 1) + kb)
    float* part;                             // [MAXG][NMAX][128] FC1 stream-K partials
    unsigned* part_flags;                    // [MAXG] epoch tags
    unsigned* xflag;                         // [MAXM] epoch tag: token quantized
    unsigned* gcnt;                          // [2] gated tokens (epoch parity)
    unsigned* pan_cnt;                       // [2][MAXP] FC1 items done per panel
    unsigned* fc2_next;                      // [4] FC2 unit queue head [par], T1 item queue head [2 + par]
    unsigned* rt_cnt;                        // [2][NRT] FC2 units stored per 256-row tile
    unsigned* work;                          // [0] epoch, [1 + parity] CTA done counters
    __nv_bfloat16* outp;                     // [MAXM][TOPK_MAX][4096] per-slot expert outputs
    uint4* rs[NDEV];                         // peer reduce-scatter buffers (two parity halves)
    uint4* ag[NDEV];                         // peer all-gather buffers (two parity halves)
    __nv_bfloat16* out;                      // [M][4096]
    int M, my_rank, ndev, oneshot;
};

// ------------------------------------------------------------------------------------------------ device helpers
__device__ __forceinline__ uint32_t su32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint32_t b, int c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(c) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint32_t b, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t parity) {
    asm volatile("{\n .reg .pred p;\nW_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n" ::"r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma3d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], %6;"
                 ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(bar), "l"(0x12F0000000000000ull) : "memory");
}
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ unsigned ld_relaxed(const unsigned* p) { unsigned v; asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void wait_ge(const unsigned* p, unsigned v) { while (ld_relaxed(p) < v) {} (void)ld_acquire(p); }
__device__ __forceinline__ void wait_eq(const unsigned* p, unsigned v) { while (ld_relaxed(p) != v) {} (void)ld_acquire(p); }
__device__ __forceinline__ void st_release(unsigned* p, unsigned v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void red_release_add(unsigned* p, unsigned v) { asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void fence_acq_rel() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void st_ll(uint4* p, uint4 v) { asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
__device__ __forceinline__ uint4 ld_ll(const uint4* p) { uint4 v; asm volatile("ld.relaxed.sys.global.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ bool elect_one() {
    uint32_t pred = 0;
    asm volatile("{\n .reg .pred p;\n elect.sync _|p, 0xffffffff;\n selp.u32 %0, 1, 0, p;\n}" : "=r"(pred));
    return pred != 0;
}
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint32_t warp_uniform(uint32_t v) { return __shfl_sync(0xffffffffu, v, 0); }
__device__ __forceinline__ float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16hi(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) { __nv_bfloat162 r = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<uint32_t*>(&r); }
// MXFP8 scale exponent: 2^ceil(log2(amax / 448)) as an unbiased exponent (amax 0 -> the smallest normal scale)
__device__ __forceinline__ int ue8m0_exp(float amax) {
    const uint32_t b = __float_as_uint(fmaxf(amax * (1.0f / 448.0f), 1.17549435e-38f));
    return static_cast<int>((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) != 0);
}
// h (FC1 output): the OCP MX rule 2^(floor(log2 amax) - 8) (e4m3 emax 8; values above 448 saturate) -- trtllm-gen's, measured
__device__ __forceinline__ int ue8m0_exp_floor(float amax) {
    return max(-127, static_cast<int>((__float_as_uint(amax) >> 23) & 0xFF) - 127 - 8);
}
__device__ __forceinline__ float exp2i(int e) { return __uint_as_float(static_cast<uint32_t>(e + 127) << 23); }
__device__ __forceinline__ uint32_t e4m3_of(float v) { return static_cast<uint32_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3)); }

// tcgen05
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ uint64_t smem_desc(uint32_t addr, uint32_t lbo, uint32_t sbo, uint32_t layout) {
    return static_cast<uint64_t>((addr >> 4) & 0x3FFF) | (static_cast<uint64_t>((lbo >> 4) & 0x3FFF) << 16) |
           (static_cast<uint64_t>((sbo >> 4) & 0x3FFF) << 32) | (1ull << 46) | (static_cast<uint64_t>(layout) << 61);
}
__device__ __forceinline__ uint64_t desc_sw128(uint32_t addr) { return smem_desc(addr, 0, 1024, 2); }
__device__ __forceinline__ uint64_t desc_sf(uint32_t addr) { return smem_desc(addr, 0, 128, 0); }
// kind::mxf8f6f4: A e2m1 (5), B e4m3 (0), fp32 D, UE8M0 scales, K-major, M = 128; N >> 3 at bit 17; scale byte ids
__device__ __forceinline__ uint32_t idesc_f4f8(int n, uint32_t sf_id) {
    return (sf_id << 4) | (5u << 7) | (static_cast<uint32_t>(n >> 3) << 17) | (1u << 23) | ((128u >> 4) << 24) | (sf_id << 29);
}
__device__ __forceinline__ void tc_mma(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t sfa, uint32_t sfb, uint32_t acc) {
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %6, 0;\n"
                 " tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32 [%0], %1, %2, %3, [%4], [%5], p;\n}"
                 ::"r"(d), "l"(a), "l"(b), "r"(idesc), "r"(sfa), "r"(sfb), "r"(acc) : "memory");
}
__device__ __forceinline__ void tc_cp_sf(uint32_t taddr, uint64_t desc) { asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(desc) : "memory"); }
__device__ __forceinline__ void tc_commit(uint32_t bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void tc_ld8(uint32_t taddr, float* v) {
    uint32_t r[8];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr) : "memory");
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __uint_as_float(r[i]);
}
__device__ __forceinline__ void tc_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
// 16 columns of the warp's TMEM lane quadrant AND the wait in one asm block: tcgen05.ld's destination registers are undefined until
// tcgen05.wait::ld, so the compiler must not see them before it (separate asm statements let it copy them early -- read stale values)
__device__ __forceinline__ void tc_ld16_wait(uint32_t taddr, float* v) {
    uint32_t r[16];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n\t"
                 "tcgen05.wait::ld.sync.aligned;"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
                 : "r"(taddr) : "memory");
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = __uint_as_float(r[i]);
}

// ------------------------------------------------------------------------------------------------ fused gate (tp4moe2, bitwise sglang moe_fused_gate)
__device__ __forceinline__ float tri_act(float x) {
    float r;
    asm("{\n\t"
        ".reg .pred ps, pf, pz, pg;\n\t"
        ".reg .b32 a, e, y, y2, ys, eb, t, mb, fe, f, p, q, lg;\n\t"
        "mul.f32 a, %1, 0f3FB8AA3B;\n\t"
        "ex2.approx.f32 e, a;\n\t"
        "add.f32 y, e, 0f3F800000;\n\t"
        "setp.lt.f32 ps, y, 0f00800000;\n\t"
        "mul.f32 y2, y, 0f4B000000;\n\t"
        "selp.f32 ys, y2, y, ps;\n\t"
        "selp.f32 eb, 0fC1B80000, 0f00000000, ps;\n\t"
        "add.s32 t, ys, -1059760811;\n\t"
        "and.b32 t, t, -8388608;\n\t"
        "sub.s32 mb, ys, t;\n\t"
        "cvt.rn.f32.s32 fe, t;\n\t"
        "fma.rn.ftz.f32 fe, fe, 0f34000000, eb;\n\t"
        "add.f32 f, mb, 0fBF800000;\n\t"
        "fma.rn.ftz.f32 p, 0fBE055027, f, 0f3E1039F6;\n\t"
        "fma.rn.ftz.f32 p, p, f, 0fBDF8CDCC;\n\t"
        "fma.rn.ftz.f32 p, p, f, 0f3E0F2955;\n\t"
        "fma.rn.ftz.f32 p, p, f, 0fBE2AD8B9;\n\t"
        "fma.rn.ftz.f32 p, p, f, 0f3E4CED0B;\n\t"
        "fma.rn.ftz.f32 p, p, f, 0fBE7FFF22;\n\t"
        "fma.rn.ftz.f32 p, p, f, 0f3EAAAA78;\n\t"
        "fma.rn.ftz.f32 p, p, f, 0fBF000000;\n\t"
        "mul.f32 q, f, p;\n\t"
        "fma.rn.ftz.f32 q, q, f, f;\n\t"
        "fma.rn.ftz.f32 lg, fe, 0f3F317218, q;\n\t"
        "setp.lt.u32 pf, ys, 2139095040;\n\t"
        "@!pf fma.rn.ftz.f32 lg, ys, 0f7F800000, 0f7F800000;\n\t"
        "setp.eq.f32 pz, ys, 0f00000000;\n\t"
        "selp.f32 lg, 0fFF800000, lg, pz;\n\t"
        "setp.gt.f32 pg, %1, 0f41A00000;\n\t"
        "selp.f32 lg, %1, lg, pg;\n\t"
        "sqrt.approx.ftz.f32 %0, lg;\n\t"
        "}" : "=f"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ float div_full(float a, float b) { float r; asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ uint32_t gate_key(float v) { const uint32_t b = __float_as_uint(v == 0.0f ? 0.0f : v); return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
__device__ __forceinline__ void gate_cas(uint64_t& a, uint64_t& b) { const uint64_t x = a > b ? a : b, y = a > b ? b : a; a = x; b = y; }
constexpr int GATE_TOPK = 6;
// one token by one warp: lanes < topk return (expert id, routing weight) of slot lane
__device__ __forceinline__ void gate_token(const Params& P, const float* lg, int lane, int& id, float& wt) {
    float sc[8];
    uint64_t h[8];
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        const float a = tri_act(__ldcg(lg + lane + 32 * w));
        float v = __fadd_rn(a, __ldg(P.gbias + lane + 32 * w));
        v = v == v ? v : -1e30f;
        sc[w] = a;
        h[w] = (static_cast<uint64_t>(gate_key(v)) << 32) | static_cast<uint64_t>(0xFFFFFFFFu - static_cast<uint32_t>(lane + 32 * w));
    }
    gate_cas(h[0], h[2]); gate_cas(h[1], h[3]); gate_cas(h[4], h[6]); gate_cas(h[5], h[7]);
    gate_cas(h[0], h[4]); gate_cas(h[1], h[5]); gate_cas(h[2], h[6]); gate_cas(h[3], h[7]);
    gate_cas(h[0], h[1]); gate_cas(h[2], h[3]); gate_cas(h[4], h[5]); gate_cas(h[6], h[7]);
    gate_cas(h[2], h[4]); gate_cas(h[3], h[5]);
    gate_cas(h[1], h[4]); gate_cas(h[3], h[6]);
    gate_cas(h[1], h[2]); gate_cas(h[3], h[4]); gate_cas(h[5], h[6]);
    int ex[GATE_TOPK];
#pragma unroll
    for (int k = 0; k < GATE_TOPK; ++k) {
        const uint32_t hk = static_cast<uint32_t>(h[0] >> 32), hl = static_cast<uint32_t>(h[0]);
        const uint32_t gm = __reduce_max_sync(0xffffffffu, hk);
        const uint32_t gl = __reduce_max_sync(0xffffffffu, hk == gm ? hl : 0u);
        ex[k] = static_cast<int>(0xFFFFFFFFu - gl);
        if (hk == gm && hl == gl) {
#pragma unroll
            for (int i = 0; i < 7; ++i) h[i] = h[i + 1];
            h[7] = 0ull;
        }
    }
    float R[GATE_TOPK];
#pragma unroll
    for (int k = 0; k < GATE_TOPK; ++k) {
        const int ew = ex[k] >> 5;
        float sv = sc[0];
#pragma unroll
        for (int w = 1; w < 8; ++w) sv = ew == w ? sc[w] : sv;
        R[k] = __shfl_sync(0xffffffffu, sv, ex[k] & 31);
    }
    const float sum = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(R[0], R[1]), R[2]), R[3]), __fadd_rn(R[4], R[5]));
    const float norm = sum > 0.0f ? sum : 1.0f;
    float r = R[0];
    id = ex[0];
#pragma unroll
    for (int k = 1; k < GATE_TOPK; ++k) { r = lane == k ? R[k] : r; id = lane == k ? ex[k] : id; }
    wt = div_full(r, norm);
}

// MXFP8 quantization of token t by one warp: lane l owns elements 128 l .. + 127 (blocks 4 l .. 4 l + 3)
__device__ __forceinline__ void quant_token(const Params& P, const uint16_t* tke, int t, int lane) {
    const uint4* src = reinterpret_cast<const uint4*>(P.xb + static_cast<size_t>(t) * DIM) + lane * 16;
    uint4 raw[16];
#pragma unroll
    for (int g = 0; g < 16; ++g) raw[g] = __ldg(src + g);
    uint32_t sfw = 0;
    uint4 code[8];
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
        float am = 0.f;
#pragma unroll
        for (int g = 4 * kb; g < 4 * kb + 4; ++g) {
            const uint32_t w4[4] = {raw[g].x, raw[g].y, raw[g].z, raw[g].w};
#pragma unroll
            for (int i = 0; i < 4; ++i) am = fmaxf(am, fmaxf(fabsf(bf16lo(w4[i])), fabsf(bf16hi(w4[i]))));
        }
        const int ex = ue8m0_exp(am);
        const float inv = exp2i(-ex);
        sfw |= static_cast<uint32_t>(ex + 127) << (8 * kb);
#pragma unroll
        for (int gg = 0; gg < 2; ++gg) {   // 16 elements per uint4 of codes: raw granules 4 kb + 2 gg, + 1
            uint32_t wv[4];
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const uint4 r = raw[4 * kb + 2 * gg + hh];
                const uint32_t w4[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
                for (int i = 0; i < 2; ++i)
                    wv[2 * hh + i] = e4m3_of(bf16lo(w4[2 * i]) * inv) | (e4m3_of(bf16hi(w4[2 * i]) * inv) << 8) |
                                     (e4m3_of(bf16lo(w4[2 * i + 1]) * inv) << 16) | (e4m3_of(bf16hi(w4[2 * i + 1]) * inv) << 24);
            }
            code[2 * kb + gg] = make_uint4(wv[0], wv[1], wv[2], wv[3]);
        }
    }
    uint4* dst = reinterpret_cast<uint4*>(P.xq + static_cast<size_t>(t) * DIM) + lane * 8;
#pragma unroll
    for (int g = 0; g < 8; ++g) dst[g] = code[g];
    reinterpret_cast<uint32_t*>(P.xs + static_cast<size_t>(t) * (DIM / 32))[lane] = sfw;
    // + the token's row in each of its panels: chunk lane of the panel's FC1 B image (SW128), its SFB word
#pragma unroll 1
    for (int k = 0; k < 8; ++k) {
        const unsigned v = tke[t * 8 + k];
        if (v == 0xFFFFu) continue;
        const int p = static_cast<int>(v >> 5), r = static_cast<int>(v & 31u);
        uint8_t* row = P.xg + ((static_cast<size_t>(p) * NCH1 + lane) * NMAX) * 128 + (r >> 3) * 1024 + (r & 7) * 128;
#pragma unroll
        for (int g = 0; g < 8; ++g) *reinterpret_cast<uint4*>(row + ((g ^ (r & 7)) << 4)) = code[g];
        *reinterpret_cast<uint32_t*>(P.xgsf + (static_cast<size_t>(p) * NSTG1 + (lane >> 1)) * 512 + r * 16 + 8 * (lane & 1)) = sfw;
    }
}

// ------------------------------------------------------------------------------------------------ routing tables
struct Tabs {
    uint16_t* ids; uint16_t* ent; uint8_t* km; uint32_t* pinf; float* srw; float* am; int* misc; uint8_t* lanes; int4* ring;
    uint16_t* tke; uint32_t* prdy; __nv_bfloat16* stg;
};
__device__ __forceinline__ Tabs tabs(uint8_t* smem) {
    uint8_t* t = smem + OFF_TAB;
    return Tabs{reinterpret_cast<uint16_t*>(t + TAB_IDS), reinterpret_cast<uint16_t*>(t + TAB_ENT), t + TAB_KM,
                reinterpret_cast<uint32_t*>(t + TAB_PINF), reinterpret_cast<float*>(t + TAB_RW), reinterpret_cast<float*>(t + TAB_AM),
                reinterpret_cast<int*>(t + TAB_MISC), t + TAB_LANES, reinterpret_cast<int4*>(t + TAB_RING),
                reinterpret_cast<uint16_t*>(t + TAB_TKE), reinterpret_cast<uint32_t*>(t + TAB_PRDY), reinterpret_cast<__nv_bfloat16*>(t + TAB_STG)};
}
// Every CTA, all THREADS: union = present experts in ascending id; entries of an expert = its tokens in ascending order (a token listing
// the same expert in several slots is ONE entry, km = its slots); panels of <= NMAX entries.  ids from smem (gids, small fused gate) or
// global (P.ids).  Writes T.ids, T.ent, T.km, T.pinf, misc 0..2.
__device__ void build_tables(const Params& P, uint8_t* smem, const Tabs& T, const int* gids) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int M = P.M, topk = P.topk, MT = M * topk, W = (M + 31) >> 5;
    uint32_t* bits = reinterpret_cast<uint32_t*>(smem + PRO_BITS);
    uint16_t* pre = reinterpret_cast<uint16_t*>(smem + PRO_PRE);
    int* offe = reinterpret_cast<int*>(smem + PRO_OFFE);
    int* pide = reinterpret_cast<int*>(smem + PRO_PIDE);
    int* ws = reinterpret_cast<int*>(smem + PRO_WS);
    for (int i = tid; i < W * NEXP; i += THREADS) bits[i] = 0u;
    for (int i = tid; i < M * 8; i += THREADS) { T.ids[i] = 0xFFFF; T.tke[i] = 0xFFFF; }
    for (int i = tid; i < MAXP / 32; i += THREADS) T.prdy[i] = 0u;
    __syncthreads();
    for (int p = tid; p < MT; p += THREADS) {
        const int e = gids ? gids[p] : __ldcg(P.ids + p);
        const int t = p / topk, k = p - t * topk;
        if (static_cast<unsigned>(e) < static_cast<unsigned>(NEXP)) {
            T.ids[t * 8 + k] = static_cast<uint16_t>(e);
            atomicOr(&bits[(t >> 5) * NEXP + e], 1u << (t & 31));
        }
    }
    __syncthreads();
    // per expert (tid == e, warps 0..7): count, word prefix; block scans of (present, count, panels)
    int c = 0, pres = 0, npan = 0, a = 0, b = 0, d = 0;
    if (tid < NEXP) {
        const int e = tid;
        for (int w = 0; w < W; ++w) { pre[w * NEXP + e] = static_cast<uint16_t>(c); c += __popc(bits[w * NEXP + e]); }
        pres = c > 0; npan = (c + NMAX - 1) / NMAX;
        a = pres; b = c; d = npan;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int ya = __shfl_up_sync(0xffffffffu, a, o), yb = __shfl_up_sync(0xffffffffu, b, o), yd = __shfl_up_sync(0xffffffffu, d, o);
            if (lane >= o) { a += ya; b += yb; d += yd; }
        }
        if (lane == 31) { ws[warp] = a; ws[8 + warp] = b; ws[16 + warp] = d; }
    }
    __syncthreads();
    if (tid < NEXP) {
        int xb = b - c, xd = d - npan;
#pragma unroll
        for (int w = 0; w < 8; ++w) if (w < warp) { xb += ws[8 + w]; xd += ws[16 + w]; }
        offe[tid] = xb; pide[tid] = xd;
        if (pres) {
            for (int p = 0; p < npan; ++p)
                T.pinf[xd + p] = static_cast<uint32_t>(tid) | (static_cast<uint32_t>(min(NMAX, c - p * NMAX)) << 8) | (static_cast<uint32_t>(xb + p * NMAX) << 14);
        }
    }
    if (tid == NEXP - 1) {
        int U = 0, E = 0, Pn = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) { U += ws[w]; E += ws[8 + w]; Pn += ws[16 + w]; }
        T.misc[0] = U; T.misc[1] = E; T.misc[2] = Pn;
    }
    __syncthreads();
    // entries: one per (token, expert) at its first slot
    for (int p = tid; p < MT; p += THREADS) {
        const int t = p / topk, k = p - t * topk;
        const uint4 row = *reinterpret_cast<const uint4*>(T.ids + t * 8);
        const uint32_t rw[4] = {row.x, row.y, row.z, row.w};
        const int e = static_cast<int>((rw[k >> 1] >> ((k & 1) * 16)) & 0xFFFFu);
        if (e == 0xFFFF) continue;
        unsigned km = 0;
#pragma unroll
        for (int k2 = 0; k2 < 8; ++k2) km |= (((rw[k2 >> 1] >> ((k2 & 1) * 16)) & 0xFFFFu) == static_cast<unsigned>(e) ? 1u : 0u) << k2;
        if (km & ((1u << k) - 1u)) continue;
        const int w = t >> 5;
        const int idx = offe[e] + pre[w * NEXP + e] + __popc(bits[w * NEXP + e] & ((1u << (t & 31)) - 1u));
        T.ent[idx] = static_cast<uint16_t>(t);
        T.km[idx] = static_cast<uint8_t>(km);
        const int j = idx - offe[e];
        T.tke[t * 8 + k] = static_cast<uint16_t>(((pide[e] + j / NMAX) << 5) | (j % NMAX));
    }
    __syncthreads();
}

// ------------------------------------------------------------------------------------------------ work
// FC1: item i = panel (i >> 3), tile (i & 7); stage s of 16 (chunks 2 s, 2 s + 1).  CTA b's stream-K range: global stages
// [total b / G, total (b + 1) / G), total = panels x 8 x 16.  Ring entries: x = kind (1 FC1 piece, 2 FC2 unit, -1 end),
// FC1: y = item, z = s0, w = s1;  FC2: y = unit (row tile major: r = unit / panels, panel = unit % panels).
__device__ __forceinline__ int fc1_lo(int total, int b, int G) { return static_cast<int>((static_cast<long long>(total) * b) / G); }
__device__ __forceinline__ void panel_of(const Tabs& T, int p, int& e, int& n, int& ebase) {
    const uint32_t v = T.pinf[p];
    e = static_cast<int>(v & 255u); n = static_cast<int>((v >> 8) & 63u); ebase = static_cast<int>(v >> 14);
}
__device__ __forceinline__ float4* part_ptr(const Params& P, int cta) { return reinterpret_cast<float4*>(P.part + static_cast<size_t>(cta) * NMAX * 128); }

// ------------------------------------------------------------------------------------------------ tail (tp4moe2)
__device__ __forceinline__ size_t rs_idx(int src, int lrt, int t, int m) { return (static_cast<size_t>((src * LRT + lrt) * MAXM + t)) * NMSG + m; }
__device__ __forceinline__ size_t ag_idx(int rtile, int t, int m) { return (static_cast<size_t>(rtile * MAXM + t)) * NMSG + m; }
constexpr int OS_MAXM = MAXM / NDEV;
__device__ __forceinline__ size_t os_idx(int src, int rtile, int t, int m) { return (static_cast<size_t>((src * NRT + rtile) * OS_MAXM + t)) * NMSG + m; }
constexpr size_t RS_PAR = static_cast<size_t>(NRT) * MAXM * NMSG;
constexpr size_t AG_PAR = static_cast<size_t>(NRT) * MAXM * NMSG;

// T1 of one (row tile, token) by one warp: lane l owns values rtile * 256 + 8 l .. + 7
__device__ __forceinline__ void t1_pair(const Params& P, const Tabs& T, bool small_rw, int rtile, int t, unsigned epoch, __nv_bfloat16* stg, int lane) {
    const int topk = P.topk;
    const uint4 idrow = *reinterpret_cast<const uint4*>(T.ids + t * 8);
    const uint32_t idw[4] = {idrow.x, idrow.y, idrow.z, idrow.w};
    const int c0 = rtile * 256 + 8 * lane;
    uint4 v[TOPK_MAX];
    float wk[TOPK_MAX];
    unsigned on = 0u;
#pragma unroll
    for (int k = 0; k < TOPK_MAX; ++k) {
        if (k < topk && ((idw[k >> 1] >> ((k & 1) * 16)) & 0xFFFFu) != 0xFFFFu) {
            on |= 1u << k;
            wk[k] = bf16r(small_rw ? T.srw[t * topk + k] : __ldcg(P.rw + t * topk + k));   // (PackTopkIds: bf16 routing weights)
            v[k] = __ldcg(reinterpret_cast<const uint4*>(P.outp + (static_cast<size_t>(t) * TOPK_MAX + k) * DIM + c0));
        }
    }
    uint4 sh = make_uint4(0u, 0u, 0u, 0u);
    if (P.shared) sh = __ldcg(reinterpret_cast<const uint4*>(P.shared + static_cast<size_t>(t) * DIM + c0));
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k = 0; k < TOPK_MAX; ++k) {
        if (!((on >> k) & 1u)) continue;
        const uint32_t w4[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[2 * j] = __fmaf_rn(bf16lo(w4[j]), wk[k], acc[2 * j]);
            acc[2 * j + 1] = __fmaf_rn(bf16hi(w4[j]), wk[k], acc[2 * j + 1]);
        }
    }
    const uint32_t s4[4] = {sh.x, sh.y, sh.z, sh.w};
    uint32_t o4[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float r0 = bf16r(acc[2 * j]), r1 = bf16r(acc[2 * j + 1]);
        const float y0 = P.shared ? __fmaf_rn(P.factor, r0, bf16lo(s4[j])) : __fmul_rn(P.factor, r0);
        const float y1 = P.shared ? __fmaf_rn(P.factor, r1, bf16hi(s4[j])) : __fmul_rn(P.factor, r1);
        o4[j] = pack_bf16x2(y0, y1);
    }
    *reinterpret_cast<uint4*>(stg + 8 * lane) = make_uint4(o4[0], o4[1], o4[2], o4[3]);
    __syncwarp();
    for (int m = lane; m < NMSG; m += 32) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(stg + 6 * m);
        const bool full6 = m < NMSG - 1;
        const uint4 msg = make_uint4(w[0], w[1], full6 ? w[2] : 0u, epoch);
        if (P.ndev == 1) {
            uint32_t* o = reinterpret_cast<uint32_t*>(P.out + static_cast<size_t>(t) * DIM + rtile * 256 + 6 * m);
            o[0] = msg.x; o[1] = msg.y;
            if (full6) o[2] = msg.z;
        } else if (P.oneshot) {
#pragma unroll
            for (int d = 0; d < NDEV; ++d) st_ll(P.rs[(P.my_rank + d) & (NDEV - 1)] + (epoch & 1u) * RS_PAR + os_idx(P.my_rank, rtile, t, m), msg);
        } else {
            st_ll(P.rs[rtile / LRT] + (epoch & 1u) * RS_PAR + rs_idx(P.my_rank, rtile % LRT, t, m), msg);
        }
    }
    __syncwarp();
}

__device__ void tail(const Params& P, const Tabs& T, bool small_rw, int tid, unsigned epoch, unsigned par, uint8_t* smem) {
    const int G = gridDim.x, b = blockIdx.x, M = P.M;
    const unsigned need = static_cast<unsigned>(T.misc[2]) * 2u;   // FC2 units (128-row tiles) per 256-row tile
    const int TCH = (M + max(1, G / NRT) - 1) / max(1, G / NRT);
    const int ntc = (M + TCH - 1) / TCH, nitem = NRT * ntc;
    volatile int* s_item = reinterpret_cast<volatile int*>(T.misc + 8);
    unsigned done_rt = 0u;
    for (;;) {
        if (tid == 0) {
            const int it = static_cast<int>(atomicAdd(P.fc2_next + 2 + par, 1u));
            if (it < nitem) {
                const int rt = it / ntc;
                if (!((done_rt >> rt) & 1u)) { wait_ge(P.rt_cnt + par * NRT + rt, need); done_rt |= 1u << rt; }
            }
            *s_item = it;
        }
        __syncthreads();
        const int it = *s_item;
        __syncthreads();
        if (it >= nitem) break;
        const int rtile = it / ntc, t0 = (it - rtile * ntc) * TCH, nt = min(TCH, M - t0);
        __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(smem) + (tid >> 5) * 256;
        for (int tt = tid >> 5; tt < nt; tt += THREADS / 32) t1_pair(P, T, small_rw, rtile, t0 + tt, epoch, stg, tid & 31);
    }
    if (P.dbg && tid == 0) P.dbg[b * 16 + 6] = gtime();
    if (P.ndev == 1) return;
    if (P.oneshot) {
        const int npair = NRT * M, mine = b < npair ? (npair - b + G - 1) / G : 0;
        const uint4* rs_me = P.rs[P.my_rank] + (epoch & 1u) * RS_PAR;
        for (int x = tid; x < mine * NMSG; x += THREADS) {
            const int pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G, rtile = j / M, t = j - rtile * M;
            uint4 v[NDEV];
            unsigned pending = (1u << NDEV) - 1u;
            do {
#pragma unroll
                for (int d = 0; d < NDEV; ++d)
                    if (pending & (1u << d)) { v[d] = ld_ll(rs_me + os_idx(d, rtile, t, m)); if (v[d].w == epoch) pending &= ~(1u << d); }
            } while (pending);
            float a[6] = {bf16lo(v[0].x), bf16hi(v[0].x), bf16lo(v[0].y), bf16hi(v[0].y), bf16lo(v[0].z), bf16hi(v[0].z)};
#pragma unroll
            for (int d = 1; d < NDEV; ++d) {
                a[0] = __fadd_rn(a[0], bf16lo(v[d].x)); a[1] = __fadd_rn(a[1], bf16hi(v[d].x));
                a[2] = __fadd_rn(a[2], bf16lo(v[d].y)); a[3] = __fadd_rn(a[3], bf16hi(v[d].y));
                a[4] = __fadd_rn(a[4], bf16lo(v[d].z)); a[5] = __fadd_rn(a[5], bf16hi(v[d].z));
            }
            uint32_t* o = reinterpret_cast<uint32_t*>(P.out + static_cast<size_t>(t) * DIM + rtile * 256 + 6 * m);
            o[0] = pack_bf16x2(a[0], a[1]); o[1] = pack_bf16x2(a[2], a[3]);
            if (m < NMSG - 1) o[2] = pack_bf16x2(a[4], a[5]);
        }
        return;
    }
    {   // T2: owner reduce in rank order, all-gather push
        const int npair = LRT * M, mine = b < npair ? (npair - b + G - 1) / G : 0;
        const uint4* rs_me = P.rs[P.my_rank] + (epoch & 1u) * RS_PAR;
        for (int x = tid; x < mine * NMSG; x += THREADS) {
            const int pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G, lrt = j / M, t = j - lrt * M;
            uint4 v[NDEV];
            unsigned pending = (1u << NDEV) - 1u;
            do {
#pragma unroll
                for (int d = 0; d < NDEV; ++d)
                    if (pending & (1u << d)) { v[d] = ld_ll(rs_me + rs_idx(d, lrt, t, m)); if (v[d].w == epoch) pending &= ~(1u << d); }
            } while (pending);
            float a[6] = {bf16lo(v[0].x), bf16hi(v[0].x), bf16lo(v[0].y), bf16hi(v[0].y), bf16lo(v[0].z), bf16hi(v[0].z)};
#pragma unroll
            for (int d = 1; d < NDEV; ++d) {
                a[0] = __fadd_rn(a[0], bf16lo(v[d].x)); a[1] = __fadd_rn(a[1], bf16hi(v[d].x));
                a[2] = __fadd_rn(a[2], bf16lo(v[d].y)); a[3] = __fadd_rn(a[3], bf16hi(v[d].y));
                a[4] = __fadd_rn(a[4], bf16lo(v[d].z)); a[5] = __fadd_rn(a[5], bf16hi(v[d].z));
            }
            const uint4 mo = make_uint4(pack_bf16x2(a[0], a[1]), pack_bf16x2(a[2], a[3]), pack_bf16x2(a[4], a[5]), epoch);
            const int rtile = P.my_rank * LRT + lrt;
#pragma unroll
            for (int d = 0; d < NDEV; ++d) st_ll(P.ag[d] + (epoch & 1u) * AG_PAR + ag_idx(rtile, t, m), mo);
        }
    }
    {   // T3: pull the reduced rows
        const int npair = NRT * M, mine = b < npair ? (npair - b + G - 1) / G : 0;
        const uint4* ag_me = P.ag[P.my_rank] + (epoch & 1u) * AG_PAR;
        for (int x = tid; x < mine * NMSG; x += THREADS) {
            const int pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G, rtile = j / M, t = j - rtile * M;
            uint4 v;
            do { v = ld_ll(ag_me + ag_idx(rtile, t, m)); } while (v.w != epoch);
            uint32_t* o = reinterpret_cast<uint32_t*>(P.out + static_cast<size_t>(t) * DIM + rtile * 256 + 6 * m);
            o[0] = v.x; o[1] = v.y;
            if (m < NMSG - 1) o[2] = v.z;
        }
    }
}

// ------------------------------------------------------------------------------------------------ kernel
__global__ void __launch_bounds__(THREADS, 1) tpb2_kernel(const __grid_constant__ Params P) {
    extern __shared__ uint8_t smem_raw[];
    const uint32_t sraw = su32(smem_raw), sb = (sraw + 1023u) & ~1023u;
    uint8_t* smem = smem_raw + (sb - sraw);
    const Tabs T = tabs(smem);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int G = gridDim.x, b = blockIdx.x, M = P.M;
    auto bar = [&](int i) { return sb + OFF_TAB + TAB_BAR + 8 * i; };
    if (P.dbg && tid == 0) P.dbg[b * 16 + 0] = gtime();
    constexpr int B_FULL = 0, B_EMPTY = NS, B_SFR = 2 * NS, B_ACCF = 3 * NS, B_ACCE = 3 * NS + ACC, B_WF = 3 * NS + 2 * ACC, B_WE = B_WF + WR;
    // ---- prologue: barriers, TMEM, lane table (before the PDL wait)
    if (tid == 0) {
        for (int i = 0; i < NS; ++i) { mbar_init(bar(B_FULL + i), 2); mbar_init(bar(B_EMPTY + i), 1); mbar_init(bar(B_SFR + i), 1); }
        for (int i = 0; i < ACC; ++i) { mbar_init(bar(B_ACCF + i), 1); mbar_init(bar(B_ACCE + i), 8); }
        for (int i = 0; i < WR; ++i) { mbar_init(bar(B_WF + i), 1); mbar_init(bar(B_WE + i), 8 + 3); }   // 8 epilogue warps, MMA, B, SF
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(&P.map1) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(&P.map2) : "memory");
    }
    if (warp == 1) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(su32(T.misc + 3)), "r"(TMEM_COLS) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
        tc_fence_before();
    }
    if (tid < 32) reinterpret_cast<uint32_t*>(T.lanes)[tid] = reinterpret_cast<const uint32_t*>(P.lanes)[tid];
    griddep_wait();
    const unsigned epoch = __ldcg(P.work) + 1u, par = epoch & 1u;
    if (b == 0) {   // the next launch's parity set (the previous launch, its user, has completed)
        for (int i = tid; i < MAXP; i += THREADS) P.pan_cnt[(par ^ 1u) * MAXP + i] = 0u;
        if (tid < NRT) P.rt_cnt[(par ^ 1u) * NRT + tid] = 0u;
        if (tid == NRT) P.work[1 + (par ^ 1u)] = 0u;
        if (tid == NRT + 1) { P.fc2_next[par ^ 1u] = 0u; P.fc2_next[2 + (par ^ 1u)] = 0u; P.gcnt[par ^ 1u] = 0u; }
    }
    // ---- routing (fused gate) + MXFP8 quantization of x
    const bool small = P.glog && M <= TPB2_SMALLM;
    int* gids = reinterpret_cast<int*>(smem + PRO_GIDS);
    if (P.glog) {
        if (small) {   // every CTA gates every token (warp w: tokens w, w + 8, ..); CTA 0 also publishes them
            for (int t = warp; t < M; t += THREADS / 32) {
                int id; float w;
                gate_token(P, P.glog + static_cast<size_t>(t) * NEXP, lane, id, w);
                if (lane < P.topk) {
                    gids[t * P.topk + lane] = id; T.srw[t * P.topk + lane] = w;
                    if (b == 0) { P.ids[t * P.topk + lane] = id; P.rw[t * P.topk + lane] = w; }
                }
            }
        } else {   // token t by CTA t % G (warp (t / G) % 8), counted on gcnt[par]
            unsigned* c = P.gcnt + par;
            for (int t = b + warp * G; t < M; t += (THREADS / 32) * G) {
                int id; float w;
                gate_token(P, P.glog + static_cast<size_t>(t) * NEXP, lane, id, w);
                if (lane < P.topk) { P.ids[t * P.topk + lane] = id; P.rw[t * P.topk + lane] = w; }
                __syncwarp();
                if (lane == 0) { fence_acq_rel(); atomicAdd(c, 1u); }
            }
        }
    }
    if (P.glog && !small) {
        if (tid == 0) wait_ge(P.gcnt + par, static_cast<unsigned>(M));
        __syncthreads();
    }
    __syncthreads();
    build_tables(P, smem, T, small ? gids : nullptr);
    if (P.dbg && tid == 0) P.dbg[b * 16 + 1] = gtime();
    for (int t = b + warp * G; t < M; t += (THREADS / 32) * G) {   // x: token t by CTA t % G -> MXFP8, scattered into its panels' images
        quant_token(P, T.tke, t, lane);
        __syncwarp();
        if (lane == 0) { fence_acq_rel(); st_release(P.xflag + t, epoch); }
    }
    if (P.dbg && tid == 0) P.dbg[b * 16 + 2] = gtime();
    const int Ptot = T.misc[2];
    const int total1 = Ptot * NT1 * NSTG1;
    const int lo = fc1_lo(total1, b, G), hi = fc1_lo(total1, b + 1, G);
    fence_proxy_async_smem();   // prologue scratch (generic) precedes the ring's async-proxy writes
    __syncthreads();
    tc_fence_after();
    const uint32_t tmem = static_cast<uint32_t>(T.misc[3]);
    int4* ring = T.ring;   // (written before the slot's WF arrive, read after its wait: the mbarrier orders it)

    if (warp == 0) {
        // ======================================================== producer: work descriptors + weights (A) + SFA
        if (lane == 0) {
            int fills = 0, w = 0;
            auto push = [&](int4 d) {
                const int slot = w % WR;
                if (w >= WR) mbar_wait(bar(B_WE + slot), ((w / WR) - 1) & 1);
                ring[slot] = d;
                mbar_arrive(bar(B_WF + slot));
                ++w;
            };
            auto stage = [&](const CUtensorMap* map, int row, int ch0, const uint8_t* sfa) {
                const int st = fills % NS;
                if (fills >= NS) mbar_wait(bar(B_EMPTY + st), ((fills / NS) - 1) & 1);
                mbar_expect_tx(bar(B_FULL + st), KC * 8192 + KC * 512);
                tma3d(sb + OFF_A + st * A_BYTES, map, 0, row, ch0, bar(B_FULL + st));
                bulk_g2s(sb + OFF_SFA + st * KC * 512, sfa, KC * 512, bar(B_FULL + st));
                ++fills;
            };
            for (int g = lo; g < hi;) {   // FC1 pieces
                const int item = g / NSTG1, s0 = g % NSTG1, s1 = min(NSTG1, hi - item * NSTG1);
                push(make_int4(1, item, s0, s1));
                const int p = item / NT1, tile = item % NT1;
                int e, n, eb;
                panel_of(T, p, e, n, eb);
                for (int s = s0; s < s1; ++s)
                    stage(&P.map1, e * (2 * INTER) + tile * 128, KC * s,
                          P.w13_sf + ((static_cast<size_t>(e) * NT1 + tile) * NCH1 + KC * s) * 512);
                g = item * NSTG1 + s1;
            }
            if (P.dbg) P.dbg[b * 16 + 3] = gtime();
            const int units = Ptot * NRG;
            int tq = static_cast<int>(atomicAdd(P.fc2_next + par, 1u));   // one ticket ahead (its round trip under the unit's loads)
            for (;;) {   // FC2 units (row group rg of FC2G row tiles, panel p) from the queue
                const int u = tq;
                if (u >= units) break;
                push(make_int4(2, u, 0, 0));
                tq = static_cast<int>(atomicAdd(P.fc2_next + par, 1u));
                const int rg = u / Ptot, p = u - rg * Ptot;
                int e, n, eb;
                panel_of(T, p, e, n, eb);
                for (int j = 0; j < FC2G; ++j) {
                    const int r = rg * FC2G + j;
                    for (int s = 0; s < NSTG2; ++s)
                        stage(&P.map2, e * DIM + r * 128, KC * s, P.w2_sf + ((static_cast<size_t>(e) * NRT2 + r) * NCH2 + KC * s) * 512);
                }
            }
            push(make_int4(-1, 0, 0, 0));
        }
        __syncwarp();
    } else if (warp == 7) {
        // ======================================================== scale copier: SFA / SFB images -> TMEM
        if (lane == 0) {
            int fills = 0, w = 0;
            for (;;) {
                const int slot = w % WR;
                mbar_wait(bar(B_WF + slot), (w / WR) & 1);
                const int4 d = ring[slot];
                mbar_arrive(bar(B_WE + slot));
                ++w;
                if (d.x < 0) break;
                const int nst = d.x == 1 ? d.w - d.z : FC2G * NSTG2;
                for (int j = 0; j < nst; ++j) {
                    const int st = fills % NS;
                    mbar_wait(bar(B_FULL + st), (fills / NS) & 1);
                    tc_fence_after();
                    const uint32_t tsf = tmem + TSF0 + st * TSF_STRIDE;
#pragma unroll
                    for (int c = 0; c < KC; ++c) tc_cp_sf(tsf + 4 * c, desc_sf(sb + OFF_SFA + (st * KC + c) * 512));
                    tc_cp_sf(tsf + TSF_B, desc_sf(sb + OFF_SFB + st * 512));
                    tc_commit(bar(B_SFR + st));
                    ++fills;
                }
            }
        }
        __syncwarp();
    } else if (warp == 6) {
        // ======================================================== B operand by bulk copies: FC1 x panel images / FC2 h panel images
        int fills = 0, w = 0;
        bool x_ready = false;
        for (;;) {
            const int slot = w % WR;
            mbar_wait(bar(B_WF + slot), (w / WR) & 1);
            const int4 d = ring[slot];
            __syncwarp();
            if (lane == 0) mbar_arrive(bar(B_WE + slot));
            ++w;
            if (d.x < 0) break;
            if (d.x == 1) {
                if (!x_ready) {   // every token quantized and scattered (by its owner CTA)
                    for (int t = lane; t < M; t += 32) wait_eq(P.xflag + t, epoch);
                    __syncwarp();
                    if (lane == 0) fence_proxy_async_global();   // generic writes -> bulk copies (async proxy)
                    if (P.dbg && lane == 0) P.dbg[b * 16 + 8] = gtime();
                    x_ready = true;
                }
                const int p = d.y / NT1;
                int e, n, eb;
                panel_of(T, p, e, n, eb);
                const int npad = (n + 7) & ~7;
                if (lane == 0) {
                    for (int s = d.z; s < d.w; ++s) {
                        const int st = fills % NS;
                        if (fills >= NS) mbar_wait(bar(B_EMPTY + st), ((fills / NS) - 1) & 1);
                        mbar_expect_tx(bar(B_FULL + st), KC * npad * 128 + 512);
#pragma unroll
                        for (int c = 0; c < KC; ++c)
                            bulk_g2s(sb + OFF_B + st * B_BYTES + c * npad * 128, P.xg + ((static_cast<size_t>(p) * NCH1 + KC * s + c) * NMAX) * 128,
                                     npad * 128, bar(B_FULL + st));
                        bulk_g2s(sb + OFF_SFB + st * 512, P.xgsf + (static_cast<size_t>(p) * NSTG1 + s) * 512, 512, bar(B_FULL + st));
                        ++fills;
                    }
                }
                fills = warp_uniform(static_cast<uint32_t>(fills));
            } else {
                const int p = d.y % Ptot;
                int e, n, eb;
                panel_of(T, p, e, n, eb);
                const int npad = (n + 7) & ~7;
                if (lane == 0) {
                    if (!((T.prdy[p >> 5] >> (p & 31)) & 1u)) {   // h of panel p: its 8 FC1 items (known ready panels skip the poll)
                        wait_ge(P.pan_cnt + par * MAXP + p, static_cast<unsigned>(NT1));
                        fence_proxy_async_global();
                        T.prdy[p >> 5] |= 1u << (p & 31);
                    }
                    for (int f = fills; f < fills + FC2G * NSTG2; ++f) {   // the panel's h once per row tile (L2)
                        const int st = f % NS, s = (f - fills) % NSTG2;
                        if (f >= NS) mbar_wait(bar(B_EMPTY + st), ((f / NS) - 1) & 1);
                        mbar_expect_tx(bar(B_FULL + st), KC * npad * 128 + 512);
#pragma unroll
                        for (int c = 0; c < KC; ++c)
                            bulk_g2s(sb + OFF_B + st * B_BYTES + c * npad * 128,
                                     P.himg + ((static_cast<size_t>(p) * NCH2 + KC * s + c) * NMAX) * 128, npad * 128, bar(B_FULL + st));
                        bulk_g2s(sb + OFF_SFB + st * 512, P.hsf + (static_cast<size_t>(p) * NSTG2 + s) * 512, 512, bar(B_FULL + st));
                    }
                }
                __syncwarp();
                fills += FC2G * NSTG2;
            }
        }
        __syncwarp();
    } else if (warp == 1) {
        // ======================================================== MMA issuer (converged warp, one elected lane)
        int fills = 0, w = 0, uses[ACC] = {0, 0, 0, 0}, ab = 0;
        for (;;) {
            const int slot = w % WR;
            mbar_wait(bar(B_WF + slot), (w / WR) & 1);
            const int4 d = ring[slot];
            __syncwarp();
            if (lane == 0) mbar_arrive(bar(B_WE + slot));
            ++w;
            if (d.x < 0) break;
            const int p = d.x == 1 ? d.y / NT1 : d.y % Ptot;
            int e, n, eb;
            panel_of(T, p, e, n, eb);
            const uint32_t npad = warp_uniform(static_cast<uint32_t>((n + 7) & ~7));
            const uint32_t id0 = idesc_f4f8(static_cast<int>(npad), 0);
            const int nacc = d.x == 1 ? 1 : FC2G, nst = d.x == 1 ? d.w - d.z : NSTG2;
            for (int a = 0; a < nacc; ++a) {   // FC1: one accumulator; FC2: one per row tile of the group
                if (uses[ab] > 0) mbar_wait(bar(B_ACCE + ab), (uses[ab] - 1) & 1);
                tc_fence_after();
                const uint32_t dacc = tmem + ab * ACC_COLS;
                for (int j = 0; j < nst; ++j) {
                    const int st = fills % NS;
                    mbar_wait(bar(B_FULL + st), (fills / NS) & 1);
                    mbar_wait(bar(B_SFR + st), (fills / NS) & 1);
                    if (P.dbg && lane == 0) { if (fills == 0) P.dbg[b * 16 + 9] = gtime(); if (d.x == 1) P.dbg[b * 16 + 10] = gtime(); }
                    tc_fence_after();
                    const uint32_t tsf = tmem + TSF0 + st * TSF_STRIDE;
                    const uint64_t ad0 = desc_sw128(sb + OFF_A + st * A_BYTES), bd0 = desc_sw128(sb + OFF_B + st * B_BYTES);
                    if (elect_one()) {
#pragma unroll
                        for (int c = 0; c < KC; ++c) {
                            const uint64_t ac = ad0 + c * (16384 >> 4), bc = bd0 + c * ((npad * 128) >> 4);
#pragma unroll
                            for (int kb = 0; kb < 4; ++kb)
                                tc_mma(dacc, ac + 2 * kb, bc + 2 * kb, id0 | (kb << 4) | (kb << 29), tsf + 4 * c, tsf + TSF_B + 2 * c, (j | c | kb) != 0);
                        }
                        tc_commit(bar(B_EMPTY + st));
                    }
                    __syncwarp();
                    ++fills;
                }
                if (elect_one()) tc_commit(bar(B_ACCF + ab));
                __syncwarp();
                ++uses[ab];
                ab = (ab + 1) % ACC;
            }
        }
    } else {
        // ======================================================== epilogue: warps 2..5 (group 0: token columns 0..15) and 8..11 (group 1:
        // 16..31); TMEM lanes 32 q .. 32 q + 31, q = warp & 3.  Named barriers: 1 + g (the group's 128 threads), 3 (both groups)
        const int q = warp & 3, L = 32 * q + lane, g = warp >= 8 ? 1 : 0, tb = 16 * g, gt = tid - (g ? 256 : 64);
        const bool up_lane = (lane & 8) == 0;
        const bool pub = warp == 2 && lane == 0;   // the publishing thread
        const int frow = T.lanes[L];
        __nv_bfloat16* stg = T.stg + g * 16 * 128;   // FC2 transpose staging [16 tokens][128 rows]
        int w = 0, uses[ACC] = {0, 0, 0, 0}, ab = 0;
        int rt_cur = -1, rt_n = 0;   // FC2 row tiles stored for 512-row group rt_cur, not yet published (pub)
        bool first_fc1 = true;
        for (;;) {
            const int slot = w % WR;
            mbar_wait(bar(B_WF + slot), (w / WR) & 1);
            const int4 d = ring[slot];
            __syncwarp();
            if (lane == 0) mbar_arrive(bar(B_WE + slot));
            ++w;
            if (d.x < 0) {
                if (pub && rt_n) {
                    fence_acq_rel();
                    red_release_add(P.rt_cnt + par * NRT + 2 * rt_cur, static_cast<unsigned>(rt_n));
                    red_release_add(P.rt_cnt + par * NRT + 2 * rt_cur + 1, static_cast<unsigned>(rt_n));
                }
                break;
            }
            const int p = d.x == 1 ? d.y / NT1 : d.y % Ptot;
            int e, n, eb;
            panel_of(T, p, e, n, eb);
            const int npad = (n + 7) & ~7;
            const int nacc = d.x == 1 ? 1 : FC2G;
            for (int a = 0; a < nacc; ++a) {
                mbar_wait(bar(B_ACCF + ab), uses[ab] & 1);
                if (P.dbg && pub && d.x == 1 && first_fc1) { P.dbg[b * 16 + 11] = gtime(); first_fc1 = false; }
                tc_fence_after();
                float v[16];
                const uint32_t ta = tmem + (static_cast<uint32_t>(32 * q) << 16) + ab * ACC_COLS + tb;
                tc_ld16_wait(ta, v);   // (columns past npad: unused garbage; the buffer has ACC_COLS = 32)
                tc_fence_before();
                __syncwarp();
                if (lane == 0) mbar_arrive(bar(B_ACCE + ab));
                ++uses[ab];
                ab = (ab + 1) % ACC;
                if (d.x == 2) {
                    // ---- FC2 row tile r: transpose through smem, 16-B rows of bf16 per (token, slot)
                    const int r = (d.y / Ptot) * FC2G + a;
#pragma unroll
                    for (int i = 0; i < 16; ++i) if (tb + i < n) stg[i * 128 + frow] = __float2bfloat16_rn(v[i]);
                    named_bar(1 + g, 128);
                    for (int x = gt; x < 16 * 16; x += 128) {   // (token i, 16-B chunk c)
                        const int i = x >> 4, c = x & 15;
                        if (tb + i >= n) break;
                        const uint4 val = *reinterpret_cast<const uint4*>(stg + i * 128 + 8 * c);
                        const int t = T.ent[eb + tb + i];
                        unsigned km = T.km[eb + tb + i];
                        while (km) {
                            const int k = __ffs(km) - 1;
                            km &= km - 1;
                            *reinterpret_cast<uint4*>(P.outp + (static_cast<size_t>(t) * TOPK_MAX + k) * DIM + r * 128 + 8 * c) = val;
                        }
                    }
                    named_bar(1 + g, 128);   // (stg reused by the next row tile)
                    continue;
                }
                // ---- FC1 piece of item d.y, stages [d.z, d.w)
                const int item = d.y, s0 = d.z, s1 = d.w;
                if (s0 > 0) {   // non-owner (this CTA's first piece): fp32 partial -> part[b], flag
                    float* pp = P.part + static_cast<size_t>(b) * NMAX * 128;
#pragma unroll
                    for (int i = 0; i < 16; ++i) if (tb + i < npad) pp[(tb + i) * 128 + L] = v[i];
                    named_bar(3, 256);
                    if (pub) { fence_acq_rel(); st_release(P.part_flags + b, epoch); }
                    continue;
                }
                if (P.dbg && pub) P.dbg[b * 16 + 12] = gtime();
                if (s1 < NSTG1) {   // owner of a split item: add the partials of the CTAs b + 1, .. covering stages s1 .. 15, in CTA order
                    const int iend = (item + 1) * NSTG1;
                    for (int c = b + 1; c < G && fc1_lo(total1, c, G) < iend; ++c) {
                        if (lane == 0) wait_eq(P.part_flags + c, epoch);
                        __syncwarp();
                        const float* pp = P.part + static_cast<size_t>(c) * NMAX * 128;
#pragma unroll
                        for (int i = 0; i < 16; ++i) if (tb + i < npad) v[i] += __ldcg(pp + (tb + i) * 128 + L);
                    }
                }
                if (P.dbg && pub) P.dbg[b * 16 + 13] = gtime();
                // SwiGLU: the gate of an up lane's h column sits in lane ^ 8 of the same warp
                float act[16];
                float am_mine = 0.f;   // lane i: the warp's amax of token tb + i
                const int ng = min(16, n - tb);   // this group's live tokens (warp-uniform; <= 0: none)
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    act[i] = 0.f;
                    if (i < ng) {
                        const float gv = __shfl_xor_sync(0xffffffffu, v[i], 8);
                        float a2 = 0.f;
                        if (up_lane) {
                            const float gg = fminf(gv, 10.0f), uu = fminf(fmaxf(v[i], -10.0f), 10.0f);
                            a2 = uu * (gg / (1.0f + __expf(-gg)));
                        }
                        act[i] = a2;
                        float m = fabsf(a2);   // amax over the warp's 16 up lanes (gate lanes hold 0)
                        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
                        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 8));    // (every lane gets it: lane i holds token i's)
                        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 16));
                        am_mine = lane == i ? m : am_mine;
                    }
                }
                if (P.dbg && pub) P.dbg[b * 16 + 14] = gtime();
                if (lane < 16) T.am[q * NMAX + tb + lane] = am_mine;   // 32-column scale blocks = warp pairs (q, q ^ 1)
                named_bar(1 + g, 128);
                if (P.dbg && pub) P.dbg[b * 16 + 15] = gtime();
                const int tile = item % NT1, q2 = tile >> 1;   // FC2 k-chunk of these 64 h columns: 64 (tile & 1) + 16 q + ..
                uint8_t* himg_p = P.himg + static_cast<size_t>(p) * NCH2 * NMAX * 128 + static_cast<size_t>(q2) * NMAX * 128;
                const int gran = 4 * (tile & 1) + q;   // 16-B granule of this warp's 16 columns in the 128-B row
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    if (i >= ng) break;
                    const int tok = tb + i;
                    const float amax = fmaxf(T.am[q * NMAX + tok], T.am[(q ^ 1) * NMAX + tok]);
                    const int ex = ue8m0_exp_floor(amax);
                    const uint32_t byte = up_lane ? e4m3_of(act[i] * exp2i(-ex)) : 0u;
                    uint32_t word = 0;   // lane wd (< 4) gathers bytes 4 wd .. 4 wd + 3 of the granule (byte jb is in up lane ((jb >> 1) & 7) | (jb & 1) << 4)
#pragma unroll
                    for (int bb = 0; bb < 4; ++bb) {
                        const int jb = 4 * (lane & 3) + bb;
                        word |= (__shfl_sync(0xffffffffu, byte, ((jb >> 1) & 7) | ((jb & 1) << 4)) & 0xFFu) << (8 * bb);
                    }
                    if (lane < 4)
                        *reinterpret_cast<uint32_t*>(himg_p + (tok >> 3) * 1024 + (tok & 7) * 128 + ((gran ^ (tok & 7)) << 4) + 4 * lane) = word;
                    if (lane == 0 && (q & 1) == 0) {   // one scale byte per (token, 32-column block)
                        const int kb = 2 * (tile & 1) + (q >> 1);
                        P.hsf[(static_cast<size_t>(p) * NSTG2 + (q2 >> 1)) * 512 + tok * 16 + 8 * (q2 & 1) + kb] = static_cast<uint8_t>(ex + 127);
                    }
                }
                named_bar(3, 256);
                if (pub) { fence_acq_rel(); red_release_add(P.pan_cnt + par * MAXP + p, 1u); if (P.dbg) P.dbg[b * 16 + 4] = gtime(); }
            }
            if (d.x == 2) {   // FC2 unit done (both groups): count its row tiles toward T1 (published when the row group changes)
                named_bar(3, 256);
                if (pub) {
                    const int rg = d.y / Ptot;
                    if (rg != rt_cur) {
                        if (rt_n) {
                            fence_acq_rel();
                            red_release_add(P.rt_cnt + par * NRT + 2 * rt_cur, static_cast<unsigned>(rt_n));
                            red_release_add(P.rt_cnt + par * NRT + 2 * rt_cur + 1, static_cast<unsigned>(rt_n));
                        }
                        rt_cur = rg; rt_n = 0;
                    }
                    rt_n += 2;   // 2 row tiles of 128 per 256-row T1 tile
                }
            }
        }
    }
    // ---- all roles done: TMEM free, then T1 + all-reduce with every thread
    tc_fence_before();
    __syncthreads();
    if (warp == 1) {
        tc_fence_after();
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(TMEM_COLS) : "memory");
    }
    if (P.dbg && tid == 0) P.dbg[b * 16 + 5] = gtime();
    tail(P, T, small, tid, epoch, par, smem);
    __syncthreads();
    if (P.dbg && tid == 0) P.dbg[b * 16 + 7] = gtime();
    if (tid == 0) {
        fence_acq_rel();
        if (atomicAdd(P.work + 1 + par, 1u) == static_cast<unsigned>(G) - 1u) atomicAdd(P.work, 1u);   // last CTA out: next epoch
    }
}

// ------------------------------------------------------------------------------------------------ host
#ifndef B2_NO_TORCH
using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                              const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeFn encode_fn() {
    static EncodeFn fn = nullptr;
    if (!fn) {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult q;
        C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q));
        TORCH_CHECK(q == cudaDriverEntryPointSuccess && p, "cuTensorMapEncodeTiled not found");
        fn = reinterpret_cast<EncodeFn>(p);
    }
    return fn;
}
static void encode_weight_map(CUtensorMap* map, void* ptr, long long rows, int row_bytes, int chunks) {
    const cuuint64_t dims[3] = {128, static_cast<cuuint64_t>(rows), static_cast<cuuint64_t>(chunks)};
    const cuuint64_t strides[2] = {static_cast<cuuint64_t>(row_bytes), 64};
    const cuuint32_t box[3] = {128, 128, KC}, es[3] = {1, 1, 1};
    const CUresult r = encode_fn()(map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, 3, ptr, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                   CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
}
// both weight maps as 256 bytes of a CPU uint8 tensor (cached by the caller per layer)
torch::Tensor encode_maps(torch::Tensor w13, torch::Tensor w2) {
    TORCH_CHECK(w13.is_cuda() && w2.is_cuda() && w13.element_size() == 1 && w2.element_size() == 1 && w13.is_contiguous() && w2.is_contiguous());
    TORCH_CHECK(w13.dim() == 3 && w13.size(0) == NEXP && w13.size(1) == 2 * INTER && w13.size(2) == DIM / 2, "w13 [256, 1024, 2048]");
    TORCH_CHECK(w2.dim() == 3 && w2.size(0) == NEXP && w2.size(1) == DIM && w2.size(2) == INTER / 2, "w2 [256, 4096, 256]");
    auto out = torch::empty({256}, torch::dtype(torch::kUInt8));
    CUtensorMap m[2];
    encode_weight_map(&m[0], w13.data_ptr(), static_cast<long long>(NEXP) * 2 * INTER, DIM / 2, NCH1);
    encode_weight_map(&m[1], w2.data_ptr(), static_cast<long long>(NEXP) * DIM, INTER / 2, NCH2);
    std::memcpy(out.data_ptr(), m, 256);
    return out;
}
static int g_pdl = 1;
void set_pdl(int64_t v) { g_pdl = static_cast<int>(v); }
// ws: dict-free flat argument list (see tpb2.py RankState): workspace tensors + LL peer pointers
void run(torch::Tensor maps, torch::Tensor w13_sf, torch::Tensor w2_sf, torch::Tensor lanes, torch::Tensor xb, c10::optional<torch::Tensor> glog,
         c10::optional<torch::Tensor> gbias, torch::Tensor ids, torch::Tensor rw, c10::optional<torch::Tensor> shared, double factor,
         torch::Tensor xq, torch::Tensor xs, torch::Tensor xg, torch::Tensor xgsf, c10::optional<torch::Tensor> dbg, torch::Tensor himg, torch::Tensor hsf, torch::Tensor part, torch::Tensor part_flags,
         torch::Tensor xflag, torch::Tensor gcnt, torch::Tensor pan_cnt, torch::Tensor fc2_next, torch::Tensor rt_cnt, torch::Tensor work,
         torch::Tensor outp, std::vector<int64_t> rs, std::vector<int64_t> ag, torch::Tensor out, int64_t my_rank, int64_t ndev, int64_t grid) {
    const int M = static_cast<int>(xb.size(0));
    TORCH_CHECK(M >= 1 && M <= MAXM && xb.size(1) == DIM && xb.scalar_type() == torch::kBFloat16 && xb.is_contiguous());
    TORCH_CHECK(ids.size(0) == M && ids.size(1) <= 6 && ids.scalar_type() == torch::kInt32 && rw.scalar_type() == torch::kFloat32);
    TORCH_CHECK(ndev == 1 || (ndev == NDEV && static_cast<int>(rs.size()) == NDEV && static_cast<int>(ag.size()) == NDEV));
    Params P;
    std::memset(&P, 0, sizeof(P));
    std::memcpy(&P.map1, maps.data_ptr(), 256);
    P.w13_sf = w13_sf.data_ptr<uint8_t>(); P.w2_sf = w2_sf.data_ptr<uint8_t>(); P.lanes = lanes.data_ptr<uint8_t>();
    P.xb = reinterpret_cast<const __nv_bfloat16*>(xb.data_ptr());
    P.glog = glog.has_value() ? glog->data_ptr<float>() : nullptr;
    P.gbias = gbias.has_value() ? gbias->data_ptr<float>() : nullptr;
    TORCH_CHECK(!P.glog || (P.gbias && ids.size(1) == 6), "fused gate: top-6 + bias");
    P.ids = ids.data_ptr<int>(); P.rw = rw.data_ptr<float>(); P.topk = static_cast<int>(ids.size(1));
    P.shared = shared.has_value() ? reinterpret_cast<const __nv_bfloat16*>(shared->data_ptr()) : nullptr;
    P.factor = static_cast<float>(factor);
    P.xq = xq.data_ptr<uint8_t>(); P.xs = xs.data_ptr<uint8_t>(); P.xg = xg.data_ptr<uint8_t>(); P.xgsf = xgsf.data_ptr<uint8_t>();
    P.dbg = dbg.has_value() ? reinterpret_cast<unsigned long long*>(dbg->data_ptr<int64_t>()) : nullptr; P.himg = himg.data_ptr<uint8_t>(); P.hsf = hsf.data_ptr<uint8_t>();
    P.part = part.data_ptr<float>(); P.part_flags = reinterpret_cast<unsigned*>(part_flags.data_ptr<int>());
    P.xflag = reinterpret_cast<unsigned*>(xflag.data_ptr<int>()); P.gcnt = reinterpret_cast<unsigned*>(gcnt.data_ptr<int>());
    P.pan_cnt = reinterpret_cast<unsigned*>(pan_cnt.data_ptr<int>()); P.fc2_next = reinterpret_cast<unsigned*>(fc2_next.data_ptr<int>());
    P.rt_cnt = reinterpret_cast<unsigned*>(rt_cnt.data_ptr<int>()); P.work = reinterpret_cast<unsigned*>(work.data_ptr<int>());
    P.outp = reinterpret_cast<__nv_bfloat16*>(outp.data_ptr());
    for (int d = 0; d < NDEV; ++d) {
        P.rs[d] = ndev == 1 ? nullptr : reinterpret_cast<uint4*>(rs[d]);
        P.ag[d] = ndev == 1 ? nullptr : reinterpret_cast<uint4*>(ag[d]);
    }
    P.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    P.M = M; P.my_rank = static_cast<int>(my_rank); P.ndev = static_cast<int>(ndev);
    P.oneshot = M <= TPB2_ONESHOT_MAXM ? 1 : 0;
    const int G = static_cast<int>(grid);
    TORCH_CHECK(G >= 1 && G <= MAXG);
    static bool attr = false;
    if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(tpb2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_ALLOC)); attr = true; }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(G); cfg.blockDim = dim3(THREADS); cfg.dynamicSmemBytes = SMEM_ALLOC; cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = g_pdl;
    cfg.attrs = at; cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, tpb2_kernel, P));
}
int64_t smem_alloc() { return SMEM_ALLOC; }
#endif  // B2_NO_TORCH

}  // namespace TPB2_NS

#ifndef B2_NO_TORCH
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_maps", &TPB2_NS::encode_maps);
    m.def("run", &TPB2_NS::run);
    m.def("set_pdl", &TPB2_NS::set_pdl);
    m.def("smem_alloc", &TPB2_NS::smem_alloc);
    m.attr("MAXM") = TPB2_NS::MAXM; m.attr("NDEV") = TPB2_NS::NDEV; m.attr("MAXP") = TPB2_NS::MAXP; m.attr("MAXG") = TPB2_NS::MAXG;
    m.attr("NMAX") = TPB2_NS::NMAX; m.attr("NRT") = TPB2_NS::NRT; m.attr("NMSG") = TPB2_NS::NMSG; m.attr("NCH2") = TPB2_NS::NCH2;
    m.attr("NSTG2") = TPB2_NS::NSTG2; m.attr("NCH1") = TPB2_NS::NCH1; m.attr("NSTG1") = TPB2_NS::NSTG1; m.attr("RS_PAR") = static_cast<int64_t>(TPB2_NS::RS_PAR); m.attr("AG_PAR") = static_cast<int64_t>(TPB2_NS::AG_PAR);
}
#endif
