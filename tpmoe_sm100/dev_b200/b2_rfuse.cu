// Router GEMM in our own kernel, bitwise == production (cuBLAS split-K for 5 <= M <= 64): logits = sum over the M's K-parts (in order, fp32)
// of tcgen05 kind::f16 K16 chains.  One CTA per (16-expert group, K-part): A = x tokens (M <= 64 rows, M = 64 MMA, rows >= M zero-filled
// by TMA), B = W 16 expert rows (N = 16), the whole K-part's operands in smem at once (one round trip), D in TMEM -> part [p][M][256] fp32.
// The kernel triggers its PDL dependents at entry (the MoE kernel's pre-wait prologue overlaps it).
#ifndef B2_NO_TORCH
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace b2r {
__device__ __forceinline__ uint32_t su32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint32_t b, int c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(c) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint32_t b, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t parity) {
    asm volatile("{\n .reg .pred p;\nW_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n" ::"r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma3d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                 ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
}
__device__ __forceinline__ uint64_t sdesc(uint32_t addr) {   // K-major SW128, 8-row groups 1024 B apart
    return static_cast<uint64_t>((addr >> 4) & 0x3FFF) | (static_cast<uint64_t>((1024 >> 4) & 0x3FFF) << 32) | (1ull << 46) | (2ull << 61);
}
constexpr int MAXC = 22;                         // K-part chunks (64 elements) <= 1408 / 64
constexpr int GC = 2;                            // x chunks per TMA (a 3-D box: 64 elements x 64 rows x GC chunks); every K-part is a multiple
constexpr int A_OFF = 0, B_OFF = MAXC * 8192, BAR_OFF = B_OFF + MAXC * 2048, SMEM = BAR_OFF + (MAXC + 2) * 8 + 1024;

struct RP {
    CUtensorMap mx, mw;     // 3-D {64 elements, rows, 64 chunks}: x box {64, 64, GC}, W box {64, 16, chunks of the part}
    float* part;            // [S][M][256]
    float* dbg;             // [128][16] raw TMEM (layout probe) or null
    int M, S, k0[4], kn[4]; // K-part starts / lengths (multiples of 64)
    int wc;                 // W box chunks (= the longest part's)
};

__global__ void __launch_bounds__(128, 1) rfuse(const __grid_constant__ RP P) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // (the MoE kernel may start its pre-wait prologue now)
    extern __shared__ uint8_t smraw[];
    const uint32_t s0 = (su32(smraw) + 1023u) & ~1023u;
    __shared__ uint32_t tslot;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int eg = blockIdx.x & 15, p = blockIdx.x >> 4;   // expert group (16 experts), K-part
    const int nch = P.kn[p] / 64, kb0 = P.k0[p];
    const uint32_t bars = s0 + BAR_OFF, bmma = bars + 8 * MAXC;
    if (tid == 0) {
        for (int c = 0; c < nch; ++c) mbar_init(bars + 8 * c, 1);
        mbar_init(bmma, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 1) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(su32(&tslot)), "r"(32) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    const uint32_t tmem = tslot;
    asm volatile("griddepcontrol.wait;" ::: "memory");   // (x: written by the previous kernel)
    const int ng = (nch + GC - 1) / GC;   // barrier g: x chunks GC g .. (+ all of W on barrier 0)
    if (tid == 0) {   // the whole K-part at once: x in GC-chunk boxes (rows >= M zero-filled), W's 16 rows in one box
        for (int g = 0; g < ng; ++g) {
            const int cn = min(GC, nch - GC * g);
            mbar_expect_tx(bars + 8 * g, static_cast<uint32_t>(cn) * 8192 + (g == 0 ? static_cast<uint32_t>(P.wc) * 2048 : 0u));
            if (g == 0) tma3d(s0 + B_OFF, &P.mw, 0, 16 * eg, kb0 / 64, bars);
            tma3d(s0 + A_OFF + GC * g * 8192, &P.mx, 0, 0, kb0 / 64 + GC * g, bars + 8 * g);
        }
    }
    if (warp == 0) {
        // kind::f16: D f32 (bit 4), A / B bf16 (1 << 7, 1 << 10), K-major, N = 16 (>> 3 at 17), M = 64 (>> 4 at 24)
        const uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) | ((16u >> 3) << 17) | ((64u >> 4) << 24);
        for (int c = 0; c < nch; ++c) {
            if (c % GC == 0) mbar_wait(bars + 8 * (c / GC), 0);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            uint32_t pred = 0;
            asm volatile("{\n .reg .pred q;\n elect.sync _|q, 0xffffffff;\n selp.u32 %0, 1, 0, q;\n}" : "=r"(pred));
            if (pred) {
                const uint64_t ad = sdesc(s0 + A_OFF + c * 8192), bd = sdesc(s0 + B_OFF + c * 2048);
#pragma unroll
                for (int k = 0; k < 4; ++k)
                    asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, q;\n}"
                                 ::"r"(tmem), "l"(ad + 2 * k), "l"(bd + 2 * k), "r"(idesc), "r"(static_cast<uint32_t>((c | k) != 0)) : "memory");
                if (c == nch - 1) asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bmma) : "memory");
            }
            __syncwarp();
        }
    }
    mbar_wait(bmma, 0);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    uint32_t r[16];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n\ttcgen05.wait::ld.sync.aligned;"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]),
                   "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]) : "r"(tmem + ((32u * warp) << 16)) : "memory");
    const int L = 32 * warp + lane;
    if (P.dbg && blockIdx.x == 0)
        for (int j = 0; j < 16; ++j) P.dbg[L * 16 + j] = __uint_as_float(r[j]);
    // M = 64 data path (layout probe decides): token row t of D in TMEM lane 16 (t / 16) * 2 + ..? -- default assumption: rows 0..15 in
    // lanes 0..15 of warp 0, rows 16..31 in lanes 32..47, .. (the warp's low half-lanes)
    if (lane < 16) {
        const int t = 16 * warp + lane;
        if (t < P.M)
            for (int j = 0; j < 16; ++j) P.part[(static_cast<size_t>(p) * P.M + t) * 256 + 16 * eg + j] = __uint_as_float(r[j]);
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    if (warp == 1) {
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(32) : "memory");
    }
}

#ifndef B2_NO_TORCH
using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                              const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeFn encode_fn() {
    static EncodeFn fn = nullptr;
    if (!fn) {
        void* q = nullptr;
        cudaDriverEntryPointQueryResult r;
        C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &q, 12000, cudaEnableDefault, &r));
        TORCH_CHECK(r == cudaDriverEntryPointSuccess && q, "cuTensorMapEncodeTiled not found");
        fn = reinterpret_cast<EncodeFn>(q);
    }
    return fn;
}
static void map3d(CUtensorMap* m, void* ptr, int rows, int box_rows, int box_chunks) {   // {64 elements, rows, 64 chunks}: smem [chunk][row][128 B]
    const cuuint64_t dims[3] = {64, static_cast<cuuint64_t>(rows), 64}, strides[2] = {4096 * 2, 128};
    const cuuint32_t box[3] = {64, static_cast<cuuint32_t>(box_rows), static_cast<cuuint32_t>(box_chunks)}, es[3] = {1, 1, 1};
    TORCH_CHECK(encode_fn()(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, ptr, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);
}
// x [M][4096] bf16, w [256][4096] bf16, part [S][M][256] fp32; parts (k0, kn) pairs
void run(torch::Tensor x, torch::Tensor w, torch::Tensor part, std::vector<int64_t> parts, c10::optional<torch::Tensor> dbg, int64_t pdl) {
    const at::cuda::CUDAGuard g(x.device());
    RP P{};
    const int M = static_cast<int>(x.size(0)), S = static_cast<int>(parts.size() / 2);
    TORCH_CHECK(M >= 1 && M <= 64 && S >= 1 && S <= 4);
    map3d(&P.mx, x.data_ptr(), M, 64, GC);
    P.part = part.data_ptr<float>(); P.dbg = dbg ? dbg->data_ptr<float>() : nullptr; P.M = M; P.S = S;
    for (int i = 0; i < S; ++i) { P.k0[i] = static_cast<int>(parts[2 * i]); P.kn[i] = static_cast<int>(parts[2 * i + 1]); TORCH_CHECK(P.kn[i] % (64 * GC) == 0 && P.kn[i] <= 64 * MAXC); }
    // (W box = the parts' chunk count: equal-length parts, or the longest -- a shorter last part's extra chunks are loaded, never used)
    int maxc = 0;
    for (int i = 0; i < S; ++i) maxc = std::max(maxc, P.kn[i] / 64);
    map3d(&P.mw, w.data_ptr(), 256, 16, maxc);
    P.wc = maxc;
    static bool attr = false;
    if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(rfuse, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(16 * S); cfg.blockDim = dim3(128); cfg.dynamicSmemBytes = SMEM; cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = static_cast<int>(pdl);
    cfg.attrs = at; cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, rfuse, P));
}
#endif
}  // namespace b2r
#ifndef B2_NO_TORCH
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &b2r::run); }
#endif
