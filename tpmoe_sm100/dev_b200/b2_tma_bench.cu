// B200 weight-stream ceiling (no MMA): one producer thread per CTA streams 128-row weight tiles of a [E * 1024 rows][2048 B] tensor
// (the w13 layout) through an NS-stage smem ring; a consumer thread frees each stage as soon as it lands.  Map kinds: 0 = 16U4_ALIGN16B
// (FP4 unpacked to 8-bit containers in smem, as tcgen05 kind::f8f6f4 needs: 2 x smem per DRAM byte), 1 = UINT8 (same DRAM bytes, packed in
// smem).  Box: 128 rows x KC chunks of 64 B (packed) per row.  Reports GB/s of DRAM bytes.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

namespace b2tb {
__device__ __forceinline__ uint32_t su32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint32_t b, int c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(c) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint32_t b, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t parity) {
    asm volatile("{\n .reg .pred p;\nW_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n" ::"r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma3d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, uint32_t bar, uint64_t hint) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], %6;"
                 ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(bar), "l"(hint) : "memory");
}

struct Args { CUtensorMap map; int kind, kc, ns, tiles, hintsel, nchunk; };

__global__ void __launch_bounds__(64, 1) tma_bench(const __grid_constant__ Args A, unsigned long long* out) {
    extern __shared__ uint8_t smem_raw[];
    const uint32_t sraw = su32(smem_raw), sb = (sraw + 1023u) & ~1023u;
    const int tid = threadIdx.x, G = gridDim.x, b = blockIdx.x;
    const uint32_t sbytes = A.kind == 0 ? A.kc * 16384u : A.kc * 8192u;   // smem bytes per stage
    const uint32_t bars = sb + A.ns * sbytes;
    if (tid == 0) {
        for (int i = 0; i < A.ns; ++i) { mbar_init(bars + 8 * i, 1); mbar_init(bars + 8 * (16 + i), 1); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    const int nst = A.nchunk / A.kc;   // stages per tile (nchunk chunks of 64 B per row)
    const uint64_t hint = A.hintsel ? 0x12F0000000000000ull : 0x1000000000000000ull;   // evict-first / evict-normal
    unsigned long long t0 = 0;
    if (tid == 0) {
        t0 = clock64();
        int fills = 0;
        for (int t = b; t < A.tiles; t += G)
            for (int s = 0; s < nst; ++s) {
                const int st = fills % A.ns;
                if (fills >= A.ns) mbar_wait(bars + 8 * (16 + st), ((fills / A.ns) - 1) & 1);
                mbar_expect_tx(bars + 8 * st, A.kc * 8192u);   // (complete_tx counts global bytes: 16U4 lands 2x in smem)
                tma3d(sb + st * sbytes, &A.map, 0, t * 128, A.kc * s, bars + 8 * st, hint);
                ++fills;
            }
    } else if (tid == 32) {
        int fills = 0;
        for (int t = b; t < A.tiles; t += G)
            for (int s = 0; s < nst; ++s) {
                const int st = fills % A.ns;
                mbar_wait(bars + 8 * st, (fills / A.ns) & 1);
                mbar_arrive(bars + 8 * (16 + st));
                ++fills;
            }
        out[b] = clock64();
    }
    if (tid == 0) out[G + b] = t0;
}

using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                              const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
double run(torch::Tensor w, int64_t kind, int64_t kc, int64_t ns, int64_t grid, int64_t hintsel, int64_t l2promo, int64_t rowb) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q);
    auto enc = reinterpret_cast<EncodeFn>(p);
    const long long rows = w.numel() / rowb;   // [rows][rowb B]
    Args A;
    std::memset(&A, 0, sizeof(A));
    A.kind = static_cast<int>(kind); A.kc = static_cast<int>(kc); A.ns = static_cast<int>(ns); A.tiles = static_cast<int>(rows / 128);
    A.hintsel = static_cast<int>(hintsel); A.nchunk = static_cast<int>(rowb / 64);
    const CUtensorMapL2promotion pr = l2promo == 0 ? CU_TENSOR_MAP_L2_PROMOTION_NONE : (l2promo == 1 ? CU_TENSOR_MAP_L2_PROMOTION_L2_128B : CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    if (kind == 0) {
        const cuuint64_t dims[3] = {128, static_cast<cuuint64_t>(rows), static_cast<cuuint64_t>(rowb / 64)};
        const cuuint64_t strides[2] = {static_cast<cuuint64_t>(rowb), 64};
        const cuuint32_t box[3] = {128, 128, static_cast<cuuint32_t>(kc)}, es[3] = {1, 1, 1};
        TORCH_CHECK(enc(&A.map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, 3, w.data_ptr(), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                        CU_TENSOR_MAP_SWIZZLE_128B, pr, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);
    } else {
        const cuuint64_t dims[3] = {64, static_cast<cuuint64_t>(rows), 32};
        const cuuint64_t strides[2] = {2048, 64};
        const cuuint32_t box[3] = {64, 128, static_cast<cuuint32_t>(kc)}, es[3] = {1, 1, 1};
        TORCH_CHECK(enc(&A.map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, w.data_ptr(), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                        CU_TENSOR_MAP_SWIZZLE_64B, pr, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);
    }
    const uint32_t sbytes = kind == 0 ? kc * 16384 : kc * 8192;
    const int smem = static_cast<int>(ns * sbytes + 32 * 8 + 1024);
    TORCH_CHECK(smem <= 232448, "smem");
    cudaFuncSetAttribute(tma_bench, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    auto o = torch::zeros({2 * grid}, torch::dtype(torch::kInt64).device(w.device()));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto st = at::cuda::getCurrentCUDAStream();
    tma_bench<<<static_cast<int>(grid), 64, smem, st>>>(A, reinterpret_cast<unsigned long long*>(o.data_ptr<int64_t>()));   // warm-up
    cudaEventRecord(e0, st);
    for (int i = 0; i < 3; ++i) tma_bench<<<static_cast<int>(grid), 64, smem, st>>>(A, reinterpret_cast<unsigned long long*>(o.data_ptr<int64_t>()));
    cudaEventRecord(e1, st);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return static_cast<double>(rows) * static_cast<double>(rowb) * 3 / (ms * 1e-3) / 1e9;   // GB/s of DRAM bytes
}
}  // namespace b2tb
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &b2tb::run); }
