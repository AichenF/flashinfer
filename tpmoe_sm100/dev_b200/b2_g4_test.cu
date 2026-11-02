// gather4 semantics check: 2D uint8 tensor [64][4096]; gather rows {5, 9, 2, 40} of 128-B column block c into smem with SW128 (box {128, bh})
// and 16-B rows without swizzle (box {16, bh}); dump smem.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cstdint>
#include <cstring>
namespace g4t {
__device__ __forceinline__ uint32_t su32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
struct Args { CUtensorMap m1, m2; int c; int r[4]; int o1, o2; };
__global__ void k(const __grid_constant__ Args A, uint8_t* out) {
    __shared__ __align__(1024) uint8_t buf[4096];
    __shared__ __align__(8) uint64_t bar;
    const uint32_t b = su32(&bar);
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(b));
        asm volatile("fence.mbarrier_init.release.cluster;");
        asm volatile("mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(512 + 64));
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                     ::"r"(su32(buf) + A.o1), "l"(&A.m1), "r"(A.c * 128), "r"(A.r[0]), "r"(A.r[1]), "r"(A.r[2]), "r"(A.r[3]), "r"(b) : "memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                     ::"r"(su32(buf + 2048) + A.o2), "l"(&A.m2), "r"(A.c * 16), "r"(A.r[0]), "r"(A.r[1]), "r"(A.r[2]), "r"(A.r[3]), "r"(b) : "memory");
        asm volatile("{\n .reg .pred p;\nW:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n @!p bra W;\n}" ::"r"(b) : "memory");
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) out[i] = buf[i];
}
using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                              const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
torch::Tensor run(torch::Tensor x, torch::Tensor s, int64_t c, int64_t bh, int64_t o1, int64_t o2) {
    void* p = nullptr; cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q);
    auto enc = reinterpret_cast<EncodeFn>(p);
    Args A; std::memset(&A, 0, sizeof(A));
    { const cuuint64_t d[2] = {4096, 64}, st[1] = {4096}; const cuuint32_t bx[2] = {128, static_cast<cuuint32_t>(bh)}, es[2] = {1, 1};
      TORCH_CHECK(enc(&A.m1, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, x.data_ptr(), d, st, bx, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS, "enc1"); }
    { const cuuint64_t d[2] = {256, 64}, st[1] = {256}; const cuuint32_t bx[2] = {16, static_cast<cuuint32_t>(bh)}, es[2] = {1, 1};
      TORCH_CHECK(enc(&A.m2, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, s.data_ptr(), d, st, bx, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS, "enc2"); }
    A.c = static_cast<int>(c); A.o1 = static_cast<int>(o1); A.o2 = static_cast<int>(o2); A.r[0] = 5; A.r[1] = 9; A.r[2] = 2; A.r[3] = 40;
    auto out = torch::zeros({4096}, torch::dtype(torch::kUInt8).device(x.device()));
    k<<<1, 128, 0, at::cuda::getCurrentCUDAStream()>>>(A, out.data_ptr<uint8_t>());
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    return out;
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &g4t::run); }
