// B200 unit test of the cta_group::2 (CTA pair, M = 256) tcgen05 block-scaled MXFP4 x MXFP8 MMA: where the pair's A / B / SFA / SFB
// halves live.  CTA r of the cluster loads A rows 128 r .. (its own 128-row tile) + SFA, B token rows [r N/2, (r + 1) N/2) (variant 0) and
// an SFB image of either its own N/2 tokens (sfb_mode 0) or all N tokens (sfb_mode 1); the leader (rank 0) issues the scale copies
// (tcgen05.cp.cta_group::2) and the MMAs; each CTA reads D[its 128 rows][N] from its own TMEM.
#ifndef B2_NO_TORCH
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace b2p {

__device__ __forceinline__ uint32_t su32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint32_t b, int c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(c) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint32_t b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t parity) {
  asm volatile("{\n .reg .pred p;\nW_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n" ::"r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma3d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
               ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
}
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory"); }
__device__ __forceinline__ uint64_t smem_desc(uint32_t addr, uint32_t lbo, uint32_t sbo, uint32_t layout) {
  return static_cast<uint64_t>((addr >> 4) & 0x3FFF) | (static_cast<uint64_t>((lbo >> 4) & 0x3FFF) << 16) |
         (static_cast<uint64_t>((sbo >> 4) & 0x3FFF) << 32) | (1ull << 46) | (static_cast<uint64_t>(layout) << 61);
}
__device__ __forceinline__ uint64_t desc_sw128(uint32_t addr) { return smem_desc(addr, 0, 1024, 2); }
__device__ __forceinline__ uint64_t desc_sf(uint32_t addr) { return smem_desc(addr, 0, 128, 0); }
// M = 256 (M >> 4 at bit 24), N >> 3 at bit 17, A e2m1 (5), B e4m3 (0), ue8m0 scales (bit 23)
__device__ __forceinline__ uint32_t idesc2(int n, uint32_t sf_id) {
  return (sf_id << 4) | (5u << 7) | (static_cast<uint32_t>(n >> 3) << 17) | (1u << 23) | ((256u >> 4) << 24) | (sf_id << 29);
}
__device__ __forceinline__ void tc_ld8(uint32_t taddr, float* v) {
  uint32_t r[8];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n\ttcgen05.wait::ld.sync.aligned;"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr) : "memory");
#pragma unroll
  for (int i = 0; i < 8; ++i) v[i] = __uint_as_float(r[i]);
}

struct Params {
  const uint8_t* sfa;    // [2][chunks][512]: tile r's SFA images
  const uint8_t* bimg;   // [2][chunks][nb][128]: CTA r's B rows image (nb rows each)
  const uint8_t* sfb;    // [2][chunks / 2][512]: CTA r's SFB image
  float* out;            // [2][128][n]
  int row0, chunks, n, nb;
};

constexpr int SA = 0, SB = 16384, SSFA = SB + 32 * 128, SSFB = SSFA + 512, SBAR = SSFB + 512, SMEM = SBAR + 64 + 1024;
constexpr int TCOLS = 128, T_SFA = 64, T_SFB = 96;

__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(128, 1) pair_test(const __grid_constant__ CUtensorMap amap, Params p) {
  extern __shared__ uint8_t smem_raw[];
  const uint32_t sraw = su32(smem_raw), sb = (sraw + 1023u) & ~1023u;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  __shared__ uint32_t tmem_slot;
  const uint32_t bar_full = sb + SBAR, bar_mma = bar_full + 8;
  if (tid == 0) {
    mbar_init(bar_full, 1); mbar_init(bar_mma, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 1) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(su32(&tmem_slot)), "r"(TCOLS) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
  }
  tc_fence_before();
  cluster_sync();
  tc_fence_after();
  const uint32_t tmem = tmem_slot;
  const uint32_t bbytes = static_cast<uint32_t>(p.nb) * 128;
  for (int c = 0; c < p.chunks; ++c) {
    const uint32_t ph = c & 1;
    if (tid == 0) {   // every CTA: its A tile, SFA, B rows, SFB for chunk c (own barrier), then the pair syncs
      mbar_expect_tx(bar_full, 8192 + 512 + bbytes + 512);
      tma3d(sb + SA, &amap, 0, p.row0 + 128 * static_cast<int>(rank), c, bar_full);
      bulk_g2s(sb + SSFA, p.sfa + (static_cast<size_t>(rank) * p.chunks + c) * 512, 512, bar_full);
      bulk_g2s(sb + SB, p.bimg + (static_cast<size_t>(rank) * p.chunks + c) * bbytes, bbytes, bar_full);
      bulk_g2s(sb + SSFB, p.sfb + (static_cast<size_t>(rank) * ((p.chunks + 1) / 2) + (c >> 1)) * 512, 512, bar_full);
      mbar_wait(bar_full, ph);
    }
    tc_fence_before();
    cluster_sync();   // (both CTAs' operands landed)
    tc_fence_after();
    if (rank == 0 && tid == 0) {
      asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" ::"r"(tmem + T_SFA), "l"(desc_sf(sb + SSFA)) : "memory");
      asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" ::"r"(tmem + T_SFB), "l"(desc_sf(sb + SSFB)) : "memory");
      const uint64_t ad = desc_sw128(sb + SA), bd = desc_sw128(sb + SB);
#pragma unroll
      for (int b = 0; b < 4; ++b)
        asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %6, 0;\n"
                     " tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.block32 [%0], %1, %2, %3, [%4], [%5], q;\n}"
                     ::"r"(tmem), "l"(ad + 2 * b), "l"(bd + 2 * b), "r"(idesc2(p.n, b)), "r"(tmem + T_SFA), "r"(tmem + T_SFB + 2 * (c & 1)),
                       "r"(static_cast<uint32_t>((c | b) != 0)) : "memory");
      asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                   ::"r"(bar_mma), "h"(static_cast<uint16_t>(3)) : "memory");
    }
    if (tid == 0) mbar_wait(bar_mma, ph);
    tc_fence_before();
    cluster_sync();
    tc_fence_after();
  }
  const uint32_t ta = tmem + (static_cast<uint32_t>(32 * warp) << 16);
  for (int j = 0; j < p.n / 8; ++j) {
    float v[8];
    tc_ld8(ta + 8 * j, v);
#pragma unroll
    for (int i = 0; i < 8; ++i) p.out[(static_cast<size_t>(rank) * 128 + 32 * warp + lane) * p.n + 8 * j + i] = v[i];
  }
  tc_fence_before();
  cluster_sync();
  if (warp == 1) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(TCOLS) : "memory");
  }
}

#ifndef B2_NO_TORCH
using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                              const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
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

void run(torch::Tensor a, int64_t rows, int64_t row_bytes, torch::Tensor sfa, torch::Tensor bimg, torch::Tensor sfb, torch::Tensor out,
         int64_t row0, int64_t chunks, int64_t n, int64_t nb) {
  const at::cuda::CUDAGuard g(a.device());
  CUtensorMap map;
  const cuuint64_t dims[3] = {128, static_cast<cuuint64_t>(rows), static_cast<cuuint64_t>(chunks)};
  const cuuint64_t strides[2] = {static_cast<cuuint64_t>(row_bytes), 64};
  const cuuint32_t box[3] = {128, 128, 1}, es[3] = {1, 1, 1};
  TORCH_CHECK(encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, 3, a.data_ptr(), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                          CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);
  Params p{sfa.data_ptr<uint8_t>(), bimg.data_ptr<uint8_t>(), sfb.data_ptr<uint8_t>(), out.data_ptr<float>(), static_cast<int>(row0),
           static_cast<int>(chunks), static_cast<int>(n), static_cast<int>(nb)};
  C10_CUDA_CHECK(cudaFuncSetAttribute(pair_test, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  pair_test<<<2, 128, SMEM, at::cuda::getCurrentCUDAStream()>>>(map, p);
  C10_CUDA_CHECK(cudaGetLastError());
}
#endif  // B2_NO_TORCH
}  // namespace b2p

#ifndef B2_NO_TORCH
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &b2p::run); }
#endif
