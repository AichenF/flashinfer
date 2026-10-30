// B200 (sm_100a) unit test of the tcgen05 block-scaled MXFP4 x MXFP8 GEMM on SGLang's trtllm-gen expert layout, read in place:
//   D[128 x N] (fp32, TMEM) = A (one 128-row tile of an expert's weights, e2m1 + ue8m0 per 32 k) x B^T (N token rows, e4m3 + ue8m0 per 32 k)
// One CTA, 4 warps, fully serialized (one 128-k chunk per round trip): correctness of descriptors / scale images / N, not speed.
//   A: TMA 16U4_ALIGN16B (packed e2m1 -> 16-byte containers), 3-D map {128 elements, rows, chunks} (chunk stride 64 B), SWIZZLE_128B
//   SFA: SGLang's block_scale_interleave bytes = the 32x128b.warpx4 image (512 B per (128-row tile, 128-k chunk))
//   B: host-built K-major SW128 image [chunk][Npad rows][128 B]; SFB: image per chunk pair, lane = token, byte 8 (c & 1) + b
#ifndef B2_NO_TORCH   // (compile checks without a GPU / torch: -DB2_NO_TORCH builds the device code alone)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

namespace b2t {

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
__device__ __forceinline__ uint64_t smem_desc(uint32_t addr, uint32_t lbo, uint32_t sbo, uint32_t layout) {
  return static_cast<uint64_t>((addr >> 4) & 0x3FFF) | (static_cast<uint64_t>((lbo >> 4) & 0x3FFF) << 16) |
         (static_cast<uint64_t>((sbo >> 4) & 0x3FFF) << 32) | (1ull << 46) | (static_cast<uint64_t>(layout) << 61);
}
__device__ __forceinline__ uint64_t desc_sw128(uint32_t addr) { return smem_desc(addr, 0, 1024, 2); }
__device__ __forceinline__ uint64_t desc_sf(uint32_t addr) { return smem_desc(addr, 0, 128, 0); }
// kind::mxf8f6f4: A e2m1 (5) or e4m3 (0), B e4m3 (0), fp32 D, ue8m0 scales (bit 23), K-major, M = 128 (M >> 4 at bit 24), N >> 3 at bit 17
__device__ __forceinline__ uint32_t idesc_mx(int n, uint32_t atype, uint32_t sf_id) {
  return (sf_id << 4) | (atype << 7) | (0u << 10) | (static_cast<uint32_t>(n >> 3) << 17) | (1u << 23) | ((128u >> 4) << 24) | (sf_id << 29);
}
__device__ __forceinline__ void tc_mma(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t sfa, uint32_t sfb, uint32_t acc) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %6, 0;\n"
               " tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32 [%0], %1, %2, %3, [%4], [%5], p;\n}"
               ::"r"(d), "l"(a), "l"(b), "r"(idesc), "r"(sfa), "r"(sfb), "r"(acc) : "memory");
}
__device__ __forceinline__ void tc_cp_sf(uint32_t taddr, uint64_t desc) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(desc) : "memory");
}
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

struct Params {
  const uint8_t* sfa;    // the tile's SFA images: chunk c at sfa + c * 512
  const uint8_t* bimg;   // [chunks][npad][128] SW128 (A e2m1) / same for an e4m3 A via amap
  const uint8_t* sfb;    // [chunks / 2][512]
  float* out;            // [128][npad]
  int row0, chunks, npad, atype;   // atype 5: e2m1 A through the 16U4 map; 0: e4m3 A (map of plain bytes)
};

constexpr int SA = 0, SB = 16384, SSFA = SB + 32 * 128, SSFB = SSFA + 512, SBAR = SSFB + 512, SMEM = SBAR + 64 + 1024;
constexpr int TCOLS = 64, T_SFA = 32, T_SFB = 48;

__global__ void __launch_bounds__(128, 1) umma_test(const __grid_constant__ CUtensorMap amap, Params p) {
  extern __shared__ uint8_t smem_raw[];
  const uint32_t sraw = su32(smem_raw), sb = (sraw + 1023u) & ~1023u;
  uint8_t* base = smem_raw + (sb - sraw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  __shared__ uint32_t tmem_slot;
  const uint32_t bar_full = sb + SBAR, bar_sf = bar_full + 8, bar_mma = bar_full + 16;
  if (tid == 0) {
    mbar_init(bar_full, 1); mbar_init(bar_sf, 1); mbar_init(bar_mma, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 1) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(su32(&tmem_slot)), "r"(TCOLS) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = tmem_slot;
  const uint32_t bbytes = static_cast<uint32_t>(p.npad) * 128;
  const uint32_t abytes = p.atype == 5 ? 8192u : 16384u;
  for (int c = 0; c < p.chunks; ++c) {
    const uint32_t ph = c & 1;
    if (tid == 0) {
      mbar_expect_tx(bar_full, abytes + 512 + bbytes + 512);
      tma3d(sb + SA, &amap, 0, p.row0, c, bar_full);
      bulk_g2s(sb + SSFA, p.sfa + static_cast<size_t>(c) * 512, 512, bar_full);
      bulk_g2s(sb + SB, p.bimg + static_cast<size_t>(c) * bbytes, bbytes, bar_full);
      bulk_g2s(sb + SSFB, p.sfb + static_cast<size_t>(c >> 1) * 512, 512, bar_full);
      mbar_wait(bar_full, ph);
      tc_fence_after();
      tc_cp_sf(tmem + T_SFA, desc_sf(sb + SSFA));
      tc_cp_sf(tmem + T_SFB, desc_sf(sb + SSFB));
      tc_commit(bar_sf);
      mbar_wait(bar_sf, ph);
      tc_fence_after();
      const uint64_t ad = desc_sw128(sb + SA), bd = desc_sw128(sb + SB);
#pragma unroll
      for (int b = 0; b < 4; ++b)
        tc_mma(tmem, ad + 2 * b, bd + 2 * b, idesc_mx(p.npad, p.atype, b), tmem + T_SFA, tmem + T_SFB + 2 * (c & 1), (c | b) != 0);
      tc_commit(bar_mma);
      mbar_wait(bar_mma, ph);
      tc_fence_after();
    }
    __syncthreads();
  }
  tc_fence_after();
  const uint32_t ta = tmem + (static_cast<uint32_t>(32 * warp) << 16);
  for (int j = 0; j < p.npad / 8; ++j) {
    float v[8];
    tc_ld8(ta + 8 * j, v);
    tc_wait_ld();
#pragma unroll
    for (int i = 0; i < 8; ++i) p.out[(32 * warp + lane) * p.npad + 8 * j + i] = v[i];
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(TCOLS) : "memory");
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

// a: [rows][row_bytes] packed e2m1 (atype 5) or e4m3 bytes (atype 0); one 128-k chunk per TMA box of 128 rows
void run(torch::Tensor a, int64_t rows, int64_t row_bytes, int64_t atype, torch::Tensor sfa, torch::Tensor bimg, torch::Tensor sfb,
         torch::Tensor out, int64_t row0, int64_t chunks, int64_t npad) {
  const at::cuda::CUDAGuard g(a.device());
  CUtensorMap map;
  if (atype == 5) {
    const cuuint64_t dims[3] = {128, static_cast<cuuint64_t>(rows), static_cast<cuuint64_t>(chunks)};
    const cuuint64_t strides[2] = {static_cast<cuuint64_t>(row_bytes), 64};
    const cuuint32_t box[3] = {128, 128, 1}, es[3] = {1, 1, 1};
    TORCH_CHECK(encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, 3, a.data_ptr(), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);
  } else {
    const cuuint64_t dims[3] = {128, static_cast<cuuint64_t>(rows), static_cast<cuuint64_t>(chunks)};
    const cuuint64_t strides[2] = {static_cast<cuuint64_t>(row_bytes), 128};
    const cuuint32_t box[3] = {128, 128, 1}, es[3] = {1, 1, 1};
    TORCH_CHECK(encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, a.data_ptr(), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);
  }
  Params p{sfa.data_ptr<uint8_t>(), bimg.data_ptr<uint8_t>(), sfb.data_ptr<uint8_t>(), out.data_ptr<float>(), static_cast<int>(row0),
           static_cast<int>(chunks), static_cast<int>(npad), static_cast<int>(atype)};
  C10_CUDA_CHECK(cudaFuncSetAttribute(umma_test, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  umma_test<<<1, 128, SMEM, at::cuda::getCurrentCUDAStream()>>>(map, p);
  C10_CUDA_CHECK(cudaGetLastError());
}

#endif  // B2_NO_TORCH
}  // namespace b2t

#ifndef B2_NO_TORCH
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &b2t::run); }
#endif
