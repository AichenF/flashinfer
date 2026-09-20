/*
 * SM90 NVFP4 MegaMoE -- split (BN128) launcher.
 *
 * Two kernels per call: L1 dequantises the dispatched rows and runs the first
 * GEMM into the intermediate pool; L2 runs the second GEMM and scatters the
 * result back. L1 runs with a 2-CTA cluster, so it needs the extended launch.
 */
#include <cuda.h>
#include <cuda_runtime.h>

#include "megamoe_sm90_nvfp4_split_config.inc"
#include "tvm_ffi_utils.h"

namespace flashinfer::megamoe {
namespace {

namespace cfg = ::flashinfer::megamoe::config;

inline CUtensorMapDataType tma_dtype(DLDataType dt) {
  if (dt.code == kDLFloat8_e4m3fn) return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  if (dt.code == kDLFloat && dt.bits == 32) return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  return CU_TENSOR_MAP_DATA_TYPE_UINT8;
}

inline CUtensorMapSwizzle tma_swizzle(int mode) {
  return mode == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE;
}

CUtensorMap make_tma_2d_desc(void* ptr, DLDataType dtype, int gmem_inner_dim, int gmem_outer_dim,
                             int smem_inner_dim, int smem_outer_dim, int gmem_outer_stride,
                             int swizzle_mode) {
  const int elem_size = (dtype.bits * dtype.lanes) / 8;
  if (swizzle_mode != 0) smem_inner_dim = swizzle_mode / elem_size;
  CUtensorMap tm{};
  const cuuint64_t gd[2] = {static_cast<cuuint64_t>(gmem_inner_dim),
                            static_cast<cuuint64_t>(gmem_outer_dim)};
  const cuuint32_t sd[2] = {static_cast<cuuint32_t>(smem_inner_dim),
                            static_cast<cuuint32_t>(smem_outer_dim)};
  const cuuint64_t gs[1] = {
      static_cast<cuuint64_t>(static_cast<int64_t>(gmem_outer_stride) * elem_size)};
  const cuuint32_t es[2] = {1, 1};
  const CUresult res =
      cuTensorMapEncodeTiled(&tm, tma_dtype(dtype), 2, ptr, gd, gs, sd, es,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, tma_swizzle(swizzle_mode),
                             CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(res, CUDA_SUCCESS) << "cuTensorMapEncodeTiled failed: " << res;
  return tm;
}

inline int tma_aligned_size(int size, int elem_size) {
  const int t = 16 / elem_size;
  return (size + t - 1) / t * t;
}

CUtensorMap make_tma_sf_desc(void* ptr, DLDataType dtype, int shape_mn, int shape_k, int block_mn,
                             int gran_k) {
  const int elem_size = (dtype.bits * dtype.lanes) / 8;
  shape_mn = tma_aligned_size(shape_mn, elem_size);
  return make_tma_2d_desc(ptr, dtype, shape_mn, (shape_k + gran_k - 1) / gran_k, block_mn, 1,
                          shape_mn, 0);
}

void launch_clustered(void* kernel, int threads, int smem, int cluster, void** args,
                      cudaStream_t stream) {
  TVM_FFI_ICHECK_EQ(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem),
                    cudaSuccess)
      << "failed to raise dynamic shared memory to " << smem;
  cudaLaunchConfig_t lc{};
  lc.gridDim = dim3(cfg::kNumSMs);
  lc.blockDim = dim3(threads);
  lc.dynamicSmemBytes = smem;
  lc.stream = stream;
  cudaLaunchAttribute attrs[1];
  int n = 0;
  if (cluster > 1) {
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {static_cast<unsigned>(cluster), 1, 1};
    n = 1;
  }
  lc.attrs = attrs;
  lc.numAttrs = n;
  TVM_FFI_ICHECK_EQ(cudaLaunchKernelExC(&lc, kernel, args), cudaSuccess)
      << "MegaMoE split kernel launch failed";
}

}  // namespace

void sm90_nvfp4_mega_moe_split_launch(void* y, int* stats, int num_tokens,
                                      const std::vector<int64_t>& sym_buffer_ptrs, int rank_idx,
                                      void* l1_acts, void* l1_acts_sf, void* l2_acts,
                                      void* l2_acts_sf, void* l1_weights, void* l2_weights,
                                      int64_t l1_weights_inner, int64_t l1_weights_stride,
                                      int64_t l2_weights_inner, int64_t l2_weights_stride,
                                      const float* gs1, const float* gs2, cudaStream_t stream) {
  constexpr int kEPR = cfg::kNumExperts / cfg::kNumRanks;
  const DLDataType fp8{kDLFloat8_e4m3fn, 8, 1};
  const DLDataType f32{kDLFloat, 32, 1};
  const DLDataType u8{kDLUInt, 8, 1};

  const auto tm_l1_acts =
      make_tma_2d_desc(l1_acts, fp8, cfg::kHidden, cfg::kNumMaxPoolTokens, cfg::kBlockK,
                       cfg::kBlockM, cfg::kHidden, cfg::kSwizzleActsMode);
  const auto tm_l1_acts_sf = make_tma_sf_desc(l1_acts_sf, f32, cfg::kNumPaddedSFPoolTokens,
                                              cfg::kHidden, cfg::kBlockM, 128);
  const auto tm_l1_weights = make_tma_2d_desc(
      l1_weights, u8, static_cast<int>(l1_weights_inner), kEPR * cfg::kIntermediateHidden * 2,
      cfg::kWeightStoragePerKBlock, cfg::kBlockN, static_cast<int>(l1_weights_stride), 0);
  // The two M warpgroups each store a 64x64 post-SwiGLU tile, so the L1 output
  // box is 64x64 -- not (BLOCK_N/2, BLOCK_M).
  constexpr int kL1OutputStoreBlockM = 64, kL1OutputStoreBlockN = 64;
  const auto tm_l1_output =
      make_tma_2d_desc(l2_acts, fp8, cfg::kIntermediateHidden, cfg::kNumMaxPoolTokens,
                       kL1OutputStoreBlockN, kL1OutputStoreBlockM, cfg::kIntermediateHidden, 0);
  const auto tm_l2_acts =
      make_tma_2d_desc(l2_acts, fp8, cfg::kIntermediateHidden, cfg::kNumMaxPoolTokens, cfg::kBlockK,
                       cfg::kBlockM, cfg::kIntermediateHidden, cfg::kSwizzleActsMode);
  const auto tm_l2_acts_sf = make_tma_sf_desc(l2_acts_sf, f32, cfg::kNumPaddedSFPoolTokens,
                                              cfg::kIntermediateHidden, cfg::kBlockM, 128);
  const auto tm_l2_weights = make_tma_2d_desc(l2_weights, u8, static_cast<int>(l2_weights_inner),
                                              kEPR * cfg::kHidden, cfg::kWeightStoragePerKBlock,
                                              cfg::kBlockN, static_cast<int>(l2_weights_stride), 0);

  const layout::SymBuffer<8> sym_buffer(sym_buffer_ptrs, static_cast<uint32_t>(rank_idx));
  const uint32_t tokens = static_cast<uint32_t>(num_tokens);

  void* l1_args[] = {&stats,
                     const_cast<uint32_t*>(&tokens),
                     const_cast<layout::SymBuffer<8>*>(&sym_buffer),
                     const_cast<CUtensorMap*>(&tm_l1_acts),
                     const_cast<CUtensorMap*>(&tm_l1_acts_sf),
                     const_cast<CUtensorMap*>(&tm_l1_weights),
                     const_cast<CUtensorMap*>(&tm_l1_output),
                     const_cast<const float**>(&gs1)};
  launch_clustered(reinterpret_cast<void*>(cfg::kL1Kernel), cfg::kL1NumThreads, cfg::kL1SmemSize,
                   cfg::kL1ClusterSize, l1_args, stream);

  void* l2_args[] = {&y,
                     &stats,
                     const_cast<uint32_t*>(&tokens),
                     const_cast<layout::SymBuffer<8>*>(&sym_buffer),
                     const_cast<CUtensorMap*>(&tm_l2_acts),
                     const_cast<CUtensorMap*>(&tm_l2_acts_sf),
                     const_cast<CUtensorMap*>(&tm_l2_weights),
                     const_cast<const float**>(&gs2)};
  launch_clustered(reinterpret_cast<void*>(cfg::kL2Kernel), cfg::kL2NumThreads, cfg::kL2SmemSize,
                   cfg::kL2ClusterSize, l2_args, stream);
}

}  // namespace flashinfer::megamoe
