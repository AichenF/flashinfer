/*
 * SM90 NVFP4 MegaMoE -- host launcher.
 *
 * One persistent kernel fuses the cross-rank token dispatch, both FFN GEMMs
 * (NVFP4 weights x FP8 activations), SwiGLU and the combine reduction over an
 * 8-rank NVLink symmetric buffer. Everything that shapes shared memory or the
 * task schedule is a template parameter, so this file is compiled once per
 * (shape, plan) pair against the generated config header.
 */
#include <cuda.h>
#include <cuda_runtime.h>

#include "megamoe_sm90_nvfp4_config.inc"
#include "tvm_ffi_utils.h"

namespace flashinfer::megamoe {
namespace {

namespace cfg = ::flashinfer::megamoe::config;

inline CUtensorMapDataType tma_dtype(DLDataType dt) {
  if (dt.code == kDLFloat8_e4m3fn) return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  if (dt.code == kDLFloat && dt.bits == 32) return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  if (dt.code == kDLUInt && dt.bits == 8) return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  TVM_FFI_ICHECK(false) << "unsupported TMA dtype";
  return CU_TENSOR_MAP_DATA_TYPE_UINT8;
}

inline CUtensorMapSwizzle tma_swizzle(int mode) {
  switch (mode) {
    case 0:
      return CU_TENSOR_MAP_SWIZZLE_NONE;
    case 32:
      return CU_TENSOR_MAP_SWIZZLE_32B;
    case 64:
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case 128:
      return CU_TENSOR_MAP_SWIZZLE_128B;
  }
  TVM_FFI_ICHECK(false) << "unsupported swizzle mode " << mode;
  return CU_TENSOR_MAP_SWIZZLE_NONE;
}

// 2-D tiled TMA descriptor. `gmem_inner_dim` is the contiguous dim; with a
// swizzle the smem inner extent is dictated by the swizzle width.
CUtensorMap make_tma_2d_desc(void* ptr, DLDataType dtype, int gmem_inner_dim, int gmem_outer_dim,
                             int smem_inner_dim, int smem_outer_dim, int gmem_outer_stride,
                             int swizzle_mode) {
  const int elem_size = (dtype.bits * dtype.lanes) / 8;
  if (swizzle_mode != 0) smem_inner_dim = swizzle_mode / elem_size;

  CUtensorMap tensor_map{};
  const cuuint64_t gmem_dims[2] = {static_cast<cuuint64_t>(gmem_inner_dim),
                                   static_cast<cuuint64_t>(gmem_outer_dim)};
  const cuuint32_t smem_dims[2] = {static_cast<cuuint32_t>(smem_inner_dim),
                                   static_cast<cuuint32_t>(smem_outer_dim)};
  const cuuint64_t gmem_strides[1] = {
      static_cast<cuuint64_t>(static_cast<int64_t>(gmem_outer_stride) * elem_size)};
  const cuuint32_t elem_strides[2] = {1, 1};
  const CUresult res = cuTensorMapEncodeTiled(
      &tensor_map, tma_dtype(dtype), 2, ptr, gmem_dims, gmem_strides, smem_dims, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, tma_swizzle(swizzle_mode), CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(res, CUDA_SUCCESS) << "cuTensorMapEncodeTiled failed: " << res;
  return tensor_map;
}

inline int tma_aligned_size(int size, int elem_size) {
  constexpr int kAlignment = 16;
  const int t = kAlignment / elem_size;
  return (size + t - 1) / t * t;
}

// Scale-factor descriptor: MN-major, unswizzled, one column per K granule.
CUtensorMap make_tma_sf_desc(void* ptr, DLDataType dtype, int shape_mn, int shape_k, int block_mn,
                             int gran_k) {
  const int elem_size = (dtype.bits * dtype.lanes) / 8;
  shape_mn = tma_aligned_size(shape_mn, elem_size);
  const int outer = (shape_k + gran_k - 1) / gran_k;
  return make_tma_2d_desc(ptr, dtype, shape_mn, outer, block_mn, 1, shape_mn, 0);
}

}  // namespace

// Launch one MegaMoE call. Pointers are raw device addresses; `sym_buffer_ptrs`
// holds every rank's base address of the symmetric buffer (NVLink peer mapped).
void sm90_nvfp4_mega_moe_fused_launch(void* y, int* cumulative_local_expert_recv_stats,
                                      int num_tokens, const std::vector<int64_t>& sym_buffer_ptrs,
                                      int rank_idx, void* l1_acts, void* l1_acts_sf, void* l2_acts,
                                      void* l2_acts_sf, void* l1_weights, void* l2_weights,
                                      int64_t l1_acts_stride, int64_t l2_acts_stride,
                                      int64_t l1_weights_inner, int64_t l1_weights_stride,
                                      int64_t l2_weights_inner, int64_t l2_weights_stride,
                                      const float* l1_global_scales, const float* l2_global_scales,
                                      cudaStream_t stream) {
  constexpr int kBlockK = 128;
  constexpr int kSwizzleActs = 128;
  constexpr int kBStoragePerKBlock = 80;
  constexpr int kL1ScaleGranK = 128;
  constexpr int kNumExpertsPerRank = cfg::kNumExperts / 8;

  const DLDataType fp8{kDLFloat8_e4m3fn, 8, 1};
  const DLDataType f32{kDLFloat, 32, 1};
  const DLDataType u8{kDLUInt, 8, 1};

  const auto tm_l1_acts =
      make_tma_2d_desc(l1_acts, fp8, cfg::kHidden, cfg::kNumMaxPoolTokens, kBlockK, cfg::kBlockM,
                       static_cast<int>(l1_acts_stride), kSwizzleActs);
  const auto tm_l1_acts_sf = make_tma_sf_desc(l1_acts_sf, f32, cfg::kNumPaddedSFPoolTokens,
                                              cfg::kHidden, cfg::kBlockM, kL1ScaleGranK);
  const auto tm_l1_weights =
      make_tma_2d_desc(l1_weights, u8, static_cast<int>(l1_weights_inner),
                       kNumExpertsPerRank * cfg::kIntermediateHidden * 2, kBStoragePerKBlock,
                       cfg::kBlockN, static_cast<int>(l1_weights_stride), 0);
  // L1 writes its SwiGLU output as half-width blocks into the L2 activation pool.
  const auto tm_l1_output =
      make_tma_2d_desc(l2_acts, fp8, cfg::kIntermediateHidden, cfg::kNumMaxPoolTokens,
                       cfg::kBlockN / 2, cfg::kBlockM, static_cast<int>(l2_acts_stride), 0);
  const auto tm_l2_acts =
      make_tma_2d_desc(l2_acts, fp8, cfg::kIntermediateHidden, cfg::kNumMaxPoolTokens, kBlockK,
                       cfg::kBlockM, static_cast<int>(l2_acts_stride), kSwizzleActs);
  const auto tm_l2_acts_sf =
      make_tma_sf_desc(l2_acts_sf, f32, cfg::kNumPaddedSFPoolTokens, cfg::kIntermediateHidden,
                       cfg::kBlockM, cfg::kBlockN / 2);
  const auto tm_l2_weights = make_tma_2d_desc(l2_weights, u8, static_cast<int>(l2_weights_inner),
                                              kNumExpertsPerRank * cfg::kHidden, kBStoragePerKBlock,
                                              cfg::kBlockN, static_cast<int>(l2_weights_stride), 0);

  const layout::SymBuffer<8> sym_buffer(sym_buffer_ptrs, static_cast<uint32_t>(rank_idx));

  auto* kernel = reinterpret_cast<void*>(cfg::kKernel);
  TVM_FFI_ICHECK_EQ(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, cfg::kSmemSize),
      cudaSuccess)
      << "failed to raise the dynamic shared-memory limit to " << cfg::kSmemSize;

  const uint32_t tokens = static_cast<uint32_t>(num_tokens);
  unsigned long long* phase_stamps = nullptr;
  void* args[] = {&y,
                  &cumulative_local_expert_recv_stats,
                  &phase_stamps,
                  const_cast<uint32_t*>(&tokens),
                  const_cast<layout::SymBuffer<8>*>(&sym_buffer),
                  const_cast<CUtensorMap*>(&tm_l1_acts),
                  const_cast<CUtensorMap*>(&tm_l1_acts_sf),
                  const_cast<CUtensorMap*>(&tm_l1_weights),
                  const_cast<CUtensorMap*>(&tm_l1_output),
                  const_cast<CUtensorMap*>(&tm_l2_acts),
                  const_cast<CUtensorMap*>(&tm_l2_acts_sf),
                  const_cast<CUtensorMap*>(&tm_l2_weights),
                  const_cast<const float**>(&l1_global_scales),
                  const_cast<const float**>(&l2_global_scales)};

  // Persistent kernel: exactly one CTA per SM.
  TVM_FFI_ICHECK_EQ(
      cudaLaunchKernel(kernel, dim3(MEGAMOE_NUM_SMS), dim3(384), args, cfg::kSmemSize, stream),
      cudaSuccess)
      << "MegaMoE kernel launch failed";
}

}  // namespace flashinfer::megamoe
