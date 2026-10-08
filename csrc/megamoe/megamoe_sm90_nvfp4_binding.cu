/*
 * TVM-FFI bindings for the SM90 NVFP4 MegaMoE fused kernel.
 *
 * The symmetric-buffer layout is computed here too: the kernel addresses the
 * input, pool and combine regions by fixed offsets inside one allocation, so
 * Python must slice exactly the same way.
 */
#include <cuda_runtime.h>

#include "megamoe_sm90_nvfp4_config.inc"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Optional;
using tvm::ffi::TensorView;

namespace flashinfer::megamoe {

void sm90_nvfp4_mega_moe_fused_launch(void* y, int* cumulative_local_expert_recv_stats,
                                      int num_tokens, const std::vector<int64_t>& sym_buffer_ptrs,
                                      int rank_idx, void* l1_acts, void* l1_acts_sf, void* l2_acts,
                                      void* l2_acts_sf, void* l1_weights, void* l2_weights,
                                      int64_t l1_acts_stride, int64_t l2_acts_stride,
                                      int64_t l1_weights_inner, int64_t l1_weights_stride,
                                      int64_t l2_weights_inner, int64_t l2_weights_stride,
                                      const float* l1_global_scales, const float* l2_global_scales,
                                      cudaStream_t stream);

namespace {
namespace cfg = ::flashinfer::megamoe::config;

inline void* offset_ptr(TensorView t, int64_t byte_offset) {
  return static_cast<char*>(t.data_ptr()) + byte_offset;
}
}  // namespace

/*!
 * \brief Byte offsets of every region inside the symmetric buffer.
 *
 * Returns, in order:
 *   [0] total bytes required
 *   [1] x            (fp8, [num_max_tokens_per_rank, hidden])
 *   [2] x_sf         (f32, [num_max_tokens_per_rank, hidden/128])
 *   [3] topk_idx     (i64, [num_max_tokens_per_rank, num_topk])
 *   [4] topk_weights (f32, [num_max_tokens_per_rank, num_topk])
 *   [5] l1_acts      (fp8, [num_max_pool_tokens, hidden])
 *   [6] l1_acts_sf   (f32, [num_padded_sf_pool_tokens, hidden/128], MN-major)
 *   [7] l2_acts      (fp8, [num_max_pool_tokens, intermediate_hidden])
 *   [8] l2_acts_sf   (f32, [num_padded_sf_pool_tokens, intermediate_hidden/64], MN-major)
 *   [9] num_max_pool_tokens
 *  [10] num_padded_sf_pool_tokens
 */
Array<int64_t> megamoe_sm90_nvfp4_symm_buffer_layout() {
  constexpr int kNumRanks = 8;
  constexpr int kNumExpertsPerRank = cfg::kNumExperts / kNumRanks;

  const layout::Workspace workspace(nullptr, kNumRanks, cfg::kNumExperts, cfg::kNumMaxTokensPerRank,
                                    cfg::kNumTopk);

  const layout::Data fp8_token(cfg::kHidden);
  const layout::Data fp8_sf(cfg::kHidden / 32);
  const layout::Data fp8_inter_token(cfg::kIntermediateHidden);
  const layout::Data fp8_inter_sf(cfg::kIntermediateHidden / 16);
  const layout::Data topk_idx(cfg::kNumTopk * sizeof(int64_t), false);
  const layout::Data topk_weights(cfg::kNumTopk * sizeof(float), false);
  const layout::Data bf16_token(cfg::kHidden * 2);

  const layout::Buffer in_token(fp8_token, 1, cfg::kNumMaxTokensPerRank, workspace.get_end_ptr());
  const layout::Buffer in_sf(fp8_sf, 1, cfg::kNumMaxTokensPerRank, in_token.get_end_ptr());
  const layout::Buffer in_topk_idx(topk_idx, 1, cfg::kNumMaxTokensPerRank, in_sf.get_end_ptr());
  const layout::Buffer in_topk_w(topk_weights, 1, cfg::kNumMaxTokensPerRank,
                                 in_topk_idx.get_end_ptr());
  const layout::Buffer l1_token(fp8_token, 1, cfg::kNumMaxPoolTokens, in_topk_w.get_end_ptr());
  const layout::Buffer l1_sf(fp8_sf, 1, cfg::kNumPaddedSFPoolTokens, l1_token.get_end_ptr());
  const layout::Buffer l1_topk_w(layout::Data(sizeof(float), false), 1, cfg::kNumMaxPoolTokens,
                                 l1_sf.get_end_ptr());
  const layout::Buffer l2_token(fp8_inter_token, 1, cfg::kNumMaxPoolTokens,
                                l1_topk_w.get_end_ptr());
  const layout::Buffer l2_sf(fp8_inter_sf, 1, cfg::kNumPaddedSFPoolTokens, l2_token.get_end_ptr());
  const layout::Buffer combine(bf16_token, 1, cfg::kNumMaxPoolTokens, l2_sf.get_end_ptr());

  auto off = [](const void* p) { return reinterpret_cast<int64_t>(p); };
  return Array<int64_t>({off(combine.get_end_ptr()), off(in_token.base), off(in_sf.base),
                         off(in_topk_idx.base), off(in_topk_w.base), off(l1_token.base),
                         off(l1_sf.base), off(l2_token.base), off(l2_sf.base),
                         static_cast<int64_t>(cfg::kNumMaxPoolTokens),
                         static_cast<int64_t>(cfg::kNumPaddedSFPoolTokens)});
}

/*! \brief Plan constants this module was compiled for (for assertions in Python). */
Array<int64_t> megamoe_sm90_nvfp4_plan() {
  return Array<int64_t>(
      {static_cast<int64_t>(cfg::kBlockM), static_cast<int64_t>(cfg::kBlockN),
       static_cast<int64_t>(cfg::kNumStages), static_cast<int64_t>(cfg::kNumExpertsPerWave),
       static_cast<int64_t>(cfg::kPushDispatch), static_cast<int64_t>(cfg::kNoCleanBarrier),
       static_cast<int64_t>(cfg::kUseInterleavedScheduler), static_cast<int64_t>(cfg::kRSSwapAB)});
}

void megamoe_sm90_nvfp4_fused(TensorView y, TensorView symm_buffer, Array<int64_t> symm_buffer_ptrs,
                              int64_t rank_idx, TensorView l1_weights, TensorView l2_weights,
                              Array<int64_t> offsets, int64_t num_tokens,
                              Optional<TensorView> cumulative_local_expert_recv_stats,
                              Optional<TensorView> l1_global_scales,
                              Optional<TensorView> l2_global_scales) {
  TVM_FFI_ICHECK_EQ(symm_buffer_ptrs.size(), 8) << "MegaMoE is an 8-rank kernel";
  TVM_FFI_ICHECK_GE(num_tokens, 1);
  TVM_FFI_ICHECK_LE(num_tokens, cfg::kNumMaxTokensPerRank);

  const std::vector<int64_t> ptrs(symm_buffer_ptrs.begin(), symm_buffer_ptrs.end());

  int* stats = cumulative_local_expert_recv_stats.has_value()
                   ? static_cast<int*>(cumulative_local_expert_recv_stats.value().data_ptr())
                   : nullptr;
  const float* gs1 = l1_global_scales.has_value()
                         ? static_cast<const float*>(l1_global_scales.value().data_ptr())
                         : nullptr;
  const float* gs2 = l2_global_scales.has_value()
                         ? static_cast<const float*>(l2_global_scales.value().data_ptr())
                         : nullptr;

  sm90_nvfp4_mega_moe_fused_launch(
      y.data_ptr(), stats, static_cast<int>(num_tokens), ptrs, static_cast<int>(rank_idx),
      offset_ptr(symm_buffer, offsets[5]), offset_ptr(symm_buffer, offsets[6]),
      offset_ptr(symm_buffer, offsets[7]), offset_ptr(symm_buffer, offsets[8]),
      l1_weights.data_ptr(), l2_weights.data_ptr(),
      /* l1_acts_stride   */ cfg::kHidden,
      /* l2_acts_stride   */ cfg::kIntermediateHidden,
      /* l1_weights_inner */ l1_weights.size(2), l1_weights.stride(1),
      /* l2_weights_inner */ l2_weights.size(2), l2_weights.stride(1), gs1, gs2,
      get_stream(y.device()));
}

}  // namespace flashinfer::megamoe

TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_nvfp4_fused,
                              flashinfer::megamoe::megamoe_sm90_nvfp4_fused);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_nvfp4_symm_buffer_layout,
                              flashinfer::megamoe::megamoe_sm90_nvfp4_symm_buffer_layout);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_nvfp4_plan,
                              flashinfer::megamoe::megamoe_sm90_nvfp4_plan);
