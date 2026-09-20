/* TVM-FFI binding for the split (BN128) MegaMoE family. */
#include <cuda_runtime.h>

#include "megamoe_sm90_nvfp4_split_config.inc"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Optional;
using tvm::ffi::TensorView;

namespace flashinfer::megamoe {

void sm90_nvfp4_mega_moe_split_launch(void* y, int* stats, int num_tokens,
                                      const std::vector<int64_t>& sym_buffer_ptrs, int rank_idx,
                                      void* l1_acts, void* l1_acts_sf, void* l2_acts,
                                      void* l2_acts_sf, void* l1_weights, void* l2_weights,
                                      int64_t l1_weights_inner, int64_t l1_weights_stride,
                                      int64_t l2_weights_inner, int64_t l2_weights_stride,
                                      const float* gs1, const float* gs2, cudaStream_t stream);

namespace {
inline void* offset_ptr(TensorView t, int64_t off) {
  return static_cast<char*>(t.data_ptr()) + off;
}
}  // namespace

void megamoe_sm90_nvfp4_split(TensorView y, TensorView symm_buffer, Array<int64_t> symm_buffer_ptrs,
                              int64_t rank_idx, TensorView l1_weights, TensorView l2_weights,
                              Array<int64_t> offsets, int64_t num_tokens,
                              Optional<TensorView> cumulative_local_expert_recv_stats,
                              Optional<TensorView> l1_global_scales,
                              Optional<TensorView> l2_global_scales) {
  namespace cfg = ::flashinfer::megamoe::config;
  TVM_FFI_ICHECK_EQ(symm_buffer_ptrs.size(), cfg::kNumRanks);
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

  sm90_nvfp4_mega_moe_split_launch(
      y.data_ptr(), stats, static_cast<int>(num_tokens), ptrs, static_cast<int>(rank_idx),
      offset_ptr(symm_buffer, offsets[5]), offset_ptr(symm_buffer, offsets[6]),
      offset_ptr(symm_buffer, offsets[7]), offset_ptr(symm_buffer, offsets[8]),
      l1_weights.data_ptr(), l2_weights.data_ptr(), l1_weights.size(2), l1_weights.stride(1),
      l2_weights.size(2), l2_weights.stride(1), gs1, gs2, get_stream(y.device()));
}

}  // namespace flashinfer::megamoe

TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_nvfp4_split,
                              flashinfer::megamoe::megamoe_sm90_nvfp4_split);
