"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import (
    bench_gpu_time,
    attention_tflops_per_sec_with_actual_seq_lens,
)


def bench_bsr_block_sparse_attention(
    num_qo_heads,
    num_kv_heads,
    head_dim,
    seq_len,
    R,
    C,
    block_density,
):
    if num_qo_heads % num_kv_heads != 0:
        return
    MB = seq_len // R
    NB = seq_len // C
    if MB < 1 or NB < 1:
        return

    # synthesize BSR sparsity pattern
    block_mask = torch.rand(MB, NB) < block_density
    for i in range(MB):
        if not block_mask[i].any():
            block_mask[i, torch.randint(NB, (1,))] = True

    indptr_l, indices_l = [0], []
    for i in range(MB):
        for j in range(NB):
            if block_mask[i, j]:
                indices_l.append(j)
        indptr_l.append(len(indices_l))
    indptr = torch.tensor(indptr_l, dtype=torch.int32, device="cuda")
    indices = torch.tensor(indices_l, dtype=torch.int32, device="cuda")

    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"
    )

    # Benchmark sparse attention with cuTile
    q_ct = torch.randn(seq_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda")
    k_ct = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")
    v_ct = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")
    ct_wrapper = flashinfer.BlockSparseAttentionWrapper(
        float_workspace_buffer, backend="cutile"
    )
    ct_wrapper.plan(
        indptr, indices, seq_len, seq_len,
        R, C, num_qo_heads, num_kv_heads, head_dim,
        q_data_type=torch.half,
    )
    measurements_cutile = bench_gpu_time(
        lambda: ct_wrapper.run(q_ct, k_ct, v_ct),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
    sparse_ms_cutile = np.median(measurements_cutile)

    # Benchmark sparse attention with FA2 (via VariableBlockSparseAttentionWrapper with uniform blocks)
    block_row_sz = torch.full((num_kv_heads, MB), R, dtype=torch.int32)
    block_col_sz = torch.full((num_kv_heads, NB), C, dtype=torch.int32)
    block_mask_map = block_mask.cpu().unsqueeze(0).expand(num_kv_heads, -1, -1).clone()

    q = torch.randn(num_qo_heads, seq_len, head_dim, dtype=torch.half, device="cuda")
    k = torch.randn(num_kv_heads, seq_len, head_dim, dtype=torch.half, device="cuda")
    v = torch.randn(num_kv_heads, seq_len, head_dim, dtype=torch.half, device="cuda")

    sparse_wrapper_fa2 = flashinfer.sparse.VariableBlockSparseAttentionWrapper(
        float_workspace_buffer, backend="fa2"
    )
    sparse_wrapper_fa2.plan(
        block_mask_map=block_mask_map,
        block_row_sz=block_row_sz,
        block_col_sz=block_col_sz,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_data_type=torch.half,
    )
    measurements_fa2 = bench_gpu_time(
        lambda: sparse_wrapper_fa2.run(q, k, v),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
    sparse_ms_fa2 = np.median(measurements_fa2)

    # Benchmark sparse attention with FA3
    try:
        sparse_wrapper_fa3 = flashinfer.sparse.VariableBlockSparseAttentionWrapper(
            float_workspace_buffer, backend="fa3"
        )
        sparse_wrapper_fa3.plan(
            block_mask_map=block_mask_map,
            block_row_sz=block_row_sz,
            block_col_sz=block_col_sz,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_data_type=torch.half,
        )
        measurements_fa3 = bench_gpu_time(
            lambda: sparse_wrapper_fa3.run(q, k, v),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
        )
        sparse_ms_fa3 = np.median(measurements_fa3)
    except Exception as e:
        sparse_ms_fa3 = float("nan")
        print(f"  [sparse FA3 skipped: {e}]")

    # Benchmark dense attention with CUTLASS (Blackwell-native SM100a)
    q_dense = torch.randn(seq_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda")
    k_dense = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")
    v_dense = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")

    qo_segment_offsets = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)
    kv_segment_offsets = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)

    try:
        dense_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.half, device="cuda"),
            kv_layout="NHD",
            backend="cutlass",
        )
        dense_wrapper.plan(
            qo_segment_offsets,
            kv_segment_offsets,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            causal=False,
            q_data_type=torch.half,
            kv_data_type=torch.half,
        )
        dense_wrapper.run(q_dense, k_dense, v_dense)
        dense_cutlass_ms = np.median(
            bench_gpu_time(
                lambda: dense_wrapper.run(q_dense, k_dense, v_dense),
                dry_run_time_ms=100,
                repeat_time_ms=1000,
            )
        )
    except Exception as e:
        dense_cutlass_ms = float("nan")
        print(f"  [dense CUTLASS skipped: {e}]")

    def flops(ms):
        return attention_tflops_per_sec_with_actual_seq_lens(
            torch.tensor([seq_len]),
            torch.tensor([seq_len]),
            head_dim,
            head_dim,
            num_qo_heads,
            False,
            ms,
        )

    print(
        f"bench_fix_block_sparse_attention (num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}, seq_len={seq_len}, R={R}, C={C}, block_density={block_density}), sparse cutile: {flops(sparse_ms_cutile):.3f} TFLOPs/s, sparse fa2-template: {flops(sparse_ms_fa2):.3f} TFLOPs/s, sparse fa3-template: {flops(sparse_ms_fa3):.3f} TFLOPs/s, dense cutlass: {flops(dense_cutlass_ms):.3f} TFLOPs/s"
    )


if __name__ == "__main__":
    for num_qo_heads in [32]:
        for num_kv_heads in [32]:
            for head_dim in [128]:
                for seq_len in [8192, 16384, 32768]:
                    for R, C in [(64, 64), (128, 128)]:
                        for block_density in [0.1, 0.3, 0.5, 0.7, 0.9]:
                            bench_bsr_block_sparse_attention(
                                num_qo_heads,
                                num_kv_heads,
                                head_dim,
                                seq_len,
                                R,
                                C,
                                block_density,
                            )
