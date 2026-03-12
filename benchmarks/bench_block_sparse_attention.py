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


def bench_variable_block_sparse_attention(
    num_qo_heads,
    num_kv_heads,
    head_dim,
    seq_len,
    num_blocks_row,
    num_blocks_col,
    block_density,
):
    if num_qo_heads % num_kv_heads != 0:
        return
    if seq_len // num_blocks_row < 1:
        return
    if seq_len // num_blocks_col < 1:
        return

    # synthesize uniform block sz
    block_row_sz = torch.ones(num_blocks_row, dtype=torch.int32) * (
        seq_len // num_blocks_row
    )
    block_row_sz[-1] = seq_len - (seq_len // num_blocks_row) * (num_blocks_row - 1)
    block_row_sz = block_row_sz.unsqueeze(0).repeat(num_kv_heads, 1)

    block_col_sz = torch.ones(num_blocks_col, dtype=torch.int32) * (
        seq_len // num_blocks_col
    )
    block_col_sz[-1] = seq_len - (seq_len // num_blocks_col) * (num_blocks_col - 1)
    block_col_sz = block_col_sz.unsqueeze(0).repeat(num_kv_heads, 1)

    block_mask_map = (
        torch.rand(num_kv_heads, num_blocks_row, num_blocks_col) < block_density
    )

    q = torch.randn(num_qo_heads, seq_len, head_dim, dtype=torch.half, device="cuda")
    k = torch.randn(num_kv_heads, seq_len, head_dim, dtype=torch.half, device="cuda")
    v = torch.randn(num_kv_heads, seq_len, head_dim, dtype=torch.half, device="cuda")

    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"
    )
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

    # Benchmark sparse attention with FA2
    measurements_fa2 = bench_gpu_time(
        lambda: sparse_wrapper_fa2.run(q, k, v),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
    sparse_ms_fa2 = np.median(measurements_fa2)

    # Benchmark sparse attention with FA3 (may fail on Blackwell/SM100)
    sparse_ms_fa3 = float("nan")
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
        print(f"  [FA3 sparse skipped: {e}]")

    # Benchmark sparse attention with cuTile (Blackwell-native, tile-aligned R=C=128)
    R_ct, C_ct = 128, 128
    MB_ct, NB_ct = seq_len // R_ct, seq_len // C_ct
    sparse_ms_cutile = float("nan")
    if MB_ct > 0 and NB_ct > 0:
        try:
            bm = torch.rand(MB_ct, NB_ct) < block_density
            for i in range(MB_ct):
                if not bm[i].any():
                    bm[i, torch.randint(NB_ct, (1,))] = True
            indptr_l, indices_l = [0], []
            for i in range(MB_ct):
                for j in range(NB_ct):
                    if bm[i, j]:
                        indices_l.append(j)
                indptr_l.append(len(indices_l))
            indptr_ct = torch.tensor(indptr_l, dtype=torch.int32, device="cuda")
            indices_ct = torch.tensor(indices_l, dtype=torch.int32, device="cuda")
            q_ct = torch.randn(seq_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda")
            k_ct = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")
            v_ct = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")
            ct_wrapper = flashinfer.BlockSparseAttentionWrapper(
                float_workspace_buffer, backend="cutile"
            )
            ct_wrapper.plan(
                indptr_ct, indices_ct, seq_len, seq_len,
                R_ct, C_ct, num_qo_heads, num_kv_heads, head_dim,
                q_data_type=torch.half,
            )
            sparse_ms_cutile = np.median(bench_gpu_time(
                lambda: ct_wrapper.run(q_ct, k_ct, v_ct),
                dry_run_time_ms=100, repeat_time_ms=1000,
            ))
        except Exception as e:
            print(f"  [cuTile skipped: {e}]")

    q = torch.randn(seq_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda")
    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")
    v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")
    dense_sm80_ms = np.median(
        bench_gpu_time(
            lambda: flashinfer.single_prefill_with_kv_cache_return_lse(
                q, k, v, causal=False, backend="fa2"
            ),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
        )
    )
    dense_sm90_ms = float("nan")
    try:
        dense_sm90_ms = np.median(
            bench_gpu_time(
                lambda: flashinfer.single_prefill_with_kv_cache_return_lse(
                    q, k, v, causal=False, backend="fa3"
                ),
                dry_run_time_ms=100,
                repeat_time_ms=1000,
            )
        )
    except Exception as e:
        print(f"  [FA3 dense skipped: {e}]")

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
        f"bench_variable_block_sparse_attention (num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}, seq_len={seq_len}, num_blocks_row={num_blocks_row}, num_blocks_col={num_blocks_col}, block_density={block_density}), sparse cutile(R={R_ct}): {flops(sparse_ms_cutile):.3f} TFLOPs/s, sparse fa2-template: {flops(sparse_ms_fa2):.3f} TFLOPs/s, sparse fa3-template: {flops(sparse_ms_fa3):.3f} TFLOPs/s, dense fa2-template: {flops(dense_sm80_ms):.3f} TFLOPs/s, dense fa3-template: {flops(dense_sm90_ms):.3f} TFLOPs/s"
    )


if __name__ == "__main__":
    for num_qo_heads in [32]:
        for num_kv_heads in [32]:
            for head_dim in [128]:
                for seq_len in [8192, 16384, 32768]:
                    for num_blocks_row in [20]:
                        for num_blocks_col in [50]:
                            for block_density in [0.1, 0.3, 0.5, 0.7, 0.9]:
                                bench_variable_block_sparse_attention(
                                    num_qo_heads,
                                    num_kv_heads,
                                    head_dim,
                                    seq_len,
                                    num_blocks_row,
                                    num_blocks_col,
                                    block_density,
                                )