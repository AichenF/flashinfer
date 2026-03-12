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

import math

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import (
    attention_tflops_per_sec_with_actual_seq_lens,
    bench_gpu_time,
)


def _make_bsr_mask(MB, NB, density, device="cuda"):
    """Create a random block mask and return BSR (indptr, indices)."""
    block_mask = torch.rand(MB, NB, device=device) < density
    for i in range(MB):
        if not block_mask[i].any():
            block_mask[i, torch.randint(NB, (1,))] = True

    indptr = [0]
    indices_list = []
    for i in range(MB):
        for j in range(NB):
            if block_mask[i, j]:
                indices_list.append(j)
        indptr.append(len(indices_list))
    indptr_t = torch.tensor(indptr, dtype=torch.int32, device=device)
    indices_t = torch.tensor(indices_list, dtype=torch.int32, device=device)
    return block_mask, indptr_t, indices_t


def _flops(seq_len, head_dim, num_qo_heads, ms):
    return attention_tflops_per_sec_with_actual_seq_lens(
        torch.tensor([seq_len]),
        torch.tensor([seq_len]),
        head_dim,
        head_dim,
        num_qo_heads,
        False,
        ms,
    )


def bench_block_sparse_attention(
    num_qo_heads,
    num_kv_heads,
    head_dim,
    seq_len,
    R,
    C,
    block_density,
):
    """
    Benchmark block-sparse attention: cuTile vs FA2 (vs FA3 if available).

    Uses tile-aligned block sizes R, C so all backends share the same
    block partition and sparsity pattern.
    """
    if num_qo_heads % num_kv_heads != 0:
        return
    MB = seq_len // R
    NB = seq_len // C
    if MB < 1 or NB < 1:
        return

    device = "cuda"
    block_mask, indptr, indices = _make_bsr_mask(MB, NB, block_density, device)
    nnz = indices.shape[0]

    # ---- cuTile kernel: Q/K/V are (B, H, M, D) with B=1 ----
    q_cutile = torch.randn(
        1, num_qo_heads, seq_len, head_dim, dtype=torch.half, device=device
    )
    k_cutile = torch.randn(
        1, num_kv_heads, seq_len, head_dim, dtype=torch.half, device=device
    )
    v_cutile = torch.randn(
        1, num_kv_heads, seq_len, head_dim, dtype=torch.half, device=device
    )

    cutile_ok = True
    try:
        from flashinfer.cutile import block_sparse_attention

        # warmup
        block_sparse_attention(
            q_cutile, k_cutile, v_cutile, indptr, indices, R=R, C=C
        )
        measurements_cutile = bench_gpu_time(
            lambda: block_sparse_attention(
                q_cutile, k_cutile, v_cutile, indptr, indices, R=R, C=C
            ),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
        )
        cutile_ms = np.median(measurements_cutile)
    except Exception as e:
        cutile_ok = False
        cutile_ms = float("nan")
        print(f"  [cuTile skipped: {e}]")

    # ---- FA2 / FA3 via VariableBlockSparseAttentionWrapper ----
    # These use (H, M, D) layout and per-head block masks
    q_sparse = torch.randn(
        num_qo_heads, seq_len, head_dim, dtype=torch.half, device=device
    )
    k_sparse = torch.randn(
        num_kv_heads, seq_len, head_dim, dtype=torch.half, device=device
    )
    v_sparse = torch.randn(
        num_kv_heads, seq_len, head_dim, dtype=torch.half, device=device
    )

    block_row_sz = torch.full(
        (num_kv_heads, MB), R, dtype=torch.int32
    )
    block_col_sz = torch.full(
        (num_kv_heads, NB), C, dtype=torch.int32
    )
    block_mask_map = block_mask.cpu().unsqueeze(0).expand(num_kv_heads, -1, -1).clone()

    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device=device
    )

    # FA2
    fa2_ok = True
    try:
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
            lambda: sparse_wrapper_fa2.run(q_sparse, k_sparse, v_sparse),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
        )
        fa2_ms = np.median(measurements_fa2)
    except Exception as e:
        fa2_ok = False
        fa2_ms = float("nan")
        print(f"  [FA2 skipped: {e}]")

    # FA3
    fa3_ok = True
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
            lambda: sparse_wrapper_fa3.run(q_sparse, k_sparse, v_sparse),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
        )
        fa3_ms = np.median(measurements_fa3)
    except Exception as e:
        fa3_ok = False
        fa3_ms = float("nan")
        print(f"  [FA3 skipped: {e}]")

    # ---- Dense baseline: CUTLASS SM100a (Blackwell-native) ----
    q_dense = torch.randn(
        seq_len, num_qo_heads, head_dim, dtype=torch.half, device=device
    )
    k_dense = torch.randn(
        seq_len, num_kv_heads, head_dim, dtype=torch.half, device=device
    )
    v_dense = torch.randn(
        seq_len, num_kv_heads, head_dim, dtype=torch.half, device=device
    )

    qo_segment_offsets = torch.tensor(
        [0, seq_len], device=device, dtype=torch.int32
    )
    kv_segment_offsets = torch.tensor(
        [0, seq_len], device=device, dtype=torch.int32
    )

    dense_cutlass_ok = True
    try:
        dense_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.half, device=device),
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
        measurements_dense = bench_gpu_time(
            lambda: dense_wrapper.run(q_dense, k_dense, v_dense),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
        )
        dense_cutlass_ms = np.median(measurements_dense)
    except Exception as e:
        dense_cutlass_ok = False
        dense_cutlass_ms = float("nan")
        print(f"  [dense CUTLASS skipped: {e}]")

    # ---- Print results ----
    parts = []
    parts.append(
        f"seq_len={seq_len}, R={R}, C={C}, density={block_density:.0%}, "
        f"nnz={nnz}/{MB * NB}"
    )

    def fmt(label, ms, ok):
        if ok:
            tflops = _flops(seq_len, head_dim, num_qo_heads, ms)
            return f"{label}: {ms:.3f}ms ({tflops:.1f} TFLOPs/s)"
        return f"{label}: N/A"

    parts.append(fmt("cuTile", cutile_ms, cutile_ok))
    parts.append(fmt("sparse-FA2", fa2_ms, fa2_ok))
    parts.append(fmt("sparse-FA3", fa3_ms, fa3_ok))
    parts.append(fmt("dense-CUTLASS", dense_cutlass_ms, dense_cutlass_ok))

    print(
        f"[H_q={num_qo_heads}, H_kv={num_kv_heads}, D={head_dim}] "
        + " | ".join(parts)
    )


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"{'=' * 120}")
    for num_qo_heads in [32]:
        for num_kv_heads in [32]:
            for head_dim in [128]:
                for seq_len in [8192, 16384, 32768]:
                    for R, C in [(64, 64), (128, 128)]:
                        for block_density in [0.1, 0.3, 0.5, 0.7, 0.9]:
                            bench_block_sparse_attention(
                                num_qo_heads,
                                num_kv_heads,
                                head_dim,
                                seq_len,
                                R,
                                C,
                                block_density,
                            )
                    print()
