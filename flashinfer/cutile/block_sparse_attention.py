# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Block Sparse Attention kernel using cuTile (cuda.tile).

Implements FlashAttention-style tiled attention with BSR (Block Sparse Row)
masking. Only non-zero blocks in the sparse pattern are computed, skipping
empty blocks entirely for significant speedup on sparse attention patterns.

Supports variable block sizes: each KV block can have a different effective
column count via the `variable_block_sizes` tensor, following the approach
used in the CuTe DSL FlashAttention SM100 kernel.

BSR format:
  indptr:  (MB + 1,)  row-block pointers     MB = ceil(M / R)
  indices: (nnz,)     column-block indices    NB = ceil(N / C)

Inputs (BHMD layout):
  Q: (B, H_q, M, D)    query
  K: (B, H_kv, N, D)   key
  V: (B, H_kv, N, D)   value

Inputs (NHD layout):
  Q: (M, H_q, D)       query
  K: (N, H_kv, D)      key
  V: (N, H_kv, D)      value

Common:
  indptr:  (MB + 1,)    BSR row pointers   (shared across batch/head)
  indices: (nnz,)       BSR column indices

Output:
  O: same layout as Q
"""

import math
from types import SimpleNamespace
from typing import Optional

import cuda.tile as ct
import torch

INV_LOG_2 = 1.0 / math.log(2)

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


# ---------------------------------------------------------------------------
# Core block-sparse attention kernel
# ---------------------------------------------------------------------------
@ct.kernel(occupancy=2)
def block_sparse_attn_kernel(
    Q,
    K,
    V,
    Out,
    indptr,   # BSR row pointers   (MB + 1,)
    indices,  # BSR column indices  (nnz,)
    var_block_sizes,  # per-block effective column count (NB,) or dummy
    qk_scale: float,
    TILE_D: ConstInt,
    H_Q: ConstInt,
    H_KV: ConstInt,
    TILE_M: ConstInt,   # R  (block row size, must equal Q tile size)
    TILE_N: ConstInt,   # C  (block col size, must equal KV tile size)
    QUERY_GROUP_SIZE: ConstInt,
    HAS_VARIABLE_BLOCK_SIZES: ConstBool,
    CAUSAL_WITHIN_BLOCK: ConstBool,
    LAYOUT_NHD: ConstBool,
):
    """
    Block-sparse FlashAttention forward using cuTile.

    Grid:  (num_q_blocks, batch_size * H_Q, 1)  for BHMD
           (num_q_blocks, H_Q, 1)                for NHD
    Each CTA processes one Q-tile of TILE_M rows and iterates only over
    the non-zero KV blocks given by the BSR pattern.
    """
    bid_m = ct.bid(0)      # Q block index
    bid_bh = ct.bid(1)     # batch * H_Q (BHMD) or H_Q (NHD)
    batch_idx = bid_bh // H_Q
    head_idx = bid_bh % H_Q
    off_kv_h = head_idx // QUERY_GROUP_SIZE

    qk_scale_log2 = qk_scale * INV_LOG_2

    # ---- Load Q tile ----
    if LAYOUT_NHD:
        q = ct.load(
            Q,
            index=(bid_m, head_idx, 0),
            shape=(TILE_M, 1, TILE_D),
        ).reshape((TILE_M, TILE_D))
    else:
        q = ct.load(
            Q,
            index=(batch_idx, head_idx, bid_m, 0),
            shape=(1, 1, TILE_M, TILE_D),
        ).reshape((TILE_M, TILE_D))

    # ---- Online softmax accumulators ----
    m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)   # running row max
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)         # running row sum
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)    # output accumulator

    # ---- Iterate over non-zero KV blocks for this Q block ----
    block_start_idx = ct.gather(indptr, ct.arange(1, dtype=ct.int32) * 0 + bid_m)
    block_end_idx = ct.gather(indptr, ct.arange(1, dtype=ct.int32) * 0 + bid_m + 1)
    block_start = block_start_idx.item()
    block_end = block_end_idx.item()
    num_blocks = block_end - block_start

    # Q-tile row offsets (for causal masking)
    offs_m = bid_m * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    offs_m = offs_m[:, None]  # (TILE_M, 1)

    # Column offsets within a tile (for variable block size masking)
    col_offs = ct.arange(TILE_N, dtype=ct.int32)  # (TILE_N,)

    for blk_i in range(num_blocks):
        # Column block index from BSR
        col_block_scalar = ct.gather(indices, ct.arange(1, dtype=ct.int32) * 0 + block_start + blk_i)
        col_block = col_block_scalar.item()

        # ---- Load K tile: (TILE_D, TILE_N) via load + permute ----
        # NOTE: ct.load with order= and dynamic index is broken in cuTile 1.2.0
        if LAYOUT_NHD:
            k = ct.load(
                K,
                index=(col_block, off_kv_h, 0),
                shape=(TILE_N, 1, TILE_D),
                latency=2,
            ).reshape((TILE_N, TILE_D)).permute((1, 0))
        else:
            k = ct.load(
                K,
                index=(batch_idx, off_kv_h, col_block, 0),
                shape=(1, 1, TILE_N, TILE_D),
                latency=2,
            ).reshape((TILE_N, TILE_D)).permute((1, 0))

        # ---- QK = Q @ K^T : (TILE_M, TILE_N) ----
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        # ---- Optional causal masking within block ----
        if CAUSAL_WITHIN_BLOCK:
            offs_n = col_block * TILE_N + ct.arange(TILE_N, dtype=ct.int32)
            offs_n = offs_n[None, :]  # (1, TILE_N)
            causal_mask = offs_m >= offs_n  # (TILE_M, TILE_N)
            qk = ct.where(causal_mask, qk, -math.inf)

        # ---- Variable block size masking ----
        if HAS_VARIABLE_BLOCK_SIZES:
            var_col_limit_tile = ct.gather(
                var_block_sizes,
                ct.arange(1, dtype=ct.int32) * 0 + col_block,
            )
            var_col_limit = var_col_limit_tile.item()
            col_mask = col_offs[None, :] < var_col_limit  # (1, TILE_N) broadcast
            qk = ct.where(col_mask, qk, -math.inf)

        # ---- Online softmax ----
        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale_log2)
        qk = qk * qk_scale_log2 - m_ij

        p = ct.exp2(qk, flush_to_zero=True)           # (TILE_M, TILE_N)
        l_ij = ct.sum(p, axis=-1, keepdims=True)       # (TILE_M, 1)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)  # (TILE_M, 1)

        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        # ---- Load V tile: (TILE_N, TILE_D) ----
        if LAYOUT_NHD:
            v = ct.load(
                V,
                index=(col_block, off_kv_h, 0),
                shape=(TILE_N, 1, TILE_D),
                latency=4,
            ).reshape((TILE_N, TILE_D))
        else:
            v = ct.load(
                V,
                index=(batch_idx, off_kv_h, col_block, 0),
                shape=(1, 1, TILE_N, TILE_D),
                latency=4,
            ).reshape((TILE_N, TILE_D))

        # ---- Accumulate P @ V ----
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)

        m_i = m_ij

    # ---- Final normalization: O = acc / l_i ----
    # Guard against div-by-zero when a Q block has no KV blocks (l_i stays 0).
    l_i = ct.where(l_i == 0.0, ct.full((TILE_M, 1), 1.0, dtype=ct.float32), l_i)
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=ct.RoundingMode.APPROX)
    if LAYOUT_NHD:
        acc = acc.reshape((TILE_M, 1, TILE_D)).astype(Out.dtype)
        ct.store(Out, index=(bid_m, head_idx, 0), tile=acc)
    else:
        acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
        ct.store(Out, index=(batch_idx, head_idx, bid_m, 0), tile=acc)


# ---------------------------------------------------------------------------
# Tile configs for autotuning
# ---------------------------------------------------------------------------
_BSA_TILE_CONFIGS_BY_D = {
    64:  ([64, 128], [64, 128]),
    128: ([64, 128], [64, 128]),
    256: ([64, 128], [64]),
}


def _iter_bsa_configs(head_dim: int):
    key = head_dim
    if key not in _BSA_TILE_CONFIGS_BY_D:
        key = 128
    tile_ms, tile_ns = _BSA_TILE_CONFIGS_BY_D[key]
    for tm in tile_ms:
        for tn in tile_ns:
            yield SimpleNamespace(TILE_M=tm, TILE_N=tn)


# ---------------------------------------------------------------------------
# Public API: block_sparse_attention
# ---------------------------------------------------------------------------
def block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    R: int,
    C: int,
    sm_scale: float | None = None,
    causal: bool = False,
    variable_block_sizes: Optional[torch.Tensor] = None,
    layout: str = "BHMD",
) -> torch.Tensor:
    """
    Block-sparse attention using BSR mask format.

    Args:
        q: query tensor — (B, H_q, M, D) for BHMD or (M, H_q, D) for NHD
        k: key tensor   — (B, H_kv, N, D) for BHMD or (N, H_kv, D) for NHD
        v: value tensor — (B, H_kv, N, D) for BHMD or (N, H_kv, D) for NHD
        indptr:  (MB+1,) int32  BSR row pointers
        indices: (nnz,)  int32  BSR column indices
        R: block row size (must match TILE_M)
        C: block column size (must match TILE_N)
        sm_scale: softmax scale (default 1/sqrt(D))
        causal: apply causal masking within blocks
        variable_block_sizes: (NB,) int32 tensor specifying the effective
            column count for each KV block. Values must be in [1, C].
            When provided, columns beyond variable_block_sizes[j] in
            KV block j are masked out (set to -inf before softmax).
            This enables non-uniform block widths within a fixed tile grid.
        layout: tensor layout — "BHMD" (default) or "NHD"

    Returns:
        out: same layout as input q
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    indptr = indptr.contiguous().to(torch.int32)
    indices = indices.contiguous().to(torch.int32)

    use_nhd = layout.upper() == "NHD"

    if use_nhd:
        M, H_q, D = q.shape
        _, H_kv, _ = k.shape
        B = 1
    else:
        B, H_q, M, D = q.shape
        _, H_kv, N, _ = k.shape

    assert H_q % H_kv == 0
    query_group_size = H_q // H_kv

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    MB = math.ceil(M / R)
    assert indptr.shape[0] == MB + 1, (
        f"indptr length {indptr.shape[0]} != MB+1={MB + 1}"
    )

    has_var_blocks = variable_block_sizes is not None
    if has_var_blocks:
        variable_block_sizes = variable_block_sizes.contiguous().to(torch.int32)
    else:
        variable_block_sizes = torch.empty(1, dtype=torch.int32, device=q.device)

    out = torch.empty_like(q)

    TILE_M = R
    TILE_N = C

    grid = (MB, B * H_q, 1)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        block_sparse_attn_kernel,
        (
            q, k, v, out,
            indptr, indices,
            variable_block_sizes,
            sm_scale,
            D,           # TILE_D
            H_q,
            H_kv,
            TILE_M,
            TILE_N,
            query_group_size,
            has_var_blocks,      # HAS_VARIABLE_BLOCK_SIZES
            causal,              # CAUSAL_WITHIN_BLOCK
            use_nhd,             # LAYOUT_NHD
        ),
    )
    return out
