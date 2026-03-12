# SPDX-License-Identifier: Apache-2.0
"""
Tests for the cuTile block-sparse attention kernel.

Compares against a PyTorch reference implementation that applies
a dense boolean mask to standard scaled dot-product attention.
"""

import math

import torch
import torch.nn.functional as F


def _ref_attention(q, k, v, mask, sm_scale, causal=False):
    """
    Reference attention with dense boolean block mask.

    Args:
        q: (B, H_q, M, D)
        k: (B, H_kv, N, D)
        v: (B, H_kv, N, D)
        mask: (M, N) bool — True = attend
        sm_scale: softmax scale
        causal: additional causal mask
    """
    B, H_q, M, D = q.shape
    _, H_kv, N, _ = k.shape
    group_size = H_q // H_kv

    # Expand KV heads for GQA
    if group_size > 1:
        k = k.repeat_interleave(group_size, dim=1)
        v = v.repeat_interleave(group_size, dim=1)

    # QK^T
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale

    # Apply block mask
    attn_mask = mask.unsqueeze(0).unsqueeze(0).float()  # (1, 1, M, N)
    scores = scores + (1 - attn_mask) * (-1e9)

    # Apply causal mask
    if causal:
        causal_mask = torch.tril(torch.ones(M, N, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e9)

    p = F.softmax(scores, dim=-1)
    out = torch.matmul(p, v.float()).to(q.dtype)
    return out


def _make_block_sparse_mask(MB, NB, R, C, density=0.5, device="cuda"):
    """Create a random block-sparse mask with given density."""
    block_mask = torch.rand(MB, NB, device=device) < density
    # Ensure at least one block per row
    for i in range(MB):
        if not block_mask[i].any():
            block_mask[i, torch.randint(NB, (1,))] = True

    # Expand to dense mask
    M = MB * R
    N = NB * C
    dense = torch.zeros(M, N, dtype=torch.bool, device=device)
    for i in range(MB):
        for j in range(NB):
            if block_mask[i, j]:
                dense[i * R : (i + 1) * R, j * C : (j + 1) * C] = True
    return dense, block_mask


def _dense_to_bsr(block_mask, R, C, device="cuda"):
    """Convert block-level bool mask (MB, NB) → BSR indptr, indices."""
    MB, NB = block_mask.shape
    indptr = [0]
    indices_list = []
    for i in range(MB):
        for j in range(NB):
            if block_mask[i, j]:
                indices_list.append(j)
        indptr.append(len(indices_list))
    indptr = torch.tensor(indptr, dtype=torch.int32, device=device)
    indices = torch.tensor(indices_list, dtype=torch.int32, device=device)
    return indptr, indices


def test_block_sparse_vs_dense(
    B=2, H_q=4, H_kv=4, M=256, N=256, D=128, R=64, C=64, density=0.5
):
    """Test block-sparse attention matches dense reference."""
    from flashinfer.cutile import block_sparse_attention

    device = "cuda"
    torch.manual_seed(42)

    MB = M // R
    NB = N // C

    q = torch.randn(B, H_q, M, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float16)
    sm_scale = 1.0 / math.sqrt(D)

    dense_mask, block_mask = _make_block_sparse_mask(MB, NB, R, C, density, device)
    indptr, indices = _dense_to_bsr(block_mask, R, C, device)

    # cuTile block-sparse attention
    out_sparse = block_sparse_attention(
        q, k, v, indptr, indices, R=R, C=C, sm_scale=sm_scale, causal=False
    )

    # Reference dense attention with mask
    out_ref = _ref_attention(q, k, v, dense_mask, sm_scale, causal=False)

    # Compare
    torch.testing.assert_close(out_sparse, out_ref, atol=1e-2, rtol=1e-2)
    print(f"PASSED: block_sparse_vs_dense  B={B} H={H_q} M={M} N={N} D={D} "
          f"R={R} C={C} density={density}")


def test_block_sparse_causal(
    B=1, H_q=2, H_kv=2, M=128, N=128, D=64, R=64, C=64
):
    """Test block-sparse with causal masking."""
    from flashinfer.cutile import block_sparse_attention

    device = "cuda"
    torch.manual_seed(0)

    MB = M // R
    NB = N // C

    q = torch.randn(B, H_q, M, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float16)
    sm_scale = 1.0 / math.sqrt(D)

    # Full mask (all blocks present) + causal
    dense_mask = torch.ones(M, N, dtype=torch.bool, device=device)
    block_mask = torch.ones(MB, NB, dtype=torch.bool, device=device)
    indptr, indices = _dense_to_bsr(block_mask, R, C, device)

    out_sparse = block_sparse_attention(
        q, k, v, indptr, indices, R=R, C=C, sm_scale=sm_scale, causal=True
    )
    out_ref = _ref_attention(q, k, v, dense_mask, sm_scale, causal=True)

    torch.testing.assert_close(out_sparse, out_ref, atol=1e-2, rtol=1e-2)
    print(f"PASSED: block_sparse_causal  B={B} H={H_q} M={M} N={N} D={D}")


def test_block_sparse_gqa(
    B=1, H_q=8, H_kv=2, M=128, N=128, D=64, R=64, C=64, density=0.7
):
    """Test block-sparse with grouped query attention."""
    from flashinfer.cutile import block_sparse_attention

    device = "cuda"
    torch.manual_seed(7)

    MB = M // R
    NB = N // C

    q = torch.randn(B, H_q, M, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float16)
    sm_scale = 1.0 / math.sqrt(D)

    dense_mask, block_mask = _make_block_sparse_mask(MB, NB, R, C, density, device)
    indptr, indices = _dense_to_bsr(block_mask, R, C, device)

    out_sparse = block_sparse_attention(
        q, k, v, indptr, indices, R=R, C=C, sm_scale=sm_scale
    )
    out_ref = _ref_attention(q, k, v, dense_mask, sm_scale)

    torch.testing.assert_close(out_sparse, out_ref, atol=1e-2, rtol=1e-2)
    print(f"PASSED: block_sparse_gqa  B={B} H_q={H_q} H_kv={H_kv} M={M} N={N}")


def test_high_sparsity(
    B=1, H_q=4, H_kv=4, M=512, N=512, D=128, R=128, C=128, density=0.1
):
    """Test with high sparsity (only 10% of blocks non-zero)."""
    from flashinfer.cutile import block_sparse_attention

    device = "cuda"
    torch.manual_seed(99)

    MB = M // R
    NB = N // C

    q = torch.randn(B, H_q, M, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float16)
    sm_scale = 1.0 / math.sqrt(D)

    dense_mask, block_mask = _make_block_sparse_mask(MB, NB, R, C, density, device)
    indptr, indices = _dense_to_bsr(block_mask, R, C, device)

    nnz = indices.shape[0]
    total = MB * NB
    print(f"  sparsity: nnz={nnz}/{total} = {nnz / total:.1%}")

    out_sparse = block_sparse_attention(
        q, k, v, indptr, indices, R=R, C=C, sm_scale=sm_scale
    )
    out_ref = _ref_attention(q, k, v, dense_mask, sm_scale)

    torch.testing.assert_close(out_sparse, out_ref, atol=1e-2, rtol=1e-2)
    print(f"PASSED: high_sparsity  M={M} N={N} R={R} density={density}")


def _ref_attention_variable_blocks(q, k, v, indptr, indices, variable_block_sizes, R, C, sm_scale):
    """
    Reference attention for variable block sizes.

    Builds a dense mask where block (i, j) is active if it appears in the BSR
    structure, but only the first variable_block_sizes[j] columns within each
    block are unmasked.
    """
    B, H_q, M, D = q.shape
    _, H_kv, N, _ = k.shape
    group_size = H_q // H_kv

    MB = M // R
    NB = N // C

    dense_mask = torch.zeros(M, N, dtype=torch.bool, device=q.device)
    for i in range(MB):
        start = indptr[i].item()
        end = indptr[i + 1].item()
        for idx in range(start, end):
            j = indices[idx].item()
            col_limit = variable_block_sizes[j].item()
            dense_mask[i * R : (i + 1) * R, j * C : j * C + col_limit] = True

    return _ref_attention(q, k, v, dense_mask, sm_scale, causal=False)


def test_variable_block_sizes(
    B=2, H_q=4, H_kv=4, M=256, N=256, D=128, R=64, C=64, density=0.5
):
    """Test block-sparse attention with variable block sizes."""
    from flashinfer.cutile import block_sparse_attention

    device = "cuda"
    torch.manual_seed(123)

    MB = M // R
    NB = N // C

    q = torch.randn(B, H_q, M, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float16)
    sm_scale = 1.0 / math.sqrt(D)

    _, block_mask = _make_block_sparse_mask(MB, NB, R, C, density, device)
    indptr, indices = _dense_to_bsr(block_mask, R, C, device)

    # Random effective column sizes in [1, C] for each KV block
    variable_block_sizes = torch.randint(1, C + 1, (NB,), dtype=torch.int32, device=device)

    out_sparse = block_sparse_attention(
        q, k, v, indptr, indices, R=R, C=C, sm_scale=sm_scale,
        variable_block_sizes=variable_block_sizes,
    )

    out_ref = _ref_attention_variable_blocks(
        q, k, v, indptr, indices, variable_block_sizes, R, C, sm_scale,
    )

    torch.testing.assert_close(out_sparse, out_ref, atol=1e-2, rtol=1e-2)
    print(f"PASSED: variable_block_sizes  B={B} H={H_q} M={M} N={N} D={D} "
          f"R={R} C={C} density={density}")


def test_variable_block_sizes_full(
    B=1, H_q=2, H_kv=2, M=128, N=256, D=64, R=64, C=64
):
    """Test variable block sizes with all blocks present (full mask)."""
    from flashinfer.cutile import block_sparse_attention

    device = "cuda"
    torch.manual_seed(456)

    MB = M // R
    NB = N // C

    q = torch.randn(B, H_q, M, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float16)
    sm_scale = 1.0 / math.sqrt(D)

    block_mask = torch.ones(MB, NB, dtype=torch.bool, device=device)
    indptr, indices = _dense_to_bsr(block_mask, R, C, device)

    # Mix of full tiles (C) and partial tiles
    variable_block_sizes = torch.tensor(
        [C, C // 2, C, C // 4], dtype=torch.int32, device=device
    )[:NB]

    out_sparse = block_sparse_attention(
        q, k, v, indptr, indices, R=R, C=C, sm_scale=sm_scale,
        variable_block_sizes=variable_block_sizes,
    )

    out_ref = _ref_attention_variable_blocks(
        q, k, v, indptr, indices, variable_block_sizes, R, C, sm_scale,
    )

    torch.testing.assert_close(out_sparse, out_ref, atol=1e-2, rtol=1e-2)
    print(f"PASSED: variable_block_sizes_full  B={B} H={H_q} M={M} N={N} D={D}")


def test_variable_block_sizes_all_full(
    B=1, H_q=4, H_kv=4, M=128, N=128, D=128, R=64, C=64, density=0.6
):
    """When all variable_block_sizes == C, result should match fixed blocks."""
    from flashinfer.cutile import block_sparse_attention

    device = "cuda"
    torch.manual_seed(789)

    MB = M // R
    NB = N // C

    q = torch.randn(B, H_q, M, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float16)
    sm_scale = 1.0 / math.sqrt(D)

    dense_mask, block_mask = _make_block_sparse_mask(MB, NB, R, C, density, device)
    indptr, indices = _dense_to_bsr(block_mask, R, C, device)

    # All blocks have full column width → same as no variable_block_sizes
    variable_block_sizes = torch.full((NB,), C, dtype=torch.int32, device=device)

    out_var = block_sparse_attention(
        q, k, v, indptr, indices, R=R, C=C, sm_scale=sm_scale,
        variable_block_sizes=variable_block_sizes,
    )
    out_fixed = block_sparse_attention(
        q, k, v, indptr, indices, R=R, C=C, sm_scale=sm_scale,
    )

    torch.testing.assert_close(out_var, out_fixed, atol=1e-2, rtol=1e-2)
    print(f"PASSED: variable_block_sizes_all_full (matches fixed)  M={M} N={N}")


if __name__ == "__main__":
    test_block_sparse_vs_dense()
    test_block_sparse_causal()
    test_block_sparse_gqa()
    test_high_sparsity()
    test_variable_block_sizes()
    test_variable_block_sizes_full()
    test_variable_block_sizes_all_full()
    print("\nAll tests passed!")
