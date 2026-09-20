"""SM90 (Hopper) NVFP4 MegaMoE.

A single persistent kernel that fuses the cross-rank token dispatch, both FFN
GEMMs (W4A8: NVFP4 weights, FP8 activations), SwiGLU and the combine reduction
over an 8-rank NVLink symmetric buffer -- replacing an all-to-all + grouped
GEMM + all-to-all pipeline with one launch.

Usage::

    buf = MegaMoESymmBuffer(group, num_experts=256, num_max_tokens_per_rank=8448,
                            num_topk=6, hidden=4096, intermediate_hidden=2048)
    buf.stage(x_fp8, x_sf, topk_idx, topk_weights)
    nvfp4_mega_moe(y, l1_weights, l2_weights, buf)
"""

import dataclasses
import functools
from typing import Optional, Tuple

import torch

from ..jit.megamoe import (
    gen_megamoe_sm90_nvfp4_module,
    gen_megamoe_sm90_nvfp4_split_module,
    gen_megamoe_sm90_nvfp4_split_plan_module,
)
from .heuristics import (
    CANDIDATE_BLOCK_M,
    FusedPlan,
    get_num_max_pool_tokens,
    get_num_padded_sf_pool_tokens,
    select_fused_plan,
)

__all__ = [
    "MegaMoESymmBuffer",
    "nvfp4_mega_moe",
    "select_fused_plan",
    "FusedPlan",
    "SplitPlan",
    "select_split_plan",
    "FAMILY_THRESHOLD",
]


@functools.cache
def _get_module(
    plan: FusedPlan,
    num_sms: int,
    num_max_tokens_per_rank: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    activation_clamp: float,
    fast_math: bool,
):
    return gen_megamoe_sm90_nvfp4_module(
        plan,
        num_sms,
        num_max_tokens_per_rank,
        num_experts,
        num_topk,
        hidden,
        intermediate_hidden,
        activation_clamp,
        fast_math,
    ).build_and_load()


class MegaMoESymmBuffer:
    """Symmetric buffer holding the inputs, the shared expert token pool and the
    combine staging area. All ranks must allocate it identically; the kernel
    reaches peer ranks through the rendezvous'd pointers."""

    def __init__(
        self,
        group,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        activation_clamp: float = 10.0,
        fast_math: bool = True,
    ):
        import torch.distributed._symmetric_memory as symm_mem

        if group.size() != 8:
            raise ValueError(
                f"MegaMoE is an 8-rank kernel, got world size {group.size()}"
            )

        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.activation_clamp = activation_clamp
        self.fast_math = fast_math
        self.num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count

        # The buffer layout depends only on the shape, not on the per-call plan,
        # so any plan for this shape yields the same offsets. Use M=1.
        probe_plan = select_fused_plan(
            self.num_sms,
            group.size(),
            num_experts,
            num_topk,
            hidden,
            intermediate_hidden,
            num_max_tokens_per_rank,
            1,
        )
        mod = _get_module(
            probe_plan,
            self.num_sms,
            num_max_tokens_per_rank,
            num_experts,
            num_topk,
            hidden,
            intermediate_hidden,
            activation_clamp,
            fast_math,
        )
        layout = list(mod.megamoe_sm90_nvfp4_symm_buffer_layout())
        self._offsets = layout
        num_bytes = layout[0]
        self.num_max_pool_tokens = layout[9]
        self.num_padded_sf_pool_tokens = layout[10]

        self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device="cuda")
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        group.barrier()
        torch.cuda.synchronize()

        self.x = self._view(
            layout[1], (num_max_tokens_per_rank, hidden), torch.float8_e4m3fn
        )
        self.x_sf = self._view(
            layout[2], (num_max_tokens_per_rank, hidden // 128), torch.float32
        )
        self.topk_idx = self._view(
            layout[3], (num_max_tokens_per_rank, num_topk), torch.int64
        )
        self.topk_weights = self._view(
            layout[4], (num_max_tokens_per_rank, num_topk), torch.float32
        )

    def _view(self, byte_offset: int, shape: Tuple[int, ...], dtype: torch.dtype):
        itemsize = torch.empty((), dtype=dtype).element_size()
        numel = 1
        for s in shape:
            numel *= s
        flat = self.buffer[byte_offset : byte_offset + numel * itemsize]
        return flat.view(dtype).view(*shape)

    def stage(
        self,
        x_fp8: torch.Tensor,
        x_sf: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        """Copy this rank's tokens and routing into the symmetric buffer."""
        m = x_fp8.shape[0]
        self.x[:m].copy_(x_fp8)
        self.x_sf[:m].copy_(x_sf)
        self.topk_idx[:m].copy_(topk_idx)
        self.topk_weights[:m].copy_(topk_weights)

    def destroy(self) -> None:
        self.handle = None
        self.buffer = None
        self.x = self.x_sf = self.topk_idx = self.topk_weights = None


def nvfp4_mega_moe(
    y: torch.Tensor,
    l1_weights: torch.Tensor,
    l2_weights: torch.Tensor,
    symm_buffer: MegaMoESymmBuffer,
    num_tokens: Optional[int] = None,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    l1_global_scales: Optional[torch.Tensor] = None,
    l2_global_scales: Optional[torch.Tensor] = None,
    force_push_dispatch: Optional[bool] = None,
    force_no_clean_barrier: Optional[bool] = None,
    kernel_family: str = "auto",
    family_threshold: int = 256,
) -> torch.Tensor:
    """Run one fused NVFP4 MegaMoE step.

    Args:
        y: output, ``[num_tokens, hidden]`` bfloat16.
        l1_weights / l2_weights: packed NVFP4 weights with prepacked UE4M3
            scales, as produced by the MegaMoE weight transform.
        symm_buffer: staged with :meth:`MegaMoESymmBuffer.stage`.
        num_tokens: tokens on this rank; defaults to ``y.shape[0]``.
        force_push_dispatch / force_no_clean_barrier: override the M-gated
            counter-dispatch policy (A/B testing; ``None`` uses the policy).
    """
    b = symm_buffer
    m = int(num_tokens if num_tokens is not None else y.shape[0])

    if kernel_family == "auto":
        kernel_family = "fused" if m <= family_threshold else "split"
    if kernel_family not in ("fused", "split"):
        raise ValueError(
            f"kernel_family must be 'auto', 'fused' or 'split', got {kernel_family!r}"
        )
    if kernel_family == "split":
        return _run_split(
            y,
            l1_weights,
            l2_weights,
            b,
            m,
            cumulative_local_expert_recv_stats,
            l1_global_scales,
            l2_global_scales,
        )

    plan = select_fused_plan(
        b.num_sms,
        b.group.size(),
        b.num_experts,
        b.num_topk,
        b.hidden,
        b.intermediate_hidden,
        b.num_max_tokens_per_rank,
        m,
        force_push_dispatch=force_push_dispatch,
        force_no_clean_barrier=force_no_clean_barrier,
    )
    mod = _get_module(
        plan,
        b.num_sms,
        b.num_max_tokens_per_rank,
        b.num_experts,
        b.num_topk,
        b.hidden,
        b.intermediate_hidden,
        b.activation_clamp,
        b.fast_math,
    )
    mod.megamoe_sm90_nvfp4_fused(
        y,
        b.buffer,
        b.handle.buffer_ptrs,
        b.group.rank(),
        l1_weights,
        l2_weights,
        b._offsets,
        m,
        cumulative_local_expert_recv_stats,
        l1_global_scales,
        l2_global_scales,
    )
    return y


# ---------------------------------------------------------------------------
# Split (BN128) family -- selected above the fused threshold
# ---------------------------------------------------------------------------

#: Above this many tokens per rank a single fused persistent kernel no longer
#: fits the working set and the two-kernel split family takes over.
FAMILY_THRESHOLD = 256


@dataclasses.dataclass(frozen=True)
class SplitPlan:
    l1_num_stages: int
    l1_smem_size: int
    l2_num_stages: int
    l2_smem_size: int
    num_experts_per_wave: int
    num_max_pool_tokens: int
    dispatch_dequant: bool
    l2_arrival_counter: bool
    num_padded_sf_pool_tokens: int


@functools.cache
def _get_split_plan_module(
    num_sms,
    num_ranks,
    num_max_tokens_per_rank,
    num_experts,
    num_topk,
    hidden,
    intermediate_hidden,
):
    return gen_megamoe_sm90_nvfp4_split_plan_module(
        num_sms,
        num_ranks,
        num_max_tokens_per_rank,
        num_experts,
        num_topk,
        hidden,
        intermediate_hidden,
    ).build_and_load()


@functools.cache
def select_split_plan(
    num_sms,
    num_ranks,
    num_experts,
    num_topk,
    hidden,
    intermediate_hidden,
    num_max_tokens_per_rank,
    num_tokens,
) -> SplitPlan:
    """Ask the plan module for the L1/L2 pipeline configuration.

    The sizing arithmetic mirrors the kernel's shared-memory layout exactly, so
    it is evaluated by the same C++ code the kernel is built from rather than
    being re-derived here.
    """
    sf_pool = max(
        get_num_padded_sf_pool_tokens(
            get_num_max_pool_tokens(
                num_ranks, num_max_tokens_per_rank, num_topk, num_experts // num_ranks
            ),
            bm,
        )
        for bm in CANDIDATE_BLOCK_M
    )
    mod = _get_split_plan_module(
        num_sms,
        num_ranks,
        num_max_tokens_per_rank,
        num_experts,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    v = list(
        mod.megamoe_sm90_nvfp4_split_plan(
            num_sms,
            num_ranks,
            num_experts,
            num_max_tokens_per_rank,
            num_tokens,
            num_topk,
            hidden,
            intermediate_hidden,
            sf_pool,
        )
    )
    return SplitPlan(
        l1_num_stages=v[0],
        l1_smem_size=v[1],
        l2_num_stages=v[2],
        l2_smem_size=v[3],
        num_experts_per_wave=v[4],
        num_max_pool_tokens=v[5],
        dispatch_dequant=bool(v[6]),
        l2_arrival_counter=bool(v[7]),
        num_padded_sf_pool_tokens=sf_pool,
    )


@functools.cache
def _get_split_module(
    plan,
    num_sms,
    num_ranks,
    num_max_tokens_per_rank,
    num_experts,
    num_topk,
    hidden,
    intermediate_hidden,
    activation_clamp,
    fast_math,
):
    return gen_megamoe_sm90_nvfp4_split_module(
        plan,
        num_sms,
        num_ranks,
        num_max_tokens_per_rank,
        num_experts,
        num_topk,
        hidden,
        intermediate_hidden,
        activation_clamp,
        fast_math,
    ).build_and_load()


def _run_split(y, l1_weights, l2_weights, b, m, stats, gs1, gs2):
    plan = select_split_plan(
        b.num_sms,
        b.group.size(),
        b.num_experts,
        b.num_topk,
        b.hidden,
        b.intermediate_hidden,
        b.num_max_tokens_per_rank,
        m,
    )
    mod = _get_split_module(
        plan,
        b.num_sms,
        b.group.size(),
        b.num_max_tokens_per_rank,
        b.num_experts,
        b.num_topk,
        b.hidden,
        b.intermediate_hidden,
        b.activation_clamp,
        b.fast_math,
    )
    mod.megamoe_sm90_nvfp4_split(
        y,
        b.buffer,
        b.handle.buffer_ptrs,
        b.group.rank(),
        l1_weights,
        l2_weights,
        b._offsets,
        m,
        stats,
        gs1,
        gs2,
    )
    return y
