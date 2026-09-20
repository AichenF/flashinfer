"""JIT module generation for the SM90 NVFP4 MegaMoE kernel.

The kernel resolves its shared-memory layout, task schedule and dispatch
protocol at compile time, so one module is built per (shape, plan) pair. The
plan is chosen in ``flashinfer.megamoe.heuristics`` and rendered into
``megamoe_sm90_nvfp4_config.inc`` here.
"""

import os
from pathlib import Path

import jinja2

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, write_if_different


def _csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "megamoe"
    if installed.exists():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "megamoe"
    if checkout.exists():
        return checkout
    raise FileNotFoundError(f"MegaMoE CUDA sources not found (checked {installed}, {checkout})")


def get_megamoe_sm90_nvfp4_uri(plan, num_sms: int, num_max_tokens_per_rank: int,
                               num_experts: int, num_topk: int, hidden: int,
                               intermediate_hidden: int, activation_clamp: float,
                               fast_math: bool) -> str:
    return (
        f"megamoe_sm90_nvfp4_sm{num_sms}_e{num_experts}_k{num_topk}"
        f"_h{hidden}_i{intermediate_hidden}_cap{num_max_tokens_per_rank}"
        f"_bm{plan.block_m}_bn{plan.block_n}_epw{plan.num_experts_per_wave}"
        f"_st{plan.num_stages}"
        f"_{'swap' if plan.swap_ab else 'noswap'}"
        f"{'_rs' if plan.rs_swap_ab else ''}"
        f"{'_il' if plan.use_interleaved_scheduler else '_static'}"
        f"{'_push' if plan.push_dispatch else ''}"
        f"{'_noclean' if plan.no_clean_barrier else ''}"
        f"{'_mode2' if plan.use_mode2_row_decoder else ''}"
        f"{'_single' if plan.single_active_dispatch_warp else ''}"
        f"_clamp{activation_clamp:g}"
        f"{'_fm' if fast_math else ''}"
    )


def gen_megamoe_sm90_nvfp4_module(
    plan,
    num_sms: int,
    num_max_tokens_per_rank: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    activation_clamp: float,
    fast_math: bool,
) -> JitSpec:
    uri = get_megamoe_sm90_nvfp4_uri(
        plan, num_sms, num_max_tokens_per_rank, num_experts, num_topk, hidden,
        intermediate_hidden, activation_clamp, fast_math,
    )
    csrc = _csrc_dir()
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    os.makedirs(gen_directory, exist_ok=True)

    with open(csrc / "megamoe_sm90_nvfp4_config.jinja") as f:
        config_templ = jinja2.Template(f.read())
    write_if_different(
        gen_directory / "megamoe_sm90_nvfp4_config.inc",
        config_templ.render(
            num_sms=num_sms,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_experts=num_experts,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            num_experts_per_wave=plan.num_experts_per_wave,
            block_m=plan.block_m,
            block_n=plan.block_n,
            num_max_pool_tokens=plan.num_max_pool_tokens,
            num_padded_sf_pool_tokens=plan.num_padded_sf_pool_tokens,
            num_stages=plan.num_stages,
            smem_size=plan.smem_size,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
            swap_ab=plan.swap_ab,
            rs_swap_ab=plan.rs_swap_ab,
            single_active_dispatch_warp=plan.single_active_dispatch_warp,
            use_mode2_row_decoder=plan.use_mode2_row_decoder,
            use_interleaved_scheduler=plan.use_interleaved_scheduler,
            push_dispatch=plan.push_dispatch,
            no_clean_barrier=plan.no_clean_barrier,
            spin_debug=os.environ.get("FLASHINFER_MEGAMOE_SPIN_DEBUG", "0") == "1",
        ),
    )

    source_paths = []
    for filename in ("megamoe_sm90_nvfp4_fused.cu", "megamoe_sm90_nvfp4_binding.cu"):
        dest = gen_directory / filename
        write_if_different(dest, open(csrc / filename).read())
        source_paths.append(dest)

    return gen_jit_spec(
        uri,
        source_paths,
        # The kernels take a float non-type template parameter (the activation
        # clamp), which needs C++20. The register-usage level keeps the
        # persistent kernel at one CTA per SM.
        extra_cuda_cflags=[
            "-std=c++20",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--ptxas-options=--register-usage-level=5",
            "-DENABLE_BF16",
            "-DENABLE_FP8",
        ],
        extra_cflags=["-std=c++20"],
        extra_include_paths=[gen_directory, jit_env.FLASHINFER_CSRC_DIR],
    )
