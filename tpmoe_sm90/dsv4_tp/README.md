# tpmoe_sm90/dsv4_tp: TP MoE kernels for DeepSeek-V4-Flash and V4.1-Flash on H20 / H200

This directory holds the routed-expert TP MoE kernels for decode on Hopper (sm_90a). They are the final builds of the
11-10 benchmark matrix (DSV4-Flash and DSV4.1-Flash × H20 / H200 × TP4 / TP8, plus two H200 "TP8-hybrid" rows). Each
kernel is one persistent launch that does the following:

- per-token E4M3 quantization of x;
- FC1, SwiGLU (clamp 10) and FC2 over the routed experts, with MXFP4 weights in Humming's fused-E8M0 layout
  (RS-wgmma with in-register dequant);
- the weighted top-k combine;
- the TP all-reduce as low-latency messages over NVLink peer memory.

Outputs are bitwise identical on all ranks and deterministic.

Shapes:
- DSV4-Flash: H 4096, I 2048, 256 experts, top-6.
- DSV4.1-Flash: H 5120, I 2304, 384 experts, top-6. Per-rank slices are 576 (TP4: 4 items + a 64-column tail) and
  288 (TP8: 2 × 128 + 32).

The directory is a **release snapshot**. Every file is a byte copy of a pinned git blob; `PROVENANCE.md` lists the
source repository, commit and SHA-256 of each file. Sub-directories keep the layout of the benchmark suite, so the
adapters' relative paths work unchanged. Every kernel source here carries the FC1-to-FC2 staging race fix
(`TP2_B2B_FIX`, default 1: one consumer barrier between the FC1 epilogue and the FC2 unit loop). Nothing is wired into the FlashInfer
build. Kernels JIT-build through `torch.utils.cpp_extension` (`-O3 -gencode=arch=compute_90a,code=sm_90a`).

## Cells

The ratio rows are best baseline ÷ ours (> 1 means ours is faster) for M = 8 / 16 / 32 / 64 / 128 / 256 / 384 / 512.

| cell | directory | kernel | best baseline ÷ ours |
|---|---|---|---|
| H20 DSV4-Flash TP8 | `h20_dsv4f_tp8_h123/` | h123: v78h21, v94h13 at M64-127, v94h21 at M128-341 | 1.46 1.37 1.44 1.50 1.46 1.37 1.38 1.34 |
| H20 DSV4-Flash TP4 | `h20_tp4/` (`tp4moe2_v94ex.cu`) | v94ex = serving_tp4moe2_h100 kernel, ns build | 1.44 1.31 1.19 1.31 1.15 1.26 1.49 1.47 |
| H20 DSV4.1 TP8 | `h20_dsv41f_tp8/` | tp41 = tp8v41 (h123 ported to DSV4.1), T4N16 dispatch at M9-16 and M ≥ 320 | 1.35 1.27 1.32 1.35 1.30 1.37 1.22 1.21 |
| H20 DSV4.1 TP4 | `h20_tp4/` (`tp4v41e.cu`) | tp4v41e (TP4 port, T2 split tail, FC2 half k-tile first) | 1.35 1.25 1.20 1.37 1.13 1.25 1.41 1.46 |
| H200 DSV4-Flash TP4 | `h200_cells/` (`h200_dsv4f_tp4`) | v94ex | 1.36 1.32 1.21 1.28 1.14 1.28 1.34 1.34 |
| H200 DSV4-Flash TP8 | `h200_cells/` (`h200_dsv4f_tp8`) | h123 trio | 1.47 1.27 1.34 1.40 1.35 1.23 1.24 1.22 |
| H200 DSV4-Flash TP8-hybrid | `h200_cells/` (`h200_dsv4f_tp8h`) | 2 expert groups × TP4 (tpv41h MODEL 0) from M256, h123 below | 1.46 1.27 1.33 1.39 1.36 1.24 1.27 1.26 |
| H200 DSV4.1 TP4 | `h200_cells/` (`h200_dsv41f_tp4`) | tpv41 TP4 | 1.38 1.28 1.18 1.36 1.11 1.37 1.30 1.33 |
| H200 DSV4.1 TP8 | `h200_cells/` (`h200_dsv41f_tp8`) | tpv41 | 1.44 1.33 1.30 1.41 1.37 1.29 1.14 1.13 |
| H200 DSV4.1 TP8-hybrid | `h200_cells/` (`h200_dsv41f_tp8h`) | tpv41h (2 groups × TP4) for M > 8 | 1.44 1.36 1.35 1.51 1.47 1.40 1.26 1.25 |

How the numbers were measured:
- Protocol: one call per CUDA graph replay, with a 512 MiB L2 flush and an NCCL gate all-reduce plus a ~100 µs sleep
  before every replay. The statistic is the per-rank median, then the max over ranks.
- Routing: real plain-decode routing traces, 6 cases per M.
- Statistics: the median over 3 process-isolated repetitions.
- Baselines: gc-moe (EP, green contexts) and FlashInfer MegaMoE `mxfp4_mega_moe` (EP).
- Source: tpmoe_suite 753579c.

Caveat: under real routing skew, gc-moe degrades sharply on SM90 at large M (e.g. H20 DSV4.1 TP8 M512: gc 972 µs vs
FlashInfer MegaMoE 581 µs). From M64 up, the best baseline is usually FlashInfer MegaMoE.

Full layer (router GEMM + fused gate + shared expert + routed experts + all-reduce) against the best of three EP-full arms:
gc-moe with its fused shared expert, and FlashInfer MegaMoE plus a separate shared expert run either serially or on a side
stream. The tables are in `h20_dsv41f_tp8/full_layer/` (H20) and `h200_cells/full_layer/` (H200):

| cell | best EP-full ÷ TP-full, M8 … M512 |
|---|---|
| H20 DSV4.1 TP8 | 1.78 1.48 1.44 1.23 1.26 1.35 1.23 1.17 |
| H20 DSV4-Flash TP8 (M8 … M384) | 1.85 1.73 1.58 1.55 1.51 1.37 1.35 |
| H20 DSV4-Flash TP4 (M8 … M384) | 1.64 1.46 1.25 1.18 1.18 1.17 1.24 |
| H200 DSV4.1 TP8 | 1.36 1.24 1.32 1.36 1.40 1.27 1.19 1.11 |
| H200 DSV4-Flash TP8 | 1.31 1.24 1.19 1.27 1.44 1.29 1.28 1.20 |
| H200 DSV4-Flash TP4 | 1.26 1.35 1.19 1.24 1.16 1.21 1.26 1.25 |

## Pins

- `h20_dsv41f_tp8/`: the suite import of the H20 DSV4.1 TP8 adapter.
  - Its `src/` comes from development repository commit 872c14c (TP2_B2B_FIX); see `src/COMMIT` and `src/SHA256SUMS`.
  - `README_owner.md` and `full_layer/` come from that repository at d9f89d4.
- `h20_dsv4f_tp8_h123/`: the h123 served sources with the barrier (built with the served flags `-DTP2_SHARED=0
  -DTP2_NDEV=8`). The unused v28 base source of that directory is left out; see `PROVENANCE.md`.
- `h20_tp4/`: the suite import of the H20 TP4 cells.
  - `tp4moe2_v94ex.cu` is byte-identical to `../serving_tp4moe2_h100/tp4moe2_ns.cu`.
  - `tp4v41e.cu` is the H20 owner's final source plus the same one-line barrier, which the suite added. Its development
    repository does not have the barrier committed yet.
- `h200_cells/`: the suite import of the H200 owner's cells at 56f1450 (fixed sources, verify_fix passed). It contains
  only the sources the cell adapters build by default.
  - `README_owner_results.md`, `b2b_fix_README.md` and `full_layer/` come from the H200 owner's repository at 313f388.

## Build, run, gate

The kernels run through the tpmoe_suite harness (`harness/bench.py --ours-adapter <cell dir>`). The adapters import
`tps_common` / `humming_prep` from that harness, and `harness/sm90_tp4moe2.py` is included here. The per-cell READMEs give
the build flags and owner commands.

Every kept build passed this correctness gate:
- ranks bitwise identical, deterministic, rel against an exact-numerics reference, cos against dense;
- back-to-back bursts without sync between calls: rank skew, input cycling, distinct outputs, CUDA-graph bursts;
- mixed-M bursts over builds that share one workspace;
- high-count stress (≥ 1,500 rounds × 8 calls).

The race fix is verified two ways:
- On H20: unfixed TP8 builds failed 12 in 384k (h123 fused-shared) and 71 in 1.15M (tp41) back-to-back calls; fixed
  builds failed 0.
- On H200: with the fix off, routed-only TP8 failed 1-2 per 1,600 calls; with it on, 0 failures across 6 cells × 8 M ×
  1,600 calls.

Fixed and unfixed outputs are bitwise equal. The fix costs ≤ 0.2%.

## Known constraints

- **One workspace per LL buffer layout.** Builds that share a workspace must agree on the epoch / parity protocol.
  Mixed-M bursts across the dispatched builds are part of the gate.
- **`torch._scaled_mm` blockwise FP8 on SM90 is silently wrong when K/128 % 4 != 0** (torch 2.11 cu130). The DSV4.1 shared
  FC2 has K = 2304. The EP-full arms use `flashinfer.gemm.fp8_blockscale_gemm_sm90` instead. The repro is described in
  `h200_cells/full_layer/README.md`.
