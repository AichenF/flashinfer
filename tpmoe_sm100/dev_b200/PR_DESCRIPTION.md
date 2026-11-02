# tpmoe_b2: B200 TP4 fused MoE layer with per-M specialized builds (bitwise == production, ~1.24x at M <= 128)

## Summary

`tpmoe_b2` is a single-kernel DeepSeek-V4-Flash MoE layer for B200 at TP4 (tcgen05, one CTA per SM). One launch does:

- the fused gate;
- MXFP8 quantization of x;
- the routed MXFP4 experts (SGLang `flashinfer_mxfp4` / trtllm-gen weight tensors used in place);
- the shared FP8 expert with DeepGEMM numerics;
- the routed-plus-shared combine and the TP all-reduce (one-shot LL for M <= 48, otherwise reduce-scatter + all-gather).

For 5 <= M <= 64 the kernel also runs the router GEMM itself, and its logits match the production cuBLAS split-K logits bit for bit.

The layer output is **bitwise identical** to the production SGLang layer (`--moe-runner-backend flashinfer_mxfp4`) at every M tested.

This change makes the kernel **compile-time specialized per token-count bucket**:

- Each build fixes the M range, the production configuration and the grid at compile time. The fixed configuration is TP4, top-6, fused gate, fused shared expert, no dsv3 in-kernel router; the grid is 148.
- Code paths that build never takes are folded away. On the latency-bound critical path the kernel runs i-cache cold, so code size is time.
- A build checks every compile-time assumption on the host (`TORCH_CHECK`), so a wrong pick fails loudly instead of computing garbage.
- `tpb2_dispatch.py` holds the 14-entry build table: exact-M builds at 1, 2, 8, 16, 32, 64 and 128, and buckets in between. It also says which router feeds each M.

## Results

Metric: production layer time / ours (router + fused kernel), per M. Each arm is a CUDA graph over rotating weight copies (cold L2). The time is the median over rounds of the worst rank. Measured on 4x B200 (TP4) on the same node as the previous 3-build configuration.

| M | 1 | 2 | 4 | 6 | 8 | 16 | 20 | 24 | 32 | 48 | 64 | 96 | 128 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| node A, before | 0.97 | 0.97 | 1.17 | 1.22 | 1.17 | 1.15 | 1.16 | 1.18 | 1.18 | 1.17 | 1.15 | 1.14 | 1.15 | 1.137 |
| node A, this change | 1.23 | 1.15 | 1.30 | 1.27 | 1.29 | 1.29 | 1.26 | 1.26 | 1.26 | 1.25 | 1.23 | 1.19 | 1.21 | **1.245** |
| node B, before | 0.99 | 0.99 | 1.18 | 1.17 | 1.15 | 1.15 | 1.16 | 1.18 | 1.19 | 1.17 | 1.15 | 1.14 | 1.15 | 1.136 |
| node B, this change | 1.20 | 1.20 | 1.28 | 1.26 | 1.29 | 1.27 | 1.27 | 1.25 | 1.27 | 1.23 | 1.23 | 1.18 | 1.21 | **1.241** |

Absolute layer times on node A, production / ours, in us:

| M | 1 | 2 | 4 | 6 | 8 | 16 | 20 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| production | 32.7 | 35.9 | 46.8 | 50.4 | 54.7 | 64.5 | 70.4 | 71.5 | 76.1 | 90.8 | 98.1 | 116.4 | 130.9 |
| ours | 26.6 | 31.3 | 36.0 | 39.6 | 42.3 | 50.0 | 55.7 | 56.7 | 60.5 | 72.6 | 79.8 | 98.1 | 108.5 |

Run-to-run noise of identical builds is 2-4% per M, which is about 1% on the mean.

## What changed

**Compile-time M range** (`TPB2_MMIN` / `TPB2_MMAX`):
- Every device read of M goes through `pM(P)`, a clamp to the build's range, so range checks fold.
- The all-reduce mode `pOS(P)` folds the same way.
- Small-M builds lose the multi-CTA routing tables and the reduce-scatter + all-gather path. Large-M builds lose the small-gate paths.

**Production configuration fixed** (`TPB2_SPEC=1`, `TPB2_GRID=148`, `TPB2_RS=0`):
- ndev, topk, shared-expert mode, gate mode, router mode and the grid become constants.
- Divisions by the grid become multiplies.

**Dead split-K paths removed** (`TPB2_WHOLE=1`): with round-robin whole FC1 items (the bitwise configuration), the epilogue's partial-store and merge paths are compiled out.

**Smaller critical-path code:**
- `TPB2_T1CNT`: SWAR byte-field counts replace a 16x8 unrolled compare.
- `TPB2_PUT1`: one producer put site.
- `TPB2_ACTU` (M = 1): FC1 activation unrolled over the live tokens.

**Per-bucket knobs**, all re-tuned with the router in front (paired A/B): `TABW` at M1, `SHGU` up to M8 and at M17-31, static FC2 order at M9-16, `ACTX` at M >= 32.

**Smallest build:** about 13.9k SASS instructions before this change, about 7.5k after.

Harness and tools:
- `b2_layer_ab.py --xbuilds` (more builds for per-M dispatch), `--rfmax`, and `--sweep` now also covers the in-kernel router path (`--arms SF`).
- `b2_sass.py` reports SASS size per build and can prebuild in parallel.
- `b2_lines.py` reports SASS per source line, attributed to the outermost inlined line.

## Correctness

- **Per-M equality:** the dispatched layer equals production bit for bit at all 13 M, on 4 ranks (`D != A` = 0 everywhere).
- **Multi-offset sweeps:** 0 mismatching elements on all 4 ranks.
  - Coverage: every build, at 27 M values. These include off-grid M 3, 5, 7, 9, 12, 15, 17, 28, 31, 40, 56, 63, 80, 112.
  - Offsets: 20-60 consecutive capture offsets per M.
  - Production-router path: 1032 (M, offset) cases.
  - In-kernel-router path: 664 cases.
- **Router:** in-kernel logits equal the cuBLAS split-K logits at every M where the in-kernel router is used.

## Usage

```python
import tpb2, tpb2_dispatch
flags = tpb2_dispatch.build_flags(M)              # e.g. ('-DTPB2_MMIN=8', '-DTPB2_MMAX=8', ...)
E = tpb2.load_ext(tag=..., flags=flags)           # one extension per build (prebuild all: tpb2_dispatch.all_builds())
parts = tpb2_dispatch.in_kernel_router(M)         # K parts for the in-kernel router, or None
# parts:  tpb2.run(st, W, x, out, glog=rpart, gbias=bias, SW=sw, nparts=len(parts), router=(router_w, parts))
# None:   tpb2.run(st, W, x, out, glog=production_router_logits(x), gbias=bias, SW=sw)
```

Each build needs its own `tpb2.RankState` (workspace and IPC-linked all-reduce buffers).

## Reproduce

```bash
. disp_v71.sh   # X = '+'-separated builds, DISP = per-M dispatch
python3 b2_sass.py --suffix=_x --sets=$X                       # prebuild
python3 b2_layer_ab.py --arms A S DF --rounds 16 --xbuilds=$X --dispatch $DISP
```

## Limits

The original target, a 1.4x mean at M <= 128, is **not** reached under the bitwise constraint. The measured limits:

**Large M is bound by data movement.** At M128 the kernel alone takes 102.5 us.
- Removing the MMAs saves only 6.9%.
- Removing the per-stage weight scales saves 4.2%, which is just their 6.25% share of the streamed bytes.
- With FC1 isolated, each stage is held by three limits of about 0.3 us at once: MMA issue, the weight DRAM stream, and the x gather. Removing any one exposes the next.
- Unpacked FP4 costs 32 KB of shared-memory writes plus 32 KB of MMA reads per 16 KB weight stage. That is why `cta_group::2` (+8.6%) and a deeper ring (0) do not help.

**Small M is a latency chain.** At M1 about 14 serial cross-SM handoffs of roughly 1 us each, plus the all-reduce waiting for the slowest rank. Bitwise equality forbids splitting K and forbids MMA tiles below 128 rows.

**Router at M > 64.** The production cuBLAS split-K router takes 6.3 us alone and runs serially in front of the kernel. The in-kernel router is slower there.

**Still open (all estimated at about 1% or less, or non-bitwise):**
- a routing-table rewrite for M > 8;
- a cluster/DSMEM FC1 -> FC2 handoff at M <= 2;
- non-bitwise K-splits at small M;
- FP4 activations (a model-numerics change).

## Test plan

- [x] Dispatched layer == production bitwise at M 1, 2, 4, 6, 8, 16, 20, 24, 32, 48, 64, 96, 128 (4 ranks)
- [x] Multi-offset bitwise sweeps per build (27 M values incl. off-grid M; 1032 production-router + 664 in-kernel-router cases, 0 mismatches)
- [x] A/DF on two nodes (16 and 20 rounds)
- [ ] Integration into the SGLang decode path (per-batch-size build selection) -- not part of this change

🤖 Generated with [Claude Code](https://claude.com/claude-code)
