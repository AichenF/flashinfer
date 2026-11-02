# tpmoe_b2 (B200, DeepSeek-V4-Flash TP4 fused MoE) -- status at v50 (2026-11-01)

Goal set by the user: mean A/S >= 1.4 over M <= 128 (A = production flashinfer_mxfp4 layer incl. its router, shared expert and
all-reduce; S = production router + this kernel), bit-identical to production.  Not reached.

## Results (4 x B200, paired graphs, cold L2)

| M | 1 | 2 | 4 | 6 | 8 | 16 | 20 | 24 | 32 | 48 | 64 | 96 | 128 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A/S v46 (k184) | 0.90 | 0.96 | 1.10 | 1.12 | 1.11 | 1.16 | 1.13 | 1.16 | 1.13 | 1.14 | 1.12 | 1.13 | 1.12 | 1.098 |
| A/S v46 (k185) | 0.85 | 0.95 | 1.08 | 1.05 | 1.09 | 1.11 | 1.14 | 1.13 | 1.10 | 1.16 | 1.12 | 1.12 | 1.14 | 1.079 |

S == A bit for bit at every M except M128 (110 routed elements, known).  Run-to-run spread of the mean is ~2%; paired A/B
noise ~+-1.5..3% per M (M1 +-6%).  v49 adds T1L at M1 (~-1.8%).

## What moved the needle (all bitwise)

| version | change | effect |
|---|---|---|
| v41 | FC2VAR: FC2 tickets over slot-major row tiles, tail units 4 -> 2 -> 1 row tiles | M24-128 -0.4..-4.3% |
| v42 | SHW1: whole shared FC1 items (one K chain, no LASTM merge) at M >= 4 (+ rr skips the shared CTAs 3 waves) | M8-48 -3..-12%; fixed the ~1100 shared mismatches at M >= 48 |
| v43 | rr at every M (no stream-K split items / merges) | M6-16 -2..-10% |
| v46 | PGATE: M > 8, a token gated by 8 warps (per-warp top-6 + merge) | mean -1.2..-1.5% |
| v49 | T1L at M1: T1 by the last FC2 contributor's warp 7 | M1 ~-1.8% |

Common root cause of v42/v43 (found with the in-repo Xtrace reproduction): generic-memory handoffs (split-K merges, fences,
polls) inside the TMA streaming phase run 5-10x slower than idle and stall the in-order epilogue.

## Measured, not adopted (knob, effect)

In-kernel router (opt-in arm SF; bitwise == cuBLAS split-K; ~= S), separate router kernel (RR == cuBLAS time), GUNR / GWARM /
XRLX / QW811 / PGATE1 (gate / x-flag / quant tweaks: trace-faster prologue, end-to-end 0 or slower), ACTX (act ILP), HPUB (h
publish via warp 7: +0.3..7.4%), FC2RLX, FC2G8 (+0.6..2.6%), FC2GMAX=2 (~0), PFF / L2W2 (L2 prefetch of first stages / W2: 0 or
+1..5%), OUTL2, LMV, NS=5, PAIR (cta_group::2: +1.4..8.9%), WARM (fixed, +1..2%), ONESHOT_MAXM=64 (0), SHQ 2 / 8 at M1-2,
RRSPLIT=1 at M1, T1L beyond M1 (+2..20%).  dsv3 FMA router at M5-16 (arm SD) would be -3..-8% there but is not guaranteed
bitwise vs production's cuBLAS router -> rejected by the user.

## Why 1.4x is out of reach under the bitwise constraint

* Steady streaming is at the hardware limit: per 16 KB FC1 stage ~80 KB of shared-memory traffic (TMA writes unpacked e2m1 A
  + B, the MMA reads both) = ~0.34 us at 128 B/clk == the measured stage time; aggregate 5.5-5.9 TB/s.  Removing every MMA
  (PROBE=2) gains only 2-3%.
* Causal profiling (TPB2_DLY_SITE): the prologue (gate -> tables -> first push, 6-11 us) and the tail (FC2 -> T1 -> all-reduce,
  5-11 us) are on the critical path with sensitivity ~1; each cross-SM handoff costs ~1.5-2.5 us of memory latency and the
  chain has ~6 of them.  ncu PC sampling: instruction-fetch stalls 10-17% of warp time (epilogue 24% at M8).
* M1/M2 are below 1: the 4 MB FP8 shared expert must be K-split (quarters + merge) to stream fast enough; production hides it
  on a second stream.
* Remaining levers need either numerics changes (FMA router, FC2 split-K) or a different architecture for the handoffs
  (e.g. cluster/DSMEM-local FC1 -> FC2 per expert), estimated weeks of work.

## Tools

xtrace.cuh / xtrace.py (Xtrace-style phase trace; b2_layer_ab --trace [--trace-arm SF]), b2_layer_ab paired A/B
(--flags2 / --src2 + --flags2=), TPB2_DLY_SITE causal profiling, sass_lines.sh / sass_size.sh, ncu on 1-GPU b2_kdbg
(--flags=-lineinfo, SourceCounters + smsp__pcsamp_warps_issue_stalled_* metrics), panel_stats.py.
