# tp8_v41_b300 — DeepSeek-V4.1-Flash routed MoE, TP8 on 8x B300 (agent B300-TP8)

Fused routed-MoE kernel for DSV4.1-Flash (hidden 5120, I 2304, 384 experts, top-6) at TP8 on 8x B300 (sm_103a):
- numerics: MXFP8 activations x MXFP4 weights, SwiGLU clamp 10;
- one kernel does in-kernel x quant, FC1, FC2, the weighted combine (T1) and the 8-way all-reduce / all-gather;
- outputs are bitwise identical on all 8 ranks and deterministic, and match an exact-numerics torch reference
  (v41host.ref_rank / ref_rank_h; rel <= 4e-4).

## Final kernel
- **Defaults = commit 3668057.** Per-M builds (v41host.spec_flags):
  - NMAX 32 for M >= 256;
  - H2 (`TPB2_F2H`: FC2 sharded on H, an h exchange, then a local T1 and an all-gather) for M >= 128;
  - GATHER 0 for M >= 256;
  - LLS sentinel all-reduce at every M;
  - ONESHOT_MAXM 32.
- **HEAD** adds:
  - 8bbb174 (F2HParams): every F2H kernel parameter sits in a struct that only F2H builds carry, so non-F2H builds keep the pre-F2H
    Params layout. Neutral on B300; outputs identical.
  - The F2H_HSL staging-slot knob: the per-build default leaves DSV4.1 unchanged (1 slot at NMAX 32).
  - Refuted knobs, all off by default.

## Files
- `port_v41.py`: the generator. It applies exact-string patches to `../k3_b300/L/k3h.cu` (read only) and writes `tp8v41_core.cu`.
- `tail/patch_tail.py`: the suite_b200 tail stage (copied @ 4c9634b, TAILB zeroing guarded). It turns `tp8v41_core.cu` into `tp8v41.cu`.
- `gen.sh`: runs both steps.
- `v41host.py`: build flags (`spec_flags`, per-M defaults `NMAX32_MINM`, `F2H_MINM` = env `TP8_F2H_MINM`, default 128), weights,
  workspace (`RankState`), the `run` wrapper and the torch references.
- `v41tp8.py`: the TP8 harness. It runs checks, timing (protocols a/b), same-run A/B of two sources/flag sets
  (`--src2 / --flags2 / --swap`), dbg stamps, and the back-to-back stress modes.
- `suite_cell/adapter.py`: the tpmoe_suite `ours` adapter. It zero-fills the gathered-h buffer and 0xFF-fills the LL region.
- `ctr.sh`: runs a command in the CUDA 13.2.1 container on the held node. `hold.sbatch`: the node hold.
- `oNN.sh` / `sNNN.sh`: one script per experiment (logs in `runs/`, not tracked); `results_suite/*/result.json`: suite outputs.
- Some A/B scripts (o49 o50 o52 o64 o83) use an older generated source as `--src2`. Recreate it with
  `git show <sha>:tp8v41.cu > tp8v41_<x>.cu` (h2old = 3d0a801, ee6b = ee6b585, cf55 = cf554b0, e066 = e066ec9).

## Build / run
```
bash gen.sh                                   # port_v41.py + tail patch -> tp8v41.cu (extensions JIT-build per M on first use)
JOB=<held job> ./ctr.sh <log> 'python v41tp8.py --arms tp8 --skew 0 --oneshot-maxm 32 --proto b --cold-iters 25 \
    --ts 1 2 4 8 16 32 48 64 --check 1 --routing dec'                     # M8..512, 6 decode-trace cases, checks + protocol b
# same-run A/B: add --flags2=-DTPB2_X=1 (or --src2 old.cu --flags2=), second rep with --swap 1
```
Suite headline (2 reps, rep2 `--order rev`), from `/home/scratch.aichenf_wwfo/tpmoe_suite` (see `s009.sh`):
```
SUITE_BUILD=/raid/.../suite TP8_COMMIT=<sha> python3 harness/bench.py --machine b300 --model dsv41f --tp 8 --arms ours,gc,mega \
  --routing real-trace:/home/scratch.aichenf_wwfo/routing_caps/dsv41_real --protocol b --ms 8,16,32,64,128,256,384,512 \
  --ours-adapter /home/scratch.aichenf_wwfo/tp8_v41_b300/suite_cell --out results_suite/<name>
```

## Correctness gate (run for every kept change; see o57 / o63 / o75 / o82)
```
python v41tp8.py --arms tp8 --skew 1 --oneshot-maxm 32 --proto b --cold-iters 5 --ts 1 2 4 8 16 32 48 64 --check 1 \
  --b2b 500 --b2b-mix 3500 --b2b-only 1 --routing dec --dec-cases 20:32      # repeat with 10:96 / 30:96
```
- **Checks:** ranks bitwise identical, deterministic, rel vs the exact reference.
- **`--b2b R`:** R x 24 eager calls cycling 3 inputs, NaN-poisoned outputs, rank skew. The skew is B300b's pattern plus random GPU sleeps.
  Every output is compared bitwise with its isolated call. 500 rounds = 12k calls per rank per M.
- **`--b2b-mix R`:** the same, with calls drawn over all M builds on one workspace.
- **Last gate (o82, HEAD defaults):** 0 bad of 96,000 per M; b2b-mix 0 bad of 672,000.
- **Suite harness gates:** rel_ex <= 2e-4, ranks_same, b2b and mixed-M stress ok.

## Headline h9 (commit 3668057; suite harness, real trace dsv41_real L10/20/30 x s32/96, protocol b, 2 reps; record 051ac2a)
| M | ours rep1 / rep2 (us) | gc/ours | mega/ours |
|---|---|---|---|
| 8 | 53.3 / 53.2 | 1.287 / 1.283 | 1.60 / 1.61 |
| 16 | 67.6 / 67.2 | 1.203 / 1.209 | 1.52 / 1.52 |
| 32 | 86.0 / 86.0 | 1.129 / 1.116 | 1.38 / 1.39 |
| 64 | 110.2 / 109.2 | 1.048 / 1.058 | 1.26 / 1.27 |
| 128 | 136.2 / 135.5 | 1.028 / 1.030 | 1.18 / 1.19 |
| 256 | 160.8 / 160.3 | 1.001 / 1.003 | 1.10 / 1.11 |
| 384 | 176.1 / 175.6 | 1.004 / 1.005 | 1.08 / 1.07 |
| 512 | 186.0 / 186.3 | 1.039 / 1.035 | 1.08 / 1.08 |

## Kept (same-run A/B, 2 swapped reps, dec 6 cases)
- LLS sentinel all-reduce (cf554b0): M256/384/512 -5..-9.5 us.
- **H2** (177e4ec, formal signaling 09039ca):
  - race fix d91f622: fixed HGN 32 gathered-h layout; epoch-tagged red.max ready tags per (panel, source rank, tile) instead of zeroed
    per-parity counters;
  - on by default for M >= 256 at 2625d29 (-2.8..-13.7 us vs base).
- T1PAR (10ddd15): the H2 tail T1 issues all top-k loads at once. -5.1..-5.9 us at M256-512. It also makes H2 a win at M128
  (-5.4/-5.8), so H2 now starts at M >= 128.
- GATHER 0 from M256 (3668057): -3.1/-4.0 us at M256.
- F2HParams (8bbb174): no change on B300; requested for B200 (it did not fix B200's non-H2 regression).

## Refuted (knobs left in the source, default off)
| knob | what it does | result (us) |
|---|---|---|
| SIG 4 | signal agent | removed: never compiled |
| GRL | global ready list | o39 slower; removed |
| T1SPREAD | spread T1 sends | neutral |
| F2RM | block-major FC2-H tickets | +6..+12.6 |
| PT1 | progressive T1 in warp 7 | +3..+7.5 |
| EPF | prologue L2 prefetch | +1.3..+3.7 |
| SFAPF | next-item SFA L2 prefetch | 0..+1.3 |
| SFAB | SFA copies by the B warp | +1..+4.4 at GATHER-1 M |
| EARLYW 3 | one more early stage | neutral or worse |
| NMAX 24 at M256/384 | smaller panels | neutral / +1.5 |
| F2H_SPROBE, F2H_TPROBE | probes | probes only |

Budget findings:
- FC1 streams at ~5 TB/s. It is limited by the ring's DRAM round trip: 5 x 16 KB in flight, capped by smem because the FP4 A operand
  is unpacked to 2x in smem. The MMAs and SF copies are not the limit, and L2 prefetch hurts in every form tried.
- The H2 tail is 13-17 us at M256/384 (NVLink floor ~3-4 us).

## Known caveats
- **F2H x ACTX is forbidden** (static_assert): the ACTX FC1 act path does not stage h for the sender. B300b saw NaN.
- **H2 is not used for DSV4-Flash (v4ft cells).** It was +4..+6 us there at M128-512 on B300 (suite_b300b branch b300tp8).
- **F2H assumptions:**
  - the per-CTA epochs (P.work) are shared by every build on a workspace and equal across ranks: same grid, every CTA runs every call;
  - the gathered-h buffer starts zero-filled (ready tags).
- **LLS** needs its LL region 0xFF-filled once per workspace; LLS and non-LLS builds must not share a workspace.
- **B200 regression:** the B200 agent sees non-H2 M32-128 +2..3.5 us on B200 since the F2H prototype commits; cause unknown.
  It is not the Params layout. Not visible on B300 (o64 / o83).
