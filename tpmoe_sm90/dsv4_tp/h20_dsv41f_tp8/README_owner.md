# H20 · DSV4.1-Flash · TP8 — tp41 (agent H20-TP8)

Shape: H 5120, I 2304 (288 per rank = 2 x 128 + 32), E 384, top-6, MXFP4 routed experts, per-token E4M3 x (in-kernel), SwiGLU clamp 10.
Machine: 8 x H20-3e, computelab viking-prod-583 (h20-3e@cr+mp), torch 2.11 cu130 (cmp3env container).

## Kernel / build

- Source: owner repo .24 `/raid/sharedspace/fac/aichenf_v41` (git; mirror computelab `/home/scratch.aichenf_wwfo/v41_h20`).
  `tp8v41.cu` = h123 (H20 TP8 DSV4-Flash, 70dbda3) ported by `port_v41.py` (TP2_MODEL 1) + phase-2 commits.
- Kept routed-path changes (each with >= 2 agreeing 8-GPU reps): T4 split tail (236aa07; the 32 + 32-row FC1 tail item of a panel
  <= 8 tokens as one 64-row block split 4 ways over K), FC2 tail k-tile as ONE pipelined k32 slice (a40b00b, bitwise-same output),
  T4 for 9..16-token panels (9902802; 8 KB x tiles / 6 ring stages -> M-dispatched: that build at M 9..16 and M >= 320,
  `-DTP2_T4N16=0` elsewhere).  SV (full layer): fused 384-expert gate (60ed3a8, rolled 3556e2a), SH32 fused shared expert with the
  checkpoint's 32x32 UE8M0 FP8 scales (6f442aa..bbf097d; SH_DEFER 0).
- Build: `tp8v41.build('tp8v41.cu', model=1, ndev=8, shared=0|1)` (torch cpp_extension, `-O3 -gencode=arch=compute_90a,code=sm_90a
  -DTP2_MODEL=1 -DTP2_NDEV=8 -DTP2_SHARED=0|1`).  Suite adapter: `v41_h20/suite_adapter` (builds both dispatch variants).
- Flash build (TP2_MODEL 0) stays bitwise == the served h123 v78h21 .so (checked after every kernel commit).

## Results (protocol b: 1 call per graph, 512 MiB L2 flush, gate all-reduce + 100 us sleep, per-rank median -> max over ranks)

- `table_real_headline_suite.md` — HEADLINE, suite harness (tpmoe_suite 00ab127, `--routing real-trace`: DSV4.1 plain-decode trace,
  layers 10/20/30 x steps 32/96, random x, weights / 1.5; ours / gc-moe 7d0825e / flashinfer MegaMoE mxfp4; 2 reps fwd / rev).
  ours 80.8 .. 484 us at M 8..512; best baseline / ours 1.20 .. 1.37.  (Before the T4N16 dispatch; with it: M16 111.6, M384 432.6,
  M512 480.0 -- `runs/suite_disp`.)
- `table_real_headline_own.txt` — the same headline with the owner harness (v41_bench.py, tp41 / gc / PR mx), 1.19 .. 1.36x.
- `table_full_layer_real.md` — FULL LAYER on real inputs: real L20 steady decode x (steps 36, 100), checkpoint L20 router weight
  (GEMM timed) + L20 shared expert; the gate is fed the CAPTURED router logits (the captured x does not reproduce them: rel 2.4e-2 ->
  77-80 % of id rows; captured logits -> ids exact), same for both arms.  tp41full vs gcfull 1.35 .. 2.02x.
- `table_uniform_dspark_proxy.txt` — secondary: uniform routing, the DSpark verify-batch L20 trace (cap_v41_l20.pt) and the
  DSV4-Flash -> 384 proxy (3 seeds).  Under uniform routing tp41 leads only at T <= 4 (0.76x best EP at T64).
- Correctness (every run): tp41 8 ranks bitwise identical, bitwise repeatable, finite; cos 0.99968..0.99970 vs the fp32 MoE with
  per-token-quantized x (rel 0.025 = the E4M3 activation quantization every arm has); suite real-x check rel_q 0.025 / cos_dense 0.9990.

## Commands

- Suite: `cd tpmoe_suite && SUITE_BUILD=<node-local> python3 harness/bench.py --machine h20 --model dsv41f --tp 8 --arms ours,gc,fimega
  --routing real-trace:/home/scratch.aichenf_wwfo/routing_caps/dsv41_real --protocol b --ms 8,16,32,64,128,256,384,512
  --ours-adapter /home/scratch.aichenf_wwfo/v41_h20/suite_adapter --order fwd|rev --out DIR`
- Owner harness: `v41_h20/chain_trace.sh` (headline), `chain_fullreal.sh` (full layer, real inputs), `chain_rr*.sh`, `chain_ab*.sh`
  (8-GPU A/B of builds); 1-GPU: `v41_test1.py` (correctness vs the kernel-numerics emulation + h123 bitwise), `v41_time1.py`
  (rank-local timing, stamps, FC2 unit timeline, prologue sub-stamps).

## Refuted (numbers in the owner repo commit log)

FC2: wide units (back-to-back and fused 3-k-tile, + 4-accumulator), any `__noinline__` split, separate-FC2-kernel (FC2-only build:
wide units no faster even without spills; gain <= 20 us at M384), non-pipelined 1-slice tail, w2 pad-slice trim (neutral, kept),
KLOOP1, producer B-copy removal; register budget 112/32; knob sweeps (EPI_SCALE, STAGGER, PASSA, DEFER, PART_OFFLOAD, DENSE_MAXM).
SV: 2-bank slice pipelining, scale register prefetch, shared cost knobs.

## Updates (11-09 10:20 PST)

- Full layer on real inputs, rep 2 (viking-prod-586, dispatching tp41 build, `runs/f586_c*_r2`): tp41full / gcfull 1.37 .. 2.01x
  (T1 97.3 vs 168.8, T8 234.4 vs 426.9, T48 481.1 vs 796.2, T64 544.5 vs 1068.4 us); rep 1 (583) in `table_full_layer_real.md`.
- T4N16 8-GPU A/B (`runs/ab_t4n16_*`): M16 -5..-9 us, M384 -1..-4, M512 -3..-15; M8..M256 +1..+3.6 (NST 6) -> M-dispatched.
- Rolled fused gate 8-GPU A/B in the full layer (`runs/gab_*`): neutral (+-0.6 us; the router GEMM + PDL hide the gate).
- Suite adapter: `v41_h20/suite_adapter`, src at the commit in `src/COMMIT`, dispatch T4N16 build at M 9..16 / >= 320.

## Updates (11-09 / 11-10, viking-prod-586)

### Correctness: a race in the FC1-to-FC2 staging smem, now fixed (TP2_B2B_FIX, owner commit 872c14c, adapter src 32bb910)

What the race is:

- The last FC1 epilogue copies its act tile (staging +16384..+25088) to global act without a consumer barrier.
- The first two FC2 out tiles (staging + bo*16896) skip the freeo wait and overlap that act tile.
- A fast warpgroup can therefore overwrite the tile while a slow warp is still copying it.
- The result is 1-5 corrupted token rows on one rank, which the all-reduce spreads to every rank.

The fix is one `named_bar_sync(1, NCONS)` after `STAMP(P, 2)`.

Stress results (`v41_h20/b2b_stress2.py`, real L20 steady x, cases 3/11, 12k calls per mode: b2b / clone / per-rank skew / graph):

| Kernel | Unfixed | Fixed |
|---|---|---|
| tp8v41 SV + SO, M8..512 | 71 bad calls in 576k (M256..512) | 0 in 576k |
| h123 sh, M384/512 | 12 in 384k | 0 in 384k |
| served TP4 v26 sh, M ≤ 32 | 0 in 2.0M | 0 in 2.0M |
| served TP4 v26 ns + separate shared, M48..512 | 0 in 768k | 0 in 768k |
| v94ex TP4, M8..512 | 0 in 2.0M | 0 in 2.0M |
| TP8 h123 sh, M ≤ 32 | 0 in 2.0M | 0 in 2.0M |

- Fixed and unfixed outputs are bitwise equal (148/148 references).
- Cost of the fix, protocol b: tp41full +0.20%, h123 sh +0.09%, v26 +0.03% (summed over M).
- Mixed-M / input-cycling / skew bursts (builds sharing one workspace, 24k calls per target): 0 bad calls, fixed and unfixed. At this call count the pattern did not reproduce the known race in the unfixed control.
- **Every tp41 / h123full number in this cell measured before 872c14c used the racy builds.** The timing is unaffected (+0.2%). The outputs carry an error rate of about 1e-4 per call at M ≥ 256.

### Full layer on real inputs: TP vs the best EP (fr3, b2b-fixed TP builds; tables in `runs/fr3_{41,f8,f4}_table.md`)

Setup:

- Inputs: checkpoint L20 gate, bias and shared weights; real steady x; cases 3/11 × 2 reps; protocol b.
- EP arms:
  - gcfull: gc-moe fused shared expert.
  - fimxfull: flashinfer `mxfp4_mega_moe` plus a separate shared expert, `flashinfer.gemm.fp8_blockscale_gemm_sm90` ×2 with a fused SwiGLU+quant kernel.
  - fimxfull2: the same with the shared expert on a side stream.
- Code: `v41_h20/fimega_full/`.

DSV4.1 TP8 tp41full vs best EP (µs):

| M | TP | best EP | ratio |
|---|---|---|---|
| 8 | 94.8 | 168.6 (gcfull) | 1.78 |
| 16 | 130.4 | 193.6 (gcfull) | 1.48 |
| 32 | 170.5 | 246.3 (fimxfull2) | 1.44 |
| 64 | 231.5 | 284.2 | 1.23 |
| 128 | 309.6 | 391.0 | 1.26 |
| 256 | 409.1 | 552.1 | 1.35 |
| 384 | 477.5 | 588.4 | 1.23 |
| 512 | 541.0 | 630.9 | 1.17 |

DSV4-Flash TP8 h123full (fixed sh builds) vs best EP (µs):

| M | TP | best EP | ratio |
|---|---|---|---|
| 8 | 72.3 | 133.8 | 1.85 |
| 16 | 90.6 | 156.9 | 1.73 |
| 32 | 112.3 | 177.6 | 1.58 |
| 64 | 146.7 | 228.0 | 1.55 |
| 128 | 175.0 | 263.9 | 1.51 |
| 256 | 216.7 | 296.5 | 1.37 |
| 384 | 252.1 | 340.9 | 1.35 |

DSV4-Flash TP4 v94ex SV (fixed) vs EP4 (µs; gcfull is the best EP at every M):

| M | TP | best EP | ratio |
|---|---|---|---|
| 8 | 95.4 | 156.8 | 1.64 |
| 16 | 137.4 | 200.9 | 1.46 |
| 32 | 180.1 | 225.1 | 1.25 |
| 64 | 244.4 | 288.4 | 1.18 |
| 128 | 293.3 | 345.4 | 1.18 |
| 256 | 363.6 | 426.7 | 1.17 |
| 384 | 421.1 | 520.3 | 1.24 |

Correctness checks:

- TP is bitwise repeatable with ranks identical.
- EP is repeatable.
- EP routing rows equal the TP gate (1.0).
- Cross-arm rel_rms: 0.062-0.065 TP vs EP, 0.035 gcfull vs fimxfull.
