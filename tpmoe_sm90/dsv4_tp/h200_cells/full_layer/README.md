# H200 full-layer TP vs EP on real inputs

Full MoE layer per call. TP-full is router GEMM (M tokens, fp32 logits) + one TP kernel: fused gate, per-token quant, routed
experts, sliced shared expert and in-kernel all-reduce. EP-full arms do the same layer on T = M/8 (TP8) or M/4 (TP4) tokens per rank:
- gcfull: router GEMM + Triton gate + quant + gc-moe with its own fused shared expert (gc-moe's SE, so not affected by the
  _scaled_mm issue below).
- fimxfull: router GEMM + gate + quant + flashinfer mxfp4_mega_moe + an unsharded shared expert on every rank. The SE is
  flashinfer.gemm.fp8_blockscale_gemm_sm90 x2 + fused Triton SwiGLU-clamp + per-128 FP8 quant, then y += SE.
- fimxfull2: the same, with the SE on a side stream overlapping MegaMoE (sglang dual-stream pattern).

Everything is in one CUDA graph per call. Protocol b: 512 MiB L2 flush, gate all-reduce + sleep, per-rank median, max over ranks.
Inputs are real steady x (routing_caps/dsv4f_real or dsv41_real, case 3 = decode step 36, case 11 = step 100). The router and
shared-expert weights are the layer-20 checkpoint extracts (v41_h20/l20_real_cl.pt, l20_v4f.pt), and the gate runs on our own router
output (--gate-input ours).

Bench: H20-TP8's v41_bench.py (fimega_full drop, sha256 d37920de44768685 = full/ref/h20_v41_bench_snap_1929b.py) +
full/patch_full2.py, which only sets env paths and adds the b2b8 stress (8 back-to-back eager calls, B2B_ROUNDS rounds, bitwise vs
the arm's own first output, ranks compared).

## Shared-expert GEMM on SM90 (found here, 11-09)

torch._scaled_mm blockwise (cuBLASLt, 1x128 act / 128x128 weight scales, torch 2.11 cu130) is **silently wrong when K/128 % 4 != 0**:
- K/128 = 12, 16, 20, 24: rel 1.7e-3.
- K/128 = 17, 18, 22: rel 0.40-0.57, with no error raised.

The DSV4.1 SE FC2 has K = 2304 (K/128 = 18), so it is affected. Repro: ub/scaled_mm_k2304.py, runs/smm_k2304.log.

fp8_blockscale_gemm_sm90 is correct for all K and faster at T = 1..64 per rank (1 call per graph, L2 flushed):

| GEMM | fp8_blockscale_gemm_sm90 (us) | cuBLASLt (us) |
|---|---|---|
| v41 FC1 [4608 x 5120] | 14.0-16.1 | 19.5-20.0 |
| v41 FC2 [5120 x 2304] | 9.6-10.3 | 12.4-12.7 (wrong) |
| Flash FC1 | 11.6-14.0 | 16.9-17.0 |
| Flash FC2 | 8.6-9.2 | 11.6-11.8 |

Source: ub/se_gemm_bench.py, runs/se_gemm_bench2.log. Both arms sit at 1-1.7 TB/s, well below HBM.

## Node incident

viking-prod-322 (hold 5757216) stopped responding between 19:30 and 19:34 PST on 11-09, during the SUITE agent's lock window
(its matrix_h200_s3 "r1_ours" step started 19:30:26). Symptoms: no ping; srun failed with "Communication connection failure"; Slurm
marked the node DOWN+NOT_RESPONDING ("Not responding", 19:45:34) and requeued the job. I scancelled 5757216 at 19:46:34. The cause is
unknown (no host access). The replacement hold is 5776731, and chain_fr.sh redoes setup, weights, builds and runs there.

GPU 6 of 322 ran hot all session: 55-81 C at run start (the nvidia-smi line in every log header) while the other seven sat at
29-44 C. The same is visible in the morning c1_* logs. In the fl chain, GPU 6 never set the reported time: across 256 arm x M
points the slowest rank (the reported max over per-rank medians) was rank 0 176 times, rank 1 79 times and rank 2 once. Rank 6
was a median 0.22% (max 1.45%) above the fastest rank, and the per-rank spread was a median 1.6% (p90 3.3%). So the earlier 322
numbers are not inflated by GPU 6. Still, 322 must not be used for measurements again. If it is ever reallocated, check GPU 6
temps and ECC first.
The h123full M384 b2b8 miss ran at 19:23:11, about 8 min before the node degraded.

## TP-full correctness after the race fix (provisional mark dropped 11-10)

H20-TP8 reproduced a real SM90 race in the fused-shared-expert TP builds (v78h21_sh at M384/512, about 1 in 1,000-1,600
back-to-back calls, identical wrong rows on every rank). The fix is one named_bar_sync(1, NCONS) after STAMP(P,2), behind
TP2_B2B_FIX (default 1). On H200 it is verified in results/h200/b2b_fix/README.md: the suite cells pass with the fix ON, and with it
OFF the race also shows on routed-only TP8.

The fr chain below runs every TP-full build with the fix and 200 b2b8 rounds. tp41full, h123full (TP8) and h123full (TP4) each
passed 51,200 back-to-back eager calls bitwise (all ranks identical, 0 failures), so the TP-full numbers are no longer provisional.
The fl chain's TP-full numbers (unfixed builds, on 322) are superseded by fr.

## Results so far

- table_fl.md: the fl chain on 322, run just before the node died. TP-full vs gcfull (+ routed-only TP vs gc), DSV4.1 TP8 and
  DSV4-Flash TP8, cases 3/11 x 2 reps.
  - Correctness: TP-full outputs are bitwise across ranks and repeats with no non-finites, and gcfull routing rows are 100% equal to
    the TP fused gate.
  - Harness bug: the b2b8 flag for the routed-only arms (tp41, h123) in that chain compared against another arm's output, so ignore
    it; fixed in patch_full2.
  - One real b2b8 miss: h123full at M384, case 11 rep 1. One call out of 8 was not bitwise vs its own first output; the other 3 runs
    at M384 passed. Handed to H20-TP8 to reproduce on H20, where it is the production kernel.
- **table_fr.md: the final full-layer result**, run on viking-cr-196 (the suite's hold 5775142, shared via the h200_196 lock) on
  11-10 00:49-01:01, with fixed builds and the H20-TP8 bench. Arms: TP-full vs gcfull / fimxfull / fimxfull2, cases 3/11 x 2 reps.
  Best EP-full / TP-full, M = 8 .. 512:

  | cell | ratios, M = 8 .. 512 | mean | min..max |
  |---|---|---|---|
  | DSV4.1 TP8 | 1.36 1.24 1.32 1.36 1.40 1.27 1.19 1.11 | 1.28 | 1.11..1.40 |
  | DSV4-Flash TP8 | 1.31 1.24 1.19 1.27 1.44 1.29 1.28 1.20 | 1.28 | 1.19..1.44 |
  | DSV4-Flash TP4 | 1.26 1.35 1.19 1.24 1.16 1.21 1.26 1.25 | 1.24 | 1.16..1.35 |

  - Best EP arm: gcfull at small M (TP8 M <= 32 for DSV4.1, M <= 64 for Flash; TP4 M <= 256). fimxfull / fimxfull2 at large M.
  - fimxfull2 (SE on a side stream) is 0-2% faster than fimxfull and bitwise identical to it.
  - Correctness, all runs:
    - TP-full: repeat bitwise, ranks identical, b2b8 51,200 calls bitwise, 0 non-finite.
    - gcfull, fimxfull, fimxfull2: deterministic. Their routing rows are 100% equal to the TP fused gate.
    - cross rel_rms: TP-full vs EP 0.062-0.066; gcfull vs fimxfull 0.035.
    - The routed-only h123 arm (included by the bench) has cos 0.99967-0.99969 vs the fp32 quant reference.
  - TP4 weights: v41_prep.py --model flash --ndev 4 (H20-TP8's method). Its content sha (7b1c35a9..) equals my
    prep_humming_synth output but not H20-TP8's printed sha (1ebdedfe..), most likely a hash-method difference. The values agree with
    the EP weights (h123 TP4 cos 0.99967, cross rel 0.063).
  - Node: 8x H200 on viking-cr-196: 34-48 C idle, 0 ECC, NV18 full mesh, p2p all OK. torch SymmetricMemory printed
    init_multicast_for_block warnings (multicast unavailable); the runs were unaffected.
