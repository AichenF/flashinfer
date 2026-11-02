# tpmoe_b2 (B200 TP4, DSV4-Flash) -- status v51 (11-01)

Defaults unchanged from v50 (mean A/S ~1.08-1.10 over M 1..128, bitwise except M128's 110 known routed elements).

## Tried this round (all bitwise; none faster -> off or opt-in)
| idea | result |
|---|---|
| CSPLIT (2-CTA cluster K-halves, DSMEM merge) M1-only dispatch | M1 -1.8% (noise level), M2 +3..10%, M4/8 ~0 -> not worth two builds |
| 64-row FC1 items (M64 MMA) for 2x parallelism at M1/M2 | block-scaled 1-CTA tcgen05 MMA is M128 only (illegal instruction) |
| swapped roles (tokens = A padded to 128, weights = B N256) | math bound: 128 clk/MMA = 64 clk per 128 weight rows vs 42 clk marginal now |
| in-kernel dsv3 router M<=4 (exact sglang dsv3_router_gemm order) | logits bitwise, but +0.5..1.4 us: grid-wide logits exchange > separate kernel + PDL |
| T1LL (LL-tagged FC2 outputs, T1 polls, no counts / fences) at M1 | +0.6 us: poll detection 1.7-3 us, cold sum, larger AR rank skew |

## Limits (measured)
- Large M: FC1 stage needs ~10 KB smem traffic per M128xK32 MMA-equivalent (unpacked e2m1 A written by TMA + read by MMA, B both ways);
  at 7 TB/s HBM that is ~95% of the 128 B/clk smem port -> FC1 runs ~5.5 TB/s. fp4 padding is inherent to kind::mxf8f6f4 (and to
  tcgen05.cp b4x16_p64), so neither TMEM-A nor swapped roles reduce it.
- Small M: every cross-SM handoff (partial merge, h publish, FC2 ready, T1 count, AR) costs 1-2 us whichever protocol (atomics, LL polls,
  DSMEM needs clusters). M1 chain: prologue 7 us, FC1 stream 4.3 + epilogue chain 6, FC2 4.6, T1 + AR 5.
- 1.4x mean is not reachable bitwise; remaining levers are numerics changes (user decision) or a cluster-wide FC1->FC2 rewrite.

## v53 (11-01 evening, dev git 66b8f3d) -- full run k228 (umb-b200-238, 8 rounds)
| M | 1 | 2 | 4 | 6 | 8 | 16 | 20 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 33.3 | 36.7 | 45.9 | 51.8 | 55.5 | 64.5 | 71.8 | 73.3 | 78.0 | 89.8 | 98.6 | 114.8 | 130.1 |
| S | 36.6 | 37.8 | 40.6 | 45.5 | 49.8 | 57.5 | 63.3 | 63.1 | 69.3 | 79.6 | 86.6 | 102.5 | 115.6 |
| A/S | 0.91 | 0.97 | 1.13 | 1.14 | 1.11 | 1.12 | 1.13 | 1.16 | 1.13 | 1.13 | 1.14 | 1.12 | 1.13 |
Mean A/S 1.10; S == A bitwise except M128 (110 routed elements). New default: FC2STATIC at M4 (paired -4.7%).
Also measured (off): IDPRE, OUTPAD ~0; EPIFIRST +1.5..5%; T1L at M>=48 2x slower; EARLYTRIG ~0; WARM_WAIT=2 +4% at M1.
Roofline (b2_roofline.py): per-rank bytes / 6.9 TB/s = 3.8 (M1) .. 63 us (M128); KS reaches ~58% of it at M64-128.

## v54 (11-01, dev git 11c2cf2) -- full run k234 (A / S / SF, 8 rounds)
| M | 1 | 2 | 4 | 6 | 8 | 16 | 20 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 33.3 | 38.5 | 46.8 | 52.5 | 56.2 | 64.1 | 71.0 | 73.2 | 77.4 | 89.5 | 98.9 | 116.0 | 129.2 |
| S (production router + kernel) | 37.1 | 38.2 | 40.5 | 48.2 | 50.8 | 57.3 | 61.8 | 64.2 | 69.6 | 78.6 | 87.5 | 102.9 | 115.2 |
| SF (in-kernel router M5-64) | - | - | - | 47.8 | 50.5 | 56.6 | 61.7 | 64.1 | 69.2 | 78.4 | 87.8 | - | - |
Mean A/S 1.100; with SF at M5-64 (fused router, bitwise vs this cuBLAS split table) 1.103. Run-to-run noise per M +-3..5%.
In-kernel router now covers M5-128 (x sub-parts double-buffered; M128 MMA for R > 64) -- slower at M96/128 (512 KB of x per unit).

## v55 (dev git a352fe5) -- full run k239 (A / S / SF, 10 rounds)
| M | 1 | 2 | 4 | 6 | 8 | 16 | 20 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 32.7 | 36.0 | 45.4 | 50.6 | 54.2 | 64.5 | 70.6 | 74.1 | 77.2 | 89.1 | 98.1 | 115.1 | 130.1 |
| S | 35.8 | 37.8 | 40.2 | 46.3 | 49.4 | 58.1 | 62.1 | 63.6 | 69.3 | 79.9 | 88.1 | 102.7 | 115.4 |
| SF | - | - | - | 47.7 | 48.8 | 57.0 | 61.2 | 63.0 | 69.1 | 78.0 | 86.6 | - | - |
Mean A/S 1.091; SF rule (M5-64) 1.098. Between-run spread of single M values +-3..5% (M2 A/S 0.95 here, 1.01 in k234).
v55 adds LLRR at M1 (paired -2.0%). Tried and off this round: FC2L (M1 local T1) +14%, T1L at M>=48 2x slower, EPIFIRST +1.5..5%,
FC2S_SHF worse, IDPRE / OUTPAD / EARLYTRIG ~0, in-kernel router at M96/128 slower.

## v56 (dev git 9ed105d) -- full run k245 (A / S / SF, 10 rounds)
| M | 1 | 2 | 4 | 6 | 8 | 16 | 20 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 32.8 | 36.4 | 46.8 | 50.9 | 55.3 | 64.9 | 70.1 | 72.9 | 77.5 | 90.4 | 97.8 | 115.1 | 129.3 |
| S | 35.3 | 38.7 | 41.5 | 46.5 | 49.6 | 56.8 | 62.3 | 63.8 | 68.2 | 78.1 | 87.8 | 103.0 | 114.1 |
| A/S | 0.93 | 0.94 | 1.13 | 1.09 | 1.12 | 1.14 | 1.13 | 1.14 | 1.14 | 1.16 | 1.11 | 1.12 | 1.13 |
Mean A/S 1.098 (SF this run within noise of S). v56 = v55 + TABW at M1 (paired -1.2..-2.5%). Also off: PGATE1 at M1 (~0),
SMALLM 4 (M6/M8 +0..4%). M1 kernel alone (KS ~33 us) now equals A; the remaining M1 gap is the production router (~2.5 us).

## v57 CORRECTION (dev git 9005566) -- shared quarters were not bitwise
`b2_layer_ab.py --xoff N` (other capture rows) showed ~1100 mismatches (rel 2e-3) at M1/M2 for xoff 37: the shared FC1 K-quarters
(SHQ=4 at M < 4) sum partials, DeepGEMM runs one K chain. The earlier "bitwise at M1/M2" held only for the default rows. v57 runs
whole shared items at every M (SHW1=1): 0 mismatches over 8 offsets x M1-3; costs M1 ~+10%, M2 ~+5%. Routed K-halves at M1: 0/24.
Large M: ~110 routed elements can differ (M64/96/128 at xoff 37, M128 at 0/211) -- production's tile configs, routing dependent.
| M | 1 | 2 | 4 | 6 | 8 | 16 | 20 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 32.8 | 38.7 | 46.4 | 49.8 | 55.2 | 64.3 | 71.4 | 72.8 | 76.6 | 89.2 | 98.8 | 114.0 | 130.0 |
| S | 37.5 | 40.8 | 42.9 | 47.1 | 49.2 | 58.5 | 61.8 | 64.1 | 70.0 | 78.2 | 86.6 | 103.6 | 114.9 |
| A/S | 0.87 | 0.95 | 1.08 | 1.06 | 1.12 | 1.10 | 1.16 | 1.14 | 1.09 | 1.14 | 1.14 | 1.10 | 1.13 |
Mean A/S 1.083 (k250, bitwise-robust config). Run-to-run noise per M +-3..5%.

## v58 (dev git ce8925d) -- SHGU packed at M1; k255 (A / S, 8 rounds); bitwise at xoff 0 / 37 / 99 / 211 (M1, M2, M4)
| M | 1 | 2 | 4 | 6 | 8 | 16 | 20 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 33.6 | 36.8 | 47.2 | 51.3 | 55.1 | 64.2 | 71.1 | 72.1 | 77.0 | 89.6 | 100.0 | 114.3 | 130.0 |
| S | 36.0 | 41.0 | 42.0 | 47.9 | 50.1 | 58.7 | 62.5 | 64.3 | 69.6 | 80.0 | 88.4 | 103.4 | 116.1 |
| A/S | 0.93 | 0.90 | 1.12 | 1.07 | 1.10 | 1.09 | 1.14 | 1.12 | 1.11 | 1.12 | 1.13 | 1.11 | 1.12 |
Mean A/S 1.082 (bitwise-robust; M2 pays the whole-shared-item correction ~5%, SHGU neutral there).

## v59 (dev git 22768fa) -- BITWISE == production at every M (validated by sweeps)
`b2_layer_ab.py --sweep N` compares production vs ours over N consecutive capture offsets per M.
- Root cause of the long-known "110 routed mismatches" (M128, and rarely every M): our routed SwiGLU used u * (g / (1 + __expf(-g)))
  (IEEE division); trtllm-gen computes g * rcp(1 + exp(-g)). TPB2_SILU=1 (u * __fdividef(g, 1 + __expf(-g))): M128 40/40 -> 0/40.
- Routed K-halves at M1 (RRSPLIT) were not trtllm-gen's chain (1/200 offsets) -> RRSPLIT=1 (whole items; ~0..1.7% at M1).
- Shared K-quarters at M<4 (v57 fix SHW1=1) + SHGU packed at M1 (gate / up chains on two CTAs, each one K chain).
Sweeps (0 mismatching offsets): M1 300, M2 300, M4/6/8 200, M16/20/24/32/48 60, M64/96/128 40.
Timing node umb-b200-239 (job 5496490; A/S ~3% below node -238): k268 mean A/S 1.062, k267 1.066 (13/13 M bitwise).
| M | 1 | 2 | 4 | 6 | 8 | 16 | 20 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A/S k268 | 0.97 | 0.89 | 1.05 | 1.00 | 1.05 | 1.12 | 1.11 | 1.14 | 1.08 | 1.12 | 1.11 | 1.09 | 1.09 |

## Final validation (v59+, dev git d2b6f99)
Bitwise sweeps (production layer vs ours, consecutive capture offsets, 4 ranks): 0 mismatching offsets at
M1/M2 x 300, M4/6/8 x 200, M16/20/24/32/48 x 200, M64/96/128 x 100.
Untraced phase costs (PROBE_END, KS us full / after FC2 / after FC1): M8 47.1/38.8/30.3, M32 64.7/56.7/46.6, M64 83.4/72.3/60.9,
M128 110.5/96.4/81.5. Tail (T1 + all-reduce) = ~5 serial ~1 us latencies (visibility fence, count poll, reload, NVLink, receive poll).
Mean A/S ~1.06 (node -239) .. 1.08 (node -238). Remaining structural option (cluster FC1 -> FC2 for M1/M2) estimated at ~-2 us M1.

## v61 (dev git f0bbd6e) -- k287 on node -239 (A / S / SF, 10 rounds), bitwise 13/13 (+ sweeps M3-6 x 100, M2/64/128 x 60)
| M | 1 | 2 | 4 | 6 | 8 | 16 | 20 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 32.4 | 36.6 | 44.7 | 50.1 | 55.3 | 63.2 | 71.3 | 73.4 | 77.2 | 90.0 | 99.4 | 114.5 | 129.0 |
| S | 36.5 | 38.1 | 41.5 | 48.6 | 51.3 | 58.5 | 63.3 | 64.5 | 69.9 | 79.2 | 89.6 | 104.9 | 117.4 |
| SF | - | - | - | 46.0 | 50.3 | 57.4 | 62.5 | 63.5 | 70.3 | 79.5 | 88.5 | - | - |
Mean A/S 1.071 (S), 1.081 (SF at M5-64). v60/v61 adds: SHGU at M2-6 (-1.7..-2.6%), FC2 unit cap 4 at M>=64 (~-0.7%).

## v62 (dev git b62254d) -- k293 on node -239, bitwise 13/13 (+ sweeps M1/2/6/8 x 60)
| M | 1 | 2 | 4 | 6 | 8 | 16 | 20 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 34.2 | 36.0 | 44.8 | 50.2 | 55.4 | 62.9 | 71.0 | 72.9 | 76.3 | 89.9 | 99.4 | 114.8 | 128.9 |
| S | 35.2 | 37.6 | 41.1 | 47.3 | 51.3 | 58.7 | 62.6 | 64.5 | 70.0 | 78.8 | 88.0 | 104.8 | 117.5 |
| A/S | 0.97 | 0.96 | 1.09 | 1.06 | 1.08 | 1.07 | 1.13 | 1.13 | 1.09 | 1.14 | 1.13 | 1.10 | 1.10 |
Mean A/S 1.081 (S), 1.082 (SF rule). v62 = static FC2 order at M1-8.

## v64 (dev git b882ae5) -- average of 3 full runs (k298, k299_fa, k299_fb) on node -239, all bitwise
| M | 1 | 2 | 4 | 6 | 8 | 16 | 20 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 32.7 | 36.9 | 45.6 | 50.9 | 55.6 | 64.4 | 70.2 | 71.9 | 76.8 | 89.9 | 99.3 | 114.7 | 130.0 |
| S | 36.2 | 38.5 | 40.7 | 47.8 | 50.6 | 58.9 | 63.9 | 64.1 | 70.5 | 81.5 | 89.4 | 104.9 | 116.9 |
| SF | - | - | - | 47.2 | 50.3 | 57.2 | 62.8 | 64.0 | 69.5 | 79.6 | 88.7 | - | - |
| A/S | 0.91 | 0.96 | 1.12 | 1.07 | 1.10 | 1.09 | 1.10 | 1.12 | 1.09 | 1.10 | 1.11 | 1.09 | 1.11 |
Mean A/S 1.075 (S), 1.084 (SF at M5-64). v63/v64: T1L and TABW off at M1 (the small-M restructuring flipped their sign).

## v65 (dev git 06a2ed4) -- average of 2 full runs (k303_fa/fb) on node -239, all bitwise (+ sweeps M1/2/4/8 x 60)
| M | 1 | 2 | 4 | 6 | 8 | 16 | 20 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A/S | 0.93 | 0.93 | 1.14 | 1.08 | 1.09 | 1.10 | 1.11 | 1.11 | 1.11 | 1.12 | 1.11 | 1.10 | 1.11 |
Mean A/S 1.081 (S), 1.087 (SF at M5-64). v65 = v64 + SHEARLY off at M<=2 + FC2PAIR off at M<=8.

## v66 (dev git 6cd4c53) -- M-dispatched builds; average of 2 full runs (k308_fa/fb), node -239, all bitwise
Build 1 = default (ROLL act); build 2 = -DTPB2_ACTX=1 (4-token act rounds), used for M >= 32. SF = in-kernel router (M5-64).
| M | 1 | 2 | 4 | 6 | 8 | 16 | 20 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 32.1 | 35.7 | 47.2 | 51.0 | 55.1 | 64.3 | 70.5 | 72.5 | 76.2 | 89.0 | 97.7 | 114.2 | 129.6 |
| best ours | 35.7 | 38.4 | 40.7 | 47.1 | 49.6 | 56.8 | 61.9 | 62.9 | 66.4 | 77.7 | 85.9 | 103.3 | 114.3 |
| A/best | 0.90 | 0.93 | 1.16 | 1.08 | 1.11 | 1.13 | 1.14 | 1.15 | 1.15 | 1.15 | 1.14 | 1.11 | 1.13 |
Mean A/S: single build 1.081; dispatched builds 1.088; dispatched + SF at M5-64 1.098.

## 3-build dispatch estimate (k310 for M<=24, k308 for M>=32; node -239; all bitwise)
Builds: GUNR (-DTPB2_GUNR=1) for M <= 8, default for M 16-24, ACTX (-DTPB2_ACTX=1) for M >= 32; in-kernel router (SF) at M5-64.
| M | 1 | 2 | 4 | 6 | 8 | 16 | 20 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A/best | 0.92 | 0.96 | 1.14 | 1.11 | 1.11 | 1.12 | 1.14 | 1.13 | 1.15 | 1.15 | 1.14 | 1.11 | 1.13 |
Mean ~1.099. (Arm-to-arm noise: identical configs SF2 == S2 at M1 measured 33.9 vs 35.0 us.)
Dispatch builds validated bitwise: GUNR build sweeps M1/2/4/6/8 x 60, ACTX build sweeps M32/48/64/96/128 x 40 -> 0 mismatches (k313).
Per-build extras tested and rejected (k311/k312): LEANMMA=3, FC1PAIR, FC2RLX, OUTL2 (large-M build); XRLX, GWARM, HPUB (small-M build).

## Dispatched layer, measured in one harness (arms D / DF), average of k316 + k317 (node -239), D == A bitwise 26/26
`--flags2=-DTPB2_ACTX=1 --flags3=-DTPB2_GUNR=1 --dispatch 8:3,24:1,512:2` (GUNR build M<=8, default M16-24, ACTX build M>=32);
DF = D + in-kernel router at M5-64.
| M | 1 | 2 | 4 | 6 | 8 | 16 | 20 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 32.2 | 34.9 | 45.9 | 51.3 | 55.3 | 63.9 | 70.8 | 71.8 | 76.3 | 89.3 | 99.0 | 116.1 | 129.3 |
| DF | 34.5 | 36.5 | 39.1 | 45.8 | 49.3 | 56.8 | 62.1 | 63.2 | 65.9 | 79.2 | 86.2 | 103.0 | 114.8 |
| A/DF | 0.94 | 0.96 | 1.17 | 1.12 | 1.12 | 1.13 | 1.14 | 1.14 | 1.16 | 1.13 | 1.15 | 1.13 | 1.13 |
Mean A/S 1.078 (single build), A/D 1.093, A/DF 1.107.

## FINAL configuration (dev git 40b1042+) -- average of 3 runs (k316, k317, k321), node -239, bitwise == production at every M
Layer = per-M kernel build + optional in-kernel router:
- M <= 8: build `-DTPB2_GUNR=1`; M 16-24: default build; M >= 32: build `-DTPB2_ACTX=1` (all other defaults: SHW1=1, SILU=1,
  RRSPLIT=1, SHGU M<=6, FC2STATIC M1-8, FC2PAIR off M<=8, SHEARLY off M<=2, FC2 unit cap 4 at M>=64, T1L / TABW off).
- in-kernel router (tpb2.run(..., router=(W, router_parts(M)))) at M5-64 (bitwise with this cuBLAS's split-K table); production router otherwise.
Harness: `b2_layer_ab.py --arms A S D DF --flags2=-DTPB2_ACTX=1 --flags3=-DTPB2_GUNR=1 --dispatch 8:3,24:1,512:2`.
| M | 1 | 2 | 4 | 6 | 8 | 16 | 20 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A (us) | 32.0 | 35.2 | 45.8 | 51.2 | 55.6 | 64.0 | 70.6 | 71.9 | 76.4 | 89.4 | 98.9 | 115.7 | 129.3 |
| DF (us) | 34.5 | 36.9 | 39.3 | 46.0 | 49.2 | 56.6 | 61.8 | 63.4 | 66.2 | 79.1 | 86.4 | 103.2 | 114.7 |
| A/DF | 0.93 | 0.95 | 1.17 | 1.11 | 1.13 | 1.13 | 1.14 | 1.13 | 1.15 | 1.13 | 1.15 | 1.12 | 1.13 |
Mean: A/S 1.078 (single build), A/D 1.094 (dispatched builds), A/DF 1.106 (dispatched + in-kernel router). Target 1.4 NOT reached.

## v67-v69 (2026-11-01): compile-time specialization per M bucket -- dispatched layer A/DF ~1.25 (bitwise)

The biggest lever left after v66 was code size on the cold-i-cache critical path: every runtime branch the
production configuration never takes still sat between the hot instructions.  Device reads of M / oneshot /
ndev / topk / shw / glog / rF / rS / nparts / rank / grid now go through pM / pOS / pND / pTK / pSHW / pGL /
pRF / pRS / pNP / pRK / TPB2_GRID, so a build for an M range with TPB2_SPEC=1 folds everything else away; the
host checks every compile-time assumption (TORCH_CHECK).  Plus WHOLE (epilogue split-K merges compiled out:
rr + RRSPLIT 1 + SHW1 1 means whole items), T1CNT (SWAR T1 tile counts), PUT1, QX1, ACTU (M1).

Final dispatch = disp_v69.sh: 14 builds ({1} {2} {3-4} {5-7} {8} {9-15} {16} {17-31} {32} {33-48} {49-63} {64}
{65-127} {128}), in-kernel router at M5-64 (still 1-4% better than production's router, k397).

Same node, old v66 3-build config -> v69 (A/DF mean, 13 M, D != A 0 everywhere):
  node umb-b200-263: 1.136 -> 1.244 (k374 / k388)
  node umb-b200-236: 1.137 -> 1.258 (k355 / k387)
Per M (node -236, k387): M1 1.16 M2 1.23 M4 1.34 M6 1.31 M8 1.31 M16 1.28 M20 1.28 M24 1.30 M32 1.26 M48 1.23
M64 1.23 M96 1.20 M128 1.23.

Method lesson: knobs that touch the pre-PDL phase must be A/B'd with the router in front (S/S2 for M<=4,
SF/SF2 for M5-64); KS (precomputed logits) mis-ranks them (SHEARLY: KS -2.3%, S/S2 0, dispatched +7%).

Refuted on the specialized builds: PAIR (+8.6% M64-128), NS=5 (0), in-kernel dsv3 router (0..+4.8%), T1L,
NMAX=8, SMALLM=16 (+6.5%), ACC=6, SHW1_W=4, FC2G4 cap off, PGATE1, FC2S_SHF (+4% M1), QW811 / XRLX / IDPRE /
RANK (all within +-1%).  Remaining time is latency chains: M1 kernel ~26.5 us = prologue ~3 + FC1 chain ~7.4
(16 stages, MMA ~0.27 and TMA ~0.3 us/stage bounds coincide) + act/publish 2.6 + FC2 ~4.9 + T1 1.5 + AR 3.8
(rank skew); M128 = prologue 8 + stream ~88 (~5.4 TB/s delivered; no-MMA ~6 TB/s) + tail ~9.
1.4x is not reachable bitwise with this architecture; next levers would be structural (tables rewrite for
M > 8: ~1-2 us; cluster/DSMEM FC1->FC2 handoff at M <= 2: ~2 us; both ~+1% on the mean).

## v71 final (2026-11-02): A/DF 1.241 (node -263, 20 rounds) / 1.245 (node -236, 16 rounds), bitwise

disp_v71.sh = disp_v69.sh + SHGU_MAXM=8 in the M5-8 builds (M8 -2.0% SF/SF2).  Old v66 config on the same nodes: 1.136 / 1.137.
Identical builds vary 2-4% per M between dispatch runs (~1% on the mean), so later knob deltas below 1% are not resolvable.

Large-M limit (M128 probes, kernel alone 102.5 us): no MMA -6.9%, no routed SFA -4.2% (= its 6.25% share of the stream bytes;
L2-prefetching it is +0.9%), no x (B) copies -3.3%, no SF tcgen05.cp -1.1% -> data movement bound (~5.3 TB/s even without MMAs).
Per SM, unpacked FP4 costs 32 KB smem write + 32 KB MMA read per 16 KB stage (~0.27 us at 128 B/clk), which is why
cta_group::2 and deeper rings do not help.  Router: M96/128 cuBLAS split-K router 6.3 us alone, ~5 us serial in front of the
kernel; the in-kernel router is +6% there even in 64-row blocks (RRB, also wrong for the 2nd block -- reverted).
Reverted negatives this round: RRB, M1FAST, TCLR+IDPRE (not bitwise at M128), SFAPF, EARLY1, SHW1_W=1.
