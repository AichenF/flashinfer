# b300_dsv41f_tp4 -- v41tp4 (B200 agent's kernel, run by agent B300b)

Provenance: `/home/scratch.aichenf_wwfo/suite_b200` @ 7eaa18a (LLS hardening c54caf3): `v41tp4/v41tp4t.cu` = port_tp4.py output of tp8v41.cu
(9 full FC1 tiles, F2T tail-first FC2 with K64 tails, NDEV 4) + the tail stage `patch_tail.py` (LLS sentinel all-reduce, TAILB zeroing
guarded, gridDim trap), `v41tp4host.py` -- copied unmodified to suite_b300b/v41tp4 (suite_b300b @ f8b6cec) and compiled for sm_103a.
Builds: one per M, `v41tp4host.spec_flags(M, M, 4, extra=('-DTPB2_LLS=1',) + PER_M[M] + ('-DTPB2_ONESHOT_MAXM=64',))`, PER_M = {384: ('-DTPB2_T1PW=2',), 512: ('-DTPB2_T1PW=2',)}.
LLS: the adapter 0xFF-fills its RS buffer's LL region once (sentinel slots, re-armed by their readers); RS buffer = RS_BYTES + TB_BYTES.
Weights: `v41tp4host.kernel_layout` of the canonical MXFP4 TP slices; `ref_partial` = `v41tp4host.ref_rank`.
Validation (suite_b300b s3bench, S3_HOST=v41tp4): ranks bitwise identical, deterministic, back-to-back eager bursts (same M and cycling all M
builds, ranks skewed) bitwise equal to isolated calls.

## Correctness gate (suite_b300b s3/st01.sh, runs/st01_v414.log + st01_v414.json; node umb-b300-022, 2026-11-09)
Exact build configuration of this cell, routing = real decode trace (case L20 s32 + a skewed batch with other x), M = 8 / 64 / 256 / 384 / 512:
- isolated calls: 4 ranks bitwise identical, deterministic, rel <= 3.7e-4 vs the exact-numerics reference (ref_rank), cosine vs bf16 dense = reference's;
- b2b: 24 eager back-to-back calls cycling 3 inputs, ranks skewed by GPU sleeps, x 2 bursts -> 0 / 192 bad outputs per M;
- b2b-mix: bursts of 32 calls cycling ALL 5 M builds (one workspace / epochs / all-reduce buffer), skewed, x 2 -> 0 / 256 bad outputs;
- high-count stress: 2 cases, 2000 rounds x 8 back-to-back calls per rank, eager (random per-rank GPU sleeps before ~1/4 of the calls, outputs
  NaN-poisoned before every round) AND CUDA graph (one graph of the 8 calls replayed 2000 x) -> 16,000 eager + 16,000 graph calls per rank per M
  = 160,000 calls per rank (640,000 rank-outputs checked bitwise against isolated calls): 0 mismatches.
