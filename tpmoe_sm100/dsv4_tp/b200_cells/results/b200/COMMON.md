# B200 cells — common notes (suite_b200, agent B200)

## Release state (11-10): final kernel per cell

| cell | kernel (M range) | suite_b200 commit | headline tag (README) | best / ours, M8 .. M512 |
|---|---|---|---|---|
| DSV4-Flash TP4 (GPUs 0-3) | v77 routed-only (M < 384) + v4ft (M >= 384) | v77 build c54caf3 (+ 0682a9e SFBI guard, codegen unchanged); dispatch 6a01f29 | f4trV | 1.24 1.21 1.12 1.05 0.98 0.99 1.02 0.92 |
| DSV4-Flash TP8 | v77-ndev8 (M < 384) + v4ft (M >= 384) | v77 build c54caf3 (+ 0682a9e); dispatch 6a01f29 | f8trV | 1.23 1.20 1.17 1.08 0.99 0.95 0.95 0.85 |
| DSV4.1-Flash TP8 | tp8v41h2 = tp8_v41_b300 @3668057 + LLS grid trap (H2 from M128, NMAX 32 + GATHER 0 from M256) | 37b469a | v8trP | 1.24 1.17 1.09 1.03 1.01 0.99 1.01 1.04 |
| DSV4.1-Flash TP4 (GPUs 0-3) | v41tp4 (port_tp4 of tp8v41 @ee6b585, F2T K64 tails) + LLS | c54caf3 | v4trH (clean reps 3f/4f/5f) | 1.22 1.17 1.08 1.03 0.98 0.95 1.02 1.08 |

Here v4ft = suite_b300b/s3/port_v4f.py over tp8_v41_b300 e066ec9 (`v4ft/v4fti.cu`): LLS, NMAX 32 and GATHER 0 from M256, ACTX + T1PW2 at M384 and M512, F2H off. It runs in its own workspace, on the same stream as v77 (`arm_disp.py`, `SB_DSV4F_DISP=0` = v77 only, `SB_V4FT_MINM` moves the switch).

Build and run:
- Builds are JIT (torch cpp_extension, sm_100a) into the node-local `$SB_BUILD` on first use, so no separate build step.
- Node setup: `JOB=<hold> tpmoe_suite/env/node_setup.sh sm100`.
- Headline: `JOB=<hold> ./final_cells.sh <TP> <model>:<tag>`. This runs the gate (below), then suite reps 1f/2f (`suite.sh`) and 3f (`rep3.sh`).
- Table: `python3 -I stable.py results/b200/<cell> <tag>`. READMEs: `python3 -I mkreadme.py`.
- Development A/B (same process, bitwise compare): `JOB=<hold> GPUS=.. ./sctr.sh <out> python sb.py --model .. --tp .. --ms .. --arms ours --check 1 --routing trace --proto b`, plus one of:
  - `--flags2=..` (same kernel, another build)
  - `--src2 <file>` (tp8v41 lineage, another source)
  - `--ours2-kind v4ft --dispatch2 '<M>:<flags>,..'` (v4ft vs v77)
- XTrace: `--dbg 1 --trace 1 --xt-mask 63`, then `xt_phase.py` and `xt_cta.py` on the dumped JSON.

Gate (every kept build, every M):
- `sb.py --check 1 --b2b 12`: the check case, then back-to-back bursts of 12 calls on distinct inputs (eager rank-skewed / one rank drifting / one CUDA graph), bitwise vs sync-separated references on every rank.
- Plus `--b2b-mix 1`: odd calls at M64/128, i.e. another build, and for the DSV4-Flash dispatch also the other kernel family.
- Plus the high-count `stress_v77.py`: 1500 x 8 eager + 1500 graph replays, 2 cases. `--kern disp --mix2 64` for the dispatch.
- All final builds pass with 0 mismatches. Logs: results/b200/_stress, results/b200/_gate_v4ft and the gate_<tag> runs.

Caveats:
1. Slow-timing state. Some processes time ours 5-15% slow at the larger M while gc in the same process is normal.
   - It is per-GPU and latency-type: tables x2, main +8-16%. It is not bandwidth, SM availability, clocks, NVLS or footprint.
   - It shows up most often in the first heavy process early in a node allocation.
   - Mitigation: compare builds only within one process (ours / ours2), use >= 2 swapped reps, run a throwaway warm-up, and take medians across processes.
2. Protocol-b gate algorithm. The gate all-reduce's NCCL algorithm shifts every arm's protocol-b time. NCCL_ALGO=NVLS reads 3.5-4.4 us lower at M64-384. Headlines use the NCCL default; compare protocol-b numbers only at the same setting.
3. cudaEvent quantization is ~2.05 us, so use the trimmed means (`.bm`) or many cases for differences under 1 us.
4. A killed run leaves torch-extension `lock` files behind. Delete them before the next build (`find <build> -name lock -delete`).
5. Shared-hold lock (`node/b200lock074.sh`): my own `acquire` extends a lock I already hold, so chained window scripts must wait on the previous script's DONE marker, not on the lock.

Regenerable A/B sources (the DSV4.1 bisect `bis_<sha>.cu`, `old_t.cu`, the 8bbb174 re-measure `tp8v41g8.cu`) are not tracked. `tp8v41h2/mk_ab.sh` recreates them. The 11-10 copies are in `suite_b200/scratch/` (gitignored).


Machine: 8x B200 (computelab b200@*-1000W / umbriel-b200@ts4, 1965 MHz max SM clock, NVLink5 full mesh). Nodes used 11-09:
umbriel-b200-094, umb-b200-250, umbriel-b200-026, umbriel-b200-069, umbriel-b200-040; 11-10: umbriel-b200-074 (node-to-node spread ~4%: compare arms
only within one run).

Two harnesses, same protocol (both: CUDA graph of one call, 512 MiB L2 flush + gate all-reduce + ~100 us GPU sleep before every
replay, per-rank median then max over ranks; protocol a = gc-moe's 100-replay windows):
- SUITE harness (tpmoe_suite harness/bench.py; arms ours/gc/mega/fimega/fitrt) with my adapters
  `suite_b200/kernels/b200_<model>_tp<N>/adapter.py` -> `suite_b200/sb_adapter.py`; driver `suite_b200/suite.sh MODEL TP TAG ROUTING`
  (2 reps: rep 1 fwd (ours,gc,mega,fimega) + fitrt in its own process, rep 2 reversed order).  Headline numbers.
- my harness `suite_b200/sb.py` (arms ours/trt/gc/mega; `cell.sh MODEL TP TAG ROUTINGS`, same 2-rep scheme; `table.py TAG ROUTING`),
  used for development A/Bs (`--flags2` = paired second build, bitwise compare `ours2_vs_ours mism`), dbg stamps (`--dbg 1`) and
  XTrace (`--dbg 1 --trace 1`).
Routing headline: plain-decode real traces routing_caps/{dsv4f,dsv41}_real, layers 10/20/30 x steps 32/96 (6 cases, random x,
weights / 1.5), correctness on layer 20 step 33 real x (tpmoe_real_headline.txt v2/v3).  Secondary: the L20 serving captures
(cap_dsv4f_l20 294 rows / cap_v41_l20 732 rows), uniform.
B200 timing note: cudaEvent timestamps are quantized to ~2.05 us -> per-case medians repeat exactly; differences < ~1 us need the
trimmed means (`.bm`) or many cases.
Environment: CUDA 13.2.1 devel image, torch 2.11 cu130, cutlass-dsl 4.8.1 (cu12 libs), DeepGEMM 5081f85 + 2 kwarg shims, gc-moe
7d0825e; my harness: flashinfer 0.7.3 (trt arm); suite harness: flashinfer main c42e86fbc (fitrt / fimega).

Builds kept (suite_b200 git, nothing pushed): v77 IDV 66e06d7, LLS 06136eb, XRLX (M > 128) 88f4b3d; tp8v41 / v41tp4 F2T db4b19b,
LLS 4c9634b (patch_tail.py, shape-generic); DSV4.1 TP8 base 37b469a; DSV4-Flash v4ft dispatch (M >= 384) 6a01f29.  `gen_v77.sh` regenerates v77/tpmoe_b2.cu from the pristine copy + patches.
Per-cell READMEs are generated by `mkreadme.py` (suite tables via `stable.py results/b200/<cell> TAG [--only 2f,3f,...]`).

Harness gotchas met on 11-09:
- A killed run can leave torch-extension `lock` files in the build dir; the next process then waits forever in
  torch/utils/file_baton.py (found with py-spy).  After killing a run: `find <build dir> -name lock -delete` (no build running).
- Timing state artifact: some processes time ours (v77 and tp8v41) 5-15% slow at the larger M while gc in the same process is
  normal; not reproducible on demand (see the cell READMEs).  Compare builds within one process (ours / ours2 paired A/B) and
  across >= 2 reps; suite rep 3 (`rep3.sh`: fwd order, check 0) was added for the DSV4-Flash cells.
- `hold.sbatch` now names the B200 partitions and the account explicitly (without them a hold landed on an A10 node).

LLS hardening (c54caf3, after B300-TP8's back-to-back report -- root cause on their side: the then-unguarded TAILB counter zeroing
wrote 1280 B per call ~21 MB past an RS buffer allocated without TB_BYTES): the TAILB zeroing is now TAILB-only; lls_allreduce traps if
gridDim != TPB2_GRID (every LLS slot must have exactly one reader per call); arm_tp8v41 refuses to share a workspace between LLS and
tagged-LL builds; the host still fills the LL region with the 0xFF sentinel once.  Why LLS is race-free: a rank's T1 sends of call
e + 1 come after its griddepcontrol.wait (its call e, including the reader re-arms, complete), and no rank finishes call e + 1 before
every rank's call-(e + 1) sends arrived, so no parity-h slot is rewritten (call e + 2) before its reader re-armed it in call e.
Correctness gate from now on (every build, every M): `sb.py --check 1 --b2b 12` = check case + back-to-back bursts of 12 calls on
distinct inputs (eager, rank-skewed starts; one rank drifting; one CUDA graph of the 12 calls), bitwise vs sync-separated references
on every rank; `--b2b-mix 1` alternates two M builds (shared tp8v41 workspace).  All 4 cells, every M: 0 mismatches (LLS and LLS=0).

Protocol-b gate dependence (11-09): the cross-rank gate before every protocol-b replay is an NCCL all-reduce of one fp32 element.  Its
NCCL algorithm changes the measured protocol-b times of EVERY arm: with NCCL_ALGO=NVLS (forced) ours.b and gc.b both read 3.5-4.4 us lower
at M64-384 (M512 within 1 us), protocol a unchanged (d001 / d003 vs d002 logs) -- the gate's exit skew between ranks enters protocol b.
The headline keeps the NCCL default (NCCL_ALGO unset in tpmoe_suite env); compare protocol-b numbers only across runs with the same
gate algorithm.
v77 SFBI guard (this commit): static_assert(!TPB2_SFBI || TPB2_WHOLE) -- see the cell READMEs; codegen of every shipped build unchanged.
