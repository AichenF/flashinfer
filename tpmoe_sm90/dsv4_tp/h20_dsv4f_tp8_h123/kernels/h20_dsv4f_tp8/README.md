# h20_dsv4f_tp8 — h123 (H20 TP8 DSV4-Flash, routed-only "ns" builds)

Provenance: H20 .24 `/raid/sharedspace/fac/aichenf_cmp3/xfer/tp8` (= the served h123 binaries, dev git 70dbda3 of
`/raid/sharedspace/fac/h20-fused-moe-eaad992`; the cmp3 H20 runs used exactly these files).  Copied unmodified 2026-11-09.

| file | sha256 | served for M |
|---|---|---|
| so/tpmoe_var_ns_v78h21_tp8.so | e2816ff6bd4dd833591cbcc4a0e2541e558cdeaa0bd2a626f71c0d5f6ade69d4 | default (M < 64, M > 341) |
| so/tpmoe_var_ns_v94h13_tp8.so | ba00e7e4379ef91e66c2686f87de2be965604f731c118650a8e8215c3270a7f6 | 64..127 |
| so/tpmoe_var_ns_v94h21_tp8.so | f050fbd8e94b9c217271e46ef1adde59536d4de3878456b5cde8998fe6196548 | 128..341 |

`src/` = the sources (tp4moe2_ns.{v78h21,v94h13,v94h21}.cu + headers, sources.json); `tp4moe2.py` = the host module (sha256 matches
sources.json).  Built for torch 2.11 cu130, sm_90a.  Adapter: `adapter.py` (shared body `harness/sm90_tp4moe2.py`): weights in Humming's
layout from the canonical MXFP4 TP slices (`harness/humming_prep.py`, humming-kernels 0.1.10), RS/AG LL buffers via cudaMalloc + IPC,
`pre_replay` clears `xflag` (the host seq tag is frozen inside a one-call graph).  Numerics: Humming per-token E4M3 x (`xq_mode tok`).

---
Update 2026-11-09 21:29 (SM90 back-to-back race fix): src/tp4moe2_ns.{v78h21,v94h13,v94h21}.cu replaced by H20-TP8 fixed copies
(/home/scratch.aichenf_wwfo/v41_h20/fix_b2b/h123 = the served sources + TP2_B2B_FIX barrier, also v41_h20/cl/b2b_fix_h123_v78h21.patch); the adapter now
BUILDS these with the served flags (-DTP2_SHARED=0 -DTP2_NDEV=8); --opt prebuilt=1 = the served .so files (pre-fix).  Earlier realtrace_s3 = pre-fix.
