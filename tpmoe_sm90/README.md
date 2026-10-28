# tpmoe_sm90: TP4 fused MoE decode kernel for H20 and H100 (sm_90a)

`serving_tp4moe2_v26/` is the frozen serving package of `tp4moe2`. It is a
persistent TP4 MoE kernel for DeepSeek-V4-Flash Decode on four H20 GPUs.
One launch per MoE layer replaces SGLang's routed-expert MoE (Humming MXFP4
W4A8) plus the TP all-reduce:

- router gate (fused top-k), or the given top-k for hash-routed layers
- per-token FP8 input quantization
- FC1 + SwiGLU + FC2 over the routed experts, using the Humming fused-E8M0
  W4A8 weights in place
- shared expert, fused into the kernel for M <= 32 (the `sh` build); above
  that it runs on SGLang's alternate stream (the `ns` build)
- the cross-rank reduce, as low-latency messages over NVLink peer memory
  (IPC-linked per-rank workspace)

Shapes are hidden 4096, per-rank intermediate 512, at most 256 experts, top-k
at most 8, and M (tokens per call) at most 512. Larger calls fall back to the
original SGLang path.

The directory is self-contained and is not wired into the FlashInfer build or
Python API. Its contents are frozen: `sources.json` pins every source file by
SHA-256, and `install_tp4moe2.py` refuses to load if any file differs.

## Files

| File | Role |
|---|---|
| `tp4moe2_ns.cu` | kernel + PyTorch binding, built with `-DTP2_SHARED=0` (no shared-expert code; used for M > 32) |
| `tp4moe2_sh.cu` | kernel + binding with the fused FP8 shared expert (used for M <= 32) |
| `kpipe.cuh`, `chain4.cuh`, `chain_ss.cuh` | k-loop pipeline: MXFP4 dequant + RS-wgmma chains; SS-wgmma for the shared expert |
| `shared_num.cuh` | shared-expert numerics matching the production FP8 kernels instruction for instruction |
| `tp4moe2.py` | host side: per-rank workspace (`RankState`), LL all-reduce buffers, launch wrapper |
| `tp4_ext.py` | `torch.utils.cpp_extension.load` of both builds (`sm_90a`, build dirs `build_ns/`, `build_sh/`) |
| `build.py` | prebuild both extensions with the exact `load()` calls of the hook (no GPU needed) |
| `install_tp4moe2.py` | SGLang hook: patches `DeepseekV2MoE.forward_normal_dual_stream` / `forward_normal` and `Mxfp4HummingMoEMethod.process_weights_after_loading` |
| `tp4_lazy.py` | `sitecustomize` entry: installs the hook only after `sglang.srt.models.deepseek_v2` is imported (frontend / detokenizer processes never touch CUDA) |
| `test_v26_mix.py` | TP4 test of the hook (bitwise isolated vs mixed vs CUDA graph, gate vs production, timing); needs captured activations and the model, not included |
| `sources.json` | variant name, extension names, SHA-256 of every source |

## Use in SGLang (TP4 Decode)

```bash
# prebuild once, inside the serving image (as the user the server runs as)
python3 serving_tp4moe2_v26/build.py

# in every Decode worker process, e.g. from sitecustomize.py
export FAC_FMOE_MODE=tp4moe2 TP2_PDL=1
```

```python
import sys
sys.path.insert(0, "/path/to/tpmoe_sm90/serving_tp4moe2_v26")
import tp4_lazy   # installs the hook when deepseek_v2 is imported
```

Knobs:
- `TP2_PDL=1` uses programmatic dependent launch; it is the measured setting.
- `FAC_TP4_SHADOW=N` (default 1): the first N eager calls per (layer, M) also
  run the original path and log the difference. CUDA-graph warmup runs every
  captured M eagerly, so a server start validates every graph size.
- `FAC_TP4_SH_FUSED_MAXM` (default 32) is the largest M that uses the fused
  shared expert.

The hook logs `FAC_TP4_INSTALLED`, `FAC_TP4_LINKED`, `FAC_TP4_LAYER_READY`,
`FAC_TP4_SHAPE` and `FAC_TP4_SHADOW` JSON records. Gate a deployment on them:
- 4 linked ranks
- every MoE layer ready (46 per rank with DSpark: 43 target + 3 draft)
- no fallback M
- every shadow case within rel RMS 0.05 and per-token rel 0.25

## Measured (DeepSeek-V4-Flash, 4x H20, 1P1D with DSpark Decode)

The Decode step, independent of DSpark acceptance, is compared against the
stock Humming MoE. The first figure is real P/D; the second is fake-KV pure
Decode:

| ISL/OSL, concurrency | Step speed-up |
|---|---|
| 1K/1K, C1 | +3.0% / +2.6% |
| 1K/1K, C4 | +3.5% / +1.6% |
| 1K/1K, C16 | +8.6% / +10.1% |
| 1K/1K, C64 | +8.1% / +6.6% |
| 8K/1K, C16 | +7.7% (ITL p50) / +9.3% |

All cells are beyond the A/B/A noise floor. Start-up shadow gate against
stock Humming over 4,920 (layer, M) cases: worst rel RMS 0.0235, worst
per-token rel 0.051.

## H100 (`serving_tp4moe2_h100/`)

The same hook and kernel structure, retuned for four H100 SXM GPUs (TP4).
Differences from `serving_tp4moe2_v26/`:

- `tp4moe2_ns.cu` is a later kernel (`tpmoe_v94ex_h100`) with the
  epoch-parity LL all-reduce buffers. Both builds come from this one source:
  `sh` uses the default `TP2_SHARED=1`, `ns` adds `-DTP2_SHARED=0`. It also
  needs `kpipe_w.cuh`. Both builds use 96 registers and do not spill.
- Its output is bitwise identical to the v28 TP4 kernel. This was checked
  rank-local on all four ranks for M 1..512, and in a 4-GPU layer A/B for
  both builds.
- The shared-expert policy has a new knob, `FAC_TP4_SH_FUSED_FROMM`. The
  shared expert is fused when M <= `FAC_TP4_SH_FUSED_MAXM` or
  M >= `FAC_TP4_SH_FUSED_FROMM`. The defaults depend on the GPU, detected
  from the device name:
  - H100 TP4: 21 / 33
  - H20 TP4: 32 / off (unchanged)

  `FAC_TP4_INSTALLED` logs both values.
- The hook and `build.py` also accept TP8 (`-DTP2_NDEV=8`). This package's
  TP8 path has not been measured.

Use it as described above, with `serving_tp4moe2_h100` in the path. Build
TP4 only with `python3 serving_tp4moe2_h100/build.py 4`.

### Measured (DeepSeek-V4-Flash, 4x H100 SXM, NVLink full mesh)

The table gives the MoE layer time in us for layer 20 on captured Decode
activations:

- Runs are CUDA graphs, with rotating weight copies so L2 is cold.
- Each figure is the median of the slowest rank.
- Humming is the production path: router GEMM + `moe_fused_gate` + Humming
  routed experts, the shared expert on the alternate stream, and SGLang's
  custom all-reduce.
- Both tpmoe columns include the same router GEMM and shared expert.

| M | Humming | v28, H20 policy | this package | change vs v28 |
|---|---|---|---|---|
| 1 | 46.7 | 39.7 | 39.1 | -1.5% |
| 6 | 85.4 | 66.6 | 65.4 | -1.8% |
| 16 | 128.4 | 104.0 | 98.7 | -5.1% |
| 20 | 143.7 | 119.5 | 112.7 | -5.7% |
| 24 | 148.8 | 124.9 | 113.6 | -9.0% |
| 32 | 165.9 | 139.4 | 122.0 | -12.5% |
| 48 | 197.9 | 162.9 | 141.7 | -13.0% |
| 64 | 223.2 | 178.0 | 159.2 | -10.6% |
| 96 | 278.9 | 207.4 | 184.5 | -11.0% |
| 128 | 301.6 | 228.7 | 205.5 | -10.1% |
| 192 | 347.2 | 251.8 | 228.7 | -9.2% |
| 256 | 382.8 | 273.5 | 251.8 | -7.9% |
| 384 | 432.8 | 307.3 | 287.4 | -6.5% |
| 512 | 466.3 | 341.9 | 321.9 | -5.8% |

The mean speed-up over Humming across these 14 M values goes from 1.283x
(v28) to 1.394x (this package). End-to-end serving on H100 has not been
measured.
