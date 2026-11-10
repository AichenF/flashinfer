# _h20b — sources shared by kernels/h20_dsv4f_tp4 and kernels/h20_dsv41f_tp4

Imported 2026-11-09 (step 3) from `/home/scratch.aichenf_wwfo/suite_h20b` (H20b cell agent) commit ca18afa: `git archive ca18afa src` (all of src/,
unmodified) and `git archive ca18afa kernels/<cell>` (adapter.py + README.md), with ONE line changed in each adapter.py:
`SRC = HERE.parents[0] / '_h20b' / 'src'` (was suite_h20b/src).  Defaults: h20_dsv4f_tp4 = tp4moe2_v94ex.cu (v94ex, exact == v28);
h20_dsv41f_tp4 = tp4v41e.cu (the kept optimizations).  (Earlier import: 86ae9d6.)

Update 2026-11-09 21:29 (SM90 back-to-back race fix): src/tp4moe2_v94ex.cu replaced by H20-TP8's fixed copy /home/scratch.aichenf_wwfo/v41_h20/fix_b2b/v94ex/tp4moe2_v94ex.cu
(= ca18afa + the TP2_B2B_FIX barrier only); src/tp4v41e.cu patched here with the same one-line barrier (named_bar_sync(1, NCONS) after
STAMP(P, 2), knob TP2_B2B_FIX default 1; text identical to v41_h20 872c14c / fix_b2b).  Earlier H20 realtrace_s3 numbers of these cells are pre-fix.
