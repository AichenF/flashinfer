# _h200 — shared code of kernels/h200_* (H200 cell agent)

Imported 2026-11-09 (step 3) from `/home/scratch.aichenf_wwfo/suite_h200` commit f566acf: `git archive f566acf cells src tphost.py v41_prep.py port_v41_tp4.py`.
Changes in the copy: cells/h2cell.py git_head() returns 'f566acf (imported ...)'; kernels/h200_<cell>/adapter.py = cells/h200_<cell>/adapter.py with
sys.path pointing at kernels/_h200/cells.  Rows h200_dsv4f_tp8h / h200_dsv41f_tp8h = "TP8-hybrid (2 EP groups x TP4)" variants (separate rows).
(Earlier import: 7d6b168.)

Update 2026-11-09 (step 3): re-imported at suite_h200 b6f4263 -- f566acf..b6f4263 changes only cells/h200_dsv41f_tp8/adapter.py,
cells/h200_dsv41f_tp8h/adapter.py (TP2_TAIL4=0 build at M <= 16 / hybrid M <= 8; bitwise identical) and adds src/tpv41_fc1k.cu (unused by the
defaults); those files were copied, the tree now equals b6f4263 except the h2cell.py git_head() label.

Update 2026-11-09 21:0x (step 3): re-imported at suite_h200 56f1450 (git archive cells src tphost.py v41_prep.py port_v41_tp4.py):
SM90 race fix TP2_B2B_FIX=1 in all H200 cell sources (adapters unchanged).  The tpv41 / tpv41h cells (h200_dsv41f_*) are labelled
"fix unverified" until the H200 agent's verify_fix passes on its replacement node -- DROPPED 2026-11-09 23:4x: verify_fix complete and
verified on all 6 cells (suite_h200 56ba28b write-up; cell sources unchanged at 56f1450).  Supersedes b6f4263; the H200 dsv4f_tp4 realtrace_s3
run made before this import (f566acf sources) is superseded and re-run.
