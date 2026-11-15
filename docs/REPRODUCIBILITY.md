# Reproducibility

This file holds the technical details behind the README: the environment of the frozen runs, how the code is run, the determinism settings, the checks that compare regenerated files with the frozen ones, the guard on the test set, and the upstream code that the baselines use. The tables and figures of the paper are regenerated from frozen files; nothing in this repository contains a new result. Code comments, configuration keys and some file and folder names (for example `posthoc_R2.10`) contain internal tracking IDs from the revision process (for example, R1.23 = reviewer 1, comment 23); they do not affect behaviour.

Contents: 1 Environment and paths; 2 Running the code; 3 Determinism; 4 GPU reproduction checks; 5 Hash checks (regenerated versus frozen); 6 From paper items to scripts and files; 7 Known errors of the original code and where the code tests them; 8 Pinned upstream code; 9 Known limits; 10 Module guide.

## 1. Environment and paths

The frozen runs used Python 3.12.10, torch 2.8.0+cu128, numpy 2.1.3, pandas 2.3.1, scipy 1.16.1, scikit-learn 1.7.1, optuna 4.5.0, PyYAML 6.0.3 and matplotlib 3.10.5, on an NVIDIA GeForce RTX 5070 Ti Laptop GPU (11.94 GiB), driver 577.13, CUDA 12.8. The results of the frozen runs depend on this stack (see section 4).

```
python -m venv .venv && . .venv/bin/activate          # Git Bash on Windows: . .venv/Scripts/activate
pip install -e ".[dev]"
```

Data and code availability: the C-MAPSS turbofan data set is publicly available from NASA. The data files are not redistributed here. The repository includes the code, the configurations, the frozen tables and per-seed metrics (`results/frozen/`) and the scripts that show the errors of the original code; the prediction files, result files and checkpoints needed for the regeneration checks are a separate dataset, https://doi.org/10.5281/zenodo.24233550. The code is at https://github.com/simonchantack/SACA_Transformer. In the code and in the result files, SACA is the design named D4.

Data files: download the NASA C-MAPSS files (`train_FD00x.txt`, `test_FD00x.txt`, `RUL_FD00x.txt`, x = 1 to 4), put them in one folder and set `SACA_DATA_DIR`. The C-MAPSS files used for the frozen runs have Windows line endings (CRLF); the files from NASA's download have Unix line endings (LF). The code reads both identically (checked on the training files: identical arrays and identical regenerated results), so line endings do not matter. The `raw_file_sha256` values listed in `configs/data_splits.json` are the hashes of the CRLF files and are for information only; no check uses them, so your file hashes will differ. The frozen results folder, which is needed only for the regeneration checks, is the authors' full folder of frozen runs, and `SACA_WORKSPACE` selects it. The published dataset (https://doi.org/10.5281/zenodo.24233550) holds the part that the regeneration commands need: unzip it and set `SACA_WORKSPACE` to the unzipped folder, the one that contains `experiments/`. Not published (besides working files such as logs, scripts, READMEs and configurations): the Markdown outputs and input-hash files of the post-hoc analyses (the dataset holds one results JSON file per analysis), the folders `exp_A9` (with `table7.md`), `exp_A11`, `exp_B17`, `exp_B18` and `exp_B19`, the tuning reports, the reproduction certification, and the run folders (checkpoints and logs) of every experiment other than `exp_B16` and `exp_V1`; these are available from the corresponding author on request. `SACA_WORKSPACE` overrides the default `../workspace` of the reporting modules (`${SACA_WORKSPACE:-../workspace}`). The retrain configurations read the frozen run folders through `SACA_FROZEN_ROOT` (default `../workspace/experiments`); the dataset holds only the SACA (`exp_B16`) and no-smoothing (`exp_V1`) run folders, so set it to the dataset's `experiments/` folder only for those; the frozen-checkpoint tests of the other models, the other tests marked `frozen` and `scripts/reproduce.sh` need the authors' full project folder (with the dataset, run `python -m pytest -m "not frozen"`). The folder that holds the retrain outputs is `SACA_OUTPUT_DIR` (default `results/runs`), and the test log is `SACA_TEST_LOG` (default `results/test_eval_log.csv`). No path is hard-coded: the configuration files hold `${NAME:-default}` strings that `src/saca_rul/config.py` resolves against the repository root.

## 2. Running the code

```
scripts/reproduce.sh                # tests, table and post-hoc hash checks, shapes, figures, AST equivalence (CPU)
scripts/reproduce.sh --repro-gpu    # plus the GPU reproduction check (6 validation-only retrains)
python -m saca_rul.cli train --config configs/exp_B16_d4_main.yaml --subset FD001 --seed 0   # one retrain
python -m pytest                    # tests only
ruff check . && mypy src            # quality gates
```

`--full` retrains every configuration. It is refused unless `SACA_ALLOW_RETRAIN=1`. It was switched off while the experiments of the paper were frozen; the switch stays as a safeguard against accidental retrains.

`scripts/reproduce.sh` is the supported route. On Windows, run it from Git Bash; there is no PowerShell script. Stage A of `scripts/reproduce.sh` was run end to end.

`reproduce.sh --repro-gpu` and `cli repro` write all six cells into one run folder, so only the last cell's files are kept. To keep every cell, run `cli train` once per model with its own output folder, as the second reproduction check did (section 4).

A retrain writes `ckpt_rmse.pt`, `history.csv`, `preds_val.csv`, `train.log` and `train_meta.json` to `results/runs/<exp>/runs/<subset>_seed<k>/`.

**Test reads.** The test files are opened only by `evaluation.evaluate_test`, through `cli test-read --confirm-single-read`. It refuses unless `selection_frozen.json` holds an entry written before the read, the checkpoint hash matches, and no earlier row of the test log exists for the same configuration. The release writes to its own log (`SACA_TEST_LOG`, default `results/test_eval_log.csv`), never to the project log of the paper runs. Training, tuning and selection code runs under a guard that raises on any path containing `test_FD<d>` or `RUL_FD<d>` (`guard.py`; tested). Nothing selects on a test metric: the release has no sort, minimum or comparison on a test metric (test `test_released_tree_has_no_test_selection_code`). The lines of the submitted code that did so (`improve_transformer.py:1202/1295/1314`, `MainSingleEng_FD001_F2_FD003_F4_Final.py:1245`, the leaderboard sorts and the Optuna scripts with a test-RMSE objective) are not part of this release.

## 3. Determinism

Every final run calls `determinism.set_all(seed)`: `torch.use_deterministic_algorithms(True, warn_only=False)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.utils.deterministic.fill_uninitialized_memory = True`, cuDNN deterministic with benchmark off, and TF32 off. `assert_strict()` checks these settings, and they are written to the header of every `train.log`. The evaluation batch size is 256 on validation and 1024 on test, and the predictions are checked to be batch-invariant (a test for the batch-row error of the original code, see section 7).

## 4. GPU reproduction checks (validation only)

Both checks retrained six cells, all on FD001 with split seed 0, in strict deterministic mode: SACA (exp_B16), the corrected original model θ\* (exp_B6), PatchTST (exp_B10_topup), iTransformer (exp_B11), Crossformer (exp_B12) and the LSTM (exp_V2). The pass criterion is that the best epoch, the sha256 of the state dict and the sha256 of `preds_val.csv` equal those of the frozen run. The expected hashes are in `results/frozen/repro_expected_FD001_seed0.json`. The result of the first check is in `results/repro_check/repro_result.json` (booleans only): 6 of 6 match. The check read no test data, wrote no row to the test log and left no GPU lock behind.

### 4.1 First check, and its limit

The first check did not run on the final tree. It ran while the code in this repository was still being edited. Timeline (local time): first launch 04:34:50, aborted at epoch 0 by a code bug (a missing `batch_first` attribute), fixed at 04:37:36; relaunch 04:37:53, finished 04:44:59. After the relaunch started, the code was changed as follows: (a) `ruff check . --fix` was run over the whole tree (04:39:29), (b) typing and lint edits were made (a local rename in `attention.py`; annotation or comment edits in `data.py`, `embeddings.py`, `decoder.py`, `encoder.py`, `tuning.py`, `stats.py`, `shapes.py`, `training.py`, `figures.py`), (c) the `tune` and `freeze` sub-commands were added to `cli.py`, and (d) `verify.py` was edited. Python imports a module at first use, so a module imported lazily after an edit may have run in its edited form. The comparison below therefore covers every module, not only the ones imported at start.

**Deleted evidence.** At 04:37:36 `rm -rf results/repro_check` removed the folder of the aborted first launch. At 04:45 the run folders of the relaunch (`ckpt_rmse.pt`, `preds_val.csv`, `train.log`) and `repro_stdout.log` were removed. Because these files were deleted, the 6 of 6 match can no longer be re-checked from run artifacts: only the booleans in `repro_result.json` and the recorded launch and finish times remain.

**Equivalence evidence (CPU, no data, no number).** The tree of 04:37:53 was reconstructed by replaying the recorded file-write and shell commands up to the relaunch (they contain no in-place edits) (`results/repro_check/launch_tree/`, files with a `.txt` suffix; replay scripts in `results/repro_check/reconstruction/`). Replaying the recorded post-launch steps (`ruff --fix` with ruff 0.17.3 and the pyproject of that moment, then the scripted edits) on top of it gives the shipped `.py` files byte for byte (`results/repro_check/reconstruction/replay_check.md`; at the first release the shipped sha256 values also equalled `RELEASE_FILES.md`; this no longer holds for the six files that changed afterwards, which are listed in that file). `scripts/ast_equivalence.py` compares the launch tree with the shipped tree after removing docstrings and annotations, comparing imports as used bindings, renaming local variables consistently and inlining single-use `name = self.attr` aliases (`results/repro_check/ast_equivalence.md`). Result for the 17 modules reachable from `cli.cmd_repro` (the reproduction path): 14 are byte-identical or AST-equal, and 3 differ in lines that are all listed and explained: `attention.py` (one re-bound local split into `attn` and `attn_out`), `data.py` (a `float()` around a numpy scalar before `abs()`, and `type: ignore` comments) and `cli.py` (additive `tune` and `freeze` sub-commands, one argument hoisted into a local; the `repro` branch is untouched). Off the reproduction path, `tuning.py` and `reporting/verify.py` differ in typing-only lines, and `figures.py` (redrawn), `shapes.py` (stage shapes), `tests/test_frozen.py` (two added tests), the docstring of `posthoc_r124.py` and the usage text (docstring) of `cli.py` changed again later, and `figures.py` changed once more (plain log-axis tick labels, extended size check), so they differ from the launch tree by design. The vendored tree is byte-identical to the launch tree (`diff -r`). The snapshot of all 37 modules taken before the later edits is `results/repro_check/ast_equivalence_all_modules_pre_fix1.md`.

Limits of this evidence: the reconstruction comes from the recorded commands, not from a saved copy of the tree; "AST-equal" is not a proof of equal GPU behaviour; the `attention.py`, `data.py` and `cli.py` residuals are argued by reading, not by test. While the release folder was being assembled, a replay script overwrote five files by mistake (`figures.py`, `posthoc_r023.py`, `posthoc_r124.py`, `posthoc_r210.py` and `posthoc_valsim.py`; the last was overwritten with identical bytes). The replay rebuilt them, and their sha256 equalled the first release list until `figures.py` was intentionally rewritten and `posthoc_r124.py` later received only a docstring change ("PORT, NOT EXECUTED").

### 4.2 Second check, on the shipped code

* **What ran.** The six cells above (FD001, split seed 0, validation only, strict deterministic mode), retrained from the shipped tree through `python -m saca_rul.cli train`, one invocation per model, under a GPU lock file. The `repro` sub-command was not used, because it writes all six cells into one folder with the same run tag, so each cell would overwrite the one before. Each model has its own output folder. The comparison used the shipped `runner.compare_with_frozen`. The check was approved by the first author on 2026-10-27.
* **When.** 2026-10-27, 11:02:58 to 11:10:28 local time. The check was announced in the project log before the launch. No test data were read (the test log still had 404 rows) and no lock was left behind.
* **Shipped-tree manifest.** The sha256 of every file under `src/`, `configs/` (including `configs/selections/`) and `scripts/`: 111 files, tree sha256 `639a92e5d32b5272f15b4280f1cc039fc6adc62e858a84e3c424b4d0026eb06b`. It was recorded before the launch and re-hashed after the run, and the two are **equal**. Nothing in the tree was edited, linted or tested during the run.
* **Result** (booleans only, `results/repro_check/repro_result_2.json`): best epoch, state-dict sha256 and `preds_val.csv` sha256 equal to the frozen run for all six models (6 of 6).
* **Run folder.** It is kept whole and never deleted, outside this repository (in the project workspace). Its files hold numbers and are marked "not reported" in its README.
* **Status of the first check.** Because all six cells match on the shipped code, the limits of the first check (code edited during the run, deleted run folders) are superseded for the shipped code by the second check. The text of section 4.1 stays as the record of what happened. The second check covers FD001, seed 0 only.

## 5. Hash checks (regenerated versus frozen)

The tolerance is **zero** for tables and shipped files (sha256 equality). Float comparisons occur only in two tests: the frozen SACA checkpoint's CPU predictions versus the saved GPU validation predictions (maximum absolute difference < 1e-3 cycles, the tolerance used in `evaluation.py` and the tests; no measured value is reported here) and the sensor-order invariance test (float64, 1e-9).

| Artifact | Frozen sha256 | Regenerated | Check |
|---|---|---|---|
| `combined_table_v4.csv` | 4f4d5368…a628 | identical; the frozen copy is shipped in `results/frozen/` | direct |
| `combined_table_v4.json` | 189d784f…eb55 | identical after substituting 4 provenance fields (the regenerated copy in `results/tables/` has sha256 da80639a…); frozen copy in `results/frozen/` | `reporting/verify.py` |
| `combined_table_v4.md` | e994beca…66a6 (the hash cited in the paper materials) | identical after substituting 1 provenance field (the regenerated copy in `results/tables/` has sha256 3b28cf3a…); the frozen file itself is shipped as `results/frozen/combined_table_v4.md` | `reporting/verify.py` |
| `final_rows.json` | deaed352…c53c | shipped copy, sha checked | test |
| bootstrap analysis, test-window padding analysis and no-smoothing simulation (Markdown) | frozen files in `final/posthoc_*` of the frozen results folder | byte-identical | diff |
| the same three analyses (JSON) | frozen | equal except `log_row`, `inputs_sha256_file` and the protocol text | `verify-posthoc` |
| FD004 regime-split outputs (reads columns 1-5 of `test_FD004.txt`) | `results/frozen/R1.24_SHA256SUMS.txt` | not re-run; shipped files are hash-checked. Script of record: `results/frozen/run_regime_split.py` (573d5189…). `reporting/posthoc_r124.py` is a **port that was never executed**; its equivalence to the script of record is not demonstrated | test |
| per-seed metrics of exp_V1, A10, A11, A12, A13, B17, B18, B19 | `results/frozen/metrics/SHA256SUMS.txt` | byte copies of the frozen `metrics.json` files | test |
| SACA stage-by-stage tensor shapes | `results/shapes/d4_shapes.md` | CPU shape script re-run; earlier keys equal to the first output (kept in `results/shapes/previous/`) | test |
| data splits / tuning protocol / selections | 88840135…0eab / d41d8bd8…01e7 / `configs/selections/SHA256_selections.json` | copied, sha checked | test |

The provenance fields that differ are the generator path and sha, the `final_rows` path, and the sha of the test log at generation. The log has only grown since: its first 401 rows still hash to the recorded value and rows 402-404 are the logged post-hoc readings. The test log has 404 rows before and after this build.

`RELEASE_FILES.md` lists every file of the repository with its sha256 and has notes on individual files. To check a clone, hash each listed file and compare. The list does not contain itself.

## 6. From paper items to scripts and files

| Paper item | Script | Frozen artifact |
|---|---|---|
| Fig. 1, SACA architecture | none in this repository: the first draft was generated as an editable PowerPoint slide by a script in the project workspace (not in this repository) from a specification derived from the code; the authors edited it in PowerPoint, (a block merge, legend and label wording, line styles and layout), and exported it to PDF (page 1 copied unchanged) | `figures/fig_saca_architecture.pdf`, checked against the code and `results/shapes/d4_shapes.{md,json}` |
| Main results, all rows, seed tables and tests; Fig. 2 | `reporting/tables.py` | `combined_table_v4.{md,csv,json}`; figure `fig_v01_1_final_rmse_v02` |
| Preregistered primary comparison and non-inferiority; Fig. 3 | `reporting/tables.py` | `exp_B16/primary_test.json` (frozen results folder); `fig_v01_2_primary_pairs_v02` |
| Hierarchical bootstrap | `reporting/posthoc_r210.py` | `posthoc_R2.10/` (frozen results folder); `fig_v01_3_hier_bootstrap_v02` |
| FD004 regime split | `reporting/posthoc_r124.py` | `posthoc_R1.24/` (frozen results folder); `fig_v01_7_fd004_regime_v02` |
| Test-window padding and no-smoothing gap | `posthoc_r023.py`, `posthoc_valsim.py` | `posthoc_R0.23_nosmooth_c/`, `nosmooth_b_valsim/` (frozen results folder); `fig_v01_4_nosmooth_padding_v02` |
| Training-size sweep | `configs/exp_A10_*` | `results/frozen/metrics/exp_A10_metrics.json` (per seed); `fig_v01_5_size_curve_v02` |
| Window and dropout sensitivity | `configs/exp_A12_*`, `exp_A13_*` | `results/frozen/metrics/exp_A12_metrics.json`, `exp_A13_metrics.json`; `fig_v01_6_win_drop_v02` |
| Design chain D1 to D4 (screen) | `configs/exp_B17`; tuning in `tuning.py` | `exp_B15/stage1_summary.json` (frozen results folder), `results/frozen/metrics/exp_B17_metrics.json`; `fig_r04_1_chain_v02` (S_D is a screening statistic) |
| SACA shapes, patch counts, stage-by-stage tensors | `shapes.py` | `results/shapes/d4_shapes.{md,json}`; `fig_r04_4_window_tokens_v02` (the padding bar is N'·S − L from the shape output), `fig_r03_6_patch_coverage_v02` |
| Score function | `evaluation.py` | `bug_evidence/r06_*`; `fig_score_function_v02` |
| Attention diagnostics | `attention.py`, tests | `exp_A9` (frozen results folder); `fig_r03_1_diagnosis_v02`, `fig_r03_2_probe_v02`, `fig_r03_4_attention_illustrative_v02`, `fig_r03_5_complexity_illustrative_v02` (two are illustrative) |
| LayerNorm arm | `configs/exp_A11_layernorm.yaml` | `exp_A11` (frozen results folder); `fig_r04_5_bn_ln_illustrative_v02` (illustrative) |
| Table 7 statistics | supplement only | `exp_A9/table7.md` (frozen results folder) |

**Figure files.** The paper figures carry the suffix `_v02`: they are redrawn from the same frozen data with no new numbers. The files of the same stem without the suffix are the earlier versions; they are kept in `figures/` and are not for the paper. Ten of them were recoloured on 2026-10-31 with the same palette, data unchanged. Width: single column = 90 mm (3.5 in, 89 mm), double column = 190 mm limit (7.2 in, 183 mm). Script: `python -m saca_rul.reporting.figures_v02 --out-dir figures`. Test: `tests/test_figures_v02.py` (plotted values equal the frozen source values; no identifier in any figure text).

`fig_v01_2_primary_pairs_v02` (manuscript Fig. 3) was redrawn on 2026-10-31 in a colour-blind-safe palette (Okabe-Ito blue `#0072B2` for pairs where SACA is lower, orange `#E69F00` with a hatch where SACA is higher; margin and mean lines in dark grey). Only colours changed: the plotted arrays are identical to before.

**Colour-blind-safe recolouring of all red/green figures (2026-10-31).** The scheme of `fig_v01_2_primary_pairs_v02` (manuscript Fig. 3) is used everywhere: Okabe-Ito blue `#0072B2` and orange `#E69F00`; where a third or fourth category is needed, sky blue `#56B4E9` and grey `#666666`; no red-versus-green pairs. Every colour also has a second cue (hatch, marker or line style). Changed figures (both the `_v02` file and the version-01 file of the same stem, except where only one exists): `fig_v01_2_primary_pairs` (version 01 only; the `_v02` file was done earlier on 2026-10-31), `fig_v01_3_hier_bootstrap` (blue filled circle = CI excludes 0, orange diamond = FD001 ridge borderline, grey open square = CI covers 0), `fig_v01_4_nosmooth_padding` (orange hatched = end-padded, blue = unpadded), `fig_v01_5_size_curve` and `fig_v01_6_win_drop` (four subsets: blue circle solid, orange square dashed, sky-blue triangle dotted, grey diamond dash-dot), `fig_r03_1_diagnosis` (orange hatched bars), `fig_r03_5_complexity_illustrative` (sky-blue hatched band, blue text; the ProbSparse line is dashed grey, the exact line solid blue), `fig_r03_6_patch_coverage` (orange cross-hatched "dropped" patch), `fig_r04_1_chain` (orange hatched = median delta above 0, blue = below; subset lines use the colours and line styles of Figs. S4 and S5 (FD001 blue solid, FD003 sky blue dotted, FD004 grey dash-dot) but with their own markers: FD001 circle, FD003 square, FD004 triangle) and `fig_score_function` (blue solid versus orange dashed). Data identity: the plotted artist data (line, bar, marker and error-bar coordinates, image arrays, tick positions, axis limits) and all text were dumped from the code before and after the change and compared; they are equal in all 32 figures, except the legend note of `fig_v01_3_hier_bootstrap` (colour words only). The 13 figures that did not change regenerate with byte-identical PNG files.

**Supplementary Fig. S1** (`fig_s1_k_selection_v02`): inertia (log scale) and silhouette against k = 2 to 10 for FD002 and FD004, five split seeds each, k = 6 marked. Script: `python -m saca_rul.reporting.fig_s1_k_selection --out-dir figures` (`src/saca_rul/reporting/fig_s1_k_selection.py`). It reads only the saved outputs `k_selection_FD00{2,4}.json` of the frozen results folder and recomputes no clustering; `tests/test_fig_s1_and_palette.py` compares every plotted array with those files. k = 6 was fixed in the code before this check; the figure is a post-hoc confirmation, not a selection step. The five seeds are blue/orange/sky-blue/grey/black with different markers and line styles; the curves nearly coincide.

| Figure file (`figures/<name>.png` and `.pdf`) | Width | Illustrative |
|---|---|---|
| `fig_v01_1_final_rmse_v02`, `fig_v01_2_primary_pairs_v02`, `fig_v01_3_hier_bootstrap_v02`, `fig_v01_4_nosmooth_padding_v02`, `fig_v01_5_size_curve_v02`, `fig_v01_6_win_drop_v02`, `fig_v01_7_fd004_regime_v02` | double | no |
| `fig_r03_1_diagnosis_v02`, `fig_r03_2_probe_v02`, `fig_r03_6_patch_coverage_v02`, `fig_r04_1_chain_v02`, `fig_r04_4_window_tokens_v02` | double | no (probe: synthetic tensors, labelled) |
| `fig_r03_4_attention_illustrative_v02`, `fig_r03_5_complexity_illustrative_v02`, `fig_r04_5_bn_ln_illustrative_v02` | double | yes, labelled in the figure |
| `fig_score_function_v02` | single | no (computed from the formula) |
| `fig_s1_k_selection_v02` (supplementary Fig. S1) | double | no (saved k-selection outputs) |

**Placement widths** (figure text is at least 7 pt only at the stated width). Insert the double-column figures at 183 mm and the single-column figure at 89 mm, for example `\includegraphics[width=183mm]{...}` for the 16 double-width files and `width=89mm` for `fig_score_function_v02`. With `width=\textwidth` this holds only if the text block is at least 183 mm wide (the single-column figure only if it is about 90 mm). A narrower text block, for example a supplement with 2 cm margins on A4 (about 170 mm), scales the text to about 6.5 pt, so use `width=183mm` with `\makebox` or `adjustbox`, or request a supplement-width variant (the figure code draws at `W2 = 7.2` in; a variant needs only that constant changed).

**Caption key and notes for the figures** (typography, hatch and labels; same frozen data, no new numbers). `fig_r03_2_probe_v02` names the GPU as printed in the run-log headers of the same machine (NVIDIA GeForce RTX 5070 Ti Laptop GPU; the probe's own files do not record the device name; the environment notes list a single GPU of this model, and the header of the run log of exp_B17 names it). Its x labels read "<temporal tokens> temporal + <channel tokens> channel tokens, batch <size>"; "search-space maximum" is the largest configuration in the tuning search space. The log axes carry ticks at 10, 20, 50, 100, 200, 500 ms and 200 to 10,000 MiB. In `fig_v01_6_win_drop_v02` "tied rate" is the tied dropout rate (not a p-value). Subscripts use mathtext at full text size (matplotlib's default 70 % script scale is switched off so that subscripts also stay at 7 pt); minus signs are U+2212.

**How Fig. 1 (architecture diagram) was made.** The first draft was generated as an editable PowerPoint slide by a script in the project workspace (not in this repository), from a specification derived from the code. The authors edited the slide in PowerPoint, (a block merge, legend and label wording, line styles and layout), and exported it to PDF; page 1 of the export is copied unchanged to `figures/fig_saca_architecture.pdf`. No script in this repository generates it. It was checked against the code and `results/shapes/d4_shapes.{md,json}`.

Planning graphics (projected GPU budgets) are not regenerated: they are superseded by the measured tuning costs in the paper's table.

## 7. Known errors of the original code and where the code tests them

The table maps each topic to the code, the configuration, the result and the figure. "Frozen results folder" means the authors' full folder of frozen runs, selected with `SACA_WORKSPACE`. It is not part of this repository; the dataset at https://doi.org/10.5281/zenodo.24233550 holds the part that the regeneration commands need (section 1 lists what is not published).

| Topic | Code | Config | Result | Figure |
|---|---|---|---|---|
| Attention applied over heads, not over tokens | `attention.py`, `tests/test_attention.py::test_tokens_mix_across_the_sequence`, `bug_evidence/r01_*` | `exp_B6_theta_star` | `bug_evidence/r01_attention_axis_output.txt` | |
| Batch-row leakage (the output of batch row 0 was added to every sample) | `encoder.py` (tuple unpack), `evaluation.evaluate_test` (batch check), `bug_evidence/r02_*` | `exp_B6_theta_star` | `bug_evidence/r02_batch_dependence_output.txt`, test `test_prediction_is_independent_of_other_batch_members` | |
| Hyper-parameters tuned on the test set | `tuning.py`, `guard.py` | `protocols/protocol_P2.yaml`, `selections/exp_B6_*` | `exp_B6` tuning and the reproduction certification (frozen results folder) | |
| Test selection and the nested tuning procedure | `tuning.py`, `evaluation.py`, release-gate test | all `exp_*` | `test_eval_log.csv` (404 rows, one read per key) | |
| Data leakage between training, validation and test | `data.py` (`Pipeline`, `LeakageError`) | `data_splits.json` | `tests/test_data_eval_pipeline.py` | |
| NASA Score constants | `evaluation.py`, `bug_evidence/r06_*` | | `bug_evidence/r06_nasa_score_output.txt` | `fig_score_function_v02` |
| Traceable numbers and runs | `reporting/tables.py` | `final_rows.json`, `results/frozen/metrics/` | `combined_table_v4.*`, `results/frozen/metrics/*` (per seed, with hashes) | `fig_v01_1_final_rmse_v02` |
| Smoothing | `data.py` (`ema_causal`) | `exp_V1_nosmooth` | `posthoc_R0.23_nosmooth_c`, `nosmooth_b_valsim` (frozen results folder) | `fig_v01_4_nosmooth_padding_v02` |
| Pooling weights and the computational graph | `decoder.py`, `shapes.py` | `exp_B16_d4_main` | `results/shapes/` | `fig_r03_1_diagnosis_v02` |
| Patch count formula | `embeddings.py` (`n_patches_padded`), `shapes.py`, tests | `exp_B16_d4_main`, `exp_A12_*` | `results/shapes/d4_shapes.md` | `fig_r04_4_window_tokens_v02`, `fig_r03_6_patch_coverage_v02` |
| Training-size sweep | `data.py` (`train_engines`), `training.py` | `exp_A10_size_*` | `exp_A10/metrics.json` (frozen results folder) | `fig_v01_5_size_curve_v02` |
| Value of ProbSparse attention | `attention.py` (`ExactAttention`), design D1 | `exp_B17_d1_at_theta` | `exp_A9`, `exp_B17` (frozen results folder) | `fig_r03_2_probe_v02`, `fig_r03_4_attention_illustrative_v02`, `fig_r03_5_complexity_illustrative_v02` |
| Test-window padding | `data.py` (`testing_windows`) | | `posthoc_R0.23_nosmooth_c` (frozen results folder) | `fig_v01_4_nosmooth_padding_v02` |
| Seeding and determinism | `determinism.py` | all | certification of seed 0 of every model (20 of 20), section 4 | |
| Statistics | `stats.py`, `reporting/` | | `combined_table_v4.*`, `posthoc_R2.10` (frozen results folder) | `fig_v01_2_primary_pairs_v02`, `fig_v01_3_hier_bootstrap_v02` |
| Dropout | | `exp_A13_dropout_*` | `exp_A13` (frozen results folder) | `fig_v01_6_win_drop_v02` |
| Simple baselines | `baselines.py` | `exp_V2_lstm`, `exp_V3_ridge` | `combined_table_v4.*` | `fig_v01_1_final_rmse_v02` |
| BatchNorm versus LayerNorm | `encoder.py` (`ChannelLayerNorm`) | `exp_A11_layernorm` | `exp_A11` (frozen results folder) | `fig_r04_5_bn_ln_illustrative_v02` |
| FD004 Score by operating regime | `reporting/posthoc_r124.py` | | `posthoc_R1.24` (frozen results folder) | `fig_v01_7_fd004_regime_v02` |
| Window length | | `exp_A12_L30`, `exp_A12_L40` | `exp_A12` (frozen results folder) | `fig_v01_6_win_drop_v02` |
| Search report of the tuning studies | `tuning.py` | `protocol_P2.yaml` | `exp_B6`, `exp_B16` tuning reports (frozen results folder) | |
| Sensor order | `model.py` (`SACAModel`), test `test_d4_is_invariant_to_sensor_order` | `exp_B18`, `exp_B19` | `exp_B18`, `exp_B19` (frozen results folder) | |
| Strong baselines | `baselines.py`, `vendored/` | `exp_B10`, `exp_B11`, `exp_B12` | `combined_table_v4.*` | `fig_v01_1_final_rmse_v02` |
| Alternatives to adjacent-sensor patches | `model.py` (designs D2 to D4) | `exp_B16_d4_main` | `exp_B15`, `exp_B16` (frozen results folder) | `fig_r04_1_chain_v02` |
| Figures | `reporting/figures_v02.py` (version 01: `figures.py`; it refuses to write a figure wider than 7.2 in or with text below 7 pt; the check includes tick labels and counts mathtext exponents at 70 % of their size) | | `figures/` | all (16 plus Fig. S1; PNG 300 dpi and PDF) |
| k-means k = 6 check (supplementary Fig. S1) | `reporting/fig_s1_k_selection.py` (reads the saved k-selection outputs only) | | `k_selection_FD00{2,4}.json` (frozen results folder) | `fig_s1_k_selection_v02` |

The analysis that `encoder_fusion_experiment*.py` produced (a study of the branch fusion of the original model) is not part of the paper, and the script is left out of this repository. It is not selection code.

## 8. Pinned upstream code

The upstream code is vendored with changes limited to import paths and the replacement of einops calls (Crossformer) or a lazy einops import (iTransformer); seven files are byte copies. Each changed file carries a modification header. Full diffs and licence files are in `src/saca_rul/vendored/VENDORED.md` and `LICENSES/`:

| Repository | Commit | Licence |
|---|---|---|
| PatchTST (yuqinie98) | `204c21efe0b39603ad6e2ca640ef5896646ab1a9` | Apache-2.0 |
| iTransformer (thuml) | `c2426e68ca13f74aaec08045c5c724d8ad328124` | MIT |
| Crossformer (Thinklab-SJTU) | `c10c8eadb153d1dd9798250967747ca3ebb81383` | Apache-2.0 |
| Time-Series-Library heads (written in `baselines.py`, not vendored) | `4e938a1767106324dd753b2a44832bf870a0252e` | MIT |

No `git pull` or `git fetch` was run in the upstream clones; the vendored files are copies of the frozen vendored tree (identical in the two experiment folders that held it).

## 9. Known limits

* The frozen runs used the determinism setting of the earlier experiments; seed 0 of every model was certified bitwise under the strict setting (20 of 20), and seeds 1 to 4 were not re-run (mixed mode, disclosed in the paper).
* `reporting/tables.py`, the post-hoc ports and `figures.py` keep their reviewed code bodies; lint rules for long lines and one-line statements are relaxed for them in `pyproject.toml`, and mypy checks them with untyped defs allowed.
* The FD004 regime analysis reads the operating-setting columns of the test file, so the default checks do not re-run it; the port `reporting/posthoc_r124.py` was never executed (section 5).
* The submitted scripts in `bug_evidence/` import the submitter's original code from its own folder (not released); `bug_evidence/README.md` lists every needed file and states the re-run status. Only `r01` was re-run after the results were finalised (output identical); `r02` and `r06` read FD001 test files and were not re-run.
* The first reproduction check had the limits stated in section 4.1 (code edited during the run; run folders deleted); the second check on the shipped code supersedes them (FD001, seed 0 only).

## 10. Module guide

| Path | Content |
|---|---|
| `src/saca_rul/data.py` | training-split-only pipeline (constant-column drop, k-means operating regimes, correlation pruning, scaler), causal EMA, windows, `DataCache` |
| `src/saca_rul/embeddings.py` | patch embeddings (no padding and end padding), sinusoidal position code, window statistics, sensor ids; patch-count formulas |
| `src/saca_rul/attention.py` | ProbSparse attention (corrected), exact attention, multi-head wrapper |
| `src/saca_rul/encoder.py` | BatchNorm and LayerNorm encoder layer, the branches of the corrected original model, the branches of the designs D2 and D3, the SACA encoder `SACAEncoder` (D4) |
| `src/saca_rul/decoder.py` | fusion head (corrected original model) and sensor-indexed head (D3 and D4); there is no autoregressive decoder |
| `src/saca_rul/model.py` | `ModelConfig`, `ThetaStarModel`, the designs D1 to D4, builders, checkpoints |
| `src/saca_rul/training.py` | strict-deterministic training loop, outer fit (`run_training`, `run_ridge`) |
| `src/saca_rul/tuning.py`, `search_space.py` | validation-only studies (Optuna RandomSampler), selection, freeze of the checkpoint |
| `src/saca_rul/evaluation.py` | metrics, NASA Score, the guarded `evaluate_test` (the only reader of test files) |
| `src/saca_rul/baselines.py`, `vendored/` | baselines; official code at pinned commits with licences and modification headers |
| `src/saca_rul/stats.py`, `reporting/` | paired statistics; table generator, post-hoc analyses, figures, hash verification |
| `src/saca_rul/shapes.py` | SACA shape script (CPU) |
| `configs/` | one YAML per reported training experiment, protocols, frozen selections, arms |
| `tests/` | pytest suite |
| `scripts/reproduce.sh` | stage A (CPU checks), stage B (GPU reproduction check), `--full` (retrain, opt-in) |
| `results/` | regenerated tables, shape output with stage-by-stage tensor shapes, reproduction-check results and the AST-equivalence evidence (`results/repro_check/`), shipped frozen outputs (`results/frozen/`: frozen tables, per-seed `metrics/`, script of record `run_regime_split.py`) |
| `figures/` | 16 paper figures (`*_v02`: SACA labels, no internal identifiers, PNG 300 dpi and PDF, drawn at print size, text at least 7 pt), supplementary Fig. S1 (`fig_s1_k_selection_v02`), the architecture figure `fig_saca_architecture.pdf` (Fig. 1; first drafted by a script in the project workspace, then edited by the authors in PowerPoint; no script in this repository generates it) and the 16 version-01 figures (not for the paper; 10 of them were recoloured on 2026-10-31, data unchanged, see section 6) |
| `bug_evidence/` | `README.md` (needed files, re-run status), scripts and captured output for the three errors of the original code (attention over heads, batch-row leakage, Score constants), byte-identical to the reviewed files |
| `RELEASE_FILES.md` | the file list with sha256 and notes on individual files |
