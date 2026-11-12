# Replay check: launch tree + recorded post-launch edits == shipped tree

**Time of this check: 2026-10-27 05:07 local (fix attempt 1).** "RELEASE_FILES.md" in the table below means the RELEASE_FILES.md
of the FIRST release (before fix attempt 1), not the current one. **Six files changed after this check**, so for them the
"replayed == shipped" and "shipped == RELEASE_FILES.md" columns no longer hold and must not be read as current:
`src/saca_rul/cli.py` (usage text, docstring only), `src/saca_rul/shapes.py`, `src/saca_rul/reporting/figures.py`
(rewritten, and again in fix attempt 2), `src/saca_rul/reporting/posthoc_r124.py` (docstring only), `tests/test_frozen.py`
and `scripts/make_release_list.py` (changed 05:17 in fix attempt 1: `repro_check` dropped from `SKIP_DIRS`, two external-artifact notes extended).
The table itself is left as it was at 05:07.

The launch tree was rebuilt by replaying the Write/Edit/Bash tool calls of the build session (up to the relaunch at 2026-10-27 04:37:53 local).
The post-launch steps (ruff --fix, then the scripted typing/lint edits and the new tune/freeze sub-commands) were replayed on top of it,
with the same ruff 0.17.3 and the pyproject.toml of that moment. Result per shipped .py file (outside vendored/):

| file | replayed == shipped (bytes) | shipped sha256 == RELEASE_FILES.md sha256 |
|---|---|---|
| `bug_evidence/_fd001_pipeline.py` | True | True |
| `bug_evidence/r01_attention_axis.py` | True | True |
| `bug_evidence/r02_batch_dependence.py` | True | True |
| `bug_evidence/r06_nasa_score.py` | True | True |
| `scripts/ast_equivalence.py` | n/a (not in replay) | not listed |
| `scripts/make_configs.py` | True | True |
| `scripts/make_release_list.py` | n/a (not in replay) | True |
| `src/saca_rul/__init__.py` | True | True |
| `src/saca_rul/attention.py` | True | True |
| `src/saca_rul/baselines.py` | True | True |
| `src/saca_rul/checkpoint.py` | True | True |
| `src/saca_rul/cli.py` | True | True |
| `src/saca_rul/config.py` | True | True |
| `src/saca_rul/data.py` | True | True |
| `src/saca_rul/decoder.py` | True | True |
| `src/saca_rul/determinism.py` | True | True |
| `src/saca_rul/embeddings.py` | True | True |
| `src/saca_rul/encoder.py` | True | True |
| `src/saca_rul/evaluation.py` | True | True |
| `src/saca_rul/gpulock.py` | True | True |
| `src/saca_rul/guard.py` | True | True |
| `src/saca_rul/model.py` | True | True |
| `src/saca_rul/reporting/__init__.py` | n/a (not in replay) | True |
| `src/saca_rul/reporting/figures.py` | True | True |
| `src/saca_rul/reporting/hier_bootstrap.py` | True | True |
| `src/saca_rul/reporting/posthoc_common.py` | True | True |
| `src/saca_rul/reporting/posthoc_r023.py` | True | True |
| `src/saca_rul/reporting/posthoc_r124.py` | True | True |
| `src/saca_rul/reporting/posthoc_r210.py` | True | True |
| `src/saca_rul/reporting/posthoc_valsim.py` | True | True |
| `src/saca_rul/reporting/tables.py` | True | True |
| `src/saca_rul/reporting/verify.py` | True | True |
| `src/saca_rul/runner.py` | True | True |
| `src/saca_rul/search_space.py` | True | True |
| `src/saca_rul/shapes.py` | True | True |
| `src/saca_rul/stats.py` | True | True |
| `src/saca_rul/training.py` | True | True |
| `src/saca_rul/tuning.py` | True | True |
| `tests/conftest.py` | True | True |
| `tests/test_attention.py` | True | True |
| `tests/test_data_eval_pipeline.py` | True | True |
| `tests/test_frozen.py` | True | True |
| `tests/test_models.py` | True | True |

Vendored tree: `diff -rq` between the replayed launch tree and the shipped `src/saca_rul/vendored` printed nothing (byte-identical).
Not in the replay (written after the relaunch): `scripts/make_release_list.py`, `scripts/ast_equivalence.py`; `reporting/__init__.py` is an empty file.
