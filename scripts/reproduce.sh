#!/usr/bin/env bash
# Reproduce every reported experiment of the release, end to end, in the stages the freeze allows.
#
#   scripts/reproduce.sh                 stage A: tests, frozen-table and post-hoc hash checks, shape script, figures
#   scripts/reproduce.sh --repro-gpu     stage A + the GPU reproduction check (6 validation-only retrains)
#   SACA_ALLOW_RETRAIN=1 scripts/reproduce.sh --full
#                                        stage A + a complete retrain of every configuration of configs/exp_*.yaml
#                                        (4 subsets x 5 seeds each, strict deterministic, validation only), then the
#                                        freeze of each early-stopped checkpoint. Test reads are NOT
#                                        issued here: each is a single logged read (see README, "Test reads").
#
# Environment (all optional; defaults are relative to the release root):
#   PYTHON            interpreter with the packages of pyproject.toml (default: python)
#   SACA_DATA_DIR     folder with the C-MAPSS train_FD00x.txt files (default ../workspace/code_base/Data)
#   SACA_WORKSPACE    frozen workspace with experiments/ (default ../workspace)
#   SACA_OUTPUT_DIR   where retrains write (default results/runs)
#
# This is the supported route. On Windows run it from Git Bash (there is no PowerShell script in the release).
# Tested: stage A ran end to end with this script (bash).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
SEP=":"; SRC="$ROOT/src"
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) SEP=";"; SRC="$(cygpath -m "$ROOT/src")";; esac
export PYTHONPATH="$SRC${PYTHONPATH:+$SEP$PYTHONPATH}"
export PYTHONIOENCODING=utf-8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
MODE="${1:-}"

echo "== stage A1: tests (CPU)"
CUDA_VISIBLE_DEVICES= "$PYTHON" -m pytest

echo "== stage A2: regenerate the combined table v4 and compare with the frozen hashes"
CUDA_VISIBLE_DEVICES= "$PYTHON" -m saca_rul.cli verify-tables

echo "== stage A3: regenerate R2.10, R0.23+(c) and no-smoothing (b); compare with the frozen JSON"
CUDA_VISIBLE_DEVICES= "$PYTHON" -m saca_rul.cli verify-posthoc

echo "== stage A4: D4 shape script (CPU)"
CUDA_VISIBLE_DEVICES= "$PYTHON" -m saca_rul.cli shapes

echo "== stage A5: figures (PNG 300 dpi + PDF)"
CUDA_VISIBLE_DEVICES= "$PYTHON" -m saca_rul.cli figures
CUDA_VISIBLE_DEVICES= "$PYTHON" -m saca_rul.reporting.figures_v02 --out-dir figures   # version 02 (paper figures, suffix _v02)
CUDA_VISIBLE_DEVICES= "$PYTHON" -m saca_rul.reporting.fig_s1_k_selection --out-dir figures   # supplementary Fig. S1 (saved k-selection outputs only)

echo "== stage A6: AST equivalence of the shipped modules with the tree of the reproduction check (CPU, no data)"
"$PYTHON" scripts/ast_equivalence.py

if [[ "$MODE" == "--repro-gpu" ]]; then
  echo "== stage B: GPU reproduction check (validation only; compares best epoch, state dict and preds_val hashes)"
  "$PYTHON" -m saca_rul.cli repro
fi

if [[ "$MODE" == "--full" ]]; then
  if [[ "${SACA_ALLOW_RETRAIN:-0}" != "1" ]]; then
    echo "REFUSED: --full retrains every experiment. Not allowed during the experiment freeze (RULES 23)." >&2
    echo "Set SACA_ALLOW_RETRAIN=1 only after the freeze is lifted." >&2
    exit 2
  fi
  for cfg in configs/exp_*.yaml; do
    for sub in FD001 FD002 FD003 FD004; do
      for seed in 0 1 2 3 4; do
        echo "== train $cfg $sub seed $seed"
        "$PYTHON" -m saca_rul.cli train --config "$cfg" --subset "$sub" --seed "$seed"
        "$PYTHON" -m saca_rul.cli freeze --config "$cfg" --subset "$sub" --seed "$seed" || true
      done
    done
  done
fi
echo "done"
