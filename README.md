# Sensor-Anchored Cross-Dimension Attention for Remaining Useful Life Prediction of Turbofan Engines: A Diagnosis-Driven Redesign

[![DOI](https://zenodo.org/badge/1439133050.svg)](https://doi.org/10.5281/zenodo.24152407)

This repository holds the code of the paper above. It predicts the remaining useful life (RUL) of turbofan engines from the NASA C-MAPSS data (Commercial Modular Aero-Propulsion System Simulation). It contains SACA (sensor-anchored cross-dimension attention), the corrected original model (θ\*), five baselines (PatchTST, iTransformer, Crossformer, an LSTM and ridge regression), the data pipeline, the evaluation code, and the scripts that regenerate the tables and figures of the paper from the frozen results. SACA is named D4 in the code; D1 to D3 are the intermediate designs.

## Paper

- **Paper:** Sensor-Anchored Cross-Dimension Attention for Remaining Useful Life Prediction of Turbofan Engines: A Diagnosis-Driven Redesign.
- **Authors:** Simon Chan Tack and Subash Chandra Pakhrin.
- **Status:** under review.

## How to cite

If you use this code, please cite the archived release (version DOI): [10.5281/zenodo.24237650](https://doi.org/10.5281/zenodo.24237650). Machine-readable citation metadata are in `CITATION.cff`.

## Main result

We corrected the code and the evaluation protocol of the original model: engine-level data splits, tuning and model selection on validation data only, and one test read per final configuration. With this protocol, three Transformer baselines (PatchTST, iTransformer, Crossformer) had a lower mean test root mean square error (RMSE) than the corrected original model on all four C-MAPSS subsets. SACA had a lower test RMSE than the corrected original model in 16 of 20 subset-seed pairs (exact Wilcoxon p = 0.0004). Against the best baseline of each subset, chosen on validation data, the preregistered primary comparison was inconclusive (p = 0.0696). SACA was non-inferior to that baseline at a margin of 0.5 RMSE cycles (one-sided p = 0.00043). We do not claim that SACA outperforms the baselines. All numbers come from five split seeds on FD001 to FD004; see the paper and `results/frozen/` for the tables.

## Requirements and installation

You need Python 3.12 (tested with 3.12.10). The exact package versions are pinned in `pyproject.toml` (torch 2.8.0, numpy 2.1.3, pandas 2.3.1, scipy 1.16.1, scikit-learn 1.7.1, optuna 4.5.0, PyYAML 6.0.3, matplotlib 3.10.5). The tests and the table and figure scripts run on a CPU. Training uses a CUDA GPU by default.

Windows (PowerShell):

```powershell
git clone https://github.com/simonchantack/SACA_Transformer.git
cd SACA_Transformer
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1   # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Linux and macOS:

```bash
git clone https://github.com/simonchantack/SACA_Transformer.git
cd SACA_Transformer
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

For a CUDA build of torch, install it first from https://pytorch.org (the paper runs used torch 2.8.0 with CUDA 12.8), then run `pip install -e ".[dev]"`.

## Data

The code does not include the data. Download "CMAPSS Jet Engine Simulated Data" from the NASA Open Data Portal: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data (identifier `ff5v-kuh6`). Unpack the twelve text files `train_FD001.txt` to `train_FD004.txt`, `test_FD001.txt` to `test_FD004.txt` and `RUL_FD001.txt` to `RUL_FD004.txt` into one folder, directly inside it (no sub-folders), and tell the code where it is:

```powershell
$env:SACA_DATA_DIR = "C:\data\CMAPSS"        # Windows PowerShell
```

```bash
export SACA_DATA_DIR=/data/CMAPSS             # Linux and macOS
```

The saved predictions, result files and checkpoints behind the paper are a separate dataset, [10.5281/zenodo.24233550](https://doi.org/10.5281/zenodo.24233550); see Quick start.

## Quick start

Run the tests (no data needed; tests that need the data or the frozen results are skipped when `SACA_WORKSPACE` is not set; if you set it to the dataset, add `-m "not frozen"`):

```bash
python -m pytest -p no:cacheprovider
```

Print the tensor shapes of SACA at the frozen configurations. This needs the C-MAPSS data (`SACA_DATA_DIR`, see Data) and runs on CPU; it reproduces the shipped `results/shapes/`:

```bash
python -m saca_rul.cli shapes
```

Regenerate the tables, the post-hoc analyses and the figures from the frozen outputs. They are published as a separate dataset (saved predictions, result files and checkpoints): [10.5281/zenodo.24233550](https://doi.org/10.5281/zenodo.24233550). Download the zip file and unzip it. Set `SACA_WORKSPACE` to the unzipped folder, the one that contains `experiments/`:

```powershell
$env:SACA_WORKSPACE = "C:\data\saca_frozen_outputs"   # Windows PowerShell
```

```bash
export SACA_WORKSPACE=/data/saca_frozen_outputs            # Linux and macOS
```

`verify-posthoc` also runs a no-smoothing simulation that needs `SACA_DATA_DIR` (see Data). Then:

```bash
python -m saca_rul.cli verify-tables
python -m saca_rul.cli verify-posthoc
python -m saca_rul.reporting.figures_v02 --out-dir figures
python -m saca_rul.reporting.fig_s1_k_selection --out-dir figures
```

The dataset supports these regeneration commands. The tests marked `frozen` and `scripts/reproduce.sh` need the authors' full project folder, not only the dataset; with `SACA_WORKSPACE` set to the dataset, run `python -m pytest -m "not frozen"`.

The tables and figures written by these commands are compared with the frozen hashes; the shipped frozen tables are in `results/frozen/`. With the full project folder, `scripts/reproduce.sh` (Linux, macOS or Git Bash) runs the tests and all these steps in one go; with the dataset alone, run the commands above.

Retrain one model (needs the data and a GPU; add `--device cpu` for a slow CPU run):

```bash
python -m saca_rul.cli train --config configs/exp_B16_d4_main.yaml --subset FD001 --seed 0
```

The environment, the determinism settings, the hash checks and the test-set guard are described in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Repository layout

| Path | Content |
|---|---|
| `src/saca_rul/` | the package: `data.py` (pipeline), `embeddings.py`, `attention.py`, `encoder.py`, `decoder.py`, `model.py` (θ\* and SACA), `training.py`, `tuning.py`, `evaluation.py` (metrics and the guarded test read), `baselines.py`, `stats.py`, `shapes.py`, `reporting/` (tables, analyses, figures), `vendored/` (upstream baseline code) |
| `configs/` | one YAML file per reported experiment, the tuning protocols, the frozen selections and the data splits |
| `scripts/` | `reproduce.sh`, the release list generator and the equivalence check |
| `tests/` | the test suite |
| `results/` | frozen tables and per-seed metrics (`frozen/`), tensor shapes (`shapes/`), reproduction-check results (`repro_check/`), regenerated tables (`tables/`) |
| `figures/` | the paper figures (PNG and PDF). Fig. 1, architecture diagram; see docs/REPRODUCIBILITY.md for how it was made |
| `bug_evidence/` | short scripts and captured output showing the three errors found in the original code |
| `docs/` | `REPRODUCIBILITY.md` |
| `LICENSE`, `CITATION.cff`, `RELEASE_FILES.md` | licence, citation metadata, and the list of all files with their sha256 |

## Licence

MIT (see `LICENSE`). The upstream baseline code in `src/saca_rul/vendored/` keeps its own licence files and modification headers (see `src/saca_rul/vendored/VENDORED.md`).

## Contact

Subash Chandra Pakhrin, pakhrins@uhd.edu.
