"""Run one cell (subset, seed) of an experiment config, and the reproduction check of frozen cells.

A config names the family, the protocol YAML, the frozen selection JSON and the one-factor arm (if any). ``run_cell``
turns it into a ``RunSpec`` and calls ``training.run_training`` (or ``run_ridge``). ``repro_check`` re-trains cells in a
scratch folder and compares the validation predictions, the state dict and the best epoch with the frozen run, and
reports match or mismatch only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import training as T
from .config import load_config


def selection_for(cfg: dict[str, Any], subset: str) -> tuple[dict[str, Any], str, str]:
    """Return (selection entry for ``subset``, selection file sha256, experiment id of the selection file)."""
    path = (cfg.get("selection_by_subset") or {}).get(subset) or cfg["selection"]
    js = json.loads(Path(path).read_text(encoding="utf-8"))
    return js["per_subset"][subset], T.sha256_file(path), str(js["experiment_id"])


def make_spec(cfg: dict[str, Any], subset: str, seed: int, out_dir: str | Path | None = None, device: str = "cuda",
              max_epochs: int | None = None) -> T.RunSpec:
    ent, sel_sha, _ = selection_for(cfg, subset)
    paths = cfg["paths"]
    return T.RunSpec(experiment_id=cfg["experiment_id"], family=cfg["family"], subset=subset, seed=int(seed),
                     params=dict(ent["params"]), protocol_path=Path(cfg["protocol"]), data_dir=Path(paths["data_dir"]),
                     splits_path=Path(paths["splits"]), out_dir=Path(out_dir or paths["output_dir"]), device=device,
                     max_epochs=max_epochs, selection_sha256=sel_sha, trial_id=ent.get("trial_id"),
                     arm=dict(cfg.get("arm") or {}))


def run_cell(config_path: str | Path, subset: str, seed: int, out_dir: str | Path | None = None,
             device: str = "cuda", max_epochs: int | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    spec = make_spec(cfg, subset, seed, out_dir, device, max_epochs)
    if cfg["family"] == "ridge":
        return T.run_ridge(spec)
    return T.run_training(spec)


def _fsha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compare_with_frozen(new_run: Path, frozen_run: Path) -> dict[str, Any]:
    """Booleans only: best epoch, state-dict sha256 and preds_val.csv sha256 equal to the frozen run."""
    nm = json.loads((new_run / "train_meta.json").read_text(encoding="utf-8"))
    fm = json.loads((frozen_run / "train_meta.json").read_text(encoding="utf-8"))
    res = {"best_epoch_equal": nm["best_epoch_rmse"] == fm["best_epoch_rmse"],
           "state_dict_equal": T.state_dict_sha256(new_run / "ckpt_rmse.pt") == T.state_dict_sha256(frozen_run / "ckpt_rmse.pt"),
           "preds_val_equal": _fsha(new_run / "preds_val.csv") == _fsha(frozen_run / "preds_val.csv")}
    res["match"] = all(res.values())
    return res


def repro_check(cells: list[dict[str, Any]], scratch: str | Path, device: str = "cuda") -> list[dict[str, Any]]:
    """Re-train each cell ``{"config": ..., "subset": ..., "seed": ...}`` in ``scratch`` and compare with the frozen run
    folder ``<paths.frozen_dir>/runs/<subset>_seed<seed>``. Validation only; no test read."""
    out = []
    for c in cells:
        cfg = load_config(c["config"])
        run_cell(c["config"], c["subset"], c["seed"], out_dir=scratch, device=device)
        tag = f"{c['subset']}_seed{c['seed']}"
        fdir = cfg["paths"].get("frozen_dir_by_subset", {}).get(c["subset"], cfg["paths"]["frozen_dir"])
        cmp = compare_with_frozen(Path(scratch) / "runs" / tag, Path(fdir) / "runs" / tag)
        out.append({"config": Path(c["config"]).name, "subset": c["subset"], "seed": c["seed"], **cmp})
    return out
