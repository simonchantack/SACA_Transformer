"""Metrics and the guarded test evaluation.

Metrics (R0.6, R0.7): RMSE, MAE, R^2 and the NASA Score with ``d = RUL_pred - RUL_true`` (Saxena et al., 2008, IEEE PHM;
the printed constants contradict the paper's own text and Figure 9, and we follow the stated intent):

    d <  0 (early):  exp(-d / 13) - 1
    d >= 0 (late):   exp( d / 10) - 1

The submitted code had the two constants swapped (early /10, late /13). ``nasa_score_code_constants`` keeps the swapped
version for the comparison table only.

``evaluate_test`` is the only reader of ``test_FD00x.txt`` and ``RUL_FD00x.txt`` in the release. It refuses unless:
  1. ``selection_frozen.json`` holds an entry for the (subset, seed) with ``frozen_at`` earlier than now and a
     checkpoint sha256 equal to the checkpoint on disk;
  2. no earlier row of the test log exists for the same (experiment, config hash, subset, seed, purpose), unless a
     reason is given;
  3. the TR(seed) pipeline refit reproduces the feature list stored at training;
  4. predictions at eval batch 1024, 1, 64 and 64 with shuffled engines agree within 1e-3 cycles (R0.2 b).
It writes the prediction file and appends one row to the test log. Nothing selects on the test metrics: the checkpoint
is the early-stopped validation checkpoint frozen before the read. During the experiment freeze this function must not
be called (no new test reads).
"""
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

A_EARLY = 13.0
A_LATE = 10.0
TOL = 1e-3
LOG_FIELDS = ["timestamp", "experiment_id", "config_hash", "subset", "seed", "arm", "ckpt_sha256", "purpose",
              "reason", "n_engines", "RMSE", "Score", "batch_invariance_maxdiff", "preds_file"]


def _arr(x: Any) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if np.isnan(a).any():
        raise ValueError("NaN in input")
    return a


def nasa_score(y_true: Any, y_pred: Any) -> dict[str, float | int]:
    """NASA Score, summed over engines. d = pred - true; early (d < 0) uses /13, late (d >= 0) uses /10."""
    yt, yp = _arr(y_true), _arr(y_pred)
    if yt.shape != yp.shape:
        raise ValueError("shape mismatch")
    d = yp - yt
    early = float(np.sum(np.expm1(-d[d < 0] / A_EARLY)))
    late = float(np.sum(np.expm1(d[d >= 0] / A_LATE)))
    return {"total": early + late, "early": early, "late": late, "n": int(d.size),
            "n_early": int((d < 0).sum()), "n_late": int((d >= 0).sum())}


def nasa_score_code_constants(y_true: Any, y_pred: Any) -> float:
    """The submitted (swapped) constants, for the comparison table only: early /10, late /13."""
    d = _arr(y_pred) - _arr(y_true)
    return float(np.sum(np.expm1(-d[d < 0] / 10.0)) + np.sum(np.expm1(d[d >= 0] / 13.0)))


def rmse(y_true: Any, y_pred: Any) -> float:
    d = _arr(y_pred) - _arr(y_true)
    return float(np.sqrt(np.mean(d ** 2)))


def mae(y_true: Any, y_pred: Any) -> float:
    d = _arr(y_pred) - _arr(y_true)
    return float(np.mean(np.abs(d)))


def r2(y_true: Any, y_pred: Any) -> float:
    yt, yp = _arr(y_true), _arr(y_pred)
    sse = float(np.sum((yt - yp) ** 2))
    sst = float(np.sum((yt - yt.mean()) ** 2))
    return float(1.0 - sse / sst) if sst > 0 else float("nan")


def all_metrics(y_true: Any, y_pred: Any) -> dict[str, float | int]:
    s = nasa_score(y_true, y_pred)
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred), "R2": r2(y_true, y_pred),
            "Score": s["total"], "Score_early": s["early"], "Score_late": s["late"], "n": s["n"]}


# ------------------------------------------------------------------------------------------------ test log
def read_log(log_path: str | Path) -> list[dict[str, str]]:
    p = Path(log_path)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_log(log_path: str | Path, row: dict[str, Any]) -> None:
    p = Path(log_path)
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def load_test_split(data_dir: str | Path, subset: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Raw test rows (source = test_file) and capped RUL labels. Called only from ``evaluate_test``."""
    from .data import RUL_CAP, read_cmapss_txt

    te = read_cmapss_txt(Path(data_dir) / f"test_{subset}.txt")
    te["source"] = "test_file"
    rul = pd.read_csv(Path(data_dir) / f"RUL_{subset}.txt", header=None, names=["RUL"])
    y = np.minimum(rul["RUL"].to_numpy(dtype=np.float64), RUL_CAP)
    return te, y


class Refused(RuntimeError):
    """A pre-read check failed; nothing was read and nothing was logged."""

    def __init__(self, code: int, msg: str) -> None:
        super().__init__(msg)
        self.code = code


def evaluate_test(exp_dir: str | Path, subset: str, seed: int, data_dir: str | Path, splits_path: str | Path,
                  log_path: str | Path, purpose: str = "final", reason: str = "", norm: str = "batch"
                  ) -> dict[str, Any]:
    """One logged test read of the frozen checkpoint of ``(subset, seed)``. See the module docstring."""
    import hashlib

    from . import checkpoint as CK
    from . import data as D

    exp = Path(exp_dir).resolve()
    fz_path = exp / "selection_frozen.json"
    if not fz_path.exists():
        raise Refused(2, f"no selection_frozen.json in {exp}")
    fz = json.loads(fz_path.read_text(encoding="utf-8"))
    ent_l = [e for e in fz["entries"] if e["subset"] == subset and int(e["seed"]) == seed]
    if len(ent_l) != 1:
        raise Refused(2, f"selection_frozen.json has {len(ent_l)} entries for {subset} {seed}")
    ent = ent_l[0]
    now = dt.datetime.now()
    if dt.datetime.fromisoformat(ent["frozen_at"]) >= now:
        raise Refused(2, "frozen_at is not earlier than now")
    ckpt = exp / ent["ckpt_path"]
    if hashlib.sha256(ckpt.read_bytes()).hexdigest() != ent["ckpt_sha256"]:
        raise Refused(2, "checkpoint sha256 differs from selection_frozen.json")
    exp_id = fz["experiment_id"]
    prior = [r for r in read_log(log_path) if r["experiment_id"] == exp_id and r["config_hash"] == ent["config_hash"]
             and r["subset"] == subset and int(r["seed"]) == seed and r["purpose"] == purpose]
    if prior and not reason:
        raise Refused(3, f"repeat evaluation needs a reason; {len(prior)} prior row(s)")

    model, meta = CK.load_for_eval(ckpt, norm=norm)
    pp = meta["preprocess"]
    splits = D.load_splits(splits_path)
    raw = D.load_train_file(data_dir, subset)
    tr_ids, _ = D.split_engines(splits, subset, seed)
    if meta.get("tr_engines_override") is not None:
        tr_ids = list(meta["tr_engines_override"])
    pipe = D.Pipeline(subset, seed, smoothing=pp["smoothing"], alpha=pp["alpha"], legacy_corr=pp["legacy_corr"],
                      train_engines=tr_ids).fit(raw[raw.engine.isin(tr_ids)])
    if pipe.features != meta["features"]:
        raise Refused(4, f"refit feature list differs from training: {pipe.features} vs {meta['features']}")
    te_raw, y = load_test_split(data_dir, subset)
    te = pipe.transform(te_raw, with_labels=False)
    Xte, eng, cyc = D.testing_windows(te, pipe.features, meta["L"])
    X = np.ascontiguousarray(Xte.transpose(0, 2, 1))
    assert len(eng) == len(y), "engine count vs RUL file"
    p_main = CK.predict(model, X, 1024)
    diffs: dict[str, float] = {}
    for name, bs, order in (("b1", 1, None), ("b64", 64, None),
                            ("b64_shuffled", 64, np.random.default_rng(12345).permutation(len(X)))):
        p = CK.predict(model, X, bs, order=order)
        diffs[name] = float(np.abs(p - p_main).max())
    maxdiff = max(diffs.values())
    exp.joinpath("preds").mkdir(exist_ok=True)
    pf = Path("preds") / f"preds_{subset}_seed{seed}_test.csv"
    pd.DataFrame({"engine_id": eng, "cycle_end": cyc, "y_true_capped": y, "y_pred": p_main}).to_csv(
        exp / pf, index=False, float_format="%.6f")
    stored = pd.read_csv(exp / pf)  # the float_format rounding defines the stored prediction
    m = all_metrics(stored.y_true_capped.to_numpy(), stored.y_pred.to_numpy())
    append_log(log_path, dict(timestamp=now.isoformat(timespec="seconds"), experiment_id=exp_id,
                              config_hash=ent["config_hash"], subset=subset, seed=seed, arm=ent.get("arm", ""),
                              ckpt_sha256=ent["ckpt_sha256"], purpose=purpose, reason=reason, n_engines=len(y),
                              RMSE=f"{m['RMSE']:.6f}", Score=f"{m['Score']:.6f}",
                              batch_invariance_maxdiff=f"{maxdiff:.3e}", preds_file=str(pf).replace("\\", "/")))
    res = {"subset": subset, "seed": seed, "metrics": m, "batch_invariance": diffs,
           "batch_invariance_pass": maxdiff <= TOL, "preds_file": str(pf),
           "n_test_shorter_than_L": int((te.groupby("engine").size() < meta["L"]).sum())}
    (exp / "preds" / f"test_eval_{subset}_seed{seed}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res
