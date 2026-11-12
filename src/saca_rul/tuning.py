"""Validation-only hyper-parameter search (protocol P2, Algorithm 1 steps 1-5) and the freeze of its outcome.

One study = one (model family, subset, study seed). The sampler is ``optuna.samplers.RandomSampler(seed = study
seed)`` with no pruner; draws are taken in the YAML order of the search space; infeasible draws and out-of-memory draws
are rejected and redrawn without counting. The trial value is the best common-support validation RMSE of split seed 0
(``data.tuning_split_seed``). The test split is never opened (the guard is installed first).

``select_config`` pools the completed trials of the three studies of a subset and freezes theta* = argmin (ties go to
the configuration with fewer parameters). ``freeze_selection`` records the early-stopped checkpoint of a retrain in
``selection_frozen.json`` before any test read.
"""
from __future__ import annotations

import datetime as dt
import json
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from . import determinism, guard
from . import training as T
from .data import DataCache, load_splits
from .search_space import Infeasible, materialize, resolve_space, space_family, suggest

STUDY_SEEDS = (0, 1, 2)


def run_study(family: str, subset: str, study_seed: int, protocol_path: str | Path, data_dir: str | Path,
              splits_path: str | Path, out_root: str | Path, exp_id: str, device: str = "cuda",
              timeout_min: float | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Run one study to its target number of completed trials. Returns the ``result.json`` dict."""
    import optuna

    guard.install()
    pr = T.load_protocol(protocol_path)
    dry = dry_run
    if not dry and study_seed not in pr["study_seeds"]:
        raise ValueError(f"study seed {study_seed} not in protocol study_seeds {pr['study_seeds']}")
    space, opt = resolve_space(pr, family, dry)
    n_complete = int(pr["dry_run"]["trials_per_study"] if dry else opt["trials_per_study"])
    max_ep = int(pr["dry_run"]["max_epochs"] if dry else pr["recipe"]["max_epochs"])
    patience = int(pr["dry_run"]["patience"] if dry else pr["recipe"]["patience"])
    out = Path(out_root) / f"{family}_{subset}_study{study_seed}"
    (out / "hist").mkdir(parents=True, exist_ok=True)
    logf = open(out / "tune.log", "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{dt.datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    mem_cap_mb = T.apply_memory_cap(dev, pr)
    splits = load_splits(splits_path)
    tseed = int(splits["tuning_split_seed"])
    assert tseed == int(pr["data"]["tuning_split_seed"]), "tuning split seed differs from protocol"
    cache = DataCache(subset, tseed, data_dir, splits_path, smoothing=pr["data"]["smoothing"],
                      alpha=float(pr["data"]["ema_alpha"]),
                      tr_fraction=float(pr["dry_run"]["tr_fraction"]) if dry else 1.0,
                      support_min_label_cycle=int(pr["objective"]["support_min_label_cycle"]))
    log(f"# tuning start | model {family} subset {subset} study_seed {study_seed} dry_run {dry} device {dev}")
    log(f"# python {platform.python_version()} torch {torch.__version__} optuna {optuna.__version__} "
        f"numpy {np.__version__} pandas {pd.__version__}")
    log(f"# protocol sha256 {pr['_sha256']}; tuning split seed {tseed}; budget option {pr['budget_option']} {opt}")
    log(f"# test guard installed {guard.is_installed()}; OOM cap {mem_cap_mb} MB; eval batch {pr['_eval_bs']}")
    log(f"# C {cache.C} features {cache.features}; TR engines {len(cache.tr_ids)} VA engines {len(cache.va_ids)}")
    sids = T.sensor_ids_for(family, cache.features)
    fixed = pr["fixed"].get(space_family(family))

    storage = "sqlite:///" + str(out / "study.db").replace("\\", "/")
    study = optuna.create_study(study_name=f"P2_{family}_{subset}_s{study_seed}", storage=storage,
                                direction="minimize", load_if_exists=True,
                                sampler=optuna.samplers.RandomSampler(seed=study_seed),
                                pruner=optuna.pruners.NopPruner())
    resumed = len(study.trials) > 0
    timeout_s = (timeout_min * 60) if timeout_min else None
    t0 = time.time()
    deadline = (t0 + timeout_s) if timeout_s else None

    def objective(trial: Any) -> float:
        params = suggest(trial, space)
        try:
            cfg, L, rec = materialize(family, params, cache.C, fixed)
            tseed_tr = 1000 * study_seed + trial.number
            determinism.set_all(tseed_tr)
            model, _mc = T.build_for_family(family, cfg, cache.C, L, sids)
        except Infeasible as e:
            trial.set_user_attr("rejected_infeasible", str(e))
            raise optuna.TrialPruned(f"rejected: {e}") from e
        w = cache.windows(L)
        trial.set_user_attr("trial_seed", tseed_tr)
        trial.set_user_attr("n_params", T.n_params(model))
        trial.set_user_attr("model_cfg", json.dumps(cfg, sort_keys=True, default=str))
        trial.set_user_attr("n_train_windows", int(len(w["ytr"])))
        trial.set_user_attr("n_val_support", int(w["support"].sum()))
        ts = time.time()
        oom = None
        best, best_ep = float("nan"), 0
        hist: list[dict[str, Any]] = []
        try:
            best, best_ep, hist = T.train_one(model, w, dict(pr["recipe"], **rec), dev, tseed_tr, max_ep, patience,
                                              eval_bs=pr["_eval_bs"], deadline=deadline)
        except Exception as e:  # noqa: BLE001
            if not T.is_oom(e):
                raise
            oom = str(e).splitlines()[0][:300]
        finally:
            trial.set_user_attr("wall_s", time.time() - ts)
        if oom is not None:  # OOM rule: configuration invalid, redrawn, not counted
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            trial.set_user_attr("rejected_oom", oom)
            raise optuna.TrialPruned(f"rejected (OOM): {oom}")
        pd.DataFrame(hist).to_csv(out / "hist" / f"trial_{trial.number}.csv", index=False)
        bh = hist[best_ep - 1]
        trial.set_user_attr("best_epoch", best_ep)
        trial.set_user_attr("epochs_run", len(hist))
        trial.set_user_attr("val_RMSE_allwin_at_best", bh["val_RMSE_allwin"])
        trial.set_user_attr("s_per_epoch", float(np.mean([h["epoch_s"] for h in hist])))
        trial.set_user_attr("ms_per_step", 1000 * float(np.sum([h["epoch_s"] for h in hist])) /
                            max(1, sum(h["steps"] for h in hist)))
        trial.set_user_attr("peak_mem_MB", float(max(h["peak_mem_MB"] for h in hist)))
        return best

    def cb(st: Any, tr: Any) -> None:
        log(f"trial {tr.number} {tr.state.name} value {tr.value} params {json.dumps(tr.params)} "
            f"best_epoch {tr.user_attrs.get('best_epoch')} epochs {tr.user_attrs.get('epochs_run')} "
            f"wall {tr.user_attrs.get('wall_s', 0):.0f}s {tr.user_attrs.get('rejected_infeasible', '')}")
        n_done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in st.trials)
        last = [t.state for t in st.trials[-3:]]
        if n_done >= n_complete or (deadline and time.time() > deadline):
            st.stop()
        elif len(last) == 3 and all(x == optuna.trial.TrialState.FAIL for x in last):
            log("# 3 consecutive FAIL trials: study stopped for inspection (reported, not hidden)")
            st.stop()

    S = optuna.trial.TrialState
    n_done0 = sum(t.state == S.COMPLETE for t in study.trials)
    if n_done0 < n_complete:
        study.optimize(objective, n_trials=int(pr["max_draws_per_study"]), timeout=timeout_s, callbacks=[cb],
                       catch=(T.TrialTimeout, RuntimeError, torch.cuda.OutOfMemoryError))
    trials = study.trials
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)
    comp = [t for t in trials if t.state == S.COMPLETE]
    rej = [t for t in trials if t.state == S.PRUNED and "rejected_infeasible" in t.user_attrs]
    rej_oom = [t for t in trials if t.state == S.PRUNED and "rejected_oom" in t.user_attrs]
    fail = [t for t in trials if t.state == S.FAIL]
    wall = time.time() - t0
    best_t = min(comp, key=lambda t: float(t.value)) if comp else None  # type: ignore[arg-type]
    res = {"model": family, "experiment_id": exp_id, "subset": subset, "study_seed": study_seed, "dry_run": dry,
           "resumed": resumed, "tuning_split_seed": tseed, "budget_option": pr["budget_option"],
           "protocol_sha256": pr["_sha256"], "n_draws": len(trials), "n_complete": len(comp),
           "n_rejected_infeasible": len(rej), "n_rejected_oom": len(rej_oom), "mem_cap_MB": mem_cap_mb,
           "n_failed": len(fail), "target_complete": n_complete, "wall_s": wall,
           "gpu_h": (wall / 3600.0) if dev.type == "cuda" else 0.0, "device": str(dev),
           "best_value": best_t.value if best_t else None, "best_trial": best_t.number if best_t else None,
           "best_params": best_t.params if best_t else None, "timeout_min": timeout_min, "pruner": "none",
           "sampler": f"RandomSampler(seed={study_seed})", "test_guard_blocked": guard.blocked_count(),
           "finished": dt.datetime.now().isoformat(timespec="seconds")}
    (out / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log("# result " + json.dumps(res))
    logf.close()
    return res


def select_config(family: str, subset: str, exp_dir: str | Path, tune_root: str | Path, protocol_path: str | Path,
                  exp_id: str, study_seeds: tuple[int, ...] = STUDY_SEEDS,
                  out_name: str = "config_selection_frozen.json",
                  study_roots: dict[int, str] | None = None) -> dict[str, Any]:
    """Algorithm 1 step 5: pool the completed trials of the studies, pick theta* = argmin, freeze it.

    An existing subset entry is never changed. ``study_roots`` maps a study seed to a different tune root."""
    guard.install()
    pr = T.load_protocol(protocol_path)
    exp = Path(exp_dir)
    roots = {int(k): v for k, v in (study_roots or {}).items()}
    frames = []
    for s in study_seeds:
        d = Path(roots.get(int(s), str(tune_root))) / f"{family}_{subset}_study{s}"
        df = pd.read_csv(d / "trials.csv")
        df["study_seed"] = s
        frames.append(df)
    allt = pd.concat(frames, ignore_index=True)
    comp = allt[allt.state == "COMPLETE"].copy()
    comp["trial_id"] = comp.study_seed.astype(str) + ":" + comp.number.astype(str)
    comp = comp.sort_values(["value", "user_attrs_n_params"]).reset_index(drop=True)
    pcols = [c for c in comp.columns if c.startswith("params_")]
    win = comp.iloc[0]
    rep = exp / "report"
    rep.mkdir(parents=True, exist_ok=True)
    keep = ["trial_id", "value", "user_attrs_val_RMSE_allwin_at_best", "user_attrs_best_epoch", "user_attrs_epochs_run",
            "user_attrs_n_params", "user_attrs_wall_s"] + pcols
    keep = [k for k in keep if k in comp.columns]
    comp[keep].head(10).to_csv(rep / f"{subset}_top10.csv", index=False)
    sb = comp.groupby("study_seed").head(1).sort_values("study_seed")
    pooled_path = rep / f"{subset}_pooled_trials.csv"
    allt.to_csv(pooled_path, index=False)
    params = {str(c).replace("params_", ""): (v.item() if hasattr(v, "item") else v) for c, v in win[pcols].items()}
    for k, v in list(params.items()):  # restore the exact YAML choice object (int / bool / float)
        spec = pr["spaces"].get(space_family(family), {}).get(k) or pr["shared_space"].get(k)
        if spec and spec.get("type") == "categorical":
            ch = pr["options"][pr["budget_option"]]["batch_choices"] if spec["choices"] == "OPTION" else spec["choices"]
            m = [c for c in ch if type(c) is type(v) and c == v] or [c for c in ch if c == v]
            params[k] = m[0]
    out = exp / out_name
    js: dict[str, Any] = {"experiment_id": exp_id, "model": family, "per_subset": {},
                          "rule": pr["objective"]["selection"], "protocol_sha256": pr["_sha256"]}
    if out.exists():
        js = json.loads(out.read_text(encoding="utf-8"))
    if subset in js["per_subset"]:
        return dict(js["per_subset"][subset])
    n_by = allt.groupby(["study_seed", "state"]).size().unstack(fill_value=0).to_dict(orient="index")
    js["per_subset"][subset] = {
        "trial_id": win.trial_id, "value": float(win.value), "params": params, "n_params": int(win.user_attrs_n_params),
        "n_completed_pooled": int(len(comp)), "counts_by_study": n_by,
        "study_best_values": [float(v) for v in sb.value], "study_best_spread": float(sb.value.max() - sb.value.min()),
        "pooled_trials_sha256": T.sha256_file(pooled_path),
        "frozen_at": dt.datetime.now().isoformat(timespec="seconds"), "test_data_used": False}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(js, indent=1, default=str), encoding="utf-8")
    tmp.replace(out)
    return dict(js["per_subset"][subset])


def freeze_selection(exp_dir: str | Path, subset: str, seed: int, run_tag: str | None = None) -> dict[str, Any]:
    """Write the retrain's early-stopped checkpoint into ``selection_frozen.json`` before any test evaluation."""
    exp = Path(exp_dir).resolve()
    tag = run_tag or f"{subset}_seed{seed}"
    meta = json.loads((exp / "runs" / tag / "train_meta.json").read_text(encoding="utf-8"))
    if meta.get("smoke"):
        raise SystemExit("smoke runs are never frozen or tested")
    p = exp / "selection_frozen.json"
    js: dict[str, Any] = {"experiment_id": meta["experiment_id"],
                          "selection_rule": "val_rmse early-stopped checkpoint; configuration fixed before training; "
                                            "no test data used", "entries": []}
    if p.exists():
        js = json.loads(p.read_text(encoding="utf-8"))
    if any(e["subset"] == subset and int(e["seed"]) == int(seed) for e in js["entries"]):
        return js
    js["entries"].append({"subset": subset, "seed": int(seed), "arm": meta["preprocess"]["smoothing"],
                          "config_hash": meta["config_hash"], "ckpt_path": f"runs/{tag}/ckpt_rmse.pt",
                          "ckpt_sha256": meta["ckpt_rmse_sha256"], "best_epoch": meta["best_epoch_rmse"],
                          "val_RMSE": meta["val"]["RMSE"], "frozen_at": dt.datetime.now().isoformat(timespec="seconds")})
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(js, indent=1), encoding="utf-8")
    tmp.replace(p)
    return js


# ------------------------------------------------------------------------------------------------ ridge
def ridge_flat(X: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(X.reshape(len(X), -1), dtype=np.float64)


def ridge_fit_eval(w: dict[str, Any], alpha: float) -> tuple[Any, float, float]:
    """Fit sklearn Ridge (Cholesky, intercept) on the flattened windows; return (estimator, val RMSE support, all)."""
    from sklearn.linear_model import Ridge

    from . import evaluation as EV

    est = Ridge(alpha=float(alpha), solver="cholesky", fit_intercept=True).fit(ridge_flat(w["Xtr"]),
                                                                              w["ytr"].astype(np.float64))
    pv = est.predict(ridge_flat(w["Xva"]))
    sup = w["support"]
    return est, float(EV.rmse(w["yva"][sup], pv[sup])), float(EV.rmse(w["yva"], pv))


def ridge_tune_subset(subset: str, space_path: str | Path, protocol_path: str | Path, data_dir: str | Path,
                      splits_path: str | Path, out_root: str | Path,
                      log: Callable[[str], None] = print) -> dict[str, Any]:
    """Ridge selection for one subset: 3 studies x 4 completed trials (CPU, float64), argmin over the 12 pooled trials,
    ties -> fewer parameters (C*L + 1), then smaller alpha. Validation only."""
    import optuna
    import yaml

    guard.install()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sp = yaml.safe_load(Path(space_path).read_text(encoding="utf-8"))
    pr = T.load_protocol(protocol_path)
    splits = load_splits(splits_path)
    tseed = int(splits["tuning_split_seed"])
    cache = DataCache(subset, tseed, data_dir, splits_path, smoothing=pr["data"]["smoothing"],
                      alpha=float(pr["data"]["ema_alpha"]),
                      support_min_label_cycle=int(sp["objective"]["support_min_label_cycle"]))
    frames = []
    for k in sp["study_seeds"]:
        d = Path(out_root) / f"ridge_{subset}_study{k}"
        d.mkdir(parents=True, exist_ok=True)
        st = optuna.create_study(study_name=f"R4_ridge_{subset}_s{k}", direction="minimize",
                                 sampler=optuna.samplers.RandomSampler(seed=int(k)))

        def objective(trial: Any) -> float:
            params = suggest(trial, sp["space"])
            w = cache.windows(int(params["L"]))
            _, v, vall = ridge_fit_eval(w, params["alpha"])
            trial.set_user_attr("n_params", int(cache.C * int(params["L"]) + 1))
            trial.set_user_attr("val_RMSE_allwin", vall)
            return v

        st.optimize(objective, n_trials=int(sp["completed_trials_per_study"]))
        df = st.trials_dataframe()
        df["study_seed"] = k
        df.to_csv(d / "trials.csv", index=False)
        frames.append(df)
        for t in st.trials:
            log(f"{subset} study{k} trial {t.number} {t.state.name} value {t.value} params {json.dumps(t.params)}")
    allt = pd.concat(frames, ignore_index=True)
    comp = allt[allt.state == "COMPLETE"].copy()
    comp["trial_id"] = comp.study_seed.astype(str) + ":" + comp.number.astype(str)
    comp = comp.sort_values(["value", "user_attrs_n_params", "params_alpha"], kind="mergesort").reset_index(drop=True)
    win = comp.iloc[0]
    return {"trial_id": win.trial_id, "value": float(win.value),
            "params": {"L": int(win.params_L), "alpha": float(win.params_alpha)},
            "n_params": int(win.user_attrs_n_params), "C": int(cache.C), "test_data_used": False}
