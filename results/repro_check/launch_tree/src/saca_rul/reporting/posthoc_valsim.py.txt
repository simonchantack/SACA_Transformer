"""Port of experiments/final/nosmooth_b_valsim/run_valsim.py (sha256 ee248fec...); package imports, output folder is an
argument, no test file. No-smoothing option (b) (RULES 23; verdicts/round_04.md F1 (b)): VALIDATION-ONLY CPU simulation of test-like
truncation and padding. CPU inference of saved checkpoints only; no training; no test file (test_guard installed).

Models: D4 with causal EMA (exp_B16/runs/<S>_seed<k>/ckpt_rmse.pt) and D4 without smoothing
(exp_V1/nosmooth/runs/<S>_seed<k>/ckpt_rmse.pt), 4 subsets x 5 split seeds; checkpoint sha256 must equal the
selection_frozen / metrics.json value. Loaded through exp_B16/code/model_io.py (exp_V1/code *.py are byte-identical).
Pre-stated protocol (posted in comms.md before the run):
 1. Pipeline refit on TR(seed) with the checkpoint's preprocess (smoothing ema / none, alpha 0.24723); feature list
    must equal train_meta. Route check: full-validation windows reproduce the saved preds_val.csv (max |diff| <= 1e-3).
 2. Truncation: for every VA engine of the split (full length T_full), K = 20 cut lengths T' drawn uniformly from the
    integers {20, ..., T_full - 1}; rng = numpy default_rng([20261026, subset index, seed]); the SAME cuts for both arms.
 3. Each cut is a new pseudo-engine with rows 1..T' only, transformed by the pipeline (smoothing restarts per
    pseudo-engine, as for a test engine) and windowed by preprocess.testing_windows (L = 50): T' < L padded with the
    last row, T' > L ending one cycle before the last observation. Label = min(T_full - T', 125), the RUL at the last
    observed cycle (as the official test labels).
 4. Metrics per (subset, seed, arm): RMSE over all cuts, padded cuts (T' < 50) and unpadded cuts; gap none - EMA paired
    by seed; pooled exact Wilcoxon over 20 pairs (descriptive). Nothing is selected.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy import stats as st

from .. import checkpoint as model_io
from .. import data as P
from .. import guard
from ..config import resolve_path

guard.install()
HERE = "."  # output folder, set by main()
WS = str(resolve_path("${SACA_WORKSPACE:-../workspace}"))
SPLITS = str(resolve_path("configs/data_splits.json"))

SUBS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [0, 1, 2, 3, 4]
K, TMIN, L, CAP = 20, 20, 50, 125
ARMS = {"ema": os.path.join(WS, "experiments", "round_03", "exp_B16", "runs"),
        "none": os.path.join(WS, "experiments", "round_04", "exp_V1", "nosmooth", "runs")}
DATA = str(resolve_path("${SACA_DATA_DIR:-../workspace/code_base/Data}"))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def expected_ckpt_sha():
    out = {}
    m = json.load(open(os.path.join(WS, "experiments", "round_04", "exp_V1", "metrics.json"), encoding="utf-8"))
    for sub in SUBS:
        for r in m["per_subset"][sub]["nosmooth_per_seed"]:
            out[("none", sub, int(r["seed"]))] = r["ckpt_sha256"]
        for r in m["per_subset"][sub]["control_per_seed"]:
            out[("ema", sub, int(r["seed"]))] = r["ckpt_sha256"]
    return out


def rmse(d):
    return float(np.sqrt(np.mean(d ** 2))) if len(d) else float("nan")


def tci(v):
    v = np.asarray(v, float)
    h = st.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return [float(v.mean() - h), float(v.mean() + h)]


def _main():
    torch.manual_seed(0)
    exp_sha = expected_ckpt_sha()
    splits = P.load_splits(SPLITS)
    res = {"protocol": __doc__, "per_cell": {}, "summary": {}, "ckpt_sha256": {}, "route_check_max_abs": 0.0}
    for si, sub in enumerate(SUBS):
        raw = P.load_train_file(DATA, sub)
        for s in SEEDS:
            tr_ids, va_ids = P.split_engines(splits, sub, s)
            va_raw = raw[raw.engine.isin(va_ids)]
            Tfull = va_raw.groupby("engine")["time"].max()
            rng = np.random.default_rng([20261026, si, s])
            cuts = []
            for e in sorted(va_ids):
                tf = int(Tfull.loc[e])
                for t in rng.integers(TMIN, tf, size=K):        # uniform on {TMIN, ..., tf - 1}
                    cuts.append((e, int(t), tf))
            frames = []
            for j, (e, t, tf) in enumerate(cuts):
                g = va_raw[(va_raw.engine == e) & (va_raw.time <= t)].copy()
                g["engine"] = j + 1                                  # pseudo-engine
                frames.append(g)
            cut_df = pd.concat(frames, ignore_index=True)
            cut_df["source"] = "val_cut"
            y = np.minimum(np.array([tf - t for _, t, tf in cuts], float), CAP)
            Tp = np.array([t for _, t, _ in cuts])
            cell = {"n_va_engines": len(va_ids), "n_cuts": len(cuts), "n_padded": int((Tp < L).sum())}
            for arm, root in ARMS.items():
                ck = os.path.join(root, f"{sub}_seed{s}", "ckpt_rmse.pt")
                h = sha(ck)
                assert h == exp_sha[(arm, sub, s)], (arm, sub, s, "ckpt sha")
                res["ckpt_sha256"][f"{arm}/{sub}/s{s}"] = h
                model, meta = model_io.load_for_eval(ck)
                pp = meta["preprocess"]
                assert pp["smoothing"] == arm, (pp, arm)
                pipe = P.Pipeline(sub, s, smoothing=pp["smoothing"], alpha=pp["alpha"], legacy_corr=pp["legacy_corr"],
                                  train_engines=tr_ids).fit(raw[raw.engine.isin(tr_ids)])
                assert pipe.features == meta["features"]
                assert meta["L"] == L
                # route check on the saved validation predictions
                va_t = pipe.transform(va_raw)
                Xv, yv, ev, cv = P.training_windows(va_t, pipe.features, L)
                pv = model_io.predict(model, np.ascontiguousarray(Xv.transpose(0, 2, 1)), 1024)
                ref = pd.read_csv(os.path.join(root, f"{sub}_seed{s}", "preds_val.csv"))
                assert np.array_equal(ref.engine_id.to_numpy(), ev) and np.array_equal(ref.cycle_label.to_numpy(), cv)
                rc = float(np.abs(pv - ref.y_pred.to_numpy()).max())
                res["route_check_max_abs"] = max(res["route_check_max_abs"], rc)
                assert rc <= 1e-3, (arm, sub, s, rc)
                # test-like cuts
                te = pipe.transform(cut_df, with_labels=False)
                X, eng, ce = P.testing_windows(te, pipe.features, L)
                order = np.argsort(eng)
                assert np.array_equal(eng[order], np.arange(1, len(cuts) + 1))
                p = model_io.predict(model, np.ascontiguousarray(X.transpose(0, 2, 1)), 1024)[order]
                assert np.array_equal(ce[order][Tp < L], Tp[Tp < L])
                d = p - y
                pad = Tp < L
                cell[arm] = {"RMSE_all": rmse(d), "RMSE_padded": rmse(d[pad]), "RMSE_unpadded": rmse(d[~pad]),
                             "mean_err_padded": float(d[pad].mean()) if pad.any() else None,
                             "mean_err_unpadded": float(d[~pad].mean()), "route_check_max_abs": rc}
            res["per_cell"][f"{sub}/s{s}"] = cell
            print(sub, s, {a: round(cell[a]["RMSE_padded"], 2) for a in ARMS}, flush=True)
    # summaries
    pool = {k: [] for k in ("all", "padded", "unpadded")}
    for sub in SUBS:
        b = {}
        for part in ("all", "padded", "unpadded"):
            e = np.array([res["per_cell"][f"{sub}/s{s}"]["ema"][f"RMSE_{part}"] for s in SEEDS])
            n = np.array([res["per_cell"][f"{sub}/s{s}"]["none"][f"RMSE_{part}"] for s in SEEDS])
            g = n - e
            pool[part] += g.tolist()
            b[part] = {"ema": [float(e.mean()), float(e.std(ddof=1))], "none": [float(n.mean()), float(n.std(ddof=1))],
                       "gap_none_minus_ema": {"per_seed": g.tolist(), "mean": float(g.mean()), "t95": tci(g),
                                              "n_none_worse": int((g > 0).sum())}}
        b["n_cuts"] = int(sum(res["per_cell"][f"{sub}/s{s}"]["n_cuts"] for s in SEEDS))
        b["n_padded"] = int(sum(res["per_cell"][f"{sub}/s{s}"]["n_padded"] for s in SEEDS))
        res["summary"][sub] = b
    res["pooled"] = {}
    for part, v in pool.items():
        v = np.array(v)
        res["pooled"][part] = {"n": 20, "mean": float(v.mean()), "n_none_worse": int((v > 0).sum()),
                               "wilcoxon_exact_p_descriptive": float(st.wilcoxon(v, method="exact").pvalue)}
    res["test_guard_blocked"] = guard.blocked_count()
    res["versions"] = {"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__}
    with open(os.path.join(HERE, "valsim_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    f2 = lambda x: f"{x[0]:.2f} ± {x[1]:.2f}"   # noqa: E731
    gg = lambda b: (f"{b['mean']:+.2f} [{b['t95'][0]:+.2f}, {b['t95'][1]:+.2f}] ({b['n_none_worse']}/5)")  # noqa: E731
    M = ["# No-smoothing option (b): validation-only simulation of test-like truncation and padding (CPU, saved checkpoints)", "",
         f"VA engines of each split cut {K} times at T' ~ U{{{TMIN}, …, T_full − 1}} (same cuts for both arms), windowed with "
         "the test rule (L = 50; T' < 50 padded with the last row; otherwise ending one cycle before the last observed "
         "cycle), label = min(T_full − T', 125). D4 with EMA (exp_B16) vs D4 without smoothing (exp_V1); 5 split seeds; "
         "mean ± std over seeds; gap = none − EMA, paired t 95% CI over seeds (seeds with none worse). No test data "
         f"(test_guard blocked {res['test_guard_blocked']}); no training; route check: full-validation predictions "
         f"reproduce preds_val.csv to {res['route_check_max_abs']:.1e}.", "",
         "| Subset | cuts (padded) | RMSE all EMA / none | gap all | RMSE padded EMA / none | gap padded | RMSE unpadded EMA / none | gap unpadded |",
         "|---|---|---|---|---|---|---|---|"]
    for sub in SUBS:
        b = res["summary"][sub]
        M.append(f"| {sub} | {b['n_cuts']} ({b['n_padded']}) | {f2(b['all']['ema'])} / {f2(b['all']['none'])} | "
                 f"{gg(b['all']['gap_none_minus_ema'])} | {f2(b['padded']['ema'])} / {f2(b['padded']['none'])} | "
                 f"{gg(b['padded']['gap_none_minus_ema'])} | {f2(b['unpadded']['ema'])} / {f2(b['unpadded']['none'])} | "
                 f"{gg(b['unpadded']['gap_none_minus_ema'])} |")
    M += ["", "Pooled over 20 (subset, seed) pairs, exact Wilcoxon (descriptive):", ""]
    for part, v in res["pooled"].items():
        M.append(f"- gap {part}: mean {v['mean']:+.2f}, none worse {v['n_none_worse']}/20, p {v['wilcoxon_exact_p_descriptive']:.2g}")
    M += ["", "Validation-only and descriptive. Nothing is selected; the EMA was fixed in round 1."]
    with open(os.path.join(HERE, "valsim_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(M) + "\n")
    print("done")


def main(out_dir: str) -> None:
    """Regenerate valsim_results.{json,md} into ``out_dir`` (CPU, validation only)."""
    global HERE
    HERE = os.path.abspath(out_dir)
    os.makedirs(HERE, exist_ok=True)
    _main()


def cli(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="no-smoothing (b): validation-only simulation of test-like cuts")
    ap.add_argument("--out-dir", required=True)
    main(ap.parse_args(argv).out_dir)


if __name__ == "__main__":
    cli()
