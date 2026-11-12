"""Port of experiments/final/posthoc_R1.24/run_regime_split.py (sha256 573d5189...). Reads test_FD004.txt columns 1-5
only (unit, cycle, op_setting_1-3); RUL_FD004.txt and the sensor columns are never opened. The default frozen
check does NOT call this module: the frozen outputs are shipped with their hashes (RELEASE_FILES.md).

STATUS: PORT, NOT EXECUTED. This port has never been run: running it needs ``test_FD004.txt`` (operating-setting
columns), and no further test read is allowed after the freeze. The script of record is the frozen
``results/frozen/run_regime_split.py`` (sha256 573d5189...), which produced the shipped outputs
(``results/frozen/R1.24_SHA256SUMS.txt``). Equivalence of the port to it is not demonstrated.

R1.24 post-freeze analysis_posthoc (RULES 20, 22, 23; manifest posthoc_cpu R1.24): FD004 error analysis by
operating regime, θ* (exp_B6) vs D4 (exp_B16), 5 seeds each. Descriptive; selects nothing; single run.

Read scope (RULES 22, concerns.md R1.24 Addendum E):
- code_base/Data/test_FD004.txt: ONLY unit, cycle, op_setting_1-3. The file is read line by line and each line is
  split with maxsplit=5; only the first 5 tokens are parsed, the remainder (sensor columns) is never parsed or stored.
- code_base/Data/RUL_FD004.txt is never opened (a guard on builtins.open blocks any RUL_FD path and any test_FD00[123]).
- True RUL (y_true_capped) and cycle_end come only from the saved prediction files named in final_rows.json.
- train_FD004.txt (training file) is read to refit the TR(seed) pipeline, whose k-means (k 6, n_init 10,
  random_state = split seed, on op_setting_1-3 of TR rows) assigns the regime; inertia checked against train_meta.
Regime = k-means cluster of the test row whose cycle equals cycle_end of the prediction file (the window's last row;
R0.23: for engines with T > L the window ends one cycle before the last observation, for T ≤ L it ends at T).
Clusters are mapped to canonical regimes R1-R6 by sorting centroids (op_setting_1, then 2, then 3).
Outputs per row and seed: per-regime engine count, RMSE, mean error, Score sum and share, late-error tail counts
(d = pred − true > 10, > 20, > 30), early tail (d < −20); per-engine error quantiles; top-10 engine Score share.
"""
import builtins
import json
import os
import re

import numpy as np
import pandas as pd

from ..config import resolve_path
from . import posthoc_common as PC

HERE = "."  # output folder, set by main()

EXP_ID = "final_R1.24_FD004_regime"
SUB = "FD004"
ROWS = ["theta_star", "d4"]
NAMES = {"theta_star": "θ*", "d4": "D4"}
DATA = str(resolve_path("${SACA_DATA_DIR:-../workspace/code_base/Data}"))
SPLITS = str(resolve_path("configs/data_splits.json"))
TEST_FILE = os.path.join(DATA, "test_FD004.txt")
_open = builtins.open
_BLOCK = re.compile(r"(RUL_FD\d|test_FD00[123])", re.IGNORECASE)


def guarded_open(file, *a, **k):
    if isinstance(file, (str, bytes)) or hasattr(file, "__fspath__"):
        p = os.fsdecode(file) if not isinstance(file, str) else file
        if _BLOCK.search(p):
            raise RuntimeError(f"blocked by R1.24 read-scope guard: {p}")
        if re.search(r"test_FD004", p, re.IGNORECASE) and not getattr(guarded_open, "allow_test", False):
            raise RuntimeError(f"test_FD004 may be opened only by read_test_ops(): {p}")
    return _open(file, *a, **k)


def read_test_ops():
    """unit, cycle, op_setting_1-3 only (first 5 whitespace tokens of each line)."""
    out = []
    guarded_open.allow_test = True
    try:
        with builtins.open(TEST_FILE, encoding="utf-8") as f:
            for line in f:
                tok = line.split(maxsplit=5)[:5]
                if len(tok) < 5:
                    continue
                out.append((int(tok[0]), int(tok[1]), float(tok[2]), float(tok[3]), float(tok[4])))
    finally:
        guarded_open.allow_test = False
    return pd.DataFrame(out, columns=["engine", "time", "op_setting_1", "op_setting_2", "op_setting_3"])


def score_rows(d):
    return np.where(d < 0, np.expm1(-d / 13.0), np.expm1(d / 10.0))


def tci(v):
    from scipy import stats as st
    v = np.asarray(v, float)
    h = st.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return [float(v.mean() - h), float(v.mean() + h)]


def _main():
    builtins.open = guarded_open
    PC.refuse_if_logged(EXP_ID)
    from .. import data as P
    rows, fr = PC.load_final_rows()
    files = {PC.rel(PC.FINAL_ROWS): PC.FINAL_ROWS_SHA, PC.rel(__file__): PC.sha256(__file__),
             PC.rel(os.path.join(DATA, "train_FD004.txt")): PC.sha256(os.path.join(DATA, "train_FD004.txt")),
             PC.rel(SPLITS): PC.sha256(SPLITS)}
    guarded_open.allow_test = True
    files[PC.rel(TEST_FILE) + " (columns 1-5 only)"] = PC.sha256(TEST_FILE)
    guarded_open.allow_test = False
    preds = {}
    for k in ROWS:
        for s, p, h in PC.pred_files(rows, k, SUB):
            files[PC.rel(p)] = h
            preds[(k, s)] = pd.read_csv(p)
    inputs_sha = PC.write_inputs(os.path.join(HERE, "inputs_sha256.json"), files)
    logrow = PC.append_posthoc(EXP_ID, SUB,
                               "R1.24 FD004 regime split theta* (exp_B6) vs D4 (exp_B16): test_FD004.txt unit/cycle/"
                               "op_setting_1-3 only, regime at prediction-file cycle_end via TR-fitted k-means, true RUL "
                               "from saved prediction files only, RUL_FD004.txt never opened; inputs "
                               f"posthoc_R1.24/inputs_sha256.json {inputs_sha}; descriptive, no selection", 248, __file__)
    ops = read_test_ops()
    splits = P.load_splits(SPLITS)
    raw = P.load_train_file(DATA, SUB)
    reg_rows, kinfo, canon_names = [], {}, None
    reg_of = {}
    for s in PC.SEEDS:
        tr_ids, _ = P.split_engines(splits, SUB, s)
        pipe = P.Pipeline(SUB, s, train_engines=tr_ids).fit(raw[raw.engine.isin(tr_ids)])
        km = pipe.kmeans
        order = np.lexsort(km.cluster_centers_.T[::-1])          # sort by op1, then op2, then op3
        canon = {int(c): i + 1 for i, c in enumerate(order)}
        cents = np.round(km.cluster_centers_[order], 4).tolist()
        tm = {k: json.load(_open(os.path.join(PC.WS, rows[k]["per_subset"][SUB]["exp"], "runs", f"{SUB}_seed{s}",
                                              "train_meta.json"), encoding="utf-8")).get("kmeans_inertia")
              for k in ROWS}
        kinfo[s] = {"centroids_sorted": cents, "inertia_refit": float(km.inertia_), "inertia_train_meta": tm}
        if canon_names is None:
            canon_names = {i + 1: f"R{i + 1} ({c[0]:.0f}, {c[1]:.2f}, {c[2]:.0f})" for i, c in enumerate(cents)}
        lab = km.predict(ops[P.OPS].to_numpy())
        o = ops.assign(cluster=lab, regime=[canon[int(c)] for c in lab]).set_index(["engine", "time"])
        for k in ROWS:
            df = preds[(k, s)]
            key = list(zip(df.engine_id.astype(int), df.cycle_end.astype(int)))
            r = o.loc[key]
            reg_of[(k, s)] = r["regime"].to_numpy()
            for (e, c), (_, x) in zip(key, r.iterrows()):
                reg_rows.append({"row": k, "seed": s, "engine_id": e, "cycle_end": c, "op_setting_1": x.op_setting_1,
                                 "op_setting_2": x.op_setting_2, "op_setting_3": x.op_setting_3,
                                 "cluster": int(x.cluster), "regime": int(x.regime)})
    reg_df = pd.DataFrame(reg_rows)
    reg_path = os.path.join(HERE, "regime_table_FD004.csv")
    reg_df.to_csv(reg_path, index=False, float_format="%.4f")
    reg_sha = PC.sha256(reg_path)
    # regime agreement across seeds and rows
    base = reg_of[("d4", 0)]
    agree = {f"{k}_s{s}_vs_d4_s0": int((reg_of[(k, s)] != base).sum()) for k in ROWS for s in PC.SEEDS}
    REG = sorted(canon_names)
    TAILS = {"late_gt10": lambda d: d > 10, "late_gt20": lambda d: d > 20, "late_gt30": lambda d: d > 30,
             "early_lt_minus20": lambda d: d < -20}
    per = {k: {} for k in ROWS}
    for k in ROWS:
        for s in PC.SEEDS:
            df = preds[(k, s)]
            d = df.y_pred.to_numpy() - df.y_true_capped.to_numpy()
            sc = score_rows(d)
            rg = reg_of[(k, s)]
            b = {"RMSE": float(np.sqrt((d ** 2).mean())), "Score": float(sc.sum()),
                 "Score_late": float(sc[d >= 0].sum()), "Score_early": float(sc[d < 0].sum()),
                 "err_quantiles": {q: float(np.quantile(d, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                 "err_max": float(d.max()), "err_min": float(d.min()),
                 "top10_engine_score_share": float(np.sort(sc)[::-1][:10].sum() / sc.sum()),
                 "tails": {t: int(f(d).sum()) for t, f in TAILS.items()},
                 "tail_score_share_late_gt20": float(sc[d > 20].sum() / sc.sum()), "regimes": {}}
            for g in REG:
                m = rg == g
                b["regimes"][g] = {"n": int(m.sum()), "RMSE": float(np.sqrt((d[m] ** 2).mean())) if m.any() else None,
                                   "mean_err": float(d[m].mean()) if m.any() else None,
                                   "Score": float(sc[m].sum()), "Score_share": float(sc[m].sum() / sc.sum()),
                                   "tails": {t: int(f(d[m]).sum()) for t, f in TAILS.items()}}
            per[k][s] = b
    # summaries (mean ± std over seeds) and paired D4 − θ* (descriptive)
    def agg(fn):
        out = {}
        for k in ROWS:
            v = np.array([fn(per[k][s]) for s in PC.SEEDS], float)
            out[k] = [float(v.mean()), float(v.std(ddof=1))]
        dv = np.array([fn(per["d4"][s]) - fn(per["theta_star"][s]) for s in PC.SEEDS], float)
        out["d4_minus_theta"] = {"mean": float(dv.mean()), "t95": tci(dv), "per_seed": dv.tolist()}
        return out
    summ = {"overall": {m: agg(lambda b, m=m: b[m]) for m in ("RMSE", "Score", "Score_late", "Score_early",
                                                               "top10_engine_score_share", "tail_score_share_late_gt20")},
            "tails": {t: agg(lambda b, t=t: b["tails"][t]) for t in TAILS},
            "err_quantiles": {str(q): agg(lambda b, q=q: b["err_quantiles"][q]) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "regimes": {}}
    for g in REG:
        summ["regimes"][g] = {m: agg(lambda b, g=g, m=m: (b["regimes"][g][m] if b["regimes"][g][m] is not None
                                                          else np.nan))
                              for m in ("n", "RMSE", "mean_err", "Score", "Score_share")}
        summ["regimes"][g]["tails"] = {t: agg(lambda b, g=g, t=t: b["regimes"][g]["tails"][t]) for t in TAILS}
    res = {"experiment_id": EXP_ID, "kind": "analysis_posthoc", "log_row": logrow, "inputs_sha256_file": inputs_sha,
           "regime_table": PC.rel(reg_path), "regime_table_sha256": reg_sha, "regime_names": canon_names,
           "regime_definition": "TR(seed)-fitted k-means (k 6, n_init 10, random_state seed) on op_setting_1-3, row "
                                "at cycle == cycle_end of the prediction file; canonical order by centroid",
           "kmeans": kinfo, "regime_disagreement_vs_d4_seed0": agree, "per_seed": per, "summary": summ,
           "score": "d = pred − true; exp(−d/13)−1 if d < 0, exp(d/10)−1 if d ≥ 0 (per engine, summed)",
           "scope": "θ* (exp_B6) vs D4 (exp_B16) only; no ITT rows; no other rows (orchestrator instruction 2026-10-26)"}
    with _open(os.path.join(HERE, "regime_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=float)
    f2 = lambda x: f"{x[0]:.2f} ± {x[1]:.2f}"   # noqa: E731
    f1 = lambda x: f"{x[0]:.1f} ± {x[1]:.1f}"   # noqa: E731
    dd = lambda x, p=1: f"{x['mean']:+.{p}f} [{x['t95'][0]:+.{p}f}, {x['t95'][1]:+.{p}f}]"   # noqa: E731
    L = ["# R1.24 FD004 error analysis by operating regime: θ* vs D4 (analysis_posthoc, post-freeze)", "",
         "Test set FD004 (248 engines), 5 split seeds per row, mean ± std over seeds; Δ = D4 − θ* with paired t 95% CI "
         "over the 5 seeds (descriptive only). Regime = TR-fitted k-means cluster of the window's last row "
         "(cycle_end of the prediction file). Score per engine with early /13, late /10. d = pred − true "
         "(d > 0 = late prediction, penalised more). True RUL from the saved prediction files (capped at 125); "
         "RUL_FD004.txt not opened; only unit, cycle and op_setting_1-3 of test_FD004.txt were parsed.", "",
         f"Regime table: `{PC.rel(reg_path)}` (sha256 {reg_sha}). Regime assignment differs from D4 seed 0 for: "
         + ", ".join(f"{k} {v}" for k, v in agree.items() if v) + (" (none)" if not any(agree.values()) else "")
         + " engines.", "",
         "## Overall", "", "| Metric | θ* | D4 | Δ D4 − θ* [95% t CI] |", "|---|---|---|---|"]
    for m, p in (("RMSE", 2), ("Score", 1), ("Score_late", 1), ("Score_early", 1), ("top10_engine_score_share", 3),
                 ("tail_score_share_late_gt20", 3)):
        x = summ["overall"][m]
        L.append(f"| {m} | {x['theta_star'][0]:.{p}f} ± {x['theta_star'][1]:.{p}f} | {x['d4'][0]:.{p}f} ± "
                 f"{x['d4'][1]:.{p}f} | {dd(x['d4_minus_theta'], p)} |")
    for t in TAILS:
        x = summ["tails"][t]
        L.append(f"| engines with {t} | {f1(x['theta_star'])} | {f1(x['d4'])} | {dd(x['d4_minus_theta'])} |")
    for q in ("0.05", "0.25", "0.5", "0.75", "0.95"):
        x = summ["err_quantiles"][q]
        L.append(f"| error quantile {q} | {f2(x['theta_star'])} | {f2(x['d4'])} | {dd(x['d4_minus_theta'], 2)} |")
    L += ["", "## By regime (centroid op_setting_1, op_setting_2, op_setting_3)", "",
          "| Regime | n engines (θ* / D4) | RMSE θ* | RMSE D4 | Δ RMSE | Score θ* | Score D4 | Δ Score | Score share θ* | Score share D4 | late > 20 θ* | late > 20 D4 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in REG:
        x = summ["regimes"][g]
        L.append(f"| {canon_names[g]} | {x['n']['theta_star'][0]:.1f} / {x['n']['d4'][0]:.1f} | {f2(x['RMSE']['theta_star'])} | "
                 f"{f2(x['RMSE']['d4'])} | {dd(x['RMSE']['d4_minus_theta'], 2)} | {f1(x['Score']['theta_star'])} | "
                 f"{f1(x['Score']['d4'])} | {dd(x['Score']['d4_minus_theta'])} | {f2([100 * v for v in x['Score_share']['theta_star']])}% | "
                 f"{f2([100 * v for v in x['Score_share']['d4']])}% | {f1(x['tails']['late_gt20']['theta_star'])} | "
                 f"{f1(x['tails']['late_gt20']['d4'])} |")
    L += ["", "## k-means check", "", "| Seed | inertia refit | train_meta θ* | train_meta D4 |", "|---|---|---|---|"]
    for s in PC.SEEDS:
        ki = kinfo[s]
        L.append(f"| {s} | {ki['inertia_refit']:.6f} | {ki['inertia_train_meta']['theta_star']} | "
                 f"{ki['inertia_train_meta']['d4']} |")
    L += ["", "Descriptive only. Nothing is selected or changed from these numbers. Submitted 1390 → 1944 Score values used "
          "swapped constants (R0.6) and the removed ITT recipe; they are not reproduced here."]
    with _open(os.path.join(HERE, "regime_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("done", reg_sha)


def main(out_dir: str) -> None:
    """Regenerate the outputs into ``out_dir`` (inert test-log helpers; see posthoc_common)."""
    global HERE
    HERE = os.path.abspath(out_dir)
    os.makedirs(HERE, exist_ok=True)
    saved = builtins.open
    try:
        _main()
    finally:
        builtins.open = saved


def cli(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True, help="folder for the regenerated outputs (never the frozen folder)")
    a = ap.parse_args(argv)
    main(a.out_dir)


if __name__ == "__main__":
    cli()
