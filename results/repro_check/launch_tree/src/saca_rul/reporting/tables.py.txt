"""Combined-table generator (R2.9, R0.15, R2.8, R1.1, R0.7): one script for every final row.

Port of ``experiments/final/code/make_table_v4.py`` (frozen 2026-10-26; sha256 ef35156b...). The table bodies are
the frozen code: the acceptance test is that the regenerated outputs hash-match the frozen ones. Changed only:
imports and root paths (``SACA_WORKSPACE`` selects the frozen workspace), the removal of the ``c4`` and
``freeze-rows`` modes (their outputs are frozen artifacts, listed in RELEASE_FILES.md), and this docstring.

Modes (``python -m saca_rul.reporting.tables --version v4 --out-dir DIR``):
  --version v3   legacy v3 table (round 3), reproduced from metrics.json files and prediction CSVs.
  --version v4   combined_table_v4.{md,csv,json} from ``final_rows.json`` (the frozen row -> file mapping with sha256).
v4 rows: D4 (exp_B16), theta* (exp_B6), PatchTST (exp_B10_topup; FD003 exp_B10), iTransformer (exp_B11),
Crossformer (exp_B12), LSTM (exp_V2), ridge (exp_V3). No ITT, no theta0, no theta* fusion rows.
Reporting spec (paper_plan D29): delta = D4 - row; paired t 95% CI (Holm over 4 subsets on p); g_z with exact
noncentral-t CI; exact Wilcoxon per subset and pooled over 20 pairs; DM-HLN and engine bootstrap on seed-averaged
predictions; percentile seed bootstrap only as a labelled supplement (n = 5, under-covers).
The v4 mode is an analysis_posthoc: it reads saved test prediction files (labels inside); it evaluates no model, opens
no test_FD / RUL_FD file and writes no row to the test log.
"""
import argparse
import hashlib
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy
from scipy import stats as st
from scipy.optimize import brentq

from .. import evaluation as M
from .. import stats as S
from ..config import resolve_path

WS = str(resolve_path("${SACA_WORKSPACE:-../workspace}"))
R2 = os.path.join(WS, "experiments", "round_02")

# =====================================================================================================================
# legacy v3 (verbatim from round_02/table_r02/code/make_combined_table.py, sha256 51811cde...)
# =====================================================================================================================
SUBS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [0, 1, 2, 3, 4]
ROWS_V21 = [  # key, label, experiment dir, selection note
    ("theta_star", "Base θ* (P2 tuned, exp_B6)", os.path.join(R2, "exp_B6"), "validation (P2; target 12 trials, see Trials)"),
    ("theta0", "Base θ0 reference (round_01/exp_B3)", os.path.join(WS, "experiments", "round_01", "exp_B3"),
     "test-tuned checkpoint config (R0.3); reference only"),
    ("itt", "ITT = θ* + Scenario-A/B training recipe (single model; no capacity change, no jitter, no ensemble; lr/batch of θ*; exp_B14)",
     os.path.join(R2, "exp_B14"),
     "θ* (validation) + recipe chosen on test in the submission; 0 extra trials"),
    ("patchtst", "PatchTST (P2 tuned, full budget: exp_B10_topup; FD003 exp_B10)",
     {"FD001": os.path.join(R2, "exp_B10_topup"), "FD002": os.path.join(R2, "exp_B10_topup"),
      "FD003": os.path.join(R2, "exp_B10"), "FD004": os.path.join(R2, "exp_B10_topup")},
     "validation (P2; target 12 trials, see Trials)"),
    ("itransformer", "iTransformer (P2 tuned, exp_B11)", os.path.join(R2, "exp_B11"), "validation (P2; target 12 trials, see Trials)"),
    ("crossformer", "Crossformer (P2 tuned, exp_B12)", os.path.join(R2, "exp_B12"), "validation (P2; target 12 trials, see Trials)"),
]
THETA0_VAL = os.path.join(R2, "exp_B6", "theta0_reference.csv")
R3 = os.path.join(WS, "experiments", "round_03")
DSTAR_ROW = ("dstar_d4", "D* = D4 (round 3, preregistered; exp_B16)", os.path.join(R3, "exp_B16"),
             "validation (P2, 12 trials; design D4 chosen on validation in the preregistered stage-1 screen of 4 designs)")
# best baseline per subset, chosen on validation (preregistration 5.3; = exp_B15/code/stage2_primary.py BEST_BASELINE)
BEST_BASELINE = {"FD001": "crossformer", "FD002": "crossformer", "FD003": "crossformer", "FD004": "patchtst"}
C6_LINE = ("- Repeated test read (correction C6, added 2026-10-23; text only, no number changed): PatchTST FD002 seeds 0-4 "
           "were read on test twice for the same configuration, once in exp_B10 (2026-10-21) and once in exp_B10_topup "
           "(2026-10-23), as pre-committed in `exp_B10/topup_precommit.md` (sha256 9eb87243…d792). The top-up re-selected "
           "the same trial (2:1) and the retrains are bitwise identical, so the two reads gave bitwise-identical prediction "
           "files (`exp_B10/preds/preds_FD002_seed<k>_test.csv` = `exp_B10_topup/preds/preds_FD002_seed<k>_test.csv`, same "
           "sha256 for k = 0-4; both reads are rows of `experiments/test_eval_log.csv`). The second read informed no choice.")
V3_HEADER = [
    "v3 = v2.1 (`experiments/round_02/combined_table.md`, unchanged and still valid) plus the row \"D* = D4 (round 3, "
    "preregistered)\" (exp_B16) and the round-3 section below. Every v2.1 row and statistic is recomputed by the same code "
    "and equals v2.1.",
    "- **The preregistered primary for the D4 row is its comparison with the validation-chosen best baseline per subset "
    "(Crossformer FD001-FD003, PatchTST FD004), pooled over 20 (subset, seed) pairs: Inconclusive (exact two-sided "
    "Wilcoxon p = 0.0696). Secondary: non-inferior at margin 0.5 (one-sided p = 0.0004).** See the round-3 section.",
    "- The D4 row's Δ vs θ* columns are preregistered secondary comparisons. No significance claim of D4 vs PatchTST is "
    "allowed: the pooled p = 0.048 is uncorrected, and Holm over the 5 pooled D4 comparisons gives 0.145 "
    "(verdicts/round_03.md C3).",
    "- Disclosure (paper_plan.md D1; preregistration section 13): the round-3 designs were motivated after seeing round-2 "
    "results, including test-set summaries; the 0.5 margin was justified from round-2 baseline test-seed stds."]
SUPERSEDED = [("patchtst", "PatchTST exp_B10 (budget-short; superseded by exp_B10_topup)", os.path.join(R2, "exp_B10"),
               ["FD001", "FD002", "FD004"])]
# reviewer's CPU re-check of a test-time batch-invariance flag (verdicts/round_02.md check 7; no test read)
BI_NOTES = {("exp_B14", "FD002", 4): "validation-window re-check of the same checkpoint (review/verify_round02/bi_itt_fd002s4.py, "
            "800 VA windows, CPU): b1 vs b1024 max 3.4e-05 (float32), 5.7e-14 (float64), so no cross-sample coupling; "
            "the test-only deviation is most likely a ProbSparse top-k tie flip on padded test windows (24 engines shorter "
            "than L; unverified hypothesis)"}


def r3_tuning(sub):
    """D4 tuning = study 0 (exp_B15 stage 1; FD002: exp_B16) + studies 1-2 (exp_B16): sums of result.json n_complete and
    gpu_h (same convention as collect_p2.py)."""
    dirs = [os.path.join(R3, "exp_B16" if sub == "FD002" else "exp_B15", "tune", f"r3_D4_{sub}_study0")]
    dirs += [os.path.join(R3, "exp_B16", "tune", f"r3_D4_{sub}_study{k}") for k in (1, 2)]
    res = [json.load(open(os.path.join(d, "result.json"), encoding="utf-8")) for d in dirs]
    return {"completed_trials": int(sum(r["n_complete"] or 0 for r in res)),
            "gpu_h": float(sum(r["gpu_h"] or 0 for r in res))}


def pred_path(exp, rel):
    return os.path.join(exp, rel.replace("\\", "/"))


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def exp_of(exp, sub):
    return exp[sub] if isinstance(exp, dict) else exp


def load(exp):
    p = os.path.join(exp, "metrics.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def seed_table(mj, sub):
    """dict seed -> {test metrics, val support RMSE, preds DataFrame} for seeds with a test read."""
    if mj is None or sub not in mj["per_subset"]:
        return {}
    out = {}
    for r in mj["per_subset"][sub]["per_seed"]:
        if r.get("test") is None:
            continue
        out[r["seed"]] = r
    return out


def fmt(m, s, d=2):
    if m is None:
        return "—"
    return f"{m:.{d}f} ± {s:.{d}f}" if s is not None else f"{m:.{d}f}"


def round3_block(data):
    """Pooled D4 comparisons (20 pairs, exact Wilcoxon) vs theta*, best baseline, PatchTST, iTransformer, Crossformer;
    Holm over the 5 (reviewer's check, not preregistered); preregistered primary and NI read from exp_B16 files."""
    def pairs(ref_of_sub):
        d = []
        for sub in SUBS:
            a = seed_table(data["dstar_d4"][sub], sub)
            b = seed_table(data[ref_of_sub(sub)][sub], sub)
            d += [a[s]["test"]["RMSE"] - b[s]["test"]["RMSE"] for s in SEEDS]
        return d
    comps = [("theta_star", lambda s: "theta_star"), ("best_baseline", lambda s: BEST_BASELINE[s]),
             ("patchtst", lambda s: "patchtst"), ("itransformer", lambda s: "itransformer"),
             ("crossformer", lambda s: "crossformer")]
    pooled = {}
    for name, f in comps:
        d = pairs(f)
        w = S.pooled_wilcoxon(d)
        pooled[name] = {"n": w["n"], "mean_diff": w["mean_diff"], "median_diff": w["median_diff"],
                        "n_dstar_better": int(sum(x < 0 for x in d)), "wilcoxon_p": w["wilcoxon_p"]}
    adj = S.holm([pooled[n]["wilcoxon_p"] for n, _ in comps])
    for (n, _), q in zip(comps, adj):
        pooled[n]["holm5_p"] = q
    b16 = os.path.join(R3, "exp_B16")
    prim = json.load(open(os.path.join(b16, "primary_test.json"), encoding="utf-8"))
    sec = json.load(open(os.path.join(b16, "secondary_tests.json"), encoding="utf-8"))
    return {"best_baseline": BEST_BASELINE, "pooled_dstar_minus": pooled,
            "holm5_note": "Holm over the 5 pooled D4 comparisons: reviewer's check (verdicts/round_03.md section 1 item 5), "
                          "not preregistered",
            "primary_test_json": prim, "secondary_tests_json": sec,
            "files_sha256": {"exp_B16/primary_test.json": sha256(os.path.join(b16, "primary_test.json")),
                             "exp_B16/secondary_tests.json": sha256(os.path.join(b16, "secondary_tests.json"))}}


def _find(d, key):
    """first value of `key` anywhere in a nested dict (primary/secondary files have a fixed but nested layout)."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = _find(v, key)
            if r is not None:
                return r
    return None


def round3_md(r):
    p = r["pooled_dstar_minus"]
    lab = {"theta_star": "θ* (exp_B6)",
           "best_baseline": "best baseline B_k (validation-chosen: Crossformer FD001-FD003, PatchTST FD004)",
           "patchtst": "PatchTST", "itransformer": "iTransformer", "crossformer": "Crossformer"}
    role = {"theta_star": "secondary (4.3)", "best_baseline": "**primary (4.1)**", "patchtst": "secondary (4.3)",
            "itransformer": "secondary (4.3)", "crossformer": "secondary (4.3)"}
    pr = r["primary_test_json"]
    ni = _find(r["secondary_tests_json"], "non_inferiority") or {}
    fs = r["files_sha256"]
    L = ["Source: `experiments/round_03/exp_B16/` (primary_test.json sha256 " + fs["exp_B16/primary_test.json"] +
         ", secondary_tests.json sha256 " + fs["exp_B16/secondary_tests.json"] + "). Pooled exact two-sided Wilcoxon "
         "over 20 (subset, seed) pairs of ΔRMSE = D4 − reference, recomputed here from metrics.json.", "",
         "| D4 minus | Role | n | mean Δ | median Δ | D4 better | p (exact Wilcoxon) | Holm over 5 (not preregistered) |",
         "|---|---|---|---|---|---|---|---|"]
    for k in ["best_baseline", "theta_star", "patchtst", "itransformer", "crossformer"]:
        x = p[k]
        L.append(f"| {lab[k]} | {role[k]} | {x['n']} | {x['mean_diff']:+.3f} | {x['median_diff']:+.3f} | "
                 f"{x['n_dstar_better']}/{x['n']} | {x['wilcoxon_p']:.4f} | {x['holm5_p']:.3f} |")
    L += ["",
          f"- Primary outcome (preregistration 4.1 and 6): **{_find(pr, 'outcome')}** (p = "
          f"{_find(pr, 'wilcoxon_exact_two_sided_p'):.4f} ≥ 0.05). The direction favours D4 and may be described, not claimed.",
          f"- Non-inferiority secondary (margin {ni.get('margin_rmse')}; one-sided exact Wilcoxon on Δ − margin): "
          f"p = {ni.get('p_one_sided'):.5f}, so non-inferior at margin 0.5. It sits next to the primary and does not upgrade it.",
          "- Per-subset D4 vs B_k (paired t, Holm over 4): no subset significant (`exp_B16/stage2_table.md`).",
          "- **No significance claim of D4 vs PatchTST**: the pooled p = 0.048 is uncorrected; Holm over the 5 pooled "
          f"comparisons gives {p['patchtst']['holm5_p']:.3f}. θ* ({p['theta_star']['holm5_p']:.4f}) and iTransformer "
          f"({p['itransformer']['holm5_p']:.4f}) stay below 0.05 after this correction."]
    return L


def legacy_v3_main(out_dir):
    v3 = True
    f_md, f_csv, f_js = (("combined_table_v3.md", "combined_table_v3.csv", "combined_table_v3.json") if v3 else
                         ("combined_table.md", "combined_table.csv", "combined_stats.json"))
    ROWS = list(ROWS_V21) + ([DSTAR_ROW] if v3 else [])
    t0v = pd.read_csv(THETA0_VAL)
    data = {}
    for key, label, exp, note in ROWS:
        data[key] = {sub: load(exp_of(exp, sub)) for sub in SUBS}
    rows, stats_out = [], {"tests": "see module docstring", "comparisons": {}}
    ref = "theta_star"
    for key, label, exp, note in ROWS:
        comp = {"per_subset": {}, "pooled": None}
        pooled_d = []
        tp_raw = []
        for sub in SUBS:
            mj = data[key][sub]
            st = seed_table(mj, sub)
            row = {"row": label, "key": key, "subset": sub, "selection": note,
                   "exp": os.path.relpath(exp_of(exp, sub), WS).replace("\\", "/")}
            if len(st) == 0:
                row.update(status="pending", n=0)
                rows.append(row)
                continue
            seeds = sorted(st)
            tv = {k: [st[s]["test"][k] for s in seeds] for k in ("RMSE", "MAE", "Score")}
            if key == "theta0":
                vv = [float(t0v[(t0v.subset == sub) & (t0v.seed == s)].val_RMSE_support.iloc[0]) for s in seeds]
            else:
                vv = [st[s]["val"]["RMSE"] for s in seeds]
            blk = mj["per_subset"][sub]
            tun = r3_tuning(sub) if key == "dstar_d4" else (blk.get("tuning") or {})
            row.update(status="final" if len(seeds) == 5 else f"interim ({len(seeds)} seeds)", n=len(seeds),
                       trials=tun.get("completed_trials", 0 if key == "theta0" else None),
                       tune_gpu_h=round(tun.get("gpu_h", 0.0), 2) if tun else None,
                       val_RMSE_support=np.mean(vv), val_RMSE_support_std=np.std(vv, ddof=1) if len(vv) > 1 else None,
                       test_RMSE=np.mean(tv["RMSE"]), test_RMSE_std=np.std(tv["RMSE"], ddof=1) if len(seeds) > 1 else None,
                       test_MAE=np.mean(tv["MAE"]), test_MAE_std=np.std(tv["MAE"], ddof=1) if len(seeds) > 1 else None,
                       test_Score=np.mean(tv["Score"]),
                       test_Score_std=np.std(tv["Score"], ddof=1) if len(seeds) > 1 else None)
            if key != ref:
                rs = seed_table(data[ref][sub], sub)
                common = [s for s in seeds if s in rs]
                if len(common) >= 2:
                    a = [st[s]["test"]["RMSE"] for s in common]
                    b = [rs[s]["test"]["RMSE"] for s in common]
                    asc = [st[s]["test"]["Score"] for s in common]
                    bsc = [rs[s]["test"]["Score"] for s in common]
                    ps = S.paired_seed_tests(a, b)
                    pss = S.paired_seed_tests(asc, bsc)
                    d = np.array(a) - np.array(b)
                    pooled_d += d.tolist()
                    # engine level: seed-averaged predictions over the common seeds
                    pa = [pd.read_csv(pred_path(exp_of(exp, sub), st[s]["predictions_file_test"])) for s in common]
                    pb = [pd.read_csv(pred_path(ROWS[0][2], rs[s]["predictions_file_test"])) for s in common]
                    for x in pa[1:] + pb:
                        assert np.array_equal(x.engine_id.to_numpy(), pa[0].engine_id.to_numpy())
                        assert np.allclose(x.y_true_capped.to_numpy(), pa[0].y_true_capped.to_numpy())
                    y = pa[0].y_true_capped.to_numpy()
                    ya = np.mean([x.y_pred.to_numpy() for x in pa], 0)
                    yb = np.mean([x.y_pred.to_numpy() for x in pb], 0)
                    dm = S.dm_test(ya - y, yb - y)
                    bt = S.bootstrap_delta(y, ya, yb)
                    ci = S.t_interval(d)
                    comp["per_subset"][sub] = {"seeds": common, "dRMSE_seed": ps, "dScore_seed": pss,
                                               "dRMSE_seed_ci95_t": ci, "dm_hln_seedavg": dm,
                                               "bootstrap_seedavg": bt}
                    tp_raw.append((sub, ps.get("t_p")))
                    row.update(dRMSE=ps["mean_diff"], dRMSE_ci_lo=ci[0], dRMSE_ci_hi=ci[1], t_p=ps.get("t_p"),
                               wilcoxon_p=ps.get("wilcoxon_p"), dScore=pss["mean_diff"],
                               dm_p=dm.get("p"), boot_dRMSE_lo=bt["dRMSE_ci95"][0], boot_dRMSE_hi=bt["dRMSE_ci95"][1])
            rows.append(row)
        if key != ref and tp_raw:
            adj = S.holm([p for _, p in tp_raw])
            for (sub, _), q in zip(tp_raw, adj):
                comp["per_subset"][sub]["dRMSE_seed"]["t_p_holm4"] = q
                for r in rows:
                    if r["key"] == key and r["subset"] == sub:
                        r["t_p_holm"] = q
            if len(pooled_d) >= 5 and np.any(np.array(pooled_d) != 0):
                comp["pooled"] = S.pooled_wilcoxon(pooled_d)
            stats_out["comparisons"][f"{key}_minus_theta_star"] = comp
    df = pd.DataFrame(rows)
    if v3:
        stats_out["round3"] = round3_block(data)
    df.to_csv(os.path.join(out_dir, f_csv), index=False, float_format="%.4f")
    with open(os.path.join(out_dir, f_js), "w", encoding="utf-8") as f:
        json.dump(stats_out, f, indent=1, default=float)
    # markdown
    title = ("# Combined table v3 (round 3; test set; 5 retrain seeds = split seeds 0-4; mean ± std, ddof 1)" if v3 else
             "# Round-02 combined table (test set; 5 retrain seeds = split seeds 0-4; mean ± std, ddof 1)")
    L = [title, "",
         "Built by `table_r02/code/make_combined_table.py` from metrics.json and prediction CSVs (no test read). "
         "Score: d = pred − true; exp(−d/13)−1 if d < 0, exp(d/10)−1 if d ≥ 0. ΔRMSE = row − θ* (negative: row better), "
         "seed-paired. p(t): paired t, Holm over 4 subsets. p(W): exact Wilcoxon (n = 5 floor 0.0625). "
         "p(DM): Diebold-Mariano-HLN on seed-averaged engine predictions. Boot CI: engine-level paired bootstrap of "
         "ΔRMSE on seed-averaged predictions. Val RMSE: VA common support (label cycle ≥ 51).", ""]
    if v3:
        L += V3_HEADER + [""]
    for sub in SUBS:
        L += [f"## {sub}", "",
              "| Model | Selection | Trials (tune GPU-h) | Val RMSE | Test RMSE | Test MAE | Test Score | ΔRMSE vs θ* [95% t CI] | p(t, Holm) | p(W) | p(DM) | Boot ΔRMSE CI | ΔScore |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in [x for x in rows if x["subset"] == sub]:
            if r["status"] == "pending":
                L.append(f"| {r['row']} | {r['selection']} | pending | | | | | | | | | | |")
                continue
            tr = f"{r['trials']} ({r['tune_gpu_h']})" if r.get("trials") not in (None, 0) else "0"
            if r["key"] == "theta_star" or "dRMSE" not in r:
                dl = "—" if r["key"] == "theta_star" else "n/a"
                st_ = [dl] * 6
            else:
                st_ = [f"{r['dRMSE']:+.2f} [{r['dRMSE_ci_lo']:+.2f}, {r['dRMSE_ci_hi']:+.2f}]",
                       f"{r.get('t_p_holm', float('nan')):.3f}", f"{r['wilcoxon_p']:.4f}" if r.get('wilcoxon_p') is not None else "—",
                       f"{r['dm_p']:.3f}" if r.get("dm_p") is not None else "—",
                       f"[{r['boot_dRMSE_lo']:+.2f}, {r['boot_dRMSE_hi']:+.2f}]", f"{r['dScore']:+.1f}"]
                if r["status"] != "final":       # C5: every statistic of an interim row is marked, not only the label
                    st_ = [x + f" (interim, {r['n']} seeds)" for x in st_]
            status = "" if r["status"] == "final" else f" ({r['status']})"
            L.append(f"| {r['row']}{status} | {r['selection']} | {tr} | {fmt(r['val_RMSE_support'], r['val_RMSE_support_std'])} | "
                     f"{fmt(r['test_RMSE'], r['test_RMSE_std'])} | {fmt(r['test_MAE'], r['test_MAE_std'])} | "
                     f"{fmt(r['test_Score'], r['test_Score_std'], 1)} | " + " | ".join(st_) + " |")
        L.append("")
    L += ["## Pooled over subsets (20 seed pairs): exact Wilcoxon on ΔRMSE (row − θ*)", "",
          "| Comparison | n | mean ΔRMSE | median ΔRMSE | p |", "|---|---|---|---|---|"]
    for k, c in stats_out["comparisons"].items():
        if c.get("pooled"):
            p = c["pooled"]
            L.append(f"| {k} | {p['n']} | {p['mean_diff']:+.3f} | {p['median_diff']:+.3f} | {p['wilcoxon_p']:.4f} |")
    esp = os.path.join(R2, "exp_B6", "eval_seed_spread.json")
    if os.path.exists(esp):
        es = json.load(open(esp, encoding="utf-8"))
        L += ["", "## ProbSparse evaluation-seed spread of θ* (RULES 13; eval seed fixed at 0 in every row above)", "",
              "Validation common support, eval seeds 0-9, per checkpoint (`exp_B6/eval_seed_spread.json`; no test read).", "",
              "| Subset | mean std over eval seeds | max range | split-seed std (eval seed 0) |", "|---|---|---|---|"]
        for sub, b in es["per_subset"].items():
            L.append(f"| {sub} | {b['mean_eval_seed_std']:.3f} | {b['max_eval_seed_range']:.3f} | "
                     f"{b['split_seed_std_at_eval_seed0']:.3f} |")
    # superseded reference cells (never in the statistics)
    L += ["", "## Superseded reference (not used in any statistic above)", "",
          "Pre-committed replacement (exp_B10/topup_precommit.md): the PatchTST row above uses the full-budget top-up. "
          "The budget-short exp_B10 cells it replaced are listed here only for transparency.", "",
          "| Row | Subset | Trials (tune GPU-h) | Val RMSE | Test RMSE | Test MAE | Test Score |", "|---|---|---|---|---|---|---|"]
    for key, label, exp, subs in SUPERSEDED:
        mj = load(exp)
        for sub in subs:
            st = seed_table(mj, sub)
            if not st:
                continue
            seeds = sorted(st)
            tv = {k: [st[x]["test"][k] for x in seeds] for k in ("RMSE", "MAE", "Score")}
            vv = [st[x]["val"]["RMSE"] for x in seeds]
            tun = mj["per_subset"][sub].get("tuning") or {}
            L.append(f"| {label} | {sub} | {tun.get('completed_trials')} ({tun.get('gpu_h', 0):.2f}) | "
                     f"{fmt(np.mean(vv), np.std(vv, ddof=1))} | {fmt(np.mean(tv['RMSE']), np.std(tv['RMSE'], ddof=1))} | "
                     f"{fmt(np.mean(tv['MAE']), np.std(tv['MAE'], ddof=1))} | "
                     f"{fmt(np.mean(tv['Score']), np.std(tv['Score'], ddof=1), 1)} |")
    # C1: test-time batch-invariance flags (evaluate_test.py rc 5), listed for every row
    flags = []
    for key, label, exp, note in ROWS:
        for sub in SUBS:
            for sd, r in sorted(seed_table(data[key][sub], sub).items()):
                if r.get("batch_invariance_pass") is False:
                    ename = os.path.basename(exp_of(exp, sub))
                    flags.append((ename, sub, sd, r.get("batch_invariance_maxdiff")))
    L += ["", C6_LINE]
    if v3:
        L += ["", "## Round 3: D* = D4 row, preregistered comparison (read this before the D4 row's Δ vs θ*)", ""]
        L += round3_md(stats_out["round3"])
    L += ["", "## Test-time batch-invariance flags (R0.2 check in evaluate_test.py; tolerance 1e-3 cycles)", ""]
    if not flags:
        L.append("- None: every test read passed.")
    for ename, sub, sd, md in flags:
        L.append(f"- **{ename} {sub} seed {sd}: FAILED** (max |p_b − p_b1024| = {md:.3g} cycles; evaluate_test.py rc 5; the "
                 f"prediction file at batch 1024 is the one logged and used). Effect bound: |ΔRMSE| ≤ {md:.3g} cycles, "
                 f"immaterial to every comparison. " + BI_NOTES.get((ename, sub, sd), ""))
    n_reads = sum(len(seed_table(data[k][sub], sub)) for k, _, _, _ in ROWS for sub in SUBS)
    L.append(f"- All other test reads in this table passed ({n_reads - len(flags)} of {n_reads}).")
    L += ["", "## Budget notes", "",
          "- Completed trials per (model, subset) are in the Trials column (target 12 = 3 studies x 4). Shortfalls come from the "
          "pre-registered per-study timeout (3.0 x exp_A4 estimate); see each experiment README.",
          "- \"Same budget\" means the same pre-registered trial rule (3 studies x 4 completed trials, RandomSampler seeds 0-2, "
          "same search-space rules, objective, recipe and OOM rule) and the same timeout factor. It does not mean the same GPU-hours: "
          "tuning GPU-h differ by model and subset (Trials column), and θ* used more tuning GPU-h than every baseline.",
          "- Direction of a shortfall: the trials of a study are a fixed RandomSampler stream, so a timeout removes the cut trial and "
          "every later draw; the truncated pool is a subset of the full pool and its selected validation objective can only be equal "
          "or worse. A shortfall therefore biases selection against the truncated model (with certainty on validation, in expectation "
          "on test). The exp_A4 unit costs underestimated PatchTST by about 2x, so in exp_B10 the same timeout factor cut 5 PatchTST "
          "studies (40/48 trials). The pre-committed top-up (exp_B10_topup, timeout 180 min) re-ran them from scratch; the row above "
          "replaces exp_B10 whatever the result. θ* FD004 has 11/12 trials (one study cut), which works against θ*.",
          "- Observed in the top-up: all 5 re-run studies completed 4/4 and the 13 trials run before the original cuts reproduced "
          "their validation values exactly. The full pool selected a lower split-0 validation value on FD001 (11.573 vs 11.745) and "
          "FD004 (13.552 vs 13.603), but test RMSE rose (FD001 13.21 -> 13.37, FD004 14.02 -> 14.18); FD002 kept the same trial "
          "(bitwise-identical predictions). So the shortfall bias holds on validation, and on test it did not hold in these two cases.",
          "- θ0 and ITT rows have no own search. θ0 = checkpoint configuration (tuned on test in the submission, R0.3).",
          "- Val RMSE for θ0 = exp_B6/theta0_reference.csv (exp_B3 checkpoints on the common support)."]
    if v3:
        L += ["- D* = D4 row: 12 completed trials per subset under the same trial rule (FD001/FD003/FD004 study 0 = exp_B15 "
              "stage 1, studies 1-2 = exp_B16; FD002 studies 0-2 = exp_B16). The Trials column counts only D4's own trials. "
              "D4 was itself chosen among 4 designs on validation (stage 1: 48 trials, 2.88 GPU-h queue wall; exp_B15), so "
              "its total search is larger than any other row's (exp_B15/README.md, exp_B16/README.md)."]
    with open(os.path.join(out_dir, f_md), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("written", os.path.join(out_dir, f_md))


# =====================================================================================================================
# legacy exp_C4 (verbatim from round_04_pre/exp_C4/seed_bootstrap_effect_sizes.py, sha256 6b916631...; renamed)
# =====================================================================================================================
IDX5 = np.array(list(itertools.product(range(5), repeat=5)))     # 3125 x 5, exact bootstrap


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()




def dz_ci(d, level=0.95):
    n = len(d)
    t_obs = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    df = n - 1
    J = 1 - 3 / (4 * df - 1)
    def cdf(nc):  # nct.cdf can return NaN far in the tails; there it is 0 (nc >> t) or 1 (nc << t)
        v = st.nct.cdf(t_obs, df, nc)
        return v if np.isfinite(v) else (0.0 if nc > t_obs else 1.0)
    w = 10 + 3 * abs(t_obs)
    lo = brentq(lambda nc: cdf(nc) - (1 + level) / 2, t_obs - w, t_obs + w)
    hi = brentq(lambda nc: cdf(nc) - (1 - level) / 2, t_obs - w, t_obs + w)
    return {"d_z": float(t_obs / np.sqrt(n)), "g_z": float(J * t_obs / np.sqrt(n)),
            "d_z_ci95_noncentral_t": [float(lo / np.sqrt(n)), float(hi / np.sqrt(n))]}


def subset_block(a, b):
    d = a - b
    boots = d[IDX5].mean(1)
    g = []
    for ix in IDX5:
        aa, bb = a[ix], b[ix]
        if np.var(aa, ddof=1) + np.var(bb, ddof=1) > 0:
            g.append(S.hedges_g_av(aa, bb))
    g = np.array(g)
    return {"per_seed_diff": d.round(6).tolist(), "mean_diff": float(d.mean()), "std_diff": float(d.std(ddof=1)),
            "seed_boot_ci95_mean_diff": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "seed_boot_method": "exact enumeration of 3125 ordered resamples, percentile",
            "t_ci95_mean_diff": list(map(float, S.t_interval(d))),
            "hedges_g_av": S.hedges_g_av(a, b),
            "hedges_g_av_seed_boot_ci95": [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))],
            "hedges_g_av_boot_dropped_zero_var": int(len(IDX5) - len(g)), **dz_ci(d)}




# =====================================================================================================================
# v4 (post-freeze, RULES 23; concerns R2.9, R0.15, R2.8, R1.1, R0.7): one generator for every final row.
# =====================================================================================================================
R4 = os.path.join(WS, "experiments", "round_04")
TEST_LOG = os.path.join(WS, "experiments", "test_eval_log.csv")
V4_ROWS = [  # key, label, selection note, experiment dir (or per-subset dict), metrics.json schema
    ("d4", "D4 (main row; preregistered round 3; exp_B16)",
     "validation (P2, 12 trials; design D4 chosen on validation in the preregistered stage-1 screen of 4 designs)",
     os.path.join(R3, "exp_B16"), "std"),
    ("theta_star", "Base θ* (corrected original model; P2 tuned, exp_B6)", "validation (P2; target 12 trials, see Trials)",
     os.path.join(R2, "exp_B6"), "std"),
    ("patchtst", "PatchTST (P2 tuned, full budget: exp_B10_topup; FD003 exp_B10)",
     "validation (P2; target 12 trials, see Trials)",
     {"FD001": os.path.join(R2, "exp_B10_topup"), "FD002": os.path.join(R2, "exp_B10_topup"),
      "FD003": os.path.join(R2, "exp_B10"), "FD004": os.path.join(R2, "exp_B10_topup")}, "std"),
    ("itransformer", "iTransformer (P2 tuned, exp_B11)", "validation (P2; target 12 trials, see Trials)",
     os.path.join(R2, "exp_B11"), "std"),
    ("crossformer", "Crossformer (P2 tuned, exp_B12)", "validation (P2; target 12 trials, see Trials)",
     os.path.join(R2, "exp_B12"), "std"),
    ("lstm", "LSTM (shared 12-trial rule and search ranges; exp_V2)",
     "validation (same 12-trial rule; strict deterministic)", os.path.join(R4, "exp_V2"), "lstm"),
    ("ridge", "Ridge (12-trial rule; exp_V3)", "validation (same 12-trial rule; closed form, float64)",
     os.path.join(R4, "exp_V3"), "ridge"),
]
V4_REF = "d4"
V4_SHORT = {"d4": "D4", "theta_star": "θ*", "patchtst": "PatchTST", "itransformer": "iTransformer",
            "crossformer": "Crossformer", "lstm": "LSTM", "ridge": "Ridge", "best_baseline": "best baseline B_k"}
V4_EXCLUDED = ["ITT (θ* + submitted recipe, exp_B14): removed from the paper (RULES 21)",
               "θ0 checkpoint configuration (round_01/exp_B3): reference only, never a candidate (RULES 14)",
               "θ* fusion/branch rows: removed (RULES 22a)",
               "round-4/5 ablation arms (exp_V1, exp_A10-A13): reported in their own folders, not rows of this table"]
HOLM5 = ["theta_star", "best_baseline", "patchtst", "itransformer", "crossformer"]      # v3 family (reviewer check)
HOLM7 = ["theta_star", "best_baseline", "patchtst", "itransformer", "crossformer", "lstm", "ridge"]


def rel(p):
    return os.path.relpath(p, WS).replace("\\", "/")


def read_test_log():
    import csv
    with open(TEST_LOG, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def v4_seed_records(key, exp, schema, sub):
    mj = json.load(open(os.path.join(exp, "metrics.json"), encoding="utf-8"))
    blk = mj["per_subset"][sub]
    lst = blk["per_seed"] if schema == "std" else blk[f"{schema}_per_seed"]
    out = {}
    for r in lst:
        if r.get("test") is None:
            continue
        pf = (pred_path(exp, r["predictions_file_test"]) if schema == "std" else os.path.join(WS, r["test_preds_file"]))
        out[int(r["seed"])] = {"pred_file": os.path.normpath(pf), "test": r["test"], "val_RMSE": r["val"]["RMSE"],
                               "ckpt_sha256": r.get("ckpt_sha256"), "stored_pred_sha256": r.get("test_preds_sha256"),
                               "batch_invariance_pass": r.get("batch_invariance_pass"),
                               "batch_invariance_maxdiff": r.get("batch_invariance_maxdiff")}
    return mj, blk, out


def v4_tuning(key, exp, schema, sub, blk):
    if key == "d4":
        return r3_tuning(sub)
    if schema == "std":
        t = blk.get("tuning") or {}
        return {"completed_trials": t.get("completed_trials"), "gpu_h": t.get("gpu_h")}
    res = [json.load(open(os.path.join(exp, "tune", f"{schema}_{sub}_study{k}", "result.json"), encoding="utf-8"))
           for k in (0, 1, 2)]
    return {"completed_trials": int(sum(r.get("n_complete") or 0 for r in res)),
            "gpu_h": float(sum(r.get("gpu_h") or 0 for r in res))}




def v4_effect(d):
    d = np.asarray(d, float)
    if d.std(ddof=1) == 0:
        return {"d_z": None, "g_z": None, "d_z_ci95_noncentral_t": [None, None]}
    return dz_ci(d)


def v4_seed_boot(d):
    d = np.asarray(d, float)
    b = d[IDX5].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def generate_v4(rows_path, out_dir):
    fr = json.load(open(rows_path, encoding="utf-8"))
    rows_sha = sha256(rows_path)
    assert not fr["problems"], fr["problems"]
    # ---- load and re-verify every named file
    data, checks = {}, {"pred_files_verified": 0, "metrics_files_verified": 0, "r2_identity_max_abs": 0.0,
                        "recompute_max_abs": 0.0}
    for ent in fr["rows"]:
        k = ent["key"]
        data[k] = {}
        for sub in SUBS:
            ps = ent["per_subset"][sub]
            assert sha256(os.path.join(WS, ps["metrics_json"])) == ps["metrics_json_sha256"], (k, sub, "metrics sha")
            checks["metrics_files_verified"] += 1
            recs = {}
            for s in SEEDS:
                e = ps["seeds"][str(s)]
                p = os.path.join(WS, e["pred_file"])
                assert sha256(p) == e["pred_sha256"], (k, sub, s, "pred sha")
                checks["pred_files_verified"] += 1
                df = pd.read_csv(p)
                y, yp = df.y_true_capped.to_numpy(), df.y_pred.to_numpy()
                m = M.all_metrics(y, yp)
                r2_id = 1 - m["RMSE"] ** 2 / np.var(y)
                checks["r2_identity_max_abs"] = max(checks["r2_identity_max_abs"], abs(r2_id - m["R2"]))
                checks["recompute_max_abs"] = max(checks["recompute_max_abs"],
                                                  max(abs(m[q] - e["test_metrics_json"][q]) for q in ("RMSE", "MAE", "R2")))
                recs[s] = {"df": df, "m": m, "val": e["val_RMSE_support"], "bi": e["batch_invariance_pass"],
                           "bimax": e["batch_invariance_maxdiff"]}
            data[k][sub] = {"recs": recs, "tuning": ps["tuning"], "exp": ps["exp"]}
    # engine alignment across all rows
    for sub in SUBS:
        ref = data[V4_REF][sub]["recs"][0]["df"]
        for k in data:
            for s in SEEDS:
                df = data[k][sub]["recs"][s]["df"]
                assert np.array_equal(df.engine_id.to_numpy(), ref.engine_id.to_numpy()), (k, sub, s)
                assert np.allclose(df.y_true_capped.to_numpy(), ref.y_true_capped.to_numpy()), (k, sub, s)
    checks["engine_ids_and_labels_identical_across_rows"] = True
    labels = {e["key"]: e["label"] for e in fr["rows"]}
    notes = {e["key"]: e["selection"] for e in fr["rows"]}
    keys = [e["key"] for e in fr["rows"]]

    def vec(k, sub, q):
        return np.array([data[k][sub]["recs"][s]["m"][q] for s in SEEDS], float)

    # ---- row summaries
    summ = {}
    for k in keys:
        summ[k] = {}
        for sub in SUBS:
            v = np.array([data[k][sub]["recs"][s]["val"] for s in SEEDS], float)
            b = {"val_RMSE_support": [float(v.mean()), float(v.std(ddof=1))]}
            for q in ("RMSE", "MAE", "R2", "Score", "Score_early", "Score_late"):
                x = vec(k, sub, q)
                b["test_" + q] = [float(x.mean()), float(x.std(ddof=1))]
            b["tuning"] = data[k][sub]["tuning"]
            b["per_seed"] = {q: vec(k, sub, q).tolist() for q in ("RMSE", "MAE", "R2", "Score")}
            b["per_seed"]["val_RMSE_support"] = v.tolist()
            summ[k][sub] = b
    # ---- comparisons D4 - X
    comps_keys = [k for k in keys if k != V4_REF] + ["best_baseline"]

    def other(x, sub):
        return BEST_BASELINE[sub] if x == "best_baseline" else x

    comps = {}
    for x in comps_keys:
        c = {"per_subset": {}, "pooled": {}}
        tps = []
        for sub in SUBS:
            ok = other(x, sub)
            blk = {"vs": ok}
            for q in ("RMSE", "Score"):
                a, b = vec(V4_REF, sub, q), vec(ok, sub, q)
                d = a - b
                ps = S.paired_seed_tests(a, b)
                blk["d" + q] = {"per_seed_diff": d.tolist(), "mean_diff": float(d.mean()), "std_diff": float(d.std(ddof=1)),
                                "t_ci95": S.t_interval(d), "t_p": ps.get("t_p"), "wilcoxon_exact_p": ps.get("wilcoxon_p"),
                                "n_d4_better": int((d < 0).sum()), **v4_effect(d),
                                "supplement_seed_boot_ci95": v4_seed_boot(d)}
            y = data[V4_REF][sub]["recs"][0]["df"].y_true_capped.to_numpy()
            ya = np.mean([data[V4_REF][sub]["recs"][s]["df"].y_pred.to_numpy() for s in SEEDS], 0)
            yb = np.mean([data[ok][sub]["recs"][s]["df"].y_pred.to_numpy() for s in SEEDS], 0)
            blk["dm_hln_seedavg"] = S.dm_test(ya - y, yb - y)
            blk["engine_bootstrap_seedavg"] = S.bootstrap_delta(y, ya, yb)
            tps.append(blk["dRMSE"]["t_p"])
            c["per_subset"][sub] = blk
        for sub, q in zip(SUBS, S.holm(tps)):
            c["per_subset"][sub]["dRMSE"]["t_p_holm4"] = q
        tsc = S.holm([c["per_subset"][sub]["dScore"]["t_p"] for sub in SUBS])
        for sub, q in zip(SUBS, tsc):
            c["per_subset"][sub]["dScore"]["t_p_holm4"] = q
        for q in ("RMSE", "Score"):
            D = np.array([c["per_subset"][sub]["d" + q]["per_seed_diff"] for sub in SUBS])   # 4 x 5
            flat = D.ravel()
            w = S.pooled_wilcoxon(flat)
            rng = np.random.default_rng(0)
            draws = np.stack([D[i][rng.integers(0, 5, size=(20000, 5))] for i in range(4)], 1)
            pm = draws.reshape(20000, -1).mean(1)
            c["pooled"]["d" + q] = {"n": w["n"], "mean_diff": w["mean_diff"], "median_diff": w["median_diff"],
                                    "n_d4_better": int((flat < 0).sum()), "wilcoxon_exact_p": w["wilcoxon_p"],
                                    **v4_effect(flat), "g_z_note": "descriptive: 20 pairs from 4 subsets with different scales",
                                    "supplement_stratified_seed_boot_ci95": [float(np.percentile(pm, 2.5)),
                                                                             float(np.percentile(pm, 97.5))]}
        comps[x] = c
    for fam, name in ((HOLM5, "holm5_v3_family"), (HOLM7, "holm7_v4_family")):
        for q in ("RMSE", "Score"):
            adj = S.holm([comps[x]["pooled"]["d" + q]["wilcoxon_exact_p"] for x in fam])
            for x, a in zip(fam, adj):
                comps[x]["pooled"]["d" + q][name] = a
    # supplement: seed bootstrap narrower than t CI?
    narrower, cells = 0, 0
    for x in comps_keys:
        for sub in SUBS:
            for q in ("RMSE", "Score"):
                b = comps[x]["per_subset"][sub]["d" + q]
                cells += 1
                narrower += (b["supplement_seed_boot_ci95"][1] - b["supplement_seed_boot_ci95"][0]) < (b["t_ci95"][1] - b["t_ci95"][0])
    b16 = os.path.join(R3, "exp_B16")
    prim = json.load(open(os.path.join(b16, "primary_test.json"), encoding="utf-8"))
    sec = json.load(open(os.path.join(b16, "secondary_tests.json"), encoding="utf-8"))
    ni = _find(sec, "non_inferiority") or {}
    # primary recomputed equals the preregistered file?
    pr_re = comps["best_baseline"]["pooled"]["dRMSE"]
    checks["primary_recompute_equals_exp_B16"] = bool(abs(pr_re["wilcoxon_exact_p"] - prim["wilcoxon_exact_two_sided_p"]) < 1e-12
                                                      and abs(pr_re["mean_diff"] - prim["mean_delta"]) < 1e-9)
    flags = [(k, sub, s, data[k][sub]["recs"][s]["bimax"]) for k in keys for sub in SUBS for s in SEEDS
             if data[k][sub]["recs"][s]["bi"] is False]
    checks["test_eval_log_sha256_at_generation"] = sha256(TEST_LOG)
    checks["seed_boot_narrower_than_t_cells"] = [int(narrower), int(cells)]
    out = {"version": "v4", "generator": rel(os.path.abspath(__file__)), "generator_sha256": sha256(os.path.abspath(__file__)),
           "final_rows": rel(rows_path), "final_rows_sha256": rows_sha, "kind": "analysis_posthoc (reads saved test "
           "prediction files; no model evaluated; no test or RUL file opened; selects nothing)",
           "delta": "D4 − row (negative: D4 better)", "best_baseline_by_validation": BEST_BASELINE,
           "rows": {k: {"label": labels[k], "selection": notes[k], "per_subset": summ[k]} for k in keys},
           "comparisons_d4_minus": comps, "preregistered_primary_exp_B16": prim,
           "non_inferiority_exp_B16": ni, "batch_invariance_flags": flags, "checks": checks, "excluded": fr["excluded"],
           "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__}}
    with open(os.path.join(out_dir, "combined_table_v4.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float, ensure_ascii=False)
    # ---- csv (one row per model x subset)
    cr = []
    for k in keys:
        for sub in SUBS:
            b = summ[k][sub]
            r = {"key": k, "row": labels[k], "subset": sub, "exp": data[k][sub]["exp"],
                 "trials": b["tuning"].get("completed_trials"), "tune_gpu_h": b["tuning"].get("gpu_h"),
                 "val_RMSE_support": b["val_RMSE_support"][0], "val_RMSE_support_std": b["val_RMSE_support"][1]}
            for q in ("RMSE", "MAE", "R2", "Score"):
                r[f"test_{q}"], r[f"test_{q}_std"] = b["test_" + q]
            if k != V4_REF:
                cb = comps[k]["per_subset"][sub]
                r.update(dRMSE_d4_minus=cb["dRMSE"]["mean_diff"], dRMSE_t_lo=cb["dRMSE"]["t_ci95"][0],
                         dRMSE_t_hi=cb["dRMSE"]["t_ci95"][1], g_z=cb["dRMSE"]["g_z"],
                         g_z_lo=cb["dRMSE"]["d_z_ci95_noncentral_t"][0], g_z_hi=cb["dRMSE"]["d_z_ci95_noncentral_t"][1],
                         t_p_holm4=cb["dRMSE"]["t_p_holm4"], wilcoxon_p=cb["dRMSE"]["wilcoxon_exact_p"],
                         dm_p=cb["dm_hln_seedavg"].get("p"), engine_boot_lo=cb["engine_bootstrap_seedavg"]["dRMSE_ci95"][0],
                         engine_boot_hi=cb["engine_bootstrap_seedavg"]["dRMSE_ci95"][1],
                         dScore_d4_minus=cb["dScore"]["mean_diff"], dScore_t_lo=cb["dScore"]["t_ci95"][0],
                         dScore_t_hi=cb["dScore"]["t_ci95"][1])
            cr.append(r)
    pd.DataFrame(cr).to_csv(os.path.join(out_dir, "combined_table_v4.csv"), index=False, float_format="%.4f")
    # ---- markdown
    f2 = lambda v: "—" if v is None else f"{v:+.2f}"           # noqa: E731
    f1 = lambda v: "—" if v is None else f"{v:+.1f}"           # noqa: E731
    fp = lambda v: "—" if v is None else (f"{v:.4f}" if v >= 1e-4 else f"{v:.1e}")   # noqa: E731
    L = ["# Combined table v4 (final rows; test set; 5 retrain seeds = split seeds 0-4; mean ± std, ddof 1)", "",
         f"Built by `experiments/final/code/make_table_v4.py --version v4` from `experiments/final/final_rows.json` "
         f"(sha256 {rows_sha}). Generated once after the 2026-10-26 experiment freeze (RULES 23) as a logged "
         "`analysis_posthoc`: it reads metrics.json files and the saved test prediction files named in final_rows.json "
         "(every sha256 re-verified); it evaluates no model, opens no test_FD/RUL_FD file and selects nothing. All test "
         "numbers come from the single test read of each configuration (experiments/test_eval_log.csv).", "",
         "Definitions. Score: d = pred − true; exp(−d/13)−1 if d < 0, exp(d/10)−1 if d ≥ 0, summed over engines. "
         "**Δ = D4 − row (negative: D4 better)**, paired by split seed. Seed-level interval: paired t 95% CI (n = 5). "
         "Effect size: g_z = J·mean(Δ)/sd(Δ) with an exact noncentral-t 95% CI (CI on d_z). p(t, Holm): paired t, Holm "
         "over the 4 subsets of each comparison. p(W): exact Wilcoxon signed-rank (n = 5: floor 0.0625). p(DM): "
         "Diebold-Mariano-HLN on seed-averaged engine predictions. Engine boot CI: engine-level paired percentile "
         "bootstrap (10,000 resamples) = CI for ΔRMSE of seed-averaged predictions (not of the seed mean). Val RMSE: "
         "validation common support (label cycle ≥ 51). R² = 1 − RMSE²/Var(y_test).", "",
         "Reading rules (paper_plan.md D29, D34, D35, D36; verdicts/round_03.md C3):",
         f"- **Preregistered primary: D4 − validation-chosen best baseline (Crossformer FD001-FD003, PatchTST FD004), pooled "
         f"over 20 pairs: {prim['outcome']} (exact two-sided Wilcoxon p = {prim['wilcoxon_exact_two_sided_p']:.4f}).** "
         f"Non-inferiority at margin {ni.get('margin_rmse')}: one-sided p = {ni.get('p_one_sided'):.5f}. All other "
         "comparisons are secondary.",
         "- No significance claim of D4 vs PatchTST (pooled p uncorrected; Holm columns below).",
         "- The supplementary percentile seed bootstrap (n = 5) under-covers and never supports a superiority or "
         "significance claim. A non-rejection is never 'no difference' or 'equivalent'.",
         "- LSTM: on FD004 the LSTM has lower test RMSE than D4 although D4 is better on validation; never rank D4 above "
         "the LSTM on FD004 or write 'all subsets' (D34). 16/48 LSTM tuning trials did not converge within 150 epochs; the "
         "row is a conservative estimate for LSTMs (exp_V2/README.md).",
         "- Ridge: the α search range was binding at α = 1000 on FD001/FD003 (D35); its 5 seeds differ only through the split.",
         "- Determinism (D36): D4 seed 0 and the seed-0 cells of θ*, PatchTST, iTransformer and Crossformer are certified "
         "bitwise in strict mode (exp_CERT); seeds 1-4 of those rows are round-2/3 runs (deterministic algorithms off); "
         "LSTM and ridge rows are strict deterministic (round 4). Mixed mode in every paired comparison.",
         "- Disclosure (paper_plan D1): the round-3 designs were motivated after seeing round-2 results, including "
         "test-set summaries. D4 was chosen among 4 designs on validation, so its total search is larger than any other row's.",
         "- Rows not in this table: " + "; ".join(fr["excluded"]) + ".", ""]
    for sub in SUBS:
        L += [f"## {sub}", "", "| Model | Selection | Trials (tune GPU-h) | Val RMSE | Test RMSE | Test MAE | Test R² | Test Score |",
              "|---|---|---|---|---|---|---|---|"]
        for k in keys:
            b = summ[k][sub]
            t = b["tuning"]
            tr = (f"{t.get('completed_trials')} ({t.get('gpu_h'):.2f})" if t.get("gpu_h") is not None
                  else f"{t.get('completed_trials')}")
            L.append(f"| {labels[k]} | {notes[k]} | {tr} | {fmt(*b['val_RMSE_support'])} | {fmt(*b['test_RMSE'])} | "
                     f"{fmt(*b['test_MAE'])} | {fmt(*b['test_R2'], d=3)} | {fmt(*b['test_Score'], d=1)} |")
        L += ["", f"D4 minus row, {sub} (negative: D4 better):", "",
              "| D4 − | ΔRMSE [95% t CI] | g_z [95% CI] | p(t, Holm 4) | p(W) | D4 better | p(DM) | Engine boot ΔRMSE CI | ΔScore [95% t CI] | g_z Score [95% CI] |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for x in comps_keys:
            cb = comps[x]["per_subset"][sub]
            r_, s_ = cb["dRMSE"], cb["dScore"]
            nm = V4_SHORT[x] + (f" ({V4_SHORT[cb['vs']]})" if x == "best_baseline" else "")
            gz = lambda b, ff: ("—" if b["g_z"] is None else f"{b['g_z']:+.2f} [{b['d_z_ci95_noncentral_t'][0]:+.2f}, "  # noqa: E731
                                                              f"{b['d_z_ci95_noncentral_t'][1]:+.2f}]")
            eb = cb["engine_bootstrap_seedavg"]["dRMSE_ci95"]
            L.append(f"| {nm} | {f2(r_['mean_diff'])} [{f2(r_['t_ci95'][0])}, {f2(r_['t_ci95'][1])}] | {gz(r_, f2)} | "
                     f"{fp(r_['t_p_holm4'])} | {fp(r_['wilcoxon_exact_p'])} | {r_['n_d4_better']}/5 | "
                     f"{fp(cb['dm_hln_seedavg'].get('p'))} | [{f2(eb[0])}, {f2(eb[1])}] | "
                     f"{f1(s_['mean_diff'])} [{f1(s_['t_ci95'][0])}, {f1(s_['t_ci95'][1])}] | {gz(s_, f1)} |")
        L.append("")
    L += ["## Pooled over subsets (20 seed pairs; D4 − row)", "",
          "Exact two-sided Wilcoxon on the 20 (subset, seed) differences. Holm-5 = family of v3 (θ*, best baseline, "
          "PatchTST, iTransformer, Crossformer; reviewer's check, verdicts/round_03.md). Holm-7 = all 7 D4 comparisons "
          "of this table. Neither Holm family is preregistered. Pooled g_z is descriptive (the 20 pairs come from 4 "
          "subsets with different error scales).", "",
          "| D4 − | Metric | n | mean Δ | median Δ | D4 better | p (exact W) | Holm-5 | Holm-7 | g_z [95% CI] |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for x in comps_keys:
        for q in ("RMSE", "Score"):
            p = comps[x]["pooled"]["d" + q]
            ff = f2 if q == "RMSE" else f1
            role = " (**preregistered primary**)" if (x == "best_baseline" and q == "RMSE") else ""
            gzs = ("—" if p["g_z"] is None else f"{p['g_z']:+.2f} [{p['d_z_ci95_noncentral_t'][0]:+.2f}, "
                                                 f"{p['d_z_ci95_noncentral_t'][1]:+.2f}]")
            L.append(f"| {V4_SHORT[x]}{role} | {q} | {p['n']} | {ff(p['mean_diff'])} | {ff(p['median_diff'])} | "
                     f"{p['n_d4_better']}/{p['n']} | {fp(p['wilcoxon_exact_p'])} | {fp(p.get('holm5_v3_family'))} | "
                     f"{fp(p['holm7_v4_family'])} | {gzs} |")
    L += ["", f"- Preregistered primary recomputed here equals exp_B16/primary_test.json: "
          f"{checks['primary_recompute_equals_exp_B16']} (p {prim['wilcoxon_exact_two_sided_p']:.4f}, mean Δ "
          f"{prim['mean_delta']:+.3f}).", ""]
    L += ["## Seed-level values (test RMSE and Score per split seed)", ""]
    for sub in SUBS:
        L += [f"### {sub}", "", "| Model | Metric | s0 | s1 | s2 | s3 | s4 |", "|---|---|---|---|---|---|---|"]
        for k in keys:
            for q, ff in (("RMSE", "{:.3f}"), ("Score", "{:.1f}")):
                L.append(f"| {V4_SHORT[k]} | {q} | " + " | ".join(ff.format(v) for v in vec(k, sub, q)) + " |")
        for x in comps_keys:
            cb = comps[x]["per_subset"][sub]
            L.append(f"| Δ D4 − {V4_SHORT[x]}" + (f" ({V4_SHORT[cb['vs']]})" if x == "best_baseline" else "") +
                     " | RMSE | " + " | ".join(f"{v:+.3f}" for v in cb["dRMSE"]["per_seed_diff"]) + " |")
        L.append("")
    L += ["## Supplement: percentile seed bootstrap (n = 5, under-covers; supplementary only, paper_plan D29)", "",
          "Exact enumeration of the 3,125 ordered resamples of the 5 paired differences (per subset); pooled: "
          "subset-stratified, 20,000 draws, rng seed 0. Not a basis for any claim. The seed-bootstrap interval is "
          f"narrower than the paired t CI in {narrower} of {cells} per-subset cells (RMSE and Score).", "",
          "| D4 − | FD001 | FD002 | FD003 | FD004 | Pooled (stratified) |", "|---|---|---|---|---|---|"]
    for x in comps_keys:
        cs = [comps[x]["per_subset"][sub]["dRMSE"]["supplement_seed_boot_ci95"] for sub in SUBS]
        pc = comps[x]["pooled"]["dRMSE"]["supplement_stratified_seed_boot_ci95"]
        L.append(f"| {V4_SHORT[x]} (ΔRMSE) | " + " | ".join(f"[{f2(a)}, {f2(b)}]" for a, b in cs) +
                 f" | [{f2(pc[0])}, {f2(pc[1])}] |")
    L += ["", "## Checks (generator gates)", "",
          f"- {checks['pred_files_verified']} prediction files and {checks['metrics_files_verified']} metrics.json "
          "files re-verified against final_rows.json sha256.",
          f"- Metrics recomputed from the prediction files equal metrics.json: max abs diff (RMSE, MAE, R²) "
          f"{checks['recompute_max_abs']:.2e}.",
          f"- R0.15 gate: R² = 1 − RMSE²/Var(y) holds on every file, max abs diff {checks['r2_identity_max_abs']:.2e}.",
          "- Engine ids and capped labels are identical across all rows, seeds and subsets.",
          f"- test_eval_log.csv sha256 at generation {checks['test_eval_log_sha256_at_generation']} (this generator "
          "writes no log row and adds no test read)."]
    if flags:
        for k, sub, s, md in flags:
            L.append(f"- Batch-invariance flag: {k} {sub} seed {s} (max diff {md}).")
    else:
        L.append("- Test-time batch-invariance: no flagged read among the 140 files.")
    L += ["", "## Budget notes", "",
          "- Same budget = the same pre-registered trial rule (3 studies × 4 completed trials, RandomSampler seeds 0-2, "
          "same search-space rules, objective, recipe and OOM rule), not the same GPU-hours (Trials column).",
          "- D4: 12 own trials per subset plus the stage-1 screen of 4 designs (exp_B15).",
          "- LSTM (exp_V2): 16/48 tuning trials with validation RMSE > 30 (lr ≤ 1.2e-4); 24/48 best epoch at the cap; "
          "30/48 ran all 150 epochs; the selected configurations converged (D34).",
          "- Ridge (exp_V3): 12-trial choice; the validation-only 39-point grid is lower by 0.40 (FD001) and 0.16 (FD003) "
          "at the α = 1000 edge (D35); ridge tuning cost is CPU time recorded as gpu_h by the shared harness."]
    with open(os.path.join(out_dir, "combined_table_v4.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("written", os.path.join(out_dir, "combined_table_v4.md"))


FINAL = os.path.join(str(resolve_path("results/frozen")))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=["v3", "v4"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rows", default=os.path.join(FINAL, "final_rows.json"))
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    if a.version == "v3":
        legacy_v3_main(a.out_dir)
    else:
        if os.path.exists(os.path.join(a.out_dir, "combined_table_v4.md")):
            sys.exit("refused: combined_table_v4.md exists (single run)")
        generate_v4(a.rows, a.out_dir)


if __name__ == "__main__":
    main()
