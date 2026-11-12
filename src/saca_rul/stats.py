"""Paired statistics for the revision (used by the table, post-hoc and ablation analyses).

- paired_seed_tests(a, b): per-seed paired t-test (two-sided), exact Wilcoxon signed-rank (two-sided),
  Hedges' g_av. With n = 5 the smallest two-sided exact Wilcoxon p is 0.0625 (stated in every output).
- pooled_wilcoxon(diffs): exact Wilcoxon over pooled pairs (e.g. 20 subset x seed).
- dm_test(e1, e2): Diebold-Mariano on squared-error differential d_i = e1_i^2 - e2_i^2 across test engines,
  h = 1, with the Harvey-Leybourne-Newbold (1997) small-sample correction; t distribution, n-1 df.
- bootstrap_delta(y, p1, p2): engine-level paired bootstrap 95% CI of dRMSE and dScore (10,000 resamples).
- holm(pvals): Holm step-down adjusted p-values.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as st

from . import evaluation as M


def hedges_g_av(a, b) -> Any:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    d = a - b
    sav = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    if sav == 0:
        return float("nan")
    J = 1 - 3 / (4 * (n - 1) - 1)          # small-sample correction, df = n-1 for paired designs
    return float(J * d.mean() / sav)


def paired_seed_tests(a, b) -> Any:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    out: dict[str, Any] = {"n": int(len(d)), "mean_diff": float(d.mean()),
                           "std_diff": float(d.std(ddof=1)) if len(d) > 1 else None}
    if len(d) >= 2 and d.std(ddof=1) > 0:
        t = st.ttest_rel(a, b)
        out.update(t_stat=float(t.statistic), t_p=float(t.pvalue), t_test="paired t-test, two-sided")
    else:
        out.update(t_stat=None, t_p=None, t_test="paired t-test not computable")
    if len(d) >= 1 and np.any(d != 0):
        w = st.wilcoxon(a, b, method="exact" if np.all(d != 0) else "auto", alternative="two-sided")
        out.update(wilcoxon_stat=float(w.statistic), wilcoxon_p=float(w.pvalue),
                   wilcoxon_test="Wilcoxon signed-rank, exact, two-sided",
                   wilcoxon_min_p_possible=float(2 * 0.5 ** len(d)))
    out["hedges_g_av"] = hedges_g_av(a, b) if len(d) >= 2 else None
    return out


def pooled_wilcoxon(diffs) -> Any:
    d = np.asarray(diffs, float)
    w = st.wilcoxon(d, method="exact" if np.all(d != 0) and len(d) <= 50 else "auto")
    return {"n": int(len(d)), "median_diff": float(np.median(d)), "mean_diff": float(d.mean()),
            "wilcoxon_stat": float(w.statistic), "wilcoxon_p": float(w.pvalue),
            "test": "Wilcoxon signed-rank on pooled paired differences, two-sided"}


def dm_test(e1, e2) -> Any:
    """e1, e2: per-engine errors of model 1 and model 2. Negative DM: model 1 has lower squared error."""
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dbar = d.mean()
    var = np.var(d, ddof=0) / n                     # h = 1: gamma_0 only
    if var == 0:
        return {"n": n, "dm": None, "p": None}
    dm = dbar / np.sqrt(var)
    h = 1
    k = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)   # HLN correction
    dm_hln = dm * k
    p = 2 * st.t.sf(abs(dm_hln), df=n - 1)
    return {"n": int(n), "mean_sq_err_diff": float(dbar), "dm_hln": float(dm_hln), "p": float(p),
            "test": "Diebold-Mariano, squared-error loss, h=1, Harvey-Leybourne-Newbold correction, t(n-1)"}


def bootstrap_delta(y, p1, p2, n_boot=10000, seed=0) -> Any:
    y, p1, p2 = (np.asarray(v, float) for v in (y, p1, p2))
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.integers(0, n, size=(n_boot, n))
    e1, e2 = p1 - y, p2 - y

    def score_rows(e):
        s = np.where(e < 0, np.expm1(-e / M.A_EARLY), np.expm1(e / M.A_LATE))
        return s

    s1, s2 = score_rows(e1), score_rows(e2)
    d_rmse = np.sqrt((e1[idx] ** 2).mean(1)) - np.sqrt((e2[idx] ** 2).mean(1))
    d_score = s1[idx].sum(1) - s2[idx].sum(1)
    return {"n_boot": n_boot, "seed": seed,
            "dRMSE": float(np.sqrt((e1 ** 2).mean()) - np.sqrt((e2 ** 2).mean())),
            "dRMSE_ci95": [float(np.percentile(d_rmse, 2.5)), float(np.percentile(d_rmse, 97.5))],
            "dScore": float(s1.sum() - s2.sum()),
            "dScore_ci95": [float(np.percentile(d_score, 2.5)), float(np.percentile(d_score, 97.5))],
            "method": "engine-level paired percentile bootstrap"}


def holm(pvals) -> Any:
    p = np.asarray([np.nan if v is None else v for v in pvals], float)
    ok = ~np.isnan(p)
    adj = np.full_like(p, np.nan)
    q = p[ok]
    order = np.argsort(q)
    m = len(q)
    res = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * q[i]))
        res[i] = run
    adj[ok] = res
    return [None if np.isnan(v) else float(v) for v in adj]


def t_interval(vals, level=0.95) -> Any:
    v = np.asarray(vals, float)
    n = len(v)
    if n < 2:
        return [None, None]
    h = st.t.ppf(0.5 + level / 2, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return [float(v.mean() - h), float(v.mean() + h)]
