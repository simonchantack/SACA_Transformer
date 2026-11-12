"""Hierarchical (seeds x engines) paired bootstrap for the main comparisons (R2.10).

Port of ``round_04_pre/exp_C5/hier_bootstrap.py`` (sha256 93d9c0ff...): ``hier_boot`` and ``score_rows`` are the frozen
code. Design on test (crossed): every model and seed predicts the same official test engines. One bootstrap draw:
  1. resample seeds with replacement (same seed indices for both models: seed pairing);
  2. resample engines with replacement (same engine indices for every seed and both models: engine pairing);
  3. per resampled seed, RMSE (or Score sum) over the resampled engines; average over resampled seeds;
  4. delta = model A - model B.
Percentile 95% CI over B = 10,000 draws (rng seed 0, chunks of 500 draws to bound memory); the two-sided bootstrap
p = min(1, 2 min(P(delta* <= 0), P(delta* >= 0))) is descriptive and conservative (self-test null rejection 0.003), so a
non-rejection is never read as "no difference".
"""
from __future__ import annotations

from typing import Any

import numpy as np

A_EARLY, A_LATE = 13.0, 10.0


def score_rows(e: Any) -> np.ndarray:
    return np.where(e < 0, np.expm1(-e / A_EARLY), np.expm1(e / A_LATE))


def hier_boot(ea: Any, eb: Any, n_boot: int = 10000, seed: int = 0, chunk: int = 500) -> dict[str, Any]:
    """``ea``, ``eb``: (S seeds, E engines) errors pred - true on the SAME engines. Returns a dict for RMSE and Score."""
    ea, eb = np.asarray(ea, float), np.asarray(eb, float)
    S, E = ea.shape
    rng = np.random.default_rng(seed)
    sa, sb = score_rows(ea), score_rows(eb)
    d_r, d_s = [], []
    done = 0
    while done < n_boot:
        k = min(chunk, n_boot - done)
        si = rng.integers(0, S, size=(k, S))
        ei = rng.integers(0, E, size=(k, E))
        A = ea[si[:, :, None], ei[:, None, :]]
        B = eb[si[:, :, None], ei[:, None, :]]
        d_r.append(np.sqrt((A ** 2).mean(2)).mean(1) - np.sqrt((B ** 2).mean(2)).mean(1))
        As = sa[si[:, :, None], ei[:, None, :]]
        Bs = sb[si[:, :, None], ei[:, None, :]]
        d_s.append(As.sum(2).mean(1) - Bs.sum(2).mean(1))
        done += k
    out: dict[str, Any] = {}
    for name, d, obs in (("RMSE", np.concatenate(d_r), np.sqrt((ea ** 2).mean(1)).mean() - np.sqrt((eb ** 2).mean(1)).mean()),
                         ("Score", np.concatenate(d_s), sa.sum(1).mean() - sb.sum(1).mean())):
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        out[name] = {"delta": float(obs), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                     "boot_p_two_sided": float(p)}
    out["method"] = f"hierarchical paired bootstrap, seeds x engines (crossed), B={n_boot}, rng seed {seed}"
    return out


def selftest(reps: int = 300) -> dict[str, Any]:
    """Coverage under a null with a seed effect and an engine effect shared by both models (synthetic data only)."""
    rng = np.random.default_rng(123)
    S, E = 5, 100
    cover, rej = 0, 0
    for r in range(reps):
        eng = rng.normal(0, 10, size=E)
        seed_a = rng.normal(0, 1.0, size=(S, 1))
        seed_b = rng.normal(0, 1.0, size=(S, 1))
        ea = eng + seed_a + rng.normal(0, 6, size=(S, E))
        eb = eng + seed_b + rng.normal(0, 6, size=(S, E))
        res = hier_boot(ea, eb, n_boot=1000, seed=r)
        lo, hi = res["RMSE"]["ci95"]
        cover += lo <= 0 <= hi
        rej += res["RMSE"]["boot_p_two_sided"] < 0.05
    return {"null_reps": reps, "null_coverage_95": cover / reps, "null_rejection_rate_0.05": rej / reps,
            "setup": "S=5 seeds, E=100 engines, shared engine effect sd 10, seed effect sd 1, noise sd 6, B=1000"}
