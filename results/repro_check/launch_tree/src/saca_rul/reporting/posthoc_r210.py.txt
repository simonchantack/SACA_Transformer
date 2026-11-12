"""Port of experiments/final/posthoc_R2.10/run_hier_bootstrap.py (sha256 65ed8af3...). Changes: package imports, the
exp_C5 function comes from ``hier_bootstrap`` in this package, output folder is an argument, no test-log row.

R2.10 post-freeze analysis_posthoc (RULES 20, 23): hierarchical seeds x engines bootstrap on the final rows.

Applies hier_boot() of experiments/round_04_pre/exp_C5/hier_bootstrap.py UNCHANGED (sha256 93d9c0ff... checked before
import; frozen in freeze_manifest posthoc_cpu R2.10) to every D4 comparison of combined table v4:
D4 − θ*, PatchTST, iTransformer, Crossformer, LSTM, ridge, and D4 − validation-chosen best baseline (prereg 5.3).
Per subset: errors e = y_pred − y_true_capped from the saved test prediction files (5 seeds x all test engines; same
engines for both rows), B = 10,000 draws, rng seed 0 (the function's defaults). Output: delta, percentile 95% CI and
two-sided bootstrap p for RMSE and Score. Descriptive; the method is conservative (selftest null rejection 0.003), so a
non-rejection is never 'no difference' (paper_plan D29). Single run: refuses if its test_eval_log row exists.
Reads: final_rows.json (frozen sha) and the prediction files it names. No model, no test_FD/RUL_FD file.
"""
import json
import os

import numpy as np
import pandas as pd

from . import posthoc_common as PC

HERE = "."  # output folder, set by main()
from . import hier_bootstrap as HB

EXP_ID = "final_R2.10_hier_bootstrap"
C5 = HB.__file__
C5_SHA = "93d9c0ff0c774a56e792a17474582d613e57f9397fa59a8f33a3dd29967116c1"  # sha256 of the frozen exp_C5 source file
BEST = {"FD001": "crossformer", "FD002": "crossformer", "FD003": "crossformer", "FD004": "patchtst"}
COMPS = ["theta_star", "patchtst", "itransformer", "crossformer", "lstm", "ridge", "best_baseline"]
NAMES = {"theta_star": "θ*", "patchtst": "PatchTST", "itransformer": "iTransformer", "crossformer": "Crossformer",
         "lstm": "LSTM", "ridge": "Ridge", "best_baseline": "best baseline B_k"}


def _main():
    PC.refuse_if_logged(EXP_ID)
    rows, fr = PC.load_final_rows()
    files = {PC.rel(PC.FINAL_ROWS): PC.FINAL_ROWS_SHA, PC.rel(C5): PC.sha256(C5), PC.rel(__file__): PC.sha256(__file__)}
    errs = {}
    for k in rows:
        for sub in PC.SUBS:
            E, ref = [], None
            for s, p, h in PC.pred_files(rows, k, sub):
                files[PC.rel(p)] = h
                df = pd.read_csv(p)
                if ref is None:
                    ref = df
                assert np.array_equal(df.engine_id.to_numpy(), ref.engine_id.to_numpy())
                E.append(df.y_pred.to_numpy() - df.y_true_capped.to_numpy())
            errs[(k, sub)] = (np.array(E), ref)
    for sub in PC.SUBS:
        for k in rows:
            assert np.array_equal(errs[(k, sub)][1].engine_id.to_numpy(), errs[("d4", sub)][1].engine_id.to_numpy())
            assert np.allclose(errs[(k, sub)][1].y_true_capped.to_numpy(), errs[("d4", sub)][1].y_true_capped.to_numpy())
    inputs_sha = PC.write_inputs(os.path.join(HERE, "inputs_sha256.json"), files)
    n_eng = sum(len(errs[("d4", s)][1]) for s in PC.SUBS)
    logrow = PC.append_posthoc(EXP_ID, "FD001-FD004",
                               f"R2.10 hierarchical seeds x engines bootstrap (exp_C5 hier_boot 93d9c0ff, B 10000, rng 0) on "
                               f"the final rows of combined table v4; reads saved test prediction files only (inputs "
                               f"posthoc_R2.10/inputs_sha256.json {inputs_sha}); descriptive, no selection", n_eng, __file__)
    res = {"experiment_id": EXP_ID, "kind": "analysis_posthoc", "log_row": logrow, "inputs_sha256_file": inputs_sha,
           "method": "exp_C5 hier_boot: resample seeds (shared) and engines (shared across seeds and both rows); "
                     "delta = D4 − row; percentile 95% CI; two-sided bootstrap p (descriptive)",
           "best_baseline_by_validation": BEST, "per_subset": {}}
    for sub in PC.SUBS:
        res["per_subset"][sub] = {}
        ea = errs[("d4", sub)][0]
        for x in COMPS:
            o = BEST[sub] if x == "best_baseline" else x
            r = HB.hier_boot(ea, errs[(o, sub)][0])
            r["vs"] = o
            r["n_seeds"], r["n_engines"] = int(ea.shape[0]), int(ea.shape[1])
            res["per_subset"][sub][x] = r
    with open(os.path.join(HERE, "hier_bootstrap_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    L = ["# R2.10 hierarchical (seeds × engines) bootstrap on the final rows (analysis_posthoc, post-freeze)", "",
         f"Δ = D4 − row (negative: D4 better). exp_C5 `hier_boot` unchanged (sha256 {C5_SHA[:8]}…), B = 10,000, rng seed 0. "
         "Seeds and engines resampled jointly (crossed design). Percentile 95% CI; two-sided bootstrap p is descriptive. "
         "The method is conservative (selftest null rejection 0.003): a CI that includes 0 is not evidence of no "
         "difference (paper_plan D29).", "",
         "| Subset | D4 − | ΔRMSE | 95% CI | boot p | ΔScore | 95% CI | boot p |", "|---|---|---|---|---|---|---|---|"]
    for sub in PC.SUBS:
        for x in COMPS:
            r = res["per_subset"][sub][x]
            nm = NAMES[x] + (f" ({NAMES[r['vs']]})" if x == "best_baseline" else "")
            a, b = r["RMSE"], r["Score"]
            L.append(f"| {sub} | {nm} | {a['delta']:+.2f} | [{a['ci95'][0]:+.2f}, {a['ci95'][1]:+.2f}] | "
                     f"{a['boot_p_two_sided']:.4f} | {b['delta']:+.1f} | [{b['ci95'][0]:+.1f}, {b['ci95'][1]:+.1f}] | "
                     f"{b['boot_p_two_sided']:.4f} |")
    with open(os.path.join(HERE, "hier_bootstrap_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main(out_dir: str) -> None:
    """Regenerate the outputs into ``out_dir`` (inert test-log helpers; see posthoc_common)."""
    global HERE
    HERE = os.path.abspath(out_dir)
    os.makedirs(HERE, exist_ok=True)
    _main()


def cli(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True, help="folder for the regenerated outputs (never the frozen folder)")
    a = ap.parse_args(argv)
    main(a.out_dir)


if __name__ == "__main__":
    cli()
