# Combined table v4 (final rows; test set; 5 retrain seeds = split seeds 0-4; mean ± std, ddof 1)

Built by `experiments/final/code/make_table_v4.py --version v4` from `experiments/final/final_rows.json` (sha256 deaed352e5d4b3d0753121fb783842b4949fa564b6be519989083350d272c53c). Generated once after the 2026-10-26 experiment freeze (RULES 23) as a logged `analysis_posthoc`: it reads metrics.json files and the saved test prediction files named in final_rows.json (every sha256 re-verified); it evaluates no model, opens no test_FD/RUL_FD file and selects nothing. All test numbers come from the single test read of each configuration (experiments/test_eval_log.csv).

Definitions. Score: d = pred − true; exp(−d/13)−1 if d < 0, exp(d/10)−1 if d ≥ 0, summed over engines. **Δ = D4 − row (negative: D4 better)**, paired by split seed. Seed-level interval: paired t 95% CI (n = 5). Effect size: g_z = J·mean(Δ)/sd(Δ) with an exact noncentral-t 95% CI (CI on d_z). p(t, Holm): paired t, Holm over the 4 subsets of each comparison. p(W): exact Wilcoxon signed-rank (n = 5: floor 0.0625). p(DM): Diebold-Mariano-HLN on seed-averaged engine predictions. Engine boot CI: engine-level paired percentile bootstrap (10,000 resamples) = CI for ΔRMSE of seed-averaged predictions (not of the seed mean). Val RMSE: validation common support (label cycle ≥ 51). R² = 1 − RMSE²/Var(y_test).

Reading rules (paper_plan.md D29, D34, D35, D36; verdicts/round_03.md C3):
- **Preregistered primary: D4 − validation-chosen best baseline (Crossformer FD001-FD003, PatchTST FD004), pooled over 20 pairs: Inconclusive (exact two-sided Wilcoxon p = 0.0696).** Non-inferiority at margin 0.5: one-sided p = 0.00043. All other comparisons are secondary.
- No significance claim of D4 vs PatchTST (pooled p uncorrected; Holm columns below).
- The supplementary percentile seed bootstrap (n = 5) under-covers and never supports a superiority or significance claim. A non-rejection is never 'no difference' or 'equivalent'.
- LSTM: on FD004 the LSTM has lower test RMSE than D4 although D4 is better on validation; never rank D4 above the LSTM on FD004 or write 'all subsets' (D34). 16/48 LSTM tuning trials did not converge within 150 epochs; the row is a conservative estimate for LSTMs (exp_V2/README.md).
- Ridge: the α search range was binding at α = 1000 on FD001/FD003 (D35); its 5 seeds differ only through the split.
- Determinism (D36): D4 seed 0 and the seed-0 cells of θ*, PatchTST, iTransformer and Crossformer are certified bitwise in strict mode (exp_CERT); seeds 1-4 of those rows are round-2/3 runs (deterministic algorithms off); LSTM and ridge rows are strict deterministic (round 4). Mixed mode in every paired comparison.
- Disclosure (paper_plan D1): the round-3 designs were motivated after seeing round-2 results, including test-set summaries. D4 was chosen among 4 designs on validation, so its total search is larger than any other row's.
- Rows not in this table: ITT (θ* + submitted recipe, exp_B14): removed from the paper (RULES 21); θ0 checkpoint configuration (round_01/exp_B3): reference only, never a candidate (RULES 14); θ* fusion/branch rows: removed (RULES 22a); round-4/5 ablation arms (exp_V1, exp_A10-A13): reported in their own folders, not rows of this table.

## FD001

| Model | Selection | Trials (tune GPU-h) | Val RMSE | Test RMSE | Test MAE | Test R² | Test Score |
|---|---|---|---|---|---|---|---|
| D4 (main row; preregistered round 3; exp_B16) | validation (P2, 12 trials; design D4 chosen on validation in the preregistered stage-1 screen of 4 designs) | 12 (0.36) | 10.25 ± 0.95 | 13.61 ± 0.81 | 9.94 ± 0.43 | 0.884 ± 0.014 | 321.3 ± 76.4 |
| Base θ* (corrected original model; P2 tuned, exp_B6) | validation (P2; target 12 trials, see Trials) | 12 (1.06) | 12.71 ± 1.26 | 13.69 ± 0.76 | 10.00 ± 0.53 | 0.883 ± 0.013 | 310.3 ± 54.7 |
| PatchTST (P2 tuned, full budget: exp_B10_topup; FD003 exp_B10) | validation (P2; target 12 trials, see Trials) | 12 (0.47) | 11.25 ± 0.77 | 13.37 ± 0.27 | 10.32 ± 0.23 | 0.889 ± 0.004 | 318.6 ± 32.1 |
| iTransformer (P2 tuned, exp_B11) | validation (P2; target 12 trials, see Trials) | 12 (0.25) | 11.01 ± 0.66 | 13.60 ± 0.31 | 9.85 ± 0.18 | 0.885 ± 0.005 | 384.5 ± 33.2 |
| Crossformer (P2 tuned, exp_B12) | validation (P2; target 12 trials, see Trials) | 12 (0.25) | 10.35 ± 0.83 | 12.97 ± 0.57 | 9.28 ± 0.36 | 0.895 ± 0.009 | 385.0 ± 46.9 |
| LSTM (shared 12-trial rule and search ranges; exp_V2) | validation (same 12-trial rule; strict deterministic) | 12 (0.18) | 12.04 ± 0.77 | 14.16 ± 0.78 | 10.00 ± 0.39 | 0.875 ± 0.014 | 364.2 ± 56.3 |
| Ridge (12-trial rule; exp_V3) | validation (same 12-trial rule; closed form, float64) | 12 (0.00) | 16.53 ± 0.62 | 15.81 ± 0.17 | 12.99 ± 0.14 | 0.844 ± 0.003 | 381.8 ± 10.1 |

D4 minus row, FD001 (negative: D4 better):

| D4 − | ΔRMSE [95% t CI] | g_z [95% CI] | p(t, Holm 4) | p(W) | D4 better | p(DM) | Engine boot ΔRMSE CI | ΔScore [95% t CI] | g_z Score [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| θ* | -0.08 [-1.36, +1.20] | -0.06 [-0.95, +0.80] | 0.8672 | 1.0000 | 2/5 | 0.7334 | [-1.17, +1.65] | +11.0 [-98.5, +120.4] | +0.10 [-0.76, +1.00] |
| PatchTST | +0.24 [-0.79, +1.28] | +0.23 [-0.62, +1.17] | 1.0000 | 0.6250 | 2/5 | 0.6463 | [-1.34, +0.86] | +2.7 [-110.5, +115.9] | +0.02 [-0.85, +0.90] |
| iTransformer | +0.01 [-1.14, +1.17] | +0.01 [-0.86, +0.89] | 0.9787 | 1.0000 | 3/5 | 0.6443 | [-1.58, +1.05] | -63.2 [-176.0, +49.6] | -0.56 [-1.66, +0.33] |
| Crossformer | +0.64 [-0.24, +1.52] | +0.73 [-0.19, +1.94] | 0.2239 | 0.1875 | 1/5 | 0.6645 | [-0.91, +1.55] | -63.7 [-133.2, +5.8] | -0.91 [-2.26, +0.06] |
| LSTM | -0.55 [-1.33, +0.23] | -0.70 [-1.89, +0.21] | 0.1222 | 0.0625 | 5/5 | 0.4580 | [-1.78, +0.82] | -42.9 [-85.9, +0.1] | -0.99 [-2.41, +0.00] |
| Ridge | -2.20 [-3.25, -1.16] | -2.10 [-4.55, -0.66] | 0.0042 | 0.0625 | 5/5 | 0.0084 | [-4.92, -0.81] | -60.5 [-157.4, +36.3] | -0.62 [-1.76, +0.28] |
| best baseline B_k (Crossformer) | +0.64 [-0.24, +1.52] | +0.73 [-0.19, +1.94] | 0.1138 | 0.1875 | 1/5 | 0.6645 | [-0.91, +1.55] | -63.7 [-133.2, +5.8] | -0.91 [-2.26, +0.06] |

## FD002

| Model | Selection | Trials (tune GPU-h) | Val RMSE | Test RMSE | Test MAE | Test R² | Test Score |
|---|---|---|---|---|---|---|---|
| D4 (main row; preregistered round 3; exp_B16) | validation (P2, 12 trials; design D4 chosen on validation in the preregistered stage-1 screen of 4 designs) | 12 (0.63) | 12.98 ± 0.43 | 12.17 ± 0.06 | 8.86 ± 0.14 | 0.920 ± 0.001 | 614.4 ± 26.0 |
| Base θ* (corrected original model; P2 tuned, exp_B6) | validation (P2; target 12 trials, see Trials) | 12 (3.00) | 15.18 ± 1.45 | 14.40 ± 1.05 | 10.75 ± 0.96 | 0.887 ± 0.017 | 964.9 ± 181.5 |
| PatchTST (P2 tuned, full budget: exp_B10_topup; FD003 exp_B10) | validation (P2; target 12 trials, see Trials) | 12 (1.13) | 13.40 ± 0.41 | 12.79 ± 0.12 | 9.49 ± 0.12 | 0.911 ± 0.002 | 695.3 ± 22.9 |
| iTransformer (P2 tuned, exp_B11) | validation (P2; target 12 trials, see Trials) | 12 (0.47) | 13.87 ± 0.27 | 13.79 ± 0.36 | 9.78 ± 0.30 | 0.897 ± 0.005 | 878.8 ± 59.3 |
| Crossformer (P2 tuned, exp_B12) | validation (P2; target 12 trials, see Trials) | 12 (1.04) | 13.03 ± 0.49 | 12.58 ± 0.28 | 8.50 ± 0.26 | 0.914 ± 0.004 | 739.1 ± 64.9 |
| LSTM (shared 12-trial rule and search ranges; exp_V2) | validation (same 12-trial rule; strict deterministic) | 12 (0.33) | 14.61 ± 0.35 | 13.35 ± 0.36 | 9.73 ± 0.42 | 0.903 ± 0.005 | 876.9 ± 85.2 |
| Ridge (12-trial rule; exp_V3) | validation (same 12-trial rule; closed form, float64) | 12 (0.00) | 17.01 ± 0.33 | 15.55 ± 0.07 | 12.76 ± 0.02 | 0.869 ± 0.001 | 934.7 ± 13.4 |

D4 minus row, FD002 (negative: D4 better):

| D4 − | ΔRMSE [95% t CI] | g_z [95% CI] | p(t, Holm 4) | p(W) | D4 better | p(DM) | Engine boot ΔRMSE CI | ΔScore [95% t CI] | g_z Score [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| θ* | -2.22 [-3.47, -0.97] | -1.77 [-3.89, -0.48] | 0.0235 | 0.0625 | 5/5 | 0.0127 | [-2.12, -0.30] | -350.5 [-559.8, -141.2] | -1.66 [-3.69, -0.42] |
| PatchTST | -0.61 [-0.73, -0.50] | -5.51 [-11.56, -2.29] | 0.0004 | 0.0625 | 5/5 | 0.0280 | [-1.43, -0.11] | -80.9 [-95.8, -66.0] | -5.38 [-11.31, -2.24] |
| iTransformer | -1.62 [-2.01, -1.23] | -4.13 [-8.71, -1.66] | 0.0013 | 0.0625 | 5/5 | 0.0034 | [-2.28, -0.51] | -264.4 [-352.0, -176.8] | -3.00 [-6.38, -1.12] |
| Crossformer | -0.40 [-0.72, -0.09] | -1.26 [-2.92, -0.18] | 0.0967 | 0.0625 | 5/5 | 0.1923 | [-0.92, +0.17] | -124.7 [-186.6, -62.8] | -2.00 [-4.36, -0.61] |
| LSTM | -1.18 [-1.60, -0.76] | -2.78 [-5.94, -1.01] | 0.0044 | 0.0625 | 5/5 | 0.1290 | [-2.20, +0.26] | -262.5 [-367.9, -157.2] | -2.48 [-5.32, -0.86] |
| Ridge | -3.37 [-3.52, -3.23] | -22.67 [-47.31, -9.84] | 1.5e-06 | 0.0625 | 5/5 | 4.2e-11 | [-4.65, -2.59] | -320.3 [-354.5, -286.0] | -9.29 [-19.42, -3.98] |
| best baseline B_k (Crossformer) | -0.40 [-0.72, -0.09] | -1.26 [-2.92, -0.18] | 0.0967 | 0.0625 | 5/5 | 0.1923 | [-0.92, +0.17] | -124.7 [-186.6, -62.8] | -2.00 [-4.36, -0.61] |

## FD003

| Model | Selection | Trials (tune GPU-h) | Val RMSE | Test RMSE | Test MAE | Test R² | Test Score |
|---|---|---|---|---|---|---|---|
| D4 (main row; preregistered round 3; exp_B16) | validation (P2, 12 trials; design D4 chosen on validation in the preregistered stage-1 screen of 4 designs) | 12 (0.36) | 10.60 ± 0.64 | 11.55 ± 0.45 | 8.21 ± 0.39 | 0.913 ± 0.007 | 232.9 ± 30.5 |
| Base θ* (corrected original model; P2 tuned, exp_B6) | validation (P2; target 12 trials, see Trials) | 12 (1.00) | 11.85 ± 1.40 | 13.63 ± 0.47 | 9.52 ± 0.32 | 0.879 ± 0.008 | 405.9 ± 80.9 |
| PatchTST (P2 tuned, full budget: exp_B10_topup; FD003 exp_B10) | validation (P2; target 12 trials, see Trials) | 12 (0.58) | 10.76 ± 0.75 | 11.60 ± 0.13 | 8.64 ± 0.17 | 0.912 ± 0.002 | 224.1 ± 16.8 |
| iTransformer (P2 tuned, exp_B11) | validation (P2; target 12 trials, see Trials) | 12 (0.34) | 11.19 ± 0.60 | 12.42 ± 0.49 | 9.08 ± 0.23 | 0.899 ± 0.008 | 287.9 ± 64.3 |
| Crossformer (P2 tuned, exp_B12) | validation (P2; target 12 trials, see Trials) | 12 (0.44) | 10.75 ± 0.60 | 12.21 ± 0.19 | 8.77 ± 0.40 | 0.903 ± 0.003 | 264.8 ± 42.0 |
| LSTM (shared 12-trial rule and search ranges; exp_V2) | validation (same 12-trial rule; strict deterministic) | 12 (0.20) | 13.07 ± 0.91 | 15.23 ± 0.57 | 10.17 ± 0.39 | 0.849 ± 0.012 | 1002.2 ± 114.4 |
| Ridge (12-trial rule; exp_V3) | validation (same 12-trial rule; closed form, float64) | 12 (0.00) | 15.12 ± 0.95 | 15.99 ± 0.18 | 12.74 ± 0.14 | 0.833 ± 0.004 | 507.3 ± 28.0 |

D4 minus row, FD003 (negative: D4 better):

| D4 − | ΔRMSE [95% t CI] | g_z [95% CI] | p(t, Holm 4) | p(W) | D4 better | p(DM) | Engine boot ΔRMSE CI | ΔScore [95% t CI] | g_z Score [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| θ* | -2.08 [-3.13, -1.04] | -1.98 [-4.32, -0.59] | 0.0209 | 0.0625 | 5/5 | 0.0492 | [-2.26, -0.09] | -173.1 [-292.9, -53.2] | -1.43 [-3.25, -0.28] |
| PatchTST | -0.05 [-0.57, +0.47] | -0.09 [-0.99, +0.77] | 1.0000 | 0.8125 | 2/5 | 0.3918 | [-1.18, +0.46] | +8.8 [-28.4, +45.9] | +0.23 [-0.62, +1.17] |
| iTransformer | -0.87 [-1.92, +0.19] | -0.81 [-2.09, +0.13] | 0.1702 | 0.1250 | 4/5 | 0.1291 | [-1.90, +0.21] | -55.0 [-163.9, +53.9] | -0.50 [-1.57, +0.37] |
| Crossformer | -0.66 [-1.26, -0.06] | -1.09 [-2.59, -0.07] | 0.1138 | 0.0625 | 5/5 | 0.2872 | [-1.23, +0.35] | -31.9 [-94.8, +30.9] | -0.50 [-1.57, +0.37] |
| LSTM | -3.68 [-4.53, -2.83] | -4.32 [-9.09, -1.75] | 0.0011 | 0.0625 | 5/5 | 0.0441 | [-5.57, -0.75] | -769.4 [-884.3, -654.4] | -6.65 [-13.93, -2.81] |
| Ridge | -4.44 [-5.05, -3.83] | -7.20 [-15.07, -3.05] | 7.2e-05 | 0.0625 | 5/5 | 8.8e-05 | [-6.90, -2.73] | -274.5 [-335.0, -214.0] | -4.50 [-9.48, -1.83] |
| best baseline B_k (Crossformer) | -0.66 [-1.26, -0.06] | -1.09 [-2.59, -0.07] | 0.1138 | 0.0625 | 5/5 | 0.2872 | [-1.23, +0.35] | -31.9 [-94.8, +30.9] | -0.50 [-1.57, +0.37] |

## FD004

| Model | Selection | Trials (tune GPU-h) | Val RMSE | Test RMSE | Test MAE | Test R² | Test Score |
|---|---|---|---|---|---|---|---|
| D4 (main row; preregistered round 3; exp_B16) | validation (P2, 12 trials; design D4 chosen on validation in the preregistered stage-1 screen of 4 designs) | 12 (0.69) | 11.86 ± 1.44 | 13.80 ± 0.31 | 9.64 ± 0.21 | 0.897 ± 0.005 | 948.9 ± 100.1 |
| Base θ* (corrected original model; P2 tuned, exp_B6) | validation (P2; target 12 trials, see Trials) | 11 (2.96) | 13.20 ± 1.14 | 14.39 ± 0.57 | 10.68 ± 0.62 | 0.888 ± 0.009 | 961.4 ± 117.1 |
| PatchTST (P2 tuned, full budget: exp_B10_topup; FD003 exp_B10) | validation (P2; target 12 trials, see Trials) | 12 (0.98) | 12.48 ± 1.29 | 14.18 ± 0.06 | 10.15 ± 0.11 | 0.891 ± 0.001 | 1018.7 ± 42.1 |
| iTransformer (P2 tuned, exp_B11) | validation (P2; target 12 trials, see Trials) | 12 (0.58) | 12.65 ± 1.48 | 14.28 ± 0.22 | 9.83 ± 0.22 | 0.890 ± 0.003 | 1105.8 ± 34.4 |
| Crossformer (P2 tuned, exp_B12) | validation (P2; target 12 trials, see Trials) | 12 (1.03) | 12.54 ± 1.50 | 13.53 ± 0.27 | 9.18 ± 0.19 | 0.901 ± 0.004 | 1020.2 ± 160.6 |
| LSTM (shared 12-trial rule and search ranges; exp_V2) | validation (same 12-trial rule; strict deterministic) | 12 (0.37) | 13.23 ± 1.20 | 13.36 ± 0.40 | 9.57 ± 0.40 | 0.903 ± 0.006 | 823.0 ± 96.5 |
| Ridge (12-trial rule; exp_V3) | validation (same 12-trial rule; closed form, float64) | 12 (0.00) | 17.66 ± 1.01 | 17.52 ± 0.11 | 13.90 ± 0.12 | 0.834 ± 0.002 | 1316.9 ± 37.2 |

D4 minus row, FD004 (negative: D4 better):

| D4 − | ΔRMSE [95% t CI] | g_z [95% CI] | p(t, Holm 4) | p(W) | D4 better | p(DM) | Engine boot ΔRMSE CI | ΔScore [95% t CI] | g_z Score [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| θ* | -0.60 [-1.40, +0.20] | -0.74 [-1.96, +0.18] | 0.2150 | 0.1875 | 4/5 | 0.7908 | [-0.73, +0.56] | -12.5 [-190.7, +165.6] | -0.07 [-0.96, +0.80] |
| PatchTST | -0.38 [-0.76, +0.01] | -0.98 [-2.39, +0.01] | 0.1561 | 0.1250 | 4/5 | 0.1351 | [-1.35, +0.14] | -69.8 [-171.8, +32.3] | -0.68 [-1.86, +0.23] |
| iTransformer | -0.48 [-0.94, -0.02] | -1.05 [-2.51, -0.04] | 0.1288 | 0.0625 | 5/5 | 0.1713 | [-1.22, +0.24] | -156.9 [-262.2, -51.6] | -1.48 [-3.33, -0.31] |
| Crossformer | +0.27 [-0.40, +0.93] | +0.40 [-0.47, +1.41] | 0.3301 | 0.4375 | 1/5 | 0.1147 | [-0.08, +0.90] | -71.3 [-374.2, +231.6] | -0.23 [-1.17, +0.62] |
| LSTM | +0.44 [-0.00, +0.89] | +0.99 [-0.00, +2.40] | 0.1014 | 0.1250 | 1/5 | 0.1258 | [-0.18, +1.52] | +125.9 [-60.2, +312.1] | +0.67 [-0.24, +1.85] |
| Ridge | -3.72 [-4.15, -3.29] | -8.60 [-17.99, -3.67] | 5.3e-05 | 0.0625 | 5/5 | 2.5e-06 | [-5.65, -2.33] | -368.1 [-495.4, -240.7] | -2.87 [-6.12, -1.06] |
| best baseline B_k (PatchTST) | -0.38 [-0.76, +0.01] | -0.98 [-2.39, +0.01] | 0.1138 | 0.1250 | 4/5 | 0.1351 | [-1.35, +0.14] | -69.8 [-171.8, +32.3] | -0.68 [-1.86, +0.23] |

## Pooled over subsets (20 seed pairs; D4 − row)

Exact two-sided Wilcoxon on the 20 (subset, seed) differences. Holm-5 = family of v3 (θ*, best baseline, PatchTST, iTransformer, Crossformer; reviewer's check, verdicts/round_03.md). Holm-7 = all 7 D4 comparisons of this table. Neither Holm family is preregistered. Pooled g_z is descriptive (the 20 pairs come from 4 subsets with different error scales).

| D4 − | Metric | n | mean Δ | median Δ | D4 better | p (exact W) | Holm-5 | Holm-7 | g_z [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| θ* | RMSE | 20 | -1.25 | -1.32 | 16/20 | 0.0004 | 0.0020 | 0.0024 | -0.95 [-1.52, -0.45] |
| θ* | Score | 20 | -131.3 | -99.3 | 16/20 | 0.0042 | 0.0110 | 0.0146 | -0.66 [-1.17, -0.19] |
| PatchTST | RMSE | 20 | -0.20 | -0.28 | 13/20 | 0.0484 | 0.1453 | 0.1453 | -0.34 [-0.80, +0.10] |
| PatchTST | Score | 20 | -34.8 | -38.7 | 13/20 | 0.0441 | 0.0441 | 0.0441 | -0.47 [-0.94, -0.02] |
| iTransformer | RMSE | 20 | -0.74 | -0.83 | 17/20 | 0.0014 | 0.0057 | 0.0072 | -0.81 [-1.35, -0.33] |
| iTransformer | Score | 20 | -134.9 | -134.3 | 18/20 | 8.2e-05 | 0.0004 | 0.0005 | -1.11 [-1.72, -0.58] |
| Crossformer | RMSE | 20 | -0.04 | -0.15 | 12/20 | 0.5459 | 0.5459 | 0.5459 | -0.05 [-0.49, +0.39] |
| Crossformer | Score | 20 | -72.9 | -74.2 | 16/20 | 0.0037 | 0.0110 | 0.0146 | -0.56 [-1.06, -0.10] |
| LSTM | RMSE | 20 | -1.24 | -0.85 | 16/20 | 0.0037 | — | 0.0146 | -0.73 [-1.25, -0.25] |
| LSTM | Score | 20 | -237.2 | -114.8 | 16/20 | 0.0083 | — | 0.0166 | -0.64 [-1.14, -0.17] |
| Ridge | RMSE | 20 | -3.44 | -3.49 | 20/20 | 1.9e-06 | — | 1.3e-05 | -3.44 [-4.79, -2.37] |
| Ridge | Score | 20 | -255.8 | -282.6 | 19/20 | 5.7e-06 | — | 4.0e-05 | -1.80 [-2.60, -1.13] |
| best baseline B_k (**preregistered primary**) | RMSE | 20 | -0.20 | -0.33 | 15/20 | 0.0696 | 0.1453 | 0.1453 | -0.29 [-0.74, +0.15] |
| best baseline B_k | Score | 20 | -72.5 | -86.4 | 16/20 | 0.0003 | 0.0013 | 0.0016 | -1.06 [-1.65, -0.54] |

- Preregistered primary recomputed here equals exp_B16/primary_test.json: True (p 0.0696, mean Δ -0.200).

## Seed-level values (test RMSE and Score per split seed)

### FD001

| Model | Metric | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|
| D4 | RMSE | 12.926 | 13.430 | 12.999 | 13.781 | 14.913 |
| D4 | Score | 248.7 | 276.3 | 296.5 | 340.9 | 444.0 |
| θ* | RMSE | 14.081 | 13.148 | 12.899 | 14.781 | 13.552 |
| θ* | Score | 340.8 | 261.1 | 258.2 | 386.9 | 304.6 |
| PatchTST | RMSE | 13.477 | 13.382 | 13.003 | 13.721 | 13.251 |
| PatchTST | Score | 354.5 | 312.2 | 277.7 | 347.1 | 301.2 |
| iTransformer | RMSE | 13.999 | 13.252 | 13.420 | 13.855 | 13.465 |
| iTransformer | Score | 416.0 | 335.5 | 406.4 | 398.5 | 366.0 |
| Crossformer | RMSE | 13.039 | 12.398 | 12.391 | 13.709 | 13.290 |
| Crossformer | Score | 373.9 | 338.5 | 345.3 | 443.4 | 423.9 |
| LSTM | RMSE | 13.306 | 13.767 | 14.647 | 13.819 | 15.261 |
| LSTM | Score | 319.4 | 324.2 | 377.4 | 343.6 | 456.3 |
| Ridge | RMSE | 16.013 | 15.870 | 15.554 | 15.865 | 15.757 |
| Ridge | Score | 393.6 | 384.4 | 365.8 | 380.8 | 384.3 |
| Δ D4 − θ* | RMSE | -1.155 | +0.282 | +0.100 | -1.000 | +1.362 |
| Δ D4 − PatchTST | RMSE | -0.550 | +0.048 | -0.004 | +0.060 | +1.662 |
| Δ D4 − iTransformer | RMSE | -1.073 | +0.178 | -0.421 | -0.074 | +1.449 |
| Δ D4 − Crossformer | RMSE | -0.113 | +1.032 | +0.607 | +0.072 | +1.623 |
| Δ D4 − LSTM | RMSE | -0.380 | -0.337 | -1.649 | -0.038 | -0.348 |
| Δ D4 − Ridge | RMSE | -3.087 | -2.440 | -2.555 | -2.084 | -0.843 |
| Δ D4 − best baseline B_k (Crossformer) | RMSE | -0.113 | +1.032 | +0.607 | +0.072 | +1.623 |

### FD002

| Model | Metric | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|
| D4 | RMSE | 12.077 | 12.152 | 12.205 | 12.236 | 12.199 |
| D4 | Score | 611.1 | 611.1 | 577.1 | 649.1 | 623.5 |
| θ* | RMSE | 13.747 | 13.688 | 14.526 | 16.174 | 13.850 |
| θ* | Score | 919.8 | 855.1 | 953.7 | 1275.7 | 820.1 |
| PatchTST | RMSE | 12.726 | 12.635 | 12.880 | 12.936 | 12.764 |
| PatchTST | Score | 705.1 | 675.9 | 668.5 | 724.6 | 702.2 |
| iTransformer | RMSE | 13.359 | 13.766 | 14.145 | 14.167 | 13.535 |
| iTransformer | Score | 786.5 | 918.0 | 936.2 | 895.2 | 858.1 |
| Crossformer | RMSE | 12.523 | 12.196 | 12.494 | 12.748 | 12.927 |
| Crossformer | Score | 762.6 | 674.6 | 667.4 | 777.4 | 813.6 |
| LSTM | RMSE | 12.991 | 13.593 | 12.987 | 13.418 | 13.782 |
| LSTM | Score | 906.1 | 902.6 | 786.6 | 797.8 | 991.5 |
| Ridge | RMSE | 15.590 | 15.628 | 15.462 | 15.495 | 15.567 |
| Ridge | Score | 927.1 | 949.2 | 923.1 | 924.5 | 949.4 |
| Δ D4 − θ* | RMSE | -1.670 | -1.536 | -2.321 | -3.938 | -1.651 |
| Δ D4 − PatchTST | RMSE | -0.649 | -0.483 | -0.675 | -0.700 | -0.565 |
| Δ D4 − iTransformer | RMSE | -1.281 | -1.614 | -1.939 | -1.931 | -1.336 |
| Δ D4 − Crossformer | RMSE | -0.446 | -0.044 | -0.289 | -0.512 | -0.727 |
| Δ D4 − LSTM | RMSE | -0.913 | -1.441 | -0.782 | -1.182 | -1.583 |
| Δ D4 − Ridge | RMSE | -3.513 | -3.476 | -3.256 | -3.259 | -3.368 |
| Δ D4 − best baseline B_k (Crossformer) | RMSE | -0.446 | -0.044 | -0.289 | -0.512 | -0.727 |

### FD003

| Model | Metric | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|
| D4 | RMSE | 11.706 | 10.780 | 11.826 | 11.877 | 11.571 |
| D4 | Score | 246.9 | 181.6 | 231.7 | 243.2 | 260.9 |
| θ* | RMSE | 13.349 | 14.246 | 13.059 | 13.938 | 13.579 |
| θ* | Score | 322.4 | 495.8 | 338.3 | 388.0 | 485.0 |
| PatchTST | RMSE | 11.648 | 11.533 | 11.457 | 11.788 | 11.571 |
| PatchTST | Score | 240.5 | 223.1 | 197.3 | 224.1 | 235.6 |
| iTransformer | RMSE | 12.670 | 12.842 | 12.365 | 11.594 | 12.627 |
| iTransformer | Score | 297.5 | 384.7 | 243.7 | 217.0 | 296.6 |
| Crossformer | RMSE | 12.501 | 12.211 | 12.247 | 12.058 | 12.043 |
| Crossformer | Score | 335.3 | 266.6 | 238.6 | 228.8 | 254.5 |
| LSTM | RMSE | 14.661 | 14.936 | 15.204 | 15.169 | 16.182 |
| LSTM | Score | 984.5 | 857.9 | 940.4 | 1148.4 | 1079.9 |
| Ridge | RMSE | 15.694 | 16.062 | 16.086 | 16.149 | 15.979 |
| Ridge | Score | 476.0 | 513.5 | 548.3 | 512.0 | 486.9 |
| Δ D4 − θ* | RMSE | -1.643 | -3.466 | -1.232 | -2.061 | -2.008 |
| Δ D4 − PatchTST | RMSE | +0.058 | -0.753 | +0.370 | +0.089 | -0.000 |
| Δ D4 − iTransformer | RMSE | -0.964 | -2.061 | -0.538 | +0.283 | -1.056 |
| Δ D4 − Crossformer | RMSE | -0.795 | -1.430 | -0.420 | -0.181 | -0.472 |
| Δ D4 − LSTM | RMSE | -2.955 | -4.155 | -3.378 | -3.292 | -4.611 |
| Δ D4 − Ridge | RMSE | -3.988 | -5.282 | -4.260 | -4.272 | -4.408 |
| Δ D4 − best baseline B_k (Crossformer) | RMSE | -0.795 | -1.430 | -0.420 | -0.181 | -0.472 |

### FD004

| Model | Metric | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|
| D4 | RMSE | 13.426 | 14.249 | 13.945 | 13.740 | 13.626 |
| D4 | Score | 833.3 | 1077.4 | 1024.3 | 922.6 | 886.9 |
| θ* | RMSE | 13.711 | 13.950 | 14.627 | 15.153 | 14.532 |
| θ* | Score | 847.9 | 880.2 | 1031.7 | 1130.1 | 917.2 |
| PatchTST | RMSE | 14.228 | 14.249 | 14.132 | 14.111 | 14.163 |
| PatchTST | Score | 1009.2 | 1084.4 | 994.0 | 1031.1 | 974.7 |
| iTransformer | RMSE | 14.409 | 14.468 | 14.012 | 14.438 | 14.065 |
| iTransformer | Score | 1100.7 | 1165.7 | 1085.5 | 1081.3 | 1095.7 |
| Crossformer | RMSE | 13.871 | 13.187 | 13.674 | 13.583 | 13.343 |
| Crossformer | Score | 1273.2 | 837.4 | 1027.5 | 1017.7 | 945.1 |
| LSTM | RMSE | 12.725 | 13.478 | 13.562 | 13.252 | 13.759 |
| LSTM | Score | 711.9 | 760.7 | 803.7 | 886.5 | 952.0 |
| Ridge | RMSE | 17.412 | 17.478 | 17.452 | 17.597 | 17.670 |
| Ridge | Score | 1261.8 | 1308.1 | 1314.1 | 1341.5 | 1359.3 |
| Δ D4 − θ* | RMSE | -0.284 | +0.299 | -0.682 | -1.413 | -0.906 |
| Δ D4 − PatchTST | RMSE | -0.802 | +0.000 | -0.188 | -0.371 | -0.537 |
| Δ D4 − iTransformer | RMSE | -0.982 | -0.219 | -0.067 | -0.698 | -0.440 |
| Δ D4 − Crossformer | RMSE | -0.444 | +1.063 | +0.271 | +0.157 | +0.282 |
| Δ D4 − LSTM | RMSE | +0.701 | +0.771 | +0.383 | +0.488 | -0.133 |
| Δ D4 − Ridge | RMSE | -3.985 | -3.229 | -3.507 | -3.857 | -4.044 |
| Δ D4 − best baseline B_k (PatchTST) | RMSE | -0.802 | +0.000 | -0.188 | -0.371 | -0.537 |

## Supplement: percentile seed bootstrap (n = 5, under-covers; supplementary only, paper_plan D29)

Exact enumeration of the 3,125 ordered resamples of the 5 paired differences (per subset); pooled: subset-stratified, 20,000 draws, rng seed 0. Not a basis for any claim. The seed-bootstrap interval is narrower than the paired t CI in 56 of 56 per-subset cells (RMSE and Score).

| D4 − | FD001 | FD002 | FD003 | FD004 | Pooled (stratified) |
|---|---|---|---|---|---|
| θ* (ΔRMSE) | [-0.84, +0.68] | [-3.13, -1.61] | [-2.82, -1.54] | [-1.09, -0.06] | [-1.61, -0.91] |
| PatchTST (ΔRMSE) | [-0.31, +1.01] | [-0.68, -0.54] | [-0.43, +0.23] | [-0.63, -0.15] | [-0.38, +0.01] |
| iTransformer (ΔRMSE) | [-0.67, +0.82] | [-1.87, -1.37] | [-1.56, -0.21] | [-0.77, -0.20] | [-1.00, -0.47] |
| Crossformer (ΔRMSE) | [+0.11, +1.19] | [-0.60, -0.19] | [-1.05, -0.33] | [-0.16, +0.73] | [-0.24, +0.17] |
| LSTM (ΔRMSE) | [-1.13, -0.17] | [-1.45, -0.91] | [-4.26, -3.17] | [+0.14, +0.69] | [-1.46, -1.05] |
| Ridge (ΔRMSE) | [-2.77, -1.48] | [-3.47, -3.28] | [-4.88, -4.13] | [-3.98, -3.44] | [-3.63, -3.23] |
| best baseline B_k (ΔRMSE) | [+0.11, +1.19] | [-0.60, -0.19] | [-1.05, -0.33] | [-0.63, -0.15] | [-0.39, -0.02] |

## Checks (generator gates)

- 140 prediction files and 28 metrics.json files re-verified against final_rows.json sha256.
- Metrics recomputed from the prediction files equal metrics.json: max abs diff (RMSE, MAE, R²) 0.00e+00.
- R0.15 gate: R² = 1 − RMSE²/Var(y) holds on every file, max abs diff 1.11e-16.
- Engine ids and capped labels are identical across all rows, seeds and subsets.
- test_eval_log.csv sha256 at generation adfb5dcec327edf118854ada3e1e8120460ccf2c6ecb244236ed19ac909c1c15 (this generator writes no log row and adds no test read).
- Test-time batch-invariance: no flagged read among the 140 files.

## Budget notes

- Same budget = the same pre-registered trial rule (3 studies × 4 completed trials, RandomSampler seeds 0-2, same search-space rules, objective, recipe and OOM rule), not the same GPU-hours (Trials column).
- D4: 12 own trials per subset plus the stage-1 screen of 4 designs (exp_B15).
- LSTM (exp_V2): 16/48 tuning trials with validation RMSE > 30 (lr ≤ 1.2e-4); 24/48 best epoch at the cap; 30/48 ran all 150 epochs; the selected configurations converged (D34).
- Ridge (exp_V3): 12-trial choice; the validation-only 39-point grid is lower by 0.40 (FD001) and 0.16 (FD003) at the α = 1000 edge (D35); ridge tuning cost is CPU time recorded as gpu_h by the shared harness.
