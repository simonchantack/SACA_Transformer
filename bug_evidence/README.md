# Evidence for three errors in the originally submitted code

We found three errors in the originally submitted code:

1. Its attention ran over the attention heads instead of over the tokens.
2. The attention output of the first sample in a batch was added to every sample in that batch.
3. The two constants of the NASA Score were swapped.

This folder holds a short script for each error and the output that the script printed. Each script imports the submitted code unchanged. The submitted code is not part of this release. The outputs were captured on 2026-10-20 with torch 2.8.0+cu128, on the CPU. The released code fixes all three errors.

The file names start with `r01`, `r02` and `r06`. These are only file numbers; they have no other meaning. The sha256 checksum of every file in this folder is listed in `RELEASE_FILES.md` in the main folder.

## What the scripts need (not released)

The scripts set `CODE_DIR = os.getcwd()` and import from it, so you must start them from the folder that holds the originally submitted code. That code and the saved checkpoint are not included here. They are available from the corresponding author on request. You also need the NASA C-MAPSS files for the FD001 subset (see the main README, section "Data").

| Needed file (relative to the folder with the submitted code) | Used by |
|---|---|
| `Encoder_Layers.py` | r01, r02 |
| `attn.py` (also imported by `Encoder_Layers.py`) | r01 |
| `utils/masking.py` (imported by `attn.py`) | r01 |
| `improve_transformer.py` (the function `score_nasa`, and `PatchTST_RUL_Model` for `_fd001_pipeline.py`) | r06 |
| `CommonFunctions.py` | `_fd001_pipeline.py` |
| `smooth_timeseries.py` | `_fd001_pipeline.py` |
| `BestModelsImproved/1 Improved Transformer/FD001/RMSE_11-948 Ensemble Lowest RMSE 11-7/Models/BEST_improved_FD001.pt` (the saved FD001 ensemble: 3 members, d_model 64, 2 layers, window 40) | r02, r06 |
| `Data/train_FD001.txt` | r02, r06 |
| `Data/test_FD001.txt` | r02, r06 |
| `Data/RUL_FD001.txt` | r02, r06 |

## Files in this folder

| File | What it is |
|---|---|
| `_fd001_pipeline.py` | helper shared by r02 and r06 (loads the FD001 data and the saved ensemble) |
| `r01_attention_axis.py` | script for error 1 |
| `r01_attention_axis_output.txt` | its saved output |
| `r02_batch_dependence.py` | script for error 2 |
| `r02_batch_dependence_output.txt` | its saved output |
| `r06_nasa_score.py` | script for error 3 |
| `r06_nasa_score_output.txt` | its saved output |
| `rerun/r01_attention_axis_output_rerun_20261027.txt` | output of a later re-run of r01 (identical to the saved output) |
| `rerun/r01_stderr.txt` | what the re-run wrote to the error stream (an empty file: nothing) |

## How to run

Start from the folder with the submitted code, with the GPU hidden:

```bash
cd <folder with the submitted code>
CUDA_VISIBLE_DEVICES= python <path to this release>/bug_evidence/r01_attention_axis.py    > r01_attention_axis_output.txt
CUDA_VISIBLE_DEVICES= python <path to this release>/bug_evidence/r02_batch_dependence.py  > r02_batch_dependence_output.txt
CUDA_VISIBLE_DEVICES= python <path to this release>/bug_evidence/r06_nasa_score.py        > r06_nasa_score_output.txt
```

The runtime is about 2 s, 7 s and 7 s.

## Re-run status

- `r01_attention_axis.py` was re-run on the CPU on 2026-10-27 (from the folder with the submitted code, with the GPU hidden). It reads no data file. The new output is byte-identical to the saved output; it is kept in `rerun/`.
- `r02_batch_dependence.py` was not re-run after the results were finalised. It reads `test_FD001.txt` and `RUL_FD001.txt` and evaluates the saved ensemble on them, and a new read of the test set is not allowed after the results were finalised. The saved output of 2026-10-20 stands.
- `r06_nasa_score.py` was not re-run, for the same reason: it scores the saved ensemble on the FD001 test engines. The saved output of 2026-10-20 stands.

No claim in this release depends on a new run of r02 or r06.

## Key results (copied from the saved outputs)

| Script | Key result |
|---|---|
| `r01_attention_axis.py` | The attention matrix is 8 x 8 (heads x heads) for 7, 20, and 440 tokens. Replacing every other token changes token 0's output by 3.0e-08 (round-off). With the axis corrected, the change is 0.66. |
| `r02_batch_dependence.py` | Saved FD001 ensemble: test RMSE 11.949 at batch 64, 12.297 at batch 1, 11.757 at batch 100, and 11.717 at batch 64 with the engines shuffled. |
| `r06_nasa_score.py` | A single late error of +10 scores 1.158 in the code and 1.718 by the standard formula. The saved FD001 ensemble scores 228.1 in the code (the value reported in the originally submitted paper) and 205.6 by the standard formula. |

### Error 1: attention runs over heads, not tokens

`ProbSparseMultiheadAttention.forward` (`Encoder_Layers.py:192-197`) reshapes q, k and v to `(B, nhead, L, d_k)`. But `ProbAttention.forward` (`attn.py:70-76`) unpacks its input as `(B, L_Q, H, D)` and does the transpose itself, so it treats the 8 heads as the sequence and the L tokens as heads. The script shows:

1. The softmax is always over 8 items, whatever the token count.
2. Replacing all tokens except token 0 changes token 0's output by at most 3.0e-08. Running each token alone gives the same output (maximum difference 1.8e-07). Shuffling the other tokens changes no per-token output (8.9e-08).
3. With the layout that `ProbAttention` expects, the same weights give a `(B, 8, 7, 7)` token-by-token matrix, and the change is 0.66.
4. In a full encoder layer at batch size 1, perturbing token 3 changes only token 3's output.

### Error 2: the attention output of sample 0 is added to every sample in the batch

`TransformerBatchNormEncoderLayer.forward` (`Encoder_Layers.py:547-552`) takes `self.self_attn(...)[0]`. For `ProbSparseMultiheadAttention`, which returns one `(B, L, d)` tensor, `[0]` is batch row 0, and `src + src2` adds it to every sample. The saved FD001 ensemble on the 100 FD001 test engines gives test RMSE 11.949 (batch 64), 12.297 (batch 1), 11.757 (batch 100) and 11.717 (batch 64, engines shuffled, random seed 0). At batch 64 compared with batch 1, 62 of 100 engines move by more than 1 cycle (maximum 14.4). Replacing only test engine 1 changes the other engines in its batch by up to 3.29 cycles. The released code (`encoder.py` and `evaluation.evaluate_test`) removes this pattern and checks that predictions do not depend on the batch.

### Error 3: the NASA Score constants are swapped

The standard C-MAPSS Score is defined by Saxena et al., 2008 ("Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM 2008; IEEE DOI 10.1109/PHM.2008.4711414). With e = y_pred - y_true, the early side (e < 0) uses exp(-e/13) - 1 and the late side (e >= 0) uses exp(e/10) - 1. The submitted code swaps the constants (`improve_transformer.py:704-713` and `MainSingleEng_FD001_F2_FD003_F4_Final.py:1199-1209`; the script copies the second one verbatim because that file cannot be imported). Hand-computable cases:

| e | Submitted code | Standard formula |
|---|---|---|
| -10 | 1.7183 | 1.1581 |
| +10 | 1.1581 | 1.7183 |
| +20 | 3.6574 | 6.3891 |
| [-10, +20] (summed) | 5.3757 | 7.5472 |

The saved FD001 ensemble at batch 64 scores 228.1 in the code and 205.6 by the standard formula. At batch 1 the values are 236.1 and 222.1. The released `evaluation.py` uses 13 (early) and 10 (late).
