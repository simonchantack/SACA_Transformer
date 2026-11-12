# Bug evidence for the response letter: R0.1, R0.2, R0.6

Each script is a short, CPU-only reproduction that imports the **submitted** code, unchanged. The submitted code is not part
of this release (it contains the test-selection lines removed under R0.4). Each `*_output.txt` is the output captured on
2026-10-20 (torch 2.8.0+cu128, CPU, the frozen venv2 Python). The scripts and outputs here are byte-identical to the reviewed
files in `workspace/review/bug_evidence/` (sha256 below).

## 1. What the scripts need (not released)

The scripts set `CODE_DIR = os.getcwd()` and import from it, so they must be started **from the folder that holds the
submitted code**. In this project that folder is `workspace/code_base/`; its files named below are byte-identical to the
originals in `inputs/code/` (same sha256).

| Needed file (relative to the submitted-code folder) | Used by | sha256 |
|---|---|---|
| `Encoder_Layers.py` | r01, r02 | 6ffb15fdd6f4e18790e14b4470e23726484eda21dc0dce5ea22e7904ddfe1592 |
| `attn.py` | r01 (and `Encoder_Layers.py`) | e22588f85062dfbb8752c803607d6fce5e8e06043c30c5b6ee03d7ccfd04bb74 |
| `utils/masking.py` | `attn.py` | c5f95590ed9e586e405ced3f9902ea5f1cac3ce94f22df5af02a23352beed84a |
| `improve_transformer.py` | r06 (`score_nasa`), `_fd001_pipeline.py` (`PatchTST_RUL_Model`) | a9a590a65aaa77b7f642cf425be8ee6ab1a865a533e699cbf4560780618695b3 |
| `CommonFunctions.py` | `_fd001_pipeline.py` | 075c0012347d761e5d6199af61f843e386889b795bf2e601db21f2365d07425c |
| `smooth_timeseries.py` | `_fd001_pipeline.py` | ec968c2b5aac310d6810c8acfef9292a700b9a4723379a73198d792f54a7396e |
| `BestModelsImproved/1 Improved Transformer/FD001/RMSE_11-948 Ensemble Lowest RMSE 11-7/Models/BEST_improved_FD001.pt` (saved FD001 ensemble, 3 members, d_model 64, 2 layers, window 40) | r02, r06 | 1f21dc028c1872e49cd6572d458deeeb60091819c5d99140d97fe814cf0962ae |
| `Data/train_FD001.txt` | r02, r06 | b26c80e34451ffeb0f80e878a37c078c733539dc43c40283ef3aedb389c113d4 |
| `Data/test_FD001.txt` | r02, r06 | df6fbba80b3ad86b89c911faa389faecab99a719f2f8a52a065e01077ab5ee76 |
| `Data/RUL_FD001.txt` | r02, r06 | 8110d1571bc63a3d071f1a16da4befafcb4274bce76f04167ee827cb60bf8d48 |

The three data hashes are copied from `workspace/data_splits.json` (`raw_file_sha256`); the files were not opened for this
README. The first seven rows (code files and checkpoint) were hashed on 2026-10-27 (byte hash only).

Released files of this folder:

| File | Concern | sha256 |
|---|---|---|
| `_fd001_pipeline.py` (shared helper) | R0.2, R0.6 | 44b466ec5605576ccb8bb3b9c347080d2cfb4ce02b84bb09cadf5d30b2f3f715 |
| `r01_attention_axis.py` | R0.1 | df6f9fb62d7ce37ad83934f77424bfba7b58ce794076c2906184df7b941d8d3a |
| `r01_attention_axis_output.txt` | R0.1 | a9e1648f2bfe2d9a11d430069ef5bbd51b0711dba55f8b99643044bc4435464c |
| `r02_batch_dependence.py` | R0.2 | ef14048328d3a2ff32fc8ed30953bdc32a432cc21f2904f53bd5e863e7297a3b |
| `r02_batch_dependence_output.txt` | R0.2 | 757c610032f0a8a11dcb6a7d241929381f8d8d0b50b21a52a7df8f9ee4b608f1 |
| `r06_nasa_score.py` | R0.6 | 36a5ce6576896e02d7484255a2c22c0559bf573d23605b577864a1d676b1791d |
| `r06_nasa_score_output.txt` | R0.6 | 0436921a74b04b8d80ccdc6c24fa15fa5233a07d7f6c743830ea5d7eba21260b |
| `rerun/r01_attention_axis_output_rerun_20261027.txt` | R0.1 | a9e1648f2bfe2d9a11d430069ef5bbd51b0711dba55f8b99643044bc4435464c |

## 2. How to run

From the submitted-code folder, with the GPU hidden:

```bash
cd <submitted-code folder>          # here: workspace/code_base
CUDA_VISIBLE_DEVICES= python <path to final>/bug_evidence/r01_attention_axis.py    > r01_attention_axis_output.txt
CUDA_VISIBLE_DEVICES= python <path to final>/bug_evidence/r02_batch_dependence.py  > r02_batch_dependence_output.txt
CUDA_VISIBLE_DEVICES= python <path to final>/bug_evidence/r06_nasa_score.py        > r06_nasa_score_output.txt
```

Runtime: about 2 s, 7 s, 7 s.

## 3. Re-run status after the experiment freeze (2026-10-27)

| Script | Re-run post-freeze? | Result |
|---|---|---|
| `r01_attention_axis.py` | Yes, CPU, 2026-10-27, from `workspace/code_base` with `-B` and the GPU hidden. It reads no data file. | Output byte-identical to the saved output (same sha256, `diff` empty). Kept in `rerun/`. |
| `r02_batch_dependence.py` | **No** | Not re-run. It reads `Data/test_FD001.txt` and `Data/RUL_FD001.txt` through the submitted pipeline and evaluates the saved ensemble on them. After the freeze a new test read is not allowed, even a descriptive one without a log row, unless Simon authorises it. The saved output of 2026-10-20 stands; its hash is above. |
| `r06_nasa_score.py` | **No** | Same reason: it scores the saved ensemble on the FD001 test engines. |

The saved r02 and r06 outputs were produced before the freeze, on the checkpoint and data files listed in section 1. No
claim in the release depends on a post-freeze run of them.

## 4. Key results (copied from the saved outputs)

| File | Concern | Key result |
|---|---|---|
| `r01_attention_axis.py` | R0.1 | Attention matrix is 8 x 8 (heads x heads) for 7, 20, and 440 tokens. Replacing every other token changes token 0's output by 3.0e-08 (round-off). With the axis corrected, the change is 0.66. |
| `r02_batch_dependence.py` | R0.2 | Saved FD001 ensemble: test RMSE 11.949 at batch 64, 12.297 at batch 1, 11.757 at batch 100, and 11.717 at batch 64 with the engines shuffled. |
| `r06_nasa_score.py` | R0.6 | A single late error of +10 scores 1.158 in the code and 1.718 by the standard formula. The saved FD001 ensemble scores 228.1 in the code (the paper's value) and 205.6 by the standard formula. |

### R0.1: attention runs over heads, not tokens

`ProbSparseMultiheadAttention.forward` (`Encoder_Layers.py:192-197`) reshapes q, k and v to `(B, nhead, L, d_k)`. But
`ProbAttention.forward` (`attn.py:70-76`) unpacks its input as `(B, L_Q, H, D)` and does the transpose itself, so it treats the
8 heads as the sequence and the L tokens as heads. The script shows: (1) the softmax is always over 8 items whatever the token
count; (2) replacing all tokens except token 0 changes token 0's output by at most 3.0e-08, running each token alone gives the
same output (max difference 1.8e-07), shuffling the other tokens changes no per-token output (8.9e-08); (3) with the layout
ProbAttention expects, the same weights give a `(B, 8, 7, 7)` token-by-token matrix and the change is 0.66; (4) in a full encoder
layer at batch size 1, perturbing token 3 changes only token 3's output.

### R0.2: sample 0's attention output is added to every sample in the batch

`TransformerBatchNormEncoderLayer.forward` (`Encoder_Layers.py:547-552`) takes `self.self_attn(...)[0]`. For
`ProbSparseMultiheadAttention`, which returns one `(B, L, d)` tensor, `[0]` is batch row 0, and `src + src2` adds it to every
sample. The saved FD001 ensemble on the 100 FD001 test engines gives test RMSE 11.949 (batch 64), 12.297 (batch 1), 11.757
(batch 100), 11.717 (batch 64, engines shuffled, rng seed 0). At batch 64 vs batch 1, 62 of 100 engines move by more than 1
cycle (maximum 14.4). Replacing only test engine 1 changes the other engines in its batch by up to 3.29 cycles. The released
code (`encoder.py`, `evaluation.evaluate_test`) removes the pattern and checks batch invariance.

### R0.6: NASA Score constants are swapped

The standard C-MAPSS Score is defined by Saxena et al., 2008 ("Damage Propagation Modeling for Aircraft Engine
Run-to-Failure Simulation", PHM 2008; IEEE DOI 10.1109/PHM.2008.4711414). With e = y_pred - y_true the early side (e < 0) uses
exp(-e/13) - 1 and the late side (e >= 0) uses exp(e/10) - 1. The submitted code swaps the constants
(`improve_transformer.py:704-713`; `MainSingleEng_FD001_F2_FD003_F4_Final.py:1199-1209`, copied verbatim by the script because
that file cannot be imported). Hand-computable cases: e = -10 gives 1.7183 (code) vs 1.1581 (standard); e = +10 gives 1.1581 vs
1.7183; e = +20 gives 3.6574 vs 6.3891; [-10, +20] gives 5.3757 vs 7.5472. The saved FD001 ensemble at batch 64 scores 228.1
in the code and 205.6 by the standard formula; at batch 1 the values are 236.1 and 222.1. The released `evaluation.py` uses
13 (early) and 10 (late).
