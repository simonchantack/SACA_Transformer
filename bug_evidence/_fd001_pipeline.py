"""Shared helper for the bug-evidence scripts (eswa-reviewer, round 00).

Rebuilds the FD001 test inputs exactly as the submitted ITT script does, and loads the
saved FD001 ITT ensemble. Read-only use of workspace/code_base; nothing there is modified.

Replicated steps (MainSingleEng_FD001_F2_FD003_F4_Final_Improved.py):
  constant-feature drop (lines 156-168), correlation pruning |r| > 0.95 with RUL tie-break
  (lines 385-410), StandardScaler fit on the training file (line 540), RUL cap 125,
  Gaussian smoothing sigma=2 (lines 557-558), one test window per engine ending one cycle
  before the last observation (line 577; CommonFunctions.py:19-46).
Must be imported with the current working directory = workspace/code_base.
"""
import os
import sys

CODE_DIR = os.getcwd()
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from CommonFunctions import create_testing_sequences_sw
from smooth_timeseries import smooth_engine_data
from improve_transformer import PatchTST_RUL_Model

COLS = ['engine', 'time', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sm_{i}' for i in range(1, 22)]
ENSEMBLE_PATH = ('BestModelsImproved/1 Improved Transformer/FD001/'
                 'RMSE_11-948 Ensemble Lowest RMSE 11-7/Models/BEST_improved_FD001.pt')


def fd001_test_inputs(window=40):
    tr = pd.read_csv('Data/train_FD001.txt', sep=' ', header=None, names=COLS, index_col=False)
    te = pd.read_csv('Data/test_FD001.txt', sep=' ', header=None, names=COLS, index_col=False)
    yt = pd.read_csv('Data/RUL_FD001.txt', header=None, names=['RUL'])
    df, dft = tr.copy(), te.copy()
    const = [c for c in df.columns if abs(df[c].std() < 0.02)]
    df.drop(columns=const, inplace=True)
    dft.drop(columns=const, inplace=True)
    df['rul'] = df.groupby('engine')['time'].transform('max') - df['time']
    corr = df.corr()
    corr = corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool))
    high = [(c, r) for c in corr.columns for r in corr.index if abs(corr.loc[c, r]) > 0.95]
    drop = [p[1] if abs(corr.loc['rul', p[0]]) > abs(corr.loc['rul', p[1]]) else p[0] for p in high]
    df.drop(columns=drop, inplace=True)
    feats = list(df.columns[2:-1])
    sc = StandardScaler()
    sc.fit(df[feats])
    Xt = pd.DataFrame(np.c_[dft['engine'], sc.transform(dft[feats])], columns=['engine'] + feats)
    Xts = smooth_engine_data(Xt, feats, method='gaussian', sigma=2)
    Xtf = create_testing_sequences_sw(Xts, feats, window, 1)          # (N_engines, L, C)
    X = Xtf.transpose(0, 2, 1).astype(np.float32)                      # (N_engines, C, L)
    y = np.minimum(yt['RUL'].to_numpy(), 125).astype(np.float64)
    return X, y, feats


def build_model(sd, C, L):
    d_t = sd['temporal_encoder.patch_embed.proj.weight'].shape[0]
    n_layers = len({k.split('.')[3] for k in sd if k.startswith('temporal_encoder.encoder.layers.')})
    d_ff = sd['temporal_encoder.encoder.layers.0.linear1.weight'].shape[0]
    hh = sd['fusion_head.mlp.0.weight'].shape[0]
    m = PatchTST_RUL_Model(C=C, L=L, d_model_t=d_t, n_heads_t=8, n_layers_t=n_layers, d_ff_t=d_ff, dropout_t=0.1,
                           patch_len_t=10, stride_t=5, patch_len_c=3, stride_c=1, d_model_c=d_t, n_heads_c=8,
                           n_layers_c=n_layers, d_ff_c=d_ff, dropout_c=0.1, head_hidden=hh)
    m.load_state_dict(sd)
    m.eval()
    return m


def load_ensemble(C, L=40):
    obj = torch.load(ENSEMBLE_PATH, map_location='cpu', weights_only=False)
    sds = obj['ensemble'] if isinstance(obj, dict) and 'ensemble' in obj else [obj]
    return [build_model(sd, C, L) for sd in sds]


def predict(models, X, bs, order=None):
    """Ensemble-mean prediction in mini-batches of size bs. order: optional row permutation."""
    idx = np.arange(len(X)) if order is None else np.asarray(order)
    preds = []
    with torch.no_grad():
        for m in models:
            out = np.empty(len(X))
            for i in range(0, len(idx), bs):
                rows = idx[i:i + bs]
                out[rows] = m(torch.tensor(X[rows])).numpy().reshape(-1)
            preds.append(out)
    return np.mean(preds, axis=0)


def rmse(yp, y):
    return float(np.sqrt(np.mean((yp - y) ** 2)))
