"""R0.2 evidence: sample 0's attention output is broadcast to every sample in the batch.

Run from workspace/code_base (CPU only, ~1 min):
    python ../review/bug_evidence/r02_batch_dependence.py

Code under test (unmodified, imported from workspace/code_base):
  Encoder_Layers.py:547-552  TransformerBatchNormEncoderLayer.forward:
        src2 = self.self_attn(src, src, src, ...)[0]     # self_attn returns a (B, L, d) tensor
        src  = src + self.dropout1(src2)                  # (L, d) broadcast over all B samples
  Encoder_Layers.py:186-201  ProbSparseMultiheadAttention.forward returns one tensor (B, L, d),
                             not a tuple, so [0] selects batch row 0.
  improve_transformer.py:262-273  the temporal encoder runs on (B*C, N, d): row 0 = sample 0, sensor 0.
"""
import os
import sys

CODE_DIR = os.getcwd()
sys.path.insert(0, CODE_DIR)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import numpy as np
import torch
torch.manual_seed(0)
np.random.seed(0)
from Encoder_Layers import TransformerBatchNormEncoderLayer
import _fd001_pipeline as P

print('torch', torch.__version__, '| device cpu')

# ---------------------------------------------------------------- 1. broadcast inside one layer
print('\n[1] One TransformerBatchNormEncoderLayer (eval, dropout 0), input (B=4, L=7, d=64)')
layer = TransformerBatchNormEncoderLayer(d_model=64, nhead=8, dim_feedforward=128, dropout=0.0,
                                         activation='gelu').eval()
x = torch.randn(4, 7, 64)
grab = {}
h1 = layer.self_attn.register_forward_hook(lambda m, i, o: grab.__setitem__('attn_out', o))
h2 = layer.norm1.register_forward_pre_hook(lambda m, i: grab.__setitem__('norm1_in', i[0]))
with torch.no_grad():
    layer(x)
h1.remove(); h2.remove()
a = grab['attn_out']                                   # (B, L, d)
resid = grab['norm1_in'].permute(2, 0, 1)              # undo permute(1, 2, 0) -> (B, L, d) = src + src2
added = resid - x                                      # the term actually added to each sample
print('  self_attn output shape:', tuple(a.shape), '| shape after [0]:', tuple(a[0].shape))
for i in range(4):
    print(f'  sample {i}: |added term - attn_out[sample 0]| = {float((added[i] - a[0]).abs().max()):.1e}; '
          f'|added term - attn_out[sample {i}] (its own)| = {float((added[i] - a[i]).abs().max()):.3f}')
print('  Every sample receives sample 0\'s attention output; samples 1-3 never receive their own.')
with torch.no_grad():
    z0 = layer(x)
    x2 = x.clone()
    x2[0] += 5.0
    z1 = layer(x2)
print('  per-sample |output change| after perturbing sample 0 only:',
      (z1 - z0).abs().sum((-1, -2)).numpy().round(3).tolist())

# ---------------------------------------------------------------- 2. saved FD001 ITT ensemble on the test set
print('\n[2] Saved FD001 ITT ensemble (%s), 100 FD001 test engines, window 40' % P.ENSEMBLE_PATH)
X, y, feats = P.fd001_test_inputs(40)
models = P.load_ensemble(C=X.shape[1], L=40)
print('  inputs', X.shape, '| ensemble members', len(models), '| features', len(feats))
p64 = P.predict(models, X, 64)
p1 = P.predict(models, X, 1)
p100 = P.predict(models, X, 100)
rng = np.random.default_rng(0)
order = rng.permutation(len(X))
p64_shuf = P.predict(models, X, 64, order=order)
print('  RMSE batch 64 (as in the code, DataLoader batch_size=64, shuffle=False): %.3f' % P.rmse(p64, y))
print('  RMSE batch 1  (each engine alone)                                     : %.3f' % P.rmse(p1, y))
print('  RMSE batch 100 (all engines in one batch)                              : %.3f' % P.rmse(p100, y))
print('  RMSE batch 64, engines in shuffled order (rng seed 0)                  : %.3f' % P.rmse(p64_shuf, y))
d = np.abs(p64 - p1)
print('  per-engine |pred(batch 64) - pred(batch 1)|: max %.2f, mean %.2f cycles; engines differing > 1 cycle: %d / %d'
      % (d.max(), d.mean(), int((d > 1).sum()), len(d)))
# first sample of each batch is its own "sample 0", so it matches batch-1 prediction
print('  engines that are first in their batch of 64 (rows 0 and 64): |diff| = %.2e, %.2e'
      % (d[0], d[64]))

# ---------------------------------------------------------------- 3. changing one engine changes the others
Xmod = X.copy()
Xmod[0] = X[0][:, ::-1]                                   # replace engine 1's window (time-reversed)
pmod = P.predict(models, Xmod, 64)
print('\n[3] Replace only test engine 1 (row 0) and re-predict at batch 64:')
print('  max |pred change| on rows 1..63 (same batch): %.2f cycles; rows 64..99 (other batch): %.2e'
      % (np.abs(pmod - p64)[1:64].max(), np.abs(pmod - p64)[64:].max()))

print('\nKEY RESULT: the [0] index adds sample 0\'s attention output to every sample; the saved FD001 ensemble '
      'gives test RMSE %.3f at batch 64 vs %.3f at batch 1 (and %.3f when the same engines are shuffled), and '
      'changing one engine moves the predictions of the other engines in its batch by up to %.2f cycles.'
      % (P.rmse(p64, y), P.rmse(p1, y), P.rmse(p64_shuf, y), np.abs(pmod - p64)[1:64].max()))
