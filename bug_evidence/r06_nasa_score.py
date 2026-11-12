"""R0.6 evidence: NASA Score constants are swapped in the code.

Run from workspace/code_base (CPU only, ~10 s):
    python ../review/bug_evidence/r06_nasa_score.py

Standard C-MAPSS Score (Saxena et al., 2008) and the paper's own equation
(access_ESWA_v2.tex:425-426), with e = y_pred - y_true:
    e <  0 (early, under-estimate RUL): exp(-e / 13) - 1
    e >= 0 (late,  over-estimate RUL) : exp( e / 10) - 1
Code under test (unmodified):
    improve_transformer.py:704-713          score_nasa: a1=10 for e<0, a2=13 for e>=0  (used by all ITT scenarios)
    MainSingleEng_FD001_F2_FD003_F4_Final.py:1199-1209  score: same swap (base model); a top-level
                                            script, so its function is copied verbatim below.
"""
import os
import sys
import math

CODE_DIR = os.getcwd()
sys.path.insert(0, CODE_DIR)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import numpy as np
import torch
torch.manual_seed(0)
from improve_transformer import score_nasa          # code version, imported unchanged
import _fd001_pipeline as P


def score_final_py(errors):                         # verbatim copy of Final.py:1199-1209
    a1 = 10
    a2 = 13
    s1 = 0
    s2 = 0
    for err in errors:
        if err < 0:
            s1 += (np.exp(-1 * (err / a1))) - 1
        if ((err > 0) or (err == 0)):
            s2 += (np.exp(err / a2)) - 1
    return [s1, s2]


def score_standard(errors):
    return float(sum(math.exp(-e / 13) - 1 if e < 0 else math.exp(e / 10) - 1 for e in errors))


print('[1] Hand-computable examples (e = y_pred - y_true, in cycles)')
print('    exp(10/13)-1 = %.4f   exp(10/10)-1 = %.4f   exp(20/13)-1 = %.4f   exp(20/10)-1 = %.4f'
      % (math.exp(10 / 13) - 1, math.exp(1) - 1, math.exp(20 / 13) - 1, math.exp(2) - 1))
print('    %-22s %14s %14s %14s %14s' % ('errors', 'code score_nasa', 'code Final.py', 'standard', 'code/standard'))
for e in ([-10.0], [10.0], [-10.0, 10.0], [20.0], [-20.0], [-10.0, 20.0], [-30.0, -5.0, 5.0, 30.0]):
    sc = score_nasa(np.array(e)); sf = sum(score_final_py(np.array(e))); ss = score_standard(e)
    print('    %-22s %14.4f %14.4f %14.4f %14.3f' % (e, sc, sf, ss, sc / ss))
print('    Per-element: the code charges an early error of -10 as 1.7183 (standard 1.1581) and a late error of')
print('    +10 as 1.1581 (standard 1.7183): late predictions are penalised less than early ones, the opposite of')
print('    the metric\'s intent. For a symmetric pair such as [-10, +10] the two swaps cancel in the sum, so the')
print('    totals agree only by coincidence; any asymmetric error set differs.')

print('\n[2] Saved FD001 ITT ensemble on the 100 FD001 test engines (batch 64, as in the code)')
X, y, _ = P.fd001_test_inputs(40)
models = P.load_ensemble(C=X.shape[1], L=40)
for bs in (64, 1):
    yp = P.predict(models, X, bs)
    e = yp - y
    sc = score_nasa(e); ss = score_standard(e)
    print('    batch %-3d RMSE %.3f | Score code %.1f | Score standard %.1f | early (e<0) %d, late (e>=0) %d engines'
          % (bs, P.rmse(yp, y), sc, ss, int((e < 0).sum()), int((e >= 0).sum())))
yp = P.predict(models, X, 64)
e = yp - y
print('\nKEY RESULT: code Score uses /10 for early and /13 for late (swapped); FD001 ITT ensemble Score is %.1f with '
      'the code vs %.1f with the standard formula (paper reports 228.1).' % (score_nasa(e), score_standard(e)))
