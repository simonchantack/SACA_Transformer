"""R0.1 evidence: the ProbSparse wrapper attends across heads, not across tokens.

Run from workspace/code_base (CPU only):
    python ../review/bug_evidence/r01_attention_axis.py

Code under test (unmodified, imported from workspace/code_base):
  Encoder_Layers.py:192-197  ProbSparseMultiheadAttention.forward builds q, k, v as
                             (B, nhead, L, d_k) and passes them to ProbAttention.
  attn.py:70-76              ProbAttention.forward unpacks its input as (B, L_Q, H, D)
                             and transposes to (B, H, L_Q, D) itself.
So ProbAttention's "sequence" axis is the 8 heads and its "head" axis is the L tokens.
"""
import os
import sys

CODE_DIR = os.getcwd()
sys.path.insert(0, CODE_DIR)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import torch
torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False)
from Encoder_Layers import ProbSparseMultiheadAttention, TransformerBatchNormEncoderLayer
import attn as attn_mod

print('torch', torch.__version__, '| device cpu')
B, L, D_MODEL, H = 2, 7, 64, 8          # L = 7 = temporal-encoder token count N_t at window 40

mha = ProbSparseMultiheadAttention(d_model=D_MODEL, nhead=H, dropout=0.0).eval()

# ---------------------------------------------------------------- 1. shape trace
captured = {}
orig_forward = attn_mod.ProbAttention.forward


def traced_forward(self, queries, keys, values, attn_mask=None):
    captured['input_shape'] = tuple(queries.shape)
    out, attns = orig_forward(self, queries, keys, values, attn_mask)
    captured['attn_shape'] = tuple(attns.shape) if attns is not None else None
    captured['attn'] = attns
    captured['out_shape'] = tuple(out.shape)
    return out, attns


attn_mod.ProbAttention.forward = traced_forward
print('\n[1] Shape trace (B=%d samples, L=%d tokens, d_model=%d, nhead=%d, d_k=%d)' % (B, L, D_MODEL, H, D_MODEL // H))
for L_try in (7, 20, 440):
    x = torch.randn(B, L_try, D_MODEL)
    with torch.no_grad():
        mha(x, x, x)
    bq, lq, hq, dq = captured['input_shape']
    print(f'  tokens L={L_try:3d}: tensor passed to ProbAttention = {captured["input_shape"]} '
          f'-> ProbAttention reads B={bq}, L_Q={lq} (sequence), H={hq} (heads), D={dq}; '
          f'attention matrix {captured["attn_shape"]} -> each softmax is over {captured["attn_shape"][-1]} items')
print('  Result: the softmax is always over 8 = nhead items, whatever the token count. '
      'The token count appears in the "heads" slot.')
a = captured['attn'][0, 0]
print('  Example: attention matrix of sample 0, "head slot" 0 (= token 0), rows/cols = the 8 heads; rows sum to',
      a.sum(-1).numpy().round(4).tolist())
attn_mod.ProbAttention.forward = orig_forward

# ---------------------------------------------------------------- 2. no token mixing
print('\n[2] Token-mixing tests on ProbSparseMultiheadAttention (eval, dropout 0)')
x = torch.randn(B, L, D_MODEL)
with torch.no_grad():
    y = mha(x, x, x)
    # 2a: replace every token except token 0 with fresh random values
    x_other = torch.randn(B, L, D_MODEL)
    x_other[:, 0] = x[:, 0]
    y_other = mha(x_other, x_other, x_other)
    # 2b: each token processed alone (sequence of length 1)
    y_alone = torch.cat([mha(x[:, j:j + 1], x[:, j:j + 1], x[:, j:j + 1]) for j in range(L)], dim=1)
    # 2c: shuffle the other tokens; output for token 0 compared
    perm = torch.cat([torch.tensor([0]), torch.randperm(L - 1) + 1])
    xp = x[:, perm]
    yp = mha(xp, xp, xp)
print('  2a max |change in token-0 output| after replacing all other tokens with noise: %.3e' % (y_other[:, 0] - y[:, 0]).abs().max())
print('  2b max |output(token in full sequence) - output(token alone)| over all tokens: %.3e' % (y - y_alone).abs().max())
print('  2c token order', perm.tolist(), '-> max |per-token output change| (after undoing the permutation): %.3e'
      % (yp - y[:, perm]).abs().max())

# ---------------------------------------------------------------- 3. reference: same weights, correct axis
print('\n[3] Reference: same q/k/v/out weights, ProbAttention given (B, L, H, D) as it expects')


def fixed_axis(m, x):
    b, l, _ = x.shape
    q = m.q_proj(x).view(b, l, m.nhead, m.d_k)          # (B, L, H, d_k) -- no transpose
    k = m.k_proj(x).view(b, l, m.nhead, m.d_k)
    v = m.v_proj(x).view(b, l, m.nhead, m.d_k)
    out, att = m.attn(q, k, v)                           # out (B, L, H, d_k)
    return m.out_proj(out.contiguous().view(b, l, m.d_model)), att


with torch.no_grad():
    f, att = fixed_axis(mha, x)
    f_other, _ = fixed_axis(mha, x_other)
print('  attention matrix shape', tuple(att.shape), '-> softmax over the', att.shape[-1], 'tokens')
print('  3a max |change in token-0 output| after replacing other tokens: %.3e' % (f_other[:, 0] - f[:, 0]).abs().max())
print('  (A length-1 check is not possible here: ProbAttention samples c*ceil(ln 1) = 0 keys for L = 1.)')

# ---------------------------------------------------------------- 4. full encoder layer, batch of 1
print('\n[4] Full TransformerBatchNormEncoderLayer (eval), batch size 1, perturb token 3 only')
layer = TransformerBatchNormEncoderLayer(d_model=D_MODEL, nhead=H, dim_feedforward=128, dropout=0.0,
                                         activation='gelu').eval()
x1 = torch.randn(1, L, D_MODEL)
with torch.no_grad():
    z0 = layer(x1)
    x2 = x1.clone()
    x2[0, 3] += 5.0
    z1 = layer(x2)
print('  per-token |output change|:', (z1 - z0).abs().sum(-1)[0].numpy().round(4).tolist())
print('  Only token 3 changes: the encoder layer has no token-to-token information flow.')

bug_change = float((y_other[:, 0] - y[:, 0]).abs().max())
ref_change = float((f_other[:, 0] - f[:, 0]).abs().max())
print('\nKEY RESULT: ProbAttention receives (B, nhead, L, d_k) but reads it as (B, L, H, D); its attention matrix '
      'is nhead x nhead (8 x 8) for every token count. Replacing all other tokens changes token 0 by %.1e '
      '(float32 round-off) in the submitted code vs %.2f with the axis corrected -> the submitted attention '
      'mixes heads, never tokens.' % (bug_change, ref_change))
