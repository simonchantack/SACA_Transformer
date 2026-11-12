# Vendored by eswa-experimenter-arch (exp_A4, ESWA revision) on 2026-10-20 from https://github.com/Thinklab-SJTU/Crossformer
# at commit c10c8eadb153d1dd9798250967747ca3ebb81383, file cross_models/cross_embed.py (Apache-2.0; licence in vendored/LICENSES/Crossformer_LICENSE).
# MODIFIED: import paths made package-relative / optional; einops calls replaced by equivalent torch reshape/permute/repeat; Full diff: vendored/VENDORED.md.

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len

        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        batch, ts_len, ts_dim = x.shape

        x_segment = x.reshape(batch, ts_len // self.seg_len, self.seg_len, ts_dim).permute(0, 3, 1, 2).reshape(-1, self.seg_len)  # = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len = self.seg_len)
        x_embed = self.linear(x_segment)
        x_embed = x_embed.reshape(batch, ts_dim, -1, x_embed.shape[-1])  # = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b = batch, d = ts_dim)
        
        return x_embed