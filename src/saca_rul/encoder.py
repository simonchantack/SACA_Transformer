"""Encoders: the post-norm batch-norm Transformer layer and the stacks built from it.

``TransformerBatchNormEncoderLayer``: self-attention, residual, BatchNorm (or LayerNorm for the LN arm), GELU
feed-forward, residual, norm. It differs from ``nn.TransformerEncoderLayer`` in the norm only.

Stacks
  * ``PatchTSTEncoder``  (theta*, D1, D2 temporal branch): instance norm, patch + sinusoidal PE, encoder, mean over
    sensors (channel independent, shared weights).
  * ``SensorChannelTransformerEncoder`` (theta*, D1): patches along the sensor axis, one token per (cycle, patch).
  * ``SensorTokenEncoder`` (D2, D3): one token per sensor with a sensor-identity embedding.
  * ``PerSensorTemporalEncoder`` (D3): per-sensor temporal summary, no mean over sensors.
  * ``SACAEncoder`` (D4): K coupled layers; per layer a time stage (anchor + patch tokens of each sensor) and a sensor
    stage (exact attention among the C anchors).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiheadSelfAttention
from .embeddings import (
    EndPaddedPatchEmbedding,
    PatchEmbedding,
    SinusoidalPositionEncoding,
    check_sensor_ids,
    make_sensor_embedding,
    window_stats,
)


class ChannelLayerNorm(nn.LayerNorm):
    """LayerNorm over dim 1 of an (S, d, B) tensor (the layout the layer passes to its norms).

    ``LN(Z)[s, :, b] = gamma * (Z[s, :, b] - mean_j Z[s, j, b]) / sqrt(var_j Z[s, j, b] + eps) + beta``, eps = 1e-5.
    State-dict keys stay ``weight`` / ``bias``; BatchNorm's running statistics are absent, so a BN checkpoint cannot be
    loaded strictly into an LN model (R1.20)."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__(d_model, eps=eps, elementwise_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != self.normalized_shape[0]:
            raise ValueError(f"ChannelLayerNorm expects (S, d={self.normalized_shape[0]}, B), got {tuple(x.shape)}")
        y = F.layer_norm(x.transpose(1, 2), self.normalized_shape, self.weight, self.bias, self.eps)
        return y.transpose(1, 2)


class TransformerBatchNormEncoderLayer(nn.Module):
    """Post-norm Transformer encoder layer with BatchNorm (default) or LayerNorm.

    ``attention`` is ``"prob"`` (ProbSparse, theta*) or ``"exact"`` (D1-D4). Input and output are (B, S, d)."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 256, dropout: float = 0.1,
                 activation: str = "gelu", attention: str = "prob", norm: str = "batch",
                 eval_seed: int = 0, exact_when_full: bool = True) -> None:
        super().__init__()
        self.self_attn = MultiheadSelfAttention(d_model, nhead, dropout=dropout, kind=attention,
                                                eval_seed=eval_seed, exact_when_full=exact_when_full)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1: nn.Module
        self.norm2: nn.Module
        if norm == "batch":
            self.norm1 = nn.BatchNorm1d(d_model, eps=1e-5)
            self.norm2 = nn.BatchNorm1d(d_model, eps=1e-5)
        elif norm == "layer":
            self.norm1 = ChannelLayerNorm(d_model)
            self.norm2 = ChannelLayerNorm(d_model)
        else:
            raise ValueError(f"unknown norm {norm!r}")
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation: Callable[..., torch.Tensor]
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"activation should be relu/gelu, not {activation}")

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor | None = None, is_causal: bool | None = None,
                src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        assert src2.shape == src.shape, (src2.shape, src.shape)
        src = src + self.dropout1(src2)
        src = src.permute(1, 2, 0)  # (S, d, B): BatchNorm1d(d) normalises over S and B
        src = self.norm1(src)
        src = src.permute(2, 0, 1)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = src.permute(1, 2, 0)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)
        return src


def encoder_stack(d_model: int, n_heads: int, d_ff: int, dropout: float, n_layers: int, attention: str = "prob",
                  norm: str = "batch", **kw: Any) -> nn.TransformerEncoder:
    """``n_layers`` copies of one freshly initialised layer (``nn.TransformerEncoder`` clones the layer, so all layers
    start from identical weights, as in the submitted code)."""
    layer = TransformerBatchNormEncoderLayer(d_model, n_heads, d_ff, dropout, "gelu", attention=attention,
                                             norm=norm, **kw)
    return nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)  # type: ignore[arg-type]


class PatchTSTEncoder(nn.Module):
    """Channel-independent temporal encoder (shared weights across sensors). x (B, C, L) -> (B, N, d)."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float, patch_len: int,
                 stride: int, use_batchnorm_out: bool = False, attention: str = "prob", **kw: Any) -> None:
        super().__init__()
        self.inst_norm = nn.InstanceNorm1d(1, affine=False, eps=1e-6)
        self.patch_embed = PatchEmbedding(patch_len=patch_len, stride=stride, d_model=d_model)
        self.encoder = encoder_stack(d_model, n_heads, d_ff, dropout, n_layers, attention=attention, **kw)
        self.pos_enc = SinusoidalPositionEncoding(d_model)
        self.use_bn = use_batchnorm_out
        if self.use_bn:
            self.bn_out = nn.BatchNorm1d(d_model)
        else:
            self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        x = x.reshape(B * C, 1, L)
        x = self.inst_norm(x).squeeze(1)
        tokens = self.pos_enc(self.patch_embed(x))
        enc = self.encoder(tokens)
        _, N, D = enc.shape
        enc = enc.view(B, C, N, D).mean(dim=1)  # (B, N, d): mean over sensors
        if self.use_bn:
            enc = self.bn_out(enc.transpose(1, 2)).transpose(1, 2)
        else:
            enc = self.ln_out(enc)
        return enc


class SensorChannelTransformerEncoder(nn.Module):
    """Channel branch of theta*: patches along the sensor axis at every cycle; tokens = (cycle, patch).

    x (B, C, L) -> (B, L * N_c, d)."""

    def __init__(self, C: int, L: int, patch_len: int, stride: int, d_model: int = 128, n_heads: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.1,
                 use_batchnorm_out: bool = False, attention: str = "prob", **kw: Any) -> None:
        super().__init__()
        self.C = C
        self.L = L
        self.patch_embed = PatchEmbedding(patch_len=patch_len, stride=stride, d_model=d_model)
        self.pos_encoder = SinusoidalPositionEncoding(d_model)
        self.transformer_encoder = encoder_stack(d_model, n_heads, dim_feedforward, dropout, num_layers,
                                                 attention=attention, **kw)
        self.use_bn = use_batchnorm_out
        if self.use_bn:
            self.bn_out = nn.BatchNorm1d(d_model)
        else:
            self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        assert C == self.C and L == self.L, "Shape mismatch for SensorChannelTransformerEncoder"
        x = x.permute(0, 2, 1).reshape(B * L, C)
        tokens = self.patch_embed(x)
        num_patches = tokens.size(1)
        tokens = tokens.view(B, L, num_patches, -1).view(B, L * num_patches, -1)
        tokens = self.pos_encoder(tokens)
        enc = self.transformer_encoder(tokens)
        if self.use_bn:
            enc = self.bn_out(enc.transpose(1, 2)).transpose(1, 2)
        else:
            enc = self.ln_out(enc)
        return enc


class SensorTokenEncoder(nn.Module):
    """D2/D3 channel branch. x (B, C, L) -> (B, C, d).

    ``z_i = W_in x_i + b_in + E[id(i)]``; exact attention over the C tokens, no positional encoding; BatchNorm1d(d)
    over (batch, sensors) at the output."""

    def __init__(self, C: int, L: int, sensor_ids: tuple[int, ...] | list[int], d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.C, self.L = int(C), int(L)
        sids = check_sensor_ids(sensor_ids, self.C)
        self.inp = nn.Linear(self.L, d_model)
        self.sensor_emb = make_sensor_embedding(d_model)
        self.register_buffer("ids0", torch.tensor([s - 1 for s in sids], dtype=torch.long), persistent=False)
        self.encoder = encoder_stack(d_model, n_heads, d_ff, dropout, n_layers, attention="exact")
        self.bn_out = nn.BatchNorm1d(d_model)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.inp(x) + self.sensor_emb(self.ids0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        assert C == self.C and L == self.L, (x.shape, self.C, self.L)
        z = self.encoder(self.tokens(x))
        return self.bn_out(z.transpose(1, 2)).transpose(1, 2)


class PerSensorTemporalEncoder(nn.Module):
    """D3 temporal branch. x (B, C, L) -> (h_t (B, C, d), mu (B, C), sigma (B, C)); no mean over sensors."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float, patch_len: int,
                 stride: int) -> None:
        super().__init__()
        self.patch_embed = EndPaddedPatchEmbedding(patch_len, stride, d_model)
        self.pos_enc = SinusoidalPositionEncoding(d_model)
        self.encoder = encoder_stack(d_model, n_heads, d_ff, dropout, n_layers, attention="exact")
        self.stats_proj = nn.Linear(2, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, L = x.shape
        xt, mu, sigma = window_stats(x)
        tok = self.pos_enc(self.patch_embed(xt.reshape(B * C, L)))
        H = self.encoder(tok)
        h = H.mean(dim=1).view(B, C, -1)
        st = torch.stack([mu, torch.log(sigma)], dim=-1)
        return h + self.stats_proj(st), mu, sigma


class SACAEncoder(nn.Module):
    """D4 SACA encoder (sensor-anchored cross-dimension attention).

    Per layer l = 1..K:
      time stage   ``[a_i ; p_i,1..N'] <- TBN_t([a_i ; p_i,1..N'])`` per sensor (shared weights), exact attention;
      sensor stage ``[a_1 .. a_C]      <- TBN_c([a_1 .. a_C])``, exact attention among anchors only, no PE.
    Initial anchors ``a_i = W_in x_i + b_in + E[id(i)] + W_s [mu_i, log sigma_i]``; initial patch tokens are the
    instance-normalised, end-padded patches plus sinusoidal PE. Returns the final anchors (B, C, d).
    The attribute names below are the state-dict keys of the frozen checkpoints."""

    def __init__(self, C: int, L: int, sensor_ids: tuple[int, ...] | list[int], d_model: int, patch_len_t: int,
                 stride_t: int, n_layers: int, n_heads_t: int, d_ff_t: int, dropout_t: float, n_heads_c: int,
                 d_ff_c: int, dropout_c: float, norm: str = "batch") -> None:
        super().__init__()
        self.C, self.L = C, L
        sids = check_sensor_ids(sensor_ids, C)
        d = d_model
        self.inp = nn.Linear(L, d)
        self.sensor_emb = make_sensor_embedding(d)
        self.register_buffer("ids0", torch.tensor([s - 1 for s in sids], dtype=torch.long), persistent=False)
        self.stats_proj = nn.Linear(2, d, bias=False)
        self.patch_embed = EndPaddedPatchEmbedding(patch_len_t, stride_t, d)
        self.pos_enc = SinusoidalPositionEncoding(d)
        K = int(n_layers)
        self.time_layers = nn.ModuleList(
            [TransformerBatchNormEncoderLayer(d, n_heads_t, d_ff_t, dropout_t, "gelu", attention="exact", norm=norm)
             for _ in range(K)])
        self.sensor_layers = nn.ModuleList(
            [TransformerBatchNormEncoderLayer(d, n_heads_c, d_ff_c, dropout_c, "gelu", attention="exact", norm=norm)
             for _ in range(K)])

    def initial_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x (B, C, L) -> anchors a^0 (B, C, d) and patch tokens p^0 (B*C, N', d)."""
        B, C, L = x.shape
        assert C == self.C and L == self.L, (x.shape, self.C, self.L)
        xt, mu, sigma = window_stats(x)
        st = torch.stack([mu, torch.log(sigma)], dim=-1)
        a = self.inp(x) + self.sensor_emb(self.ids0).unsqueeze(0) + self.stats_proj(st)
        p = self.pos_enc(self.patch_embed(xt.reshape(B * C, L)))
        return a, p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _ = x.shape
        a, p = self.initial_tokens(x)
        d = a.shape[-1]
        for tl, sl in zip(self.time_layers, self.sensor_layers, strict=True):
            z = tl(torch.cat([a.reshape(B * C, 1, d), p], dim=1))
            a, p = z[:, 0, :].reshape(B, C, d), z[:, 1:, :]
            a = sl(a)
        return a
