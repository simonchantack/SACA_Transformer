"""External and simple baselines under the shared protocol: PatchTST, iTransformer, Crossformer, LSTM, ridge.

Contract for every baseline::

    model = build_baseline(name, cfg, C, L)     # name in NAMES
    y = model(x)                                 # x (B, C, L) float32 -> y (B,)

PatchTST, iTransformer and Crossformer backbones are the official code at pinned commits, vendored unchanged except
for import paths (``vendored/VENDORED.md``; Apache-2.0 or MIT licence files in ``vendored/LICENSES``). Only the input
adapter (permute to (B, L, C), patching, optional instance normalisation) and the task head are written here.
There are no time-mark features, no target scaling and no special bias initialisation (equal protocol with D4).

LSTM: ``x (B, C, L) -> (B, L, C) -> nn.LSTM(C, hidden, layers, batch_first, dropout if layers > 1)`` (cuDNN) ->
top-layer output at the last time step -> Dropout -> Linear(hidden, 1). Unidirectional, no norm.

Ridge: ``y = w^T vec(x) + b`` in float64, fitted by scikit-learn (closed-form Cholesky), never by gradient descent.
"""
from __future__ import annotations

import hashlib
import types
from math import ceil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .search_space import InfeasibleConfig, round_half_up
from .vendored.crossformer.cross_embed import DSW_embedding as _DSW_embedding
from .vendored.crossformer.cross_encoder import Encoder as _CrossEncoder
from .vendored.itransformer.model.iTransformer import Model as _OfficialITransformer
from .vendored.patchtst.create_patch import create_patch as _create_patch
from .vendored.patchtst.patchTST import PatchTST as _OfficialPatchTST
from .vendored.patchtst.RevIN import RevIN as _RevIN

NAMES = ("patchtst", "itransformer", "crossformer", "lstm", "ridge")


class PatchTSTRUL(nn.Module):
    """PatchTST (Nie et al., ICLR 2023), official regression head. RevIN: affine=False, subtract_last=False, no
    de-normalisation. x (B, C, L) -> (B,)."""

    FIXED: dict[str, Any] = dict(shared_embedding=True, norm="BatchNorm", pe="zeros", learn_pe=True,
                                 res_attention=True, pre_norm=False, act="gelu", revin_affine=False,
                                 revin_subtract_last=False)

    def __init__(self, C: int, L: int, patch_len: int, stride: int, d_model: int, n_heads: int, e_layers: int,
                 d_ff: int, dropout: float, revin: bool) -> None:
        super().__init__()
        C, L, patch_len, stride = int(C), int(L), int(patch_len), int(stride)
        if patch_len > L:
            raise InfeasibleConfig(f"patch_len {patch_len} > L {L}")
        num_patch = (max(L, patch_len) - patch_len) // stride + 1
        if num_patch < 2:
            raise InfeasibleConfig(f"num_patch {num_patch} < 2 (L {L}, patch_len {patch_len}, stride {stride})")
        if int(d_model) % int(n_heads):
            raise InfeasibleConfig(f"n_heads {n_heads} does not divide d_model {d_model}")
        self.C, self.L, self.patch_len, self.stride, self.num_patch = C, L, patch_len, stride, num_patch
        f = self.FIXED
        self.revin = _RevIN(C, affine=f["revin_affine"], subtract_last=f["revin_subtract_last"]) if revin else None
        self.model = _OfficialPatchTST(
            c_in=C, target_dim=1, patch_len=patch_len, stride=stride, num_patch=num_patch, n_layers=int(e_layers),
            d_model=int(d_model), n_heads=int(n_heads), shared_embedding=f["shared_embedding"], d_ff=int(d_ff),
            norm=f["norm"], attn_dropout=float(dropout), dropout=float(dropout), act=f["act"],
            res_attention=f["res_attention"], pre_norm=f["pre_norm"], store_attn=False, pe=f["pe"],
            learn_pe=f["learn_pe"], head_dropout=float(dropout), head_type="regression", y_range=None)

    def patches(self, x: torch.Tensor) -> torch.Tensor:
        z = x.permute(0, 2, 1).contiguous()
        if self.revin is not None:
            z = self.revin(z, "norm")
        zp, _ = _create_patch(z, self.patch_len, self.stride)
        return zp  # type: ignore[no-any-return]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.patches(x)).reshape(-1)  # type: ignore[no-any-return]


class ITransformerRUL(nn.Module):
    """iTransformer (Liu et al., ICLR 2024): official embedding and encoder, TSLib classification head
    (GELU, dropout, flatten C*d, Linear -> 1; TSLib commit 4e938a1). x (B, C, L) -> (B,)."""

    FIXED: dict[str, Any] = dict(activation="gelu", factor=1, embed="timeF", freq="h", class_strategy="projection")

    def __init__(self, C: int, L: int, d_model: int, d_ff: int, e_layers: int, n_heads: int, dropout: float,
                 use_norm: bool) -> None:
        super().__init__()
        if int(d_model) % int(n_heads):
            raise InfeasibleConfig(f"n_heads {n_heads} does not divide d_model {d_model}")
        self.C, self.L, self.use_norm = int(C), int(L), bool(use_norm)
        f = self.FIXED
        ns = types.SimpleNamespace(seq_len=self.L, pred_len=1, output_attention=False, use_norm=self.use_norm,
                                   d_model=int(d_model), embed=f["embed"], freq=f["freq"], dropout=float(dropout),
                                   class_strategy=f["class_strategy"], factor=f["factor"], n_heads=int(n_heads),
                                   d_ff=int(d_ff), activation=f["activation"], e_layers=int(e_layers))
        off = _OfficialITransformer(ns)
        self.enc_embedding = off.enc_embedding
        self.encoder = off.encoder
        del off
        self.act = F.gelu
        self.dropout = nn.Dropout(float(dropout))
        self.projection = nn.Linear(int(d_model) * self.C, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_enc = x.permute(0, 2, 1).contiguous()
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        return enc_out  # type: ignore[no-any-return]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.dropout(self.act(self.encode(x)))
        return self.projection(output.reshape(output.shape[0], -1)).reshape(-1)


class CrossformerRUL(nn.Module):
    """Crossformer (Zhang and Yan, ICLR 2023): official DSW embedding and hierarchical encoder with front padding,
    TSLib classification head (flatten, dropout, Linear -> 1). x (B, C, L) -> (B,)."""

    def __init__(self, C: int, L: int, seg_len: int, win_size: int, factor: int, d_model: int, d_ff: int,
                 n_heads: int, e_layers: int, dropout: float) -> None:
        super().__init__()
        C, L, seg_len, win_size, e_layers = int(C), int(L), int(seg_len), int(win_size), int(e_layers)
        if int(d_model) % int(n_heads):
            raise InfeasibleConfig(f"n_heads {n_heads} does not divide d_model {d_model}")
        if seg_len > L:
            raise InfeasibleConfig(f"seg_len {seg_len} > L {L}")
        self.C, self.L, self.seg_len, self.merge_win = C, L, seg_len, win_size
        self.pad_in_len = ceil(1.0 * L / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - L
        in_seg = self.pad_in_len // seg_len
        self.enc_value_embedding = _DSW_embedding(seg_len, int(d_model))
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, C, in_seg, int(d_model)))
        self.pre_norm = nn.LayerNorm(int(d_model))
        self.encoder = _CrossEncoder(e_layers, win_size, int(d_model), int(n_heads), int(d_ff), block_depth=1,
                                     dropout=float(dropout), in_seg_num=in_seg, factor=int(factor))
        n_last = in_seg
        for _ in range(1, e_layers):
            n_last = ceil(n_last / win_size)
        self.n_seg_in, self.n_seg_last = in_seg, n_last
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(float(dropout))
        self.projection = nn.Linear(int(d_model) * n_last * C, 1)

    def encode(self, x: torch.Tensor) -> Any:
        x_seq = x.permute(0, 2, 1).contiguous()
        if self.in_len_add != 0:
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim=1)
        x_seq = self.enc_value_embedding(x_seq)
        x_seq = x_seq + self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)
        return self.encoder(x_seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out = self.encode(x)
        output = self.dropout(self.flatten(enc_out[-1].permute(0, 1, 3, 2)))
        return self.projection(output.reshape(output.shape[0], -1)).reshape(-1)


class LSTMRUL(nn.Module):
    """Unidirectional LSTM regressor. x (B, C, L) -> (B,)."""

    def __init__(self, C: int, L: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.C, self.L = int(C), int(L)
        self.hidden, self.layers, self.p = int(hidden), int(layers), float(dropout)
        self.lstm = nn.LSTM(self.C, self.hidden, num_layers=self.layers, batch_first=True,
                            dropout=self.p if self.layers > 1 else 0.0)
        self.drop = nn.Dropout(self.p)
        self.head = nn.Linear(self.hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        assert C == self.C and L == self.L, (x.shape, self.C, self.L)
        out, _ = self.lstm(x.transpose(1, 2).contiguous())
        return self.head(self.drop(out[:, -1, :])).squeeze(-1)


def lstm_n_params(C: int, hidden: int, layers: int) -> int:
    h = int(hidden)
    return 4 * (C * h + h * h + 2 * h) + (int(layers) - 1) * 4 * (2 * h * h + 2 * h) + h + 1


class RidgeLinear(nn.Module):
    """``y = w^T vec(x) + b`` in float64; vec flattens (C, L) row-major (sensor-major)."""

    def __init__(self, C: int, L: int, alpha: float | None = None) -> None:
        super().__init__()
        self.C, self.L = int(C), int(L)
        self.alpha = None if alpha is None else float(alpha)
        self.weight = nn.Parameter(torch.zeros(self.C * self.L, dtype=torch.float64), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros((), dtype=torch.float64), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        assert C == self.C and L == self.L, (x.shape, self.C, self.L)
        return x.reshape(B, C * L).to(torch.float64) @ self.weight + self.bias


def ridge_from_sklearn(est: Any, C: int, L: int) -> RidgeLinear:
    """Copy a fitted sklearn Ridge (fit on ``X.reshape(N, C*L)`` float64) into a ``RidgeLinear`` module."""
    m = RidgeLinear(C, L, alpha=getattr(est, "alpha", None))
    coef = torch.as_tensor(est.coef_, dtype=torch.float64).reshape(-1)
    assert coef.numel() == C * L, (coef.numel(), C, L)
    with torch.no_grad():
        m.weight.copy_(coef)
        m.bias.copy_(torch.as_tensor(float(est.intercept_), dtype=torch.float64))
    return m.eval()


def _stride(patch_len: int, r: float) -> int:
    return max(1, round_half_up(float(r) * int(patch_len)))


def model_kwargs(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Map a search-space dict to the constructor arguments."""
    c = dict(cfg)
    if name == "patchtst":
        return dict(patch_len=int(c["patch_len"]), stride=_stride(c["patch_len"], c["r"]), d_model=int(c["d_model"]),
                    n_heads=int(c["n_heads"]), e_layers=int(c["e_layers"]), d_ff=int(c["d_ff"]),
                    dropout=float(c["dropout"]), revin=bool(c["revin"]))
    if name == "itransformer":
        return dict(d_model=int(c["d_model"]), d_ff=int(c["m"]) * int(c["d_model"]), e_layers=int(c["e_layers"]),
                    n_heads=int(c["n_heads"]), dropout=float(c["dropout"]), use_norm=bool(c["use_norm"]))
    if name == "crossformer":
        return dict(seg_len=int(c["seg_len"]), win_size=int(c.get("win_size", 2)), factor=int(c["factor"]),
                    d_model=int(c["d_model"]), d_ff=2 * int(c["d_model"]), n_heads=int(c["n_heads"]),
                    e_layers=int(c["e_layers"]), dropout=float(c["dropout"]))
    if name == "lstm":
        return dict(hidden=int(c["hidden"]), layers=int(c["layers"]), dropout=float(c["dropout"]))
    if name == "ridge":
        return dict(alpha=c.get("alpha"))
    raise ValueError(f"unknown baseline {name!r}; expected one of {NAMES}")


_CLASSES: dict[str, Any] = {"patchtst": PatchTSTRUL, "itransformer": ITransformerRUL,
                            "crossformer": CrossformerRUL, "lstm": LSTMRUL, "ridge": RidgeLinear}


def build_baseline(name: str, cfg: dict[str, Any], C: int, L: int) -> nn.Module:
    """Baseline ``name`` for input (B, C, L); forward -> (B,). Raises ``InfeasibleConfig`` for infeasible draws."""
    name = name.lower()
    kw = model_kwargs(name, cfg)
    model: nn.Module = _CLASSES[name](C=int(C), L=int(L), **kw)
    model.baseline_name = name  # type: ignore[assignment]
    return model


def code_sha256() -> dict[str, str]:
    """sha256 of this module and every vendored ``.py`` file (recorded in each baseline checkpoint)."""
    here = Path(__file__).resolve().parent
    files = [here / "baselines.py"] + sorted((here / "vendored").rglob("*.py"))
    return {p.relative_to(here).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
