"""Embeddings: patching, positional encoding, per-window statistics and sensor identities.

Patch counts (R1.3, submitted Eq. 10):
  * no padding (submitted temporal and channel patching): ``N = floor((L - P) / S) + 1``; the newest
    ``(L - P) mod S`` cycles are dropped.
  * end padding (PatchTST, Nie et al. 2023 Sec. 3.1; D3 and D4): ``N' = floor((L - P) / S) + 2``. S copies of the last
    value are appended, so the newest cycle ``L - 1`` always lies in the last patch.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

N_SENSOR_IDS = 21  # C-MAPSS sensors sm_1 .. sm_21
SIGMA_FLOOR_VAR = 1e-6  # sigma = sqrt(var + 1e-6), the eps of the submitted InstanceNorm1d
EMB_INIT_STD = 0.02  # sensor-ID embedding init N(0, 0.02^2)


def n_patches(L: int, P: int, S: int) -> int:
    """Patch count without padding: floor((L - P)/S) + 1 (0 when L < P)."""
    return (L - P) // S + 1 if L >= P else 0


def n_patches_padded(L: int, P: int, S: int) -> int:
    """Patch count with PatchTST end padding: floor((L - P)/S) + 2 (D4, D3)."""
    return (L - P) // S + 2


class PatchEmbedding(nn.Module):
    """(B*C, L) -> (B*C, N, d_model): unfold into length-P patches (stride S), then a shared Linear(P, d_model)."""

    def __init__(self, patch_len: int, stride: int, d_model: int) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        if L < self.patch_len:
            raise ValueError(f"Lookback L={L} < patch_len={self.patch_len}. Increase lookback or reduce patch_len.")
        n = 1 + (L - self.patch_len) // self.stride
        if n <= 0:
            raise ValueError("No patches would be created; check patch_len/stride vs lookback.")
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        return self.proj(patches)


class EndPaddedPatchEmbedding(nn.Module):
    """(B*C, L) -> (B*C, N', d): append S copies of the last value, unfold (P, S), Linear(P, d)."""

    def __init__(self, patch_len: int, stride: int, d_model: int) -> None:
        super().__init__()
        self.patch_len, self.stride = int(patch_len), int(stride)
        self.proj = nn.Linear(self.patch_len, d_model)

    def patches(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < self.patch_len:
            raise ValueError(f"L={x.shape[1]} < patch_len={self.patch_len}")
        pad = x[:, -1:].expand(x.shape[0], self.stride)
        return torch.cat([x, pad], dim=1).unfold(dimension=1, size=self.patch_len, step=self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.patches(x))


class SinusoidalPositionEncoding(nn.Module):
    """Fixed sinusoidal encoding added to (B, N, d_model) tokens (not stored in the state dict)."""

    def __init__(self, d_model: int, max_len: int = 10000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe: torch.Tensor = self.pe  # type: ignore[assignment]
        return x + pe[:, : x.size(1), :]


def window_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x (..., L) -> (x_tilde, mu, sigma) with biased variance and sigma = sqrt(var + 1e-6).

    Equals ``nn.InstanceNorm1d(affine=False, eps=1e-6)`` on the normalised series, and keeps mu and sigma."""
    mu = x.mean(dim=-1)
    var = x.var(dim=-1, unbiased=False)
    sigma = torch.sqrt(var + SIGMA_FLOOR_VAR)
    return (x - mu.unsqueeze(-1)) / sigma.unsqueeze(-1), mu, sigma


def sensor_ids_from_features(features: Sequence[str]) -> tuple[int, ...]:
    """['sm_2', 'sm_3', ...] (``Pipeline.features``) -> (2, 3, ...). The id is bound to the column."""
    out: list[int] = []
    for f in features:
        s = str(f)
        if not s.startswith("sm_"):
            raise ValueError(f"feature {s!r} is not a C-MAPSS sensor column 'sm_<k>'")
        k = int(s[3:])
        if not 1 <= k <= N_SENSOR_IDS:
            raise ValueError(f"sensor id {k} outside 1..{N_SENSOR_IDS}")
        out.append(k)
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate sensor ids {out}")
    return tuple(out)


def check_sensor_ids(sensor_ids: Sequence[int] | None, C: int) -> tuple[int, ...]:
    if sensor_ids is None:
        raise ValueError("sensor_ids are required for D2-D4 (pass sensor_ids_from_features(cache.features))")
    sids = tuple(int(s) for s in sensor_ids)
    if len(sids) != C:
        raise ValueError(f"len(sensor_ids) {len(sids)} != C {C}")
    if len(set(sids)) != C or min(sids) < 1 or max(sids) > N_SENSOR_IDS:
        raise ValueError(f"bad sensor_ids {sids}")
    return sids


def make_sensor_embedding(d_model: int) -> nn.Embedding:
    """Sensor-identity embedding E (21 x d), initialised N(0, 0.02^2)."""
    emb = nn.Embedding(N_SENSOR_IDS, d_model)
    nn.init.normal_(emb.weight, mean=0.0, std=EMB_INIT_STD)
    return emb
