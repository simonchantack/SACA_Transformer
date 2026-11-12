"""Model definitions, configuration and checkpoints.

Designs (all map a window ``(B, C, L)`` to a RUL estimate ``(B,)``):

  theta*  corrected submitted model: temporal PatchTST encoder and sensor-channel encoder in parallel, ProbSparse
          attention, concat-and-mean fusion head.
  D1      theta* with exact attention (same parameters and state-dict keys).
  D2      D1 temporal branch + one-token-per-sensor channel branch (sensor-identity embedding).
  D3      per-sensor temporal summaries + sensor-token branch, per-sensor fusion, sensor-indexed head.
  D4      SACA: one coupled encoder (per-sensor anchor inside its patch attention, exact attention among anchors).

D4 is the paper's model (frozen configuration: ``configs/selections/exp_B16_config_selection_frozen.json``).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .attention import ProbAttention, set_probsparse
from .decoder import FusionHead, SensorIndexedHead
from .embeddings import check_sensor_ids
from .encoder import (
    PatchTSTEncoder,
    PerSensorTemporalEncoder,
    SACAEncoder,
    SensorChannelTransformerEncoder,
    SensorTokenEncoder,
)

DESIGNS = ("D1", "D2", "D3", "D4")


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyper-parameters. Field names are the keys of the frozen checkpoints."""

    C: int
    L: int
    patch_len_t: int
    stride_t: int
    patch_len_c: int
    stride_c: int
    d_model_t: int
    n_heads_t: int
    n_layers_t: int
    d_ff_t: int
    dropout_t: float
    d_model_c: int
    n_heads_c: int
    n_layers_c: int
    d_ff_c: int
    dropout_c: float
    head_hidden: int
    pooling: str = "mean"
    use_bn_temporal: bool = True
    use_bn_channel: bool = True
    fusion_norm: str = "batchnorm"
    probsparse_eval_seed: int = 0
    probsparse_exact_when_full: bool = True


_MODEL_FIELDS = [f.name for f in dataclasses.fields(ModelConfig)]


def _resolve_rule(v: Any, C: int | None) -> int:
    """int, or a string rule in terms of C such as "C", "C//2", "C-1"."""
    if isinstance(v, str):
        if C is None:
            raise ValueError(f"rule {v!r} needs C")
        if not all(ch in "C/+-*0123456789 ()" for ch in v):
            raise ValueError(f"bad rule {v!r}")
        return int(eval(v, {"__builtins__": {}}, {"C": int(C)}))  # noqa: S307 (whitelisted characters)
    return int(v)


def resolve(model_dict: dict[str, Any], C: int | None = None, L: int | None = None) -> ModelConfig:
    """Build a ModelConfig from a dict. ``C`` and ``L`` override the dict."""
    d = dict(model_dict.get("model", model_dict))
    if C is not None:
        d["C"] = int(C)
    if L is not None:
        d["L"] = int(L)
    if "C" not in d:
        raise ValueError("C (number of kept sensors) is required; pass C= from the preprocessing output")
    d["patch_len_c"] = _resolve_rule(d["patch_len_c"], d["C"])
    d["stride_c"] = _resolve_rule(d["stride_c"], d["C"])
    unknown = set(d) - set(_MODEL_FIELDS)
    if unknown:
        raise ValueError(f"unknown model keys {sorted(unknown)}")
    return ModelConfig(**d)


class ThetaStarModel(nn.Module):
    """Corrected submitted model (theta*), or D1 when ``attention="exact"``. x (B, C, L) -> (B,)."""

    def __init__(self, cfg: ModelConfig, attention: str = "prob") -> None:
        super().__init__()
        if cfg.fusion_norm != "batchnorm":
            raise ValueError(cfg.fusion_norm)
        self.temporal_encoder = PatchTSTEncoder(
            d_model=cfg.d_model_t, n_heads=cfg.n_heads_t, n_layers=cfg.n_layers_t, d_ff=cfg.d_ff_t,
            dropout=cfg.dropout_t, patch_len=cfg.patch_len_t, stride=cfg.stride_t,
            use_batchnorm_out=cfg.use_bn_temporal, attention=attention)
        self.sensor_encoder = SensorChannelTransformerEncoder(
            C=cfg.C, L=cfg.L, patch_len=cfg.patch_len_c, stride=cfg.stride_c, d_model=cfg.d_model_c,
            n_heads=cfg.n_heads_c, num_layers=cfg.n_layers_c, dim_feedforward=cfg.d_ff_c, dropout=cfg.dropout_c,
            use_batchnorm_out=cfg.use_bn_channel, attention=attention)
        self.fusion_head = FusionHead(cfg.d_model_t, cfg.d_model_c, cfg.head_hidden, cfg.dropout_t, cfg.pooling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        te = self.temporal_encoder(x)
        se = self.sensor_encoder(x)
        return self.fusion_head(te, se).squeeze(-1)


class D2Model(nn.Module):
    """D2: D1 temporal branch + sensor-token channel branch, FusionHead unchanged."""

    def __init__(self, cfg: ModelConfig, sensor_ids: Sequence[int]) -> None:
        super().__init__()
        self.temporal_encoder = PatchTSTEncoder(
            d_model=cfg.d_model_t, n_heads=cfg.n_heads_t, n_layers=cfg.n_layers_t, d_ff=cfg.d_ff_t,
            dropout=cfg.dropout_t, patch_len=cfg.patch_len_t, stride=cfg.stride_t,
            use_batchnorm_out=cfg.use_bn_temporal, attention="exact")
        self.sensor_encoder = SensorTokenEncoder(cfg.C, cfg.L, list(sensor_ids), cfg.d_model_c, cfg.n_heads_c,
                                                 cfg.n_layers_c, cfg.d_ff_c, cfg.dropout_c)
        self.fusion_head = FusionHead(cfg.d_model_t, cfg.d_model_c, cfg.head_hidden, cfg.dropout_t, cfg.pooling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion_head(self.temporal_encoder(x), self.sensor_encoder(x)).squeeze(-1)


class D3Model(nn.Module):
    """D3: per-sensor fusion ``f_i = BN(W_f [h_t_i ; h_c_i] + b_f)`` and a sensor-indexed head."""

    def __init__(self, cfg: ModelConfig, sensor_ids: Sequence[int]) -> None:
        super().__init__()
        if cfg.d_model_t != cfg.d_model_c:
            raise ValueError("D3 needs d_model_t == d_model_c (true in the P2 space)")
        d = cfg.d_model_t
        self.temporal_encoder = PerSensorTemporalEncoder(d, cfg.n_heads_t, cfg.n_layers_t, cfg.d_ff_t, cfg.dropout_t,
                                                         cfg.patch_len_t, cfg.stride_t)
        self.sensor_encoder = SensorTokenEncoder(cfg.C, cfg.L, list(sensor_ids), d, cfg.n_heads_c, cfg.n_layers_c,
                                                 cfg.d_ff_c, cfg.dropout_c)
        self.fuse = nn.Linear(2 * d, d)
        self.fuse_bn = nn.BatchNorm1d(d)
        self.head = SensorIndexedHead(cfg.C, sensor_ids, d, cfg.head_hidden, cfg.dropout_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ht, _, _ = self.temporal_encoder(x)
        hc = self.sensor_encoder(x)
        f = self.fuse(torch.cat([ht, hc], dim=-1))
        f = self.fuse_bn(f.transpose(1, 2)).transpose(1, 2)
        return self.head(f)


class SACAModel(SACAEncoder):
    """D4 SACA. ``K = n_layers_t`` coupled layers; ``n_layers_c``, ``patch_c`` and ``stride_c`` are unused.

    The encoder parameters sit directly on this module (``inp``, ``sensor_emb``, ``stats_proj``, ``patch_embed``,
    ``time_layers``, ``sensor_layers``) followed by ``head``, so the state-dict keys and the initialisation order equal
    those of the frozen checkpoints. ``encode`` returns the final anchors (B, C, d); ``forward`` maps them through the
    sensor-indexed head."""

    def __init__(self, cfg: ModelConfig, sensor_ids: Sequence[int], norm: str = "batch") -> None:
        if cfg.d_model_t != cfg.d_model_c:
            raise ValueError("D4 needs d_model_t == d_model_c (true in the P2 space)")
        sids = check_sensor_ids(sensor_ids, cfg.C)
        super().__init__(cfg.C, cfg.L, sids, cfg.d_model_t, cfg.patch_len_t, cfg.stride_t, cfg.n_layers_t,
                         cfg.n_heads_t, cfg.d_ff_t, cfg.dropout_t, cfg.n_heads_c, cfg.d_ff_c, cfg.dropout_c,
                         norm=norm)
        self.head = SensorIndexedHead(cfg.C, sids, cfg.d_model_t, cfg.head_hidden, cfg.dropout_t)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


def build_model(cfg: ModelConfig, design: str = "theta", sensor_ids: Sequence[int] | None = None,
                norm: str = "batch") -> nn.Module:
    """Build ``design`` in {"theta", "D1", "D2", "D3", "D4"}; attaches ``model_config``, ``design``, ``sensor_ids``."""
    m: nn.Module
    sids: tuple[int, ...] | None = None
    if design == "theta":
        m = ThetaStarModel(cfg, attention="prob")
        set_probsparse(m, eval_seed=cfg.probsparse_eval_seed, exact_when_full=cfg.probsparse_exact_when_full,
                       output_attention=False)
    elif design == "D1":
        m = ThetaStarModel(cfg, attention="exact")
        if sensor_ids is not None:
            sids = check_sensor_ids(sensor_ids, cfg.C)
    elif design in ("D2", "D3", "D4"):
        if cfg.fusion_norm != "batchnorm" or cfg.pooling != "mean":
            raise ValueError("round-3 designs use fusion_norm batchnorm and pooling mean (P2 fixed)")
        sids = check_sensor_ids(sensor_ids, cfg.C)
        if design == "D2":
            m = D2Model(cfg, sids)
        elif design == "D3":
            m = D3Model(cfg, sids)
        else:
            m = SACAModel(cfg, sids, norm=norm)
    else:
        raise ValueError(f"unknown design {design!r}")
    m.model_config = cfg  # type: ignore[assignment]
    m.design = design  # type: ignore[assignment]
    m.sensor_ids = sids  # type: ignore[assignment]
    return m


def n_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))


def count_prob_attention(m: nn.Module) -> int:
    return sum(isinstance(x, ProbAttention) for x in m.modules())


# ------------------------------------------------------------------------------------------------ checkpoints
def _canon(o: Any) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(cfg: ModelConfig, recipe: dict[str, Any] | None = None) -> str:
    """sha256 of the canonical JSON of the model config and the train recipe (keys starting with '_' ignored)."""
    r = {k: v for k, v in (recipe or {}).items() if not str(k).startswith("_")}
    return hashlib.sha256(_canon({"model": asdict(cfg), "train": r}).encode()).hexdigest()


def code_sha256() -> dict[str, str]:
    """sha256 of the model source files of this package (recorded in each checkpoint)."""
    here = Path(__file__).resolve().parent
    files = ("model.py", "encoder.py", "attention.py", "decoder.py", "embeddings.py")
    return {f: hashlib.sha256((here / f).read_bytes()).hexdigest() for f in files}


def save_checkpoint(path: str | Path, model: nn.Module, recipe: dict[str, Any] | None = None,
                    extra: dict[str, Any] | None = None) -> None:
    """Self-describing checkpoint: design, config, sensor ids, weights, config hash, code hashes.

    theta* checkpoints have no ``family`` key; D1-D4 checkpoints carry ``family = "r3_design"`` (frozen format)."""
    cfg: ModelConfig = model.model_config  # type: ignore[assignment]
    design: str = model.design  # type: ignore[assignment]
    sids = getattr(model, "sensor_ids", None)
    r = {k: v for k, v in (recipe or {}).items() if not str(k).startswith("_")}
    obj: dict[str, Any] = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": asdict(cfg), "train_recipe": r,
        "code_sha256": code_sha256(), "torch_version": torch.__version__, "extra": extra or {},
    }
    if design == "theta":
        obj["config_hash"] = config_hash(cfg, r)
    else:
        obj["family"] = "r3_design"
        obj["design"] = design
        obj["sensor_ids"] = list(sids or [])
        obj["config_hash"] = hashlib.sha256(_canon({"design": design, "sensor_ids": sids, "model": asdict(cfg),
                                                    "train": r}).encode()).hexdigest()
    torch.save(obj, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu", norm: str = "batch"
                    ) -> tuple[nn.Module, dict[str, Any]]:
    """Load a theta* or D1-D4 checkpoint (frozen format) with a strict state-dict load; returns (model, meta).

    The code hashes stored in a frozen checkpoint refer to the pre-release source files, so they are returned in
    ``meta`` but not compared."""
    obj = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ModelConfig(**obj["model_config"])
    if obj.get("family") == "r3_design":
        design = str(obj["design"])
        sids = obj["sensor_ids"] or None
    elif obj.get("family") in (None, "dualdim"):
        design, sids = "theta", None
    else:
        raise ValueError(f"checkpoint family {obj.get('family')!r} is not a theta*/D1-D4 checkpoint")
    m = build_model(cfg, design, sids, norm=norm)
    m.load_state_dict(obj["state_dict"], strict=True)
    m.eval()
    meta = {"design": design, "sensor_ids": sids, "train_recipe": obj.get("train_recipe", {}),
            "extra": obj.get("extra", {}), "code_sha256_trained": obj.get("code_sha256")}
    return m, meta
