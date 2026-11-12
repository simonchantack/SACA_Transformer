"""Checkpoint I/O for every model family (theta*, D1-D4, PatchTST, iTransformer, Crossformer, LSTM, ridge).

The on-disk layout is the layout of the frozen checkpoints, so the released code loads them with a strict state-dict
load:
  * theta* and D1-D4: ``model.save_checkpoint`` / ``model.load_checkpoint`` (``family`` absent or ``r3_design``).
  * baselines: ``{"family", "model_cfg" (config + family, C, L), "state_dict", "train_recipe", "extra",
    "baselines_code_sha256"}``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from . import baselines as B
from . import model as M


def save_ckpt(path: str | Path, family: str, model: nn.Module, mc: dict[str, Any] | M.ModelConfig,
              recipe: dict[str, Any], extra: dict[str, Any]) -> None:
    """Save a checkpoint. ``family`` is the P2 family name (``dualdim``, ``r3_D4``, ``patchtst``, ...)."""
    if family == "dualdim" or family.startswith("r3_"):
        M.save_checkpoint(path, model, recipe, dict(extra, family_p2=family) if family != "dualdim" else extra)
        return
    assert isinstance(mc, dict)
    torch.save({"family": family, "model_cfg": dict(mc), "state_dict": model.state_dict(), "train_recipe": recipe,
                "extra": extra, "code_sha256": None, "baselines_code_sha256": B.code_sha256()}, path)


def load_for_eval(path: str | Path, device: str | torch.device = "cpu", norm: str = "batch",
                  check_baseline_code: bool = False) -> tuple[nn.Module, dict[str, Any]]:
    """Strict load in eval mode. Returns ``(model, meta)``; ``meta`` holds the training ``extra`` fields
    (``features``, ``L``, ``preprocess``, ...) plus the code hashes stored in the checkpoint."""
    obj = torch.load(path, map_location=device, weights_only=False)
    fam = obj.get("family")
    if fam in (None, "dualdim", "r3_design"):
        model, meta0 = M.load_checkpoint(path, map_location=device, norm=norm)
        model.to(device).eval()
        meta = dict(meta0.get("extra", {}), design=meta0["design"], sensor_ids=meta0["sensor_ids"],
                    code_sha256_trained=meta0["code_sha256_trained"], code_sha256_eval=M.code_sha256())
        return model, meta
    mc = dict(obj["model_cfg"])
    name, C, L = mc.pop("family"), int(mc.pop("C")), int(mc.pop("L"))
    bmodel = B.build_baseline(name, mc, C, L)
    bmodel.load_state_dict(obj["state_dict"], strict=True)
    bmodel.to(device).eval()
    meta = dict(obj.get("extra", {}))
    meta["baselines_code_sha256_trained"] = obj.get("baselines_code_sha256")
    meta["baselines_code_sha256_eval"] = B.code_sha256()
    if check_baseline_code and obj.get("baselines_code_sha256") != meta["baselines_code_sha256_eval"]:
        raise RuntimeError("baseline adapter/vendored code changed since training (sha256 mismatch)")
    return bmodel, meta


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, bs: int, order: np.ndarray | None = None,
            device: str | torch.device | None = None) -> np.ndarray:
    """``X``: float32 array (N, C, L). Returns float64 predictions (N,) in the original row order."""
    model.eval()
    dev = device or next(model.parameters()).device
    idx = np.arange(len(X)) if order is None else np.asarray(order)
    out = np.empty(len(X), dtype=np.float64)
    for i in range(0, len(idx), bs):
        rows = idx[i:i + bs]
        xb = torch.as_tensor(X[rows], dtype=torch.float32, device=dev)
        out[rows] = model(xb).reshape(-1).double().cpu().numpy()
    return out
