"""Strict determinism for every final run (RULES 18, 19).

``set_all(seed)`` seeds every RNG and enables ``torch.use_deterministic_algorithms(True, warn_only=False)``, the
cuBLAS workspace setting ``:4096:8``, deterministic memory fill, deterministic cuDNN with benchmark off, and TF32 off.
``CUBLAS_WORKSPACE_CONFIG`` is set at import time, so import this module before the first CUDA call.
The returned dict is written to the header of every run log.
"""
from __future__ import annotations

import os
import random
import warnings
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np  # noqa: E402
import torch  # noqa: E402


def assert_strict() -> None:
    """Raise AssertionError unless the strict deterministic mode is active."""
    if not torch.are_deterministic_algorithms_enabled():
        raise AssertionError("strict deterministic mode required: use_deterministic_algorithms is off")
    if torch.is_deterministic_algorithms_warn_only_enabled():
        raise AssertionError("strict deterministic mode required: warn_only must be False")
    if not bool(torch.utils.deterministic.fill_uninitialized_memory):  # type: ignore[attr-defined]
        raise AssertionError("fill_uninitialized_memory must stay True")
    if torch.backends.cudnn.benchmark or not torch.backends.cudnn.deterministic:
        raise AssertionError("cudnn must be deterministic with benchmark off")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise AssertionError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")


def set_all(seed: int) -> dict[str, Any]:
    """Seed Python, NumPy and torch (CPU and CUDA) and switch on strict deterministic mode."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    warnings.simplefilter("default")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.utils.deterministic.fill_uninitialized_memory = True  # type: ignore[attr-defined]
    assert_strict()
    return {
        "seed": seed,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "fill_uninitialized_memory": bool(torch.utils.deterministic.fill_uninitialized_memory),  # type: ignore[attr-defined]
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "tf32": False,
        "mode": "strict deterministic (saca_rul.determinism)",
    }
