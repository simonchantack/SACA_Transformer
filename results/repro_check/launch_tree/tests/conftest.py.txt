"""Shared fixtures: a tiny synthetic C-MAPSS-like dataset (no real data needed) and frozen-artifact helpers."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from saca_rul.config import ROOT, resolve_path

FROZEN_ROOT = resolve_path("${SACA_WORKSPACE:-../workspace}")
REAL_DATA = resolve_path("${SACA_DATA_DIR:-../workspace/code_base/Data}")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if (FROZEN_ROOT / "experiments").is_dir() and (REAL_DATA / "train_FD001.txt").exists():
        return
    skip = pytest.mark.skip(reason="frozen workspace artifacts or C-MAPSS data not available (SACA_WORKSPACE)")
    for it in items:
        if "frozen" in it.keywords:
            it.add_marker(skip)


def write_synthetic_fd001(dirpath: Path, n_engines: int = 12, seed: int = 0) -> list[int]:
    """Write train_FD001.txt with ``n_engines`` engines (60-90 cycles, 21 sensors, constant op settings)."""
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(1, n_engines + 1):
        T = int(rng.integers(60, 91))
        t = np.arange(1, T + 1)
        for c in t:
            sens = rng.normal(0, 1, 21) * 0.5 + (c / T) * np.linspace(0.5, 3.0, 21) * (np.arange(21) % 3 == 0)
            sens = sens + np.arange(21)
            row = [e, int(c), -0.0007, -0.0004, 100.0] + [f"{v:.4f}" for v in sens]
            rows.append(" ".join(str(x) for x in row) + " ")
    (dirpath / "train_FD001.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return list(range(1, n_engines + 1))


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("cmapss")
    ids = write_synthetic_fd001(d)
    splits = {"tuning_split_seed": 0,
              "subsets": {"FD001": {"splits": {str(s): {"train": ids[:9], "val": ids[9:]} for s in range(5)}}}}
    sp = d / "data_splits.json"
    sp.write_text(json.dumps(splits), encoding="utf-8")
    return {"data_dir": d, "splits": sp}


@pytest.fixture(scope="session")
def protocol_path() -> Path:
    return ROOT / "configs" / "protocols" / "protocol_P2.yaml"


@pytest.fixture(scope="session")
def tiny_d4_params() -> dict[str, object]:
    return {"L": 30, "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 32, "d_model": 16, "n_heads": 2,
            "n_layers_t": 1, "n_layers_c": 1, "d_ff": 32, "head_hidden": 16, "dropout_t": 0.1, "dropout_c": 0.1,
            "patch_t": 10, "r_t": 1.0, "patch_c": 3, "stride_c": 1}


def no_test_log_growth() -> int:
    p = FROZEN_ROOT / "experiments" / "test_eval_log.csv"
    return sum(1 for _ in open(p, "rb")) if p.exists() else -1


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
