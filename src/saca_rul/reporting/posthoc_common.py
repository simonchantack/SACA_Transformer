"""Shared helpers for the post-freeze analyses (R2.10, R1.24, R0.23 + no-smoothing (c), no-smoothing (b)).

Port of ``experiments/final/code/posthoc_common.py``. Differences from the frozen helper, by design:
  * the frozen workspace root comes from ``SACA_WORKSPACE`` (default ``../workspace`` next to the release root);
  * final rows come from ``results/frozen/final_rows.json`` (sha256 checked against the frozen value);
  * ``refuse_if_logged`` and ``append_posthoc`` are inert: the release never writes to a test log. Each frozen analysis
    already holds its logged row (experiments/test_eval_log.csv rows 402-404); a re-run here reads the same saved
    files and adds nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from ..config import resolve_path

WS = str(resolve_path("${SACA_WORKSPACE:-../workspace}"))
FINAL_ROWS = str(resolve_path("results/frozen/final_rows.json"))
FINAL_ROWS_SHA = "deaed352e5d4b3d0753121fb783842b4949fa564b6be519989083350d272c53c"
SUBS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [0, 1, 2, 3, 4]


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def rel(p: str) -> str:
    return os.path.relpath(p, WS).replace("\\", "/")


def load_final_rows() -> tuple[dict[str, Any], dict[str, Any]]:
    s = sha256(FINAL_ROWS)
    if s != FINAL_ROWS_SHA:
        raise RuntimeError(f"final_rows.json sha256 {s} != frozen {FINAL_ROWS_SHA}")
    with open(FINAL_ROWS, encoding="utf-8") as f:
        fr = json.load(f)
    return {r["key"]: r for r in fr["rows"]}, fr


def pred_files(rows: dict[str, Any], key: str, sub: str) -> list[tuple[int, str, str]]:
    """[(seed, abs path, sha256)] for a final row, sha verified."""
    out = []
    for s in SEEDS:
        e = rows[key]["per_subset"][sub]["seeds"][str(s)]
        p = os.path.join(WS, e["pred_file"])
        h = sha256(p)
        if h != e["pred_sha256"]:
            raise RuntimeError(f"sha mismatch {p}")
        out.append((s, p, h))
    return out


def refuse_if_logged(experiment_id: str) -> None:
    """Inert in the release (no test log is written)."""
    return None


def append_posthoc(experiment_id: str, subset: str, reason: str, n_engines: int, script: str) -> None:
    """Inert in the release (no test log is written). Returns None, so the result JSON carries ``log_row: null``."""
    return None


def write_inputs(path: str, files: dict[str, str]) -> str:
    """Write the input-file hash list; returns the sha256 of the written file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(files, f, indent=1, sort_keys=True)
    return sha256(path)
