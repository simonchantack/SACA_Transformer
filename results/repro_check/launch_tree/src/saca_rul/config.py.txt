"""YAML experiment configs with environment-expanded paths.

No path is hard-coded. A config value may contain ``${NAME}`` or ``${NAME:-default}``; ``NAME`` is read from the
environment, and a relative result is resolved against the release root (the folder that holds ``configs/``).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand(value: str, env: dict[str, str] | None = None) -> str:
    """Expand ``${NAME}`` / ``${NAME:-default}`` in ``value``. A missing name without default raises KeyError."""
    e = os.environ if env is None else env

    def sub(m: re.Match[str]) -> str:
        name, default = m.group(1), m.group(2)
        if name in e:
            return e[name]
        if default is None:
            raise KeyError(f"environment variable {name} is not set and has no default")
        return default

    return _VAR.sub(sub, value)


def resolve_path(value: str | Path, root: Path | None = None, env: dict[str, str] | None = None) -> Path:
    """Expand variables, then resolve a relative path against the release root."""
    p = Path(expand(str(value), env))
    return p if p.is_absolute() else ((root or ROOT) / p).resolve()


def _walk(obj: Any, fn: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _walk(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(v, fn) for v in obj]
    if isinstance(obj, str):
        return fn(obj)
    return obj


def load_config(path: str | Path, root: Path | None = None) -> dict[str, Any]:
    """Load a YAML experiment config and resolve its paths.

    Resolved to absolute paths: every string under ``paths``; ``protocol``; ``selection``; the values of
    ``selection_by_subset``; ``arm.tr_subsample.file``; ``arm.sensor_order``. ``${NAME}`` and ``${NAME:-default}`` are
    expanded first; relative results are resolved against the release root."""
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    cfg["_config_path"] = str(p.resolve())

    def res(v: str) -> str:
        return str(resolve_path(v, root))

    if "paths" in cfg:
        cfg["paths"] = _walk(cfg["paths"], res)
    for key in ("protocol", "selection"):
        if isinstance(cfg.get(key), str):
            cfg[key] = res(cfg[key])
    if isinstance(cfg.get("selection_by_subset"), dict):
        cfg["selection_by_subset"] = {k: res(v) for k, v in cfg["selection_by_subset"].items()}
    arm = cfg.get("arm") or {}
    if isinstance(arm.get("tr_subsample"), dict):
        arm["tr_subsample"]["file"] = res(arm["tr_subsample"]["file"])
    if isinstance(arm.get("sensor_order"), str):
        arm["sensor_order"] = res(arm["sensor_order"])
    cfg["arm"] = arm
    return cfg


def sha256_file(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
