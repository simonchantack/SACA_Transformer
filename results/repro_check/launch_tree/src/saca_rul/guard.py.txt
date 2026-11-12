"""Runtime guard: training, tuning and selection code must never open the test split (R0.3, R2.5).

``install()`` patches ``builtins.open``, ``io.open``, ``pandas.read_csv/read_table/read_fwf`` and
``numpy.loadtxt/genfromtxt``. Any path that matches ``test_FD<d>`` or ``RUL_FD<d>`` raises ``TestAccessError``.
Only ``evaluation.evaluate_test`` and the logged post-hoc analyses run without it.
"""
from __future__ import annotations

import builtins
import io
import re
from collections.abc import Callable
from typing import Any

PATTERN = re.compile(r"(test_FD\d|RUL_FD\d)", re.IGNORECASE)
_STATE: dict[str, Any] = {"installed": False, "blocked": 0}


class TestAccessError(RuntimeError):
    __test__ = False


def _check(path: Any) -> None:
    try:
        p = path.decode(errors="ignore") if isinstance(path, bytes) else str(path)
    except Exception:  # noqa: BLE001
        return
    if PATTERN.search(p):
        _STATE["blocked"] += 1
        raise TestAccessError(f"test-split access blocked by guard: {p}")


def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
    def inner(file: Any, *a: Any, **k: Any) -> Any:
        if isinstance(file, (str, bytes)) or hasattr(file, "__fspath__"):
            _check(file)
        return fn(file, *a, **k)

    inner.__wrapped_by_guard__ = True  # type: ignore[attr-defined]
    return inner


def install() -> bool:
    """Install the guard once per process. Returns True."""
    if _STATE["installed"]:
        return True
    import numpy as np
    import pandas as pd

    builtins.open = _wrap(builtins.open)
    io.open = builtins.open
    for name in ("read_csv", "read_table", "read_fwf"):
        setattr(pd, name, _wrap(getattr(pd, name)))
    for name in ("loadtxt", "genfromtxt"):
        setattr(np, name, _wrap(getattr(np, name)))
    _STATE["installed"] = True
    return True


def is_installed() -> bool:
    return bool(_STATE["installed"])


def blocked_count() -> int:
    return int(_STATE["blocked"])
