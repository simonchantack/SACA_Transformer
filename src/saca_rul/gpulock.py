"""Single-GPU lock (project protocol): a file holding ``experiment_id, pid, start_time``.

The lock path comes from ``SACA_GPU_LOCK`` (default ``results/gpu.lock``). ``acquire`` waits while a live process
holds the lock and removes a stale lock; ``release`` removes the lock only if this process owns it.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import resolve_path


def lock_path() -> Path:
    return resolve_path("${SACA_GPU_LOCK:-results/gpu.lock}")


def _alive(pid: int | None) -> bool:
    if pid is None:
        return False
    import psutil

    return bool(psutil.pid_exists(pid))


def holder() -> dict[str, Any] | None:
    try:
        txt = lock_path().read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    m = re.search(r",\s*(\d+)\s*,", txt)
    return {"text": txt.strip(), "pid": int(m.group(1)) if m else None}


def acquire(experiment_id: str, poll: float = 60, log: Callable[[str], None] = print, max_wait_h: float = 24) -> bool:
    t0 = time.time()
    p = lock_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    while True:
        h = holder()
        if h is not None and not _alive(h["pid"]):
            log(f"gpu.lock stale ({h['text']}); removed")
            p.unlink(missing_ok=True)
            continue
        if h is None:
            try:
                fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"{experiment_id}, {os.getpid()}, {dt.datetime.now().isoformat(timespec='seconds')}\n")
            return True
        log(f"gpu.lock held by: {h['text']}; polling every {poll}s")
        if time.time() - t0 > max_wait_h * 3600:
            raise TimeoutError("gpu.lock wait exceeded")
        time.sleep(poll)


def release() -> None:
    h = holder()
    if h is not None and h["pid"] == os.getpid():
        lock_path().unlink(missing_ok=True)
