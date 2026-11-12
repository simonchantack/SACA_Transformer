"""Leakage-safe C-MAPSS data pipeline (R0.5, R0.10, R0.13, R1.29, R1.37, R2.4, R2.7, R3.5).

Every learned step (constant drop, k-means regimes, per-regime statistics, correlation pruning, scaler) is fitted on
the training engines (TR) of one split seed only and then applied to TR, validation engines (VA) and the official test
file (TE) without refitting. Smoothing is the causal EMA (past-only, alpha = 0.24723) or none.
Training and validation windows use the submitted convention: label = RUL at the cycle after the window.

Nothing in this module opens a test file. ``testing_windows`` only reshapes rows that ``evaluation.evaluate_test``
passes in.
"""
from __future__ import annotations

import json
import math
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

EMA_ALPHA = 0.24723
COLS = ["engine", "time", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"sm_{i}" for i in range(1, 22)]
OPS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSORS = [f"sm_{i}" for i in range(1, 22)]
MULTI = ("FD002", "FD004")
FIXED_DROP_MULTI = ["sm_1", "sm_5", "sm_18", "sm_19"]
RUL_CAP = 125


class LeakageError(AssertionError):
    """A fit step received rows outside the training engines."""


# ------------------------------------------------------------------------------------------------ smoothing
def ema_1d(x: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    """Causal EMA of a 1-D (or (T, C)) array along axis 0 with y_0 = x_0."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] == 0:
        return x.copy()
    zi = (1.0 - alpha) * x[0:1]
    y, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], x, axis=0, zi=zi)
    return np.asarray(y)


def ema_causal(df: pd.DataFrame, features: Sequence[str], alpha: float = EMA_ALPHA, group: str = "engine",
               time: str = "time") -> pd.DataFrame:
    """Per-engine causal EMA of ``features``. Asserts that ``time`` strictly increases within each engine."""
    out = df.copy()
    feats = list(features)
    arr = out[feats].to_numpy(dtype=np.float64).copy()
    tarr = out[time].to_numpy() if time in out.columns else None
    for eng, idx in out.groupby(group, sort=False).indices.items():
        if tarr is not None:
            t = tarr[idx]
            if len(t) > 1 and not np.all(np.diff(t) > 0):
                raise ValueError(f"time not strictly increasing within engine {eng}")
        arr[idx] = ema_1d(arr[idx], alpha)
    out[feats] = arr
    return out


def apply_smoothing(df: pd.DataFrame, features: Sequence[str], method: str, alpha: float = EMA_ALPHA) -> pd.DataFrame:
    if method == "ema":
        return ema_causal(df, features, alpha=alpha)
    if method == "none":
        return df.copy()
    raise ValueError(f"unknown smoothing method {method!r} (allowed: ema, none)")


# ------------------------------------------------------------------------------------------------ files and splits
def read_cmapss_txt(path: str | Path) -> pd.DataFrame:
    """Read a C-MAPSS text file (space separated, trailing blanks)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.ParserWarning)
        df = pd.read_csv(path, sep=" ", header=None, names=COLS, index_col=False)
    return df


def load_train_file(data_dir: str | Path, subset: str) -> pd.DataFrame:
    df = read_cmapss_txt(Path(data_dir) / f"train_{subset}.txt")
    df["source"] = "train_file"
    return df


def load_splits(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def split_engines(splits: dict[str, Any], subset: str, seed: int) -> tuple[list[int], list[int]]:
    s = splits["subsets"][subset]["splits"][str(seed)]
    return list(s["train"]), list(s["val"])


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rul_uncapped"] = df.groupby("engine")["time"].transform("max") - df["time"]
    df["rul"] = np.minimum(df["rul_uncapped"], RUL_CAP)
    return df


def _keys(df: pd.DataFrame) -> set[tuple[str, int]]:
    return set(zip(df["source"].astype(str), df["engine"].astype(int), strict=True))


# ------------------------------------------------------------------------------------------------ pipeline
class Pipeline:
    """``fit(df_TR)`` then ``transform(df)`` for TR, VA or TE. ``transform`` never refits."""

    def __init__(self, subset: str, seed: int, smoothing: str = "ema", alpha: float = EMA_ALPHA,
                 legacy_corr: bool = False, corr_threshold: float = 0.95,
                 train_engines: Iterable[int] | None = None, n_clusters: int = 6) -> None:
        if smoothing not in ("ema", "none"):
            raise ValueError("smoothing must be 'ema' or 'none'")
        self.subset = subset
        self.seed = int(seed)
        self.smoothing = smoothing
        self.alpha = float(alpha)
        self.legacy_corr = bool(legacy_corr)
        self.corr_threshold = float(corr_threshold)
        self.train_engines = None if train_engines is None else {int(e) for e in train_engines}
        self.n_clusters = n_clusters
        self.multi = subset in MULTI
        self.fit_log: list[tuple[str, frozenset[tuple[str, int]]]] = []
        self.fitted = False
        self.features: list[str] = []
        self.kmeans: KMeans
        self.scaler: StandardScaler
        self.mu: pd.DataFrame
        self.sd: pd.DataFrame
        self.constant_dropped: list[str] = []
        self.sensors_nonconst: list[str] = []
        self.ops_surviving: list[str] = []
        self.corr_pairs: list[tuple[str, str]] = []
        self.corr_dropped: list[str] = []

    def _check_fit_rows(self, df: pd.DataFrame, step: str) -> None:
        if "source" not in df.columns:
            raise LeakageError(f"{step}: rows have no source tag")
        if not (df["source"] == "train_file").all():
            raise LeakageError(f"{step}: non-training-file rows passed to fit")
        if self.train_engines is not None:
            bad = set(df["engine"].astype(int)) - self.train_engines
            if bad:
                raise LeakageError(f"{step}: {len(bad)} engines outside TR passed to fit, e.g. {sorted(bad)[:5]}")
        self.fit_log.append((step, frozenset(_keys(df))))

    def _fit_kmeans(self, df: pd.DataFrame) -> None:
        self._check_fit_rows(df, "kmeans")
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.seed).fit(df[OPS].to_numpy())

    def _fit_constant(self, df: pd.DataFrame) -> None:
        self._check_fit_rows(df, "constant_drop")
        cand = OPS + SENSORS
        std = df[cand].std()
        self.constant_dropped = [c for c in cand if abs(float(std[c])) < 0.02]
        self.sensors_nonconst = [c for c in SENSORS if c not in self.constant_dropped]
        self.ops_surviving = [c for c in OPS if c not in self.constant_dropped]

    def _fit_cluster_stats(self, df: pd.DataFrame, clusters: np.ndarray) -> None:
        self._check_fit_rows(df, "cluster_stats")
        sub = df[self.sensors_nonconst].copy()
        sub["OperCluster"] = clusters
        self.mu = sub.groupby("OperCluster")[self.sensors_nonconst].mean()
        self.sd = sub.groupby("OperCluster")[self.sensors_nonconst].std(ddof=0).replace(0, 1)

    def _cluster_norm(self, df: pd.DataFrame, clusters: np.ndarray) -> np.ndarray:
        x = df[self.sensors_nonconst].to_numpy(dtype=np.float64).copy()
        missing = set(np.unique(clusters)) - set(self.mu.index)
        if missing:
            raise ValueError(f"cluster ids {missing} not seen in TR")
        mu = self.mu.loc[clusters].to_numpy()
        sd = self.sd.loc[clusters].to_numpy()
        return np.asarray((x - mu) / sd)

    def _fit_corr(self, frame: pd.DataFrame) -> None:
        """``frame``: TR rows, columns in submitted order; must contain ``rul_uncapped``."""
        self._check_fit_rows(frame, "corr_select")
        if self.legacy_corr:
            cols = [c for c in frame.columns if c not in ("source", "rul")]
        else:
            cols = [c for c in frame.columns if c in SENSORS] + ["rul_uncapped"]
        corr = frame[cols].rename(columns={"rul_uncapped": "rul"}).corr()
        mask = np.tril(np.ones(corr.shape), k=-1).astype(bool)
        corr = corr.where(mask)
        high: list[tuple[str, str]] = []
        for col in corr.columns:
            for row in corr.index:
                if abs(corr.loc[col, row]) > self.corr_threshold:
                    if not self.legacy_corr and ("rul" in (col, row)):
                        continue
                    high.append((col, row))
        drop = []
        for a, b in high:
            if abs(corr.loc["rul", a]) > abs(corr.loc["rul", b]):  # type: ignore[operator, arg-type]
                drop.append(b)
            else:
                drop.append(a)
        self.corr_pairs = high
        self.corr_dropped = drop

    def _fit_scaler(self, df: pd.DataFrame) -> None:
        self._check_fit_rows(df, "standard_scaler")
        self.scaler = StandardScaler().fit(df[self.features].to_numpy(dtype=np.float64))

    def fit(self, df_tr: pd.DataFrame) -> Pipeline:
        df = df_tr.copy()
        self._check_fit_rows(df, "fit")
        df = add_rul(df)
        clusters = np.zeros(0)
        if self.multi:
            self._fit_kmeans(df)
            clusters = self.kmeans.predict(df[OPS].to_numpy())
        self._fit_constant(df)
        if self.multi:
            self._fit_cluster_stats(df, clusters)
            xn = self._cluster_norm(df, clusters)
            frame = pd.DataFrame(xn, columns=self.sensors_nonconst, index=df.index)
            frame.insert(0, "OperCluster", clusters)
            frame.insert(0, "engine", df["engine"].to_numpy())
            frame["rul_uncapped"] = df["rul_uncapped"].to_numpy()
            frame["source"] = df["source"].to_numpy()
            self._fit_corr(frame)
            dropped = set(FIXED_DROP_MULTI) | set(self.corr_dropped)
            self.features = [c for c in self.sensors_nonconst if c not in dropped]
        else:
            frame = df[["engine", "time"] + self.ops_surviving + self.sensors_nonconst + ["rul_uncapped", "source"]]
            self._fit_corr(frame)
            self.features = [c for c in self.sensors_nonconst if c not in set(self.corr_dropped)]
            self._fit_scaler(df)
        self.fitted = True
        return self

    def transform(self, df_in: pd.DataFrame, with_labels: bool = True) -> pd.DataFrame:
        """Apply the fitted state. Returns engine, time, features (and rul if ``with_labels``)."""
        assert self.fitted, "fit first"
        df = df_in.copy()
        if self.multi:
            clusters = self.kmeans.predict(df[OPS].to_numpy())
            xn = self._cluster_norm(df, clusters)
            full = pd.DataFrame(xn, columns=self.sensors_nonconst, index=df.index)
            x = full[self.features].to_numpy()
        else:
            x = self.scaler.transform(df[self.features].to_numpy(dtype=np.float64))
        out = pd.DataFrame(x, columns=self.features, index=df.index)
        out.insert(0, "time", df["time"].to_numpy())
        out.insert(0, "engine", df["engine"].to_numpy())
        if with_labels:
            out["rul"] = add_rul(df[["engine", "time"]])["rul"].to_numpy()
        out = apply_smoothing(out, self.features, self.smoothing, self.alpha)
        return out.reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ windows
def training_windows(df: pd.DataFrame, features: Sequence[str], L: int
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows: per engine with T rows, windows i = 0..T-L-1, X = rows[i:i+L], y = rul[i+L].

    Returns X (N, L, C) float32, y (N,), engine ids (N,), label cycle (N,)."""
    Xs, ys, es, cs = [], [], [], []
    arr = df[list(features)].to_numpy(dtype=np.float32)
    rul = df["rul"].to_numpy(dtype=np.float32)
    tim = df["time"].to_numpy()
    eng = df["engine"].to_numpy()
    for _e, idx in df.groupby("engine", sort=False).indices.items():
        T = len(idx)
        n = T - L
        if n <= 0:
            continue
        a = arr[idx]
        w = np.lib.stride_tricks.sliding_window_view(a, (L, a.shape[1]))[:n, 0]
        Xs.append(w)
        ys.append(rul[idx][L:L + n])
        es.append(np.full(n, int(eng[idx][0])))
        cs.append(tim[idx][L:L + n])
    return (np.concatenate(Xs).astype(np.float32), np.concatenate(ys), np.concatenate(es), np.concatenate(cs))


def testing_windows(df: pd.DataFrame, features: Sequence[str], L: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One window per test engine, ending at the second-to-last cycle (submitted convention, R0.23).

    Engines shorter than L: all rows, padded with the last row. Returns X (N, L, C), engine ids, cycle_end."""
    Xs, es, ce = [], [], []
    for e, g in df.groupby("engine", sort=False):
        a = g[list(features)].to_numpy(dtype=np.float64)
        T = len(a)
        start = (T - 1) - L
        if T < L:
            w = np.vstack([a, np.repeat(a[-1:], L - T, axis=0)])
            end_cycle = int(g["time"].iloc[-1])
        elif start < 0:
            w = a[0:L]
            end_cycle = int(g["time"].iloc[L - 1])
        else:
            w = a[start:start + L]
            end_cycle = int(g["time"].iloc[start + L - 1])
        Xs.append(w)
        es.append(int(e))  # type: ignore[call-overload,arg-type]
        ce.append(end_cycle)
    return np.stack(Xs).astype(np.float32), np.array(es), np.array(ce)


class DataCache:
    """Fit the leakage-safe pipeline once per (subset, split seed) on TR rows, then cache windows per L.

    Preprocessing does not depend on L, so one fit serves every L in the search.
    ``train_engines`` replaces the TR engine list (the nested training-size arm); it must be a subset of TR and
    disjoint from VA. ``tr_fraction < 1`` is for dry runs only and is never reported."""

    def __init__(self, subset: str, split_seed: int, data_dir: str | Path, splits_path: str | Path,
                 smoothing: str = "ema", alpha: float = EMA_ALPHA, legacy_corr: bool = False,
                 tr_fraction: float = 1.0, support_min_label_cycle: int = 51,
                 train_engines: Sequence[int] | None = None) -> None:
        self.subset, self.split_seed = subset, int(split_seed)
        splits = load_splits(splits_path)
        raw = load_train_file(data_dir, subset)
        tr_ids, va_ids = split_engines(splits, subset, split_seed)
        if train_engines is not None:
            keep = [int(e) for e in train_engines]
            if not set(keep) <= {int(e) for e in tr_ids}:
                raise RuntimeError("sub-sample contains engines outside TR")
            if set(keep) & {int(e) for e in va_ids}:
                raise RuntimeError("sub-sample overlaps VA")
            tr_ids = keep
        assert not (set(tr_ids) & set(va_ids)), "engine overlap TR/VA"
        tr = raw[raw.engine.isin(tr_ids)].reset_index(drop=True)
        va = raw[raw.engine.isin(va_ids)].reset_index(drop=True)
        self.pipe = Pipeline(subset, split_seed, smoothing=smoothing, alpha=alpha, legacy_corr=legacy_corr,
                             train_engines=tr_ids).fit(tr)
        self.tr_t, self.va_t = self.pipe.transform(tr), self.pipe.transform(va)
        if tr_fraction < 1.0:
            keep2 = sorted(set(tr_ids))[: max(1, int(math.ceil(tr_fraction * len(tr_ids))))]
            self.tr_t = self.tr_t[self.tr_t.engine.isin(keep2)].reset_index(drop=True)
        self.tr_ids, self.va_ids = tr_ids, va_ids
        self.features: list[str] = list(self.pipe.features)
        self.support_min_label_cycle = int(support_min_label_cycle)
        self._win: dict[int, dict[str, Any]] = {}

    @property
    def C(self) -> int:
        return len(self.features)

    def apply_sensor_order(self, perm: Sequence[int], original: Sequence[str], permuted: Sequence[str]) -> None:
        """Permute the window columns (and, through ``features``, the sensor ids) after the pipeline fit."""
        if list(self.features) != list(original):
            raise ValueError(f"pipeline features {self.features} != recorded {list(original)}")
        self.features = [self.features[i] for i in perm]
        if self.features != list(permuted):
            raise ValueError("permuted feature list differs from the recorded order")
        self._win = {}

    def windows(self, L: int) -> dict[str, Any]:
        L = int(L)
        if L not in self._win:
            Xtr, ytr, etr, ctr = training_windows(self.tr_t, self.features, L)
            Xva, yva, eva, cva = training_windows(self.va_t, self.features, L)
            sup = cva >= self.support_min_label_cycle
            self._win[L] = dict(Xtr=np.ascontiguousarray(Xtr.transpose(0, 2, 1)), ytr=ytr.astype(np.float32), etr=etr,
                                ctr=ctr, Xva=np.ascontiguousarray(Xva.transpose(0, 2, 1)), yva=yva, eva=eva, cva=cva,
                                support=sup)
        return self._win[L]


def support_keys(w: dict[str, Any]) -> set[tuple[int, int]]:
    """(engine, label cycle) pairs on the common support."""
    m = w["support"]
    return set(zip(w["eva"][m].tolist(), w["cva"][m].tolist(), strict=True))
