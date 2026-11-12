"""Tests: data pipeline (leakage, causal smoothing, windows), metrics, guard, config, and a smoke test of the full
pipeline (train -> checkpoint -> reload -> validation predictions) on a tiny synthetic subset. CPU only."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from saca_rul import data as D
from saca_rul import evaluation as EV
from saca_rul.config import ROOT, expand, load_config, resolve_path


# ----------------------------------------------------------------------------- smoothing
def test_ema_is_causal_and_matches_recursion() -> None:
    x = np.random.default_rng(0).normal(size=(50, 3))
    y = D.ema_1d(x)
    ref = np.empty_like(x)
    ref[0] = x[0]
    for t in range(1, 50):
        ref[t] = D.EMA_ALPHA * x[t] + (1 - D.EMA_ALPHA) * ref[t - 1]
    assert np.allclose(y, ref)
    x2 = x.copy()
    x2[30:] += 100
    assert np.allclose(D.ema_1d(x2)[:30], y[:30])  # the future does not change the past


def test_smoothing_rejects_unknown_and_unsorted() -> None:
    df = pd.DataFrame({"engine": [1, 1], "time": [2, 1], "a": [1.0, 2.0]})
    with pytest.raises(ValueError):
        D.ema_causal(df, ["a"])
    with pytest.raises(ValueError):
        D.apply_smoothing(df, ["a"], "gaussian")


# ----------------------------------------------------------------------------- pipeline
def test_pipeline_fits_on_tr_only(tiny_data: dict[str, Path]) -> None:
    cache = D.DataCache("FD001", 0, tiny_data["data_dir"], tiny_data["splits"])
    assert not set(cache.tr_ids) & set(cache.va_ids)
    assert all(rows <= {("train_file", e) for e in cache.tr_ids} for _, rows in cache.pipe.fit_log)
    assert cache.C == len(cache.features) > 3


def test_pipeline_refuses_foreign_rows(tiny_data: dict[str, Path]) -> None:
    raw = D.load_train_file(tiny_data["data_dir"], "FD001")
    pipe = D.Pipeline("FD001", 0, train_engines=[1, 2, 3])
    with pytest.raises(D.LeakageError):
        pipe.fit(raw[raw.engine.isin([1, 2, 3, 4])])
    bad = raw[raw.engine.isin([1, 2, 3])].copy()
    bad["source"] = "test_file"
    with pytest.raises(D.LeakageError):
        D.Pipeline("FD001", 0, train_engines=[1, 2, 3]).fit(bad)


def test_windows_labels_and_shapes(tiny_data: dict[str, Path]) -> None:
    cache = D.DataCache("FD001", 0, tiny_data["data_dir"], tiny_data["splits"])
    w = cache.windows(30)
    assert w["Xtr"].shape[1:] == (cache.C, 30) and w["Xtr"].dtype == np.float32
    assert len(w["ytr"]) == len(w["Xtr"]) and w["ytr"].max() <= D.RUL_CAP
    assert not set(w["etr"].tolist()) & set(w["eva"].tolist())
    assert (w["cva"][w["support"]] >= 51).all()


def test_nosmooth_differs_from_ema(tiny_data: dict[str, Path]) -> None:
    a = D.DataCache("FD001", 0, tiny_data["data_dir"], tiny_data["splits"], smoothing="ema").windows(30)["Xtr"]
    b = D.DataCache("FD001", 0, tiny_data["data_dir"], tiny_data["splits"], smoothing="none").windows(30)["Xtr"]
    assert a.shape == b.shape and not np.allclose(a, b)


def test_testing_windows_padding_rule() -> None:
    feats = ["f"]
    short = pd.DataFrame({"engine": 1, "time": range(1, 21), "f": np.arange(20.0)})
    X, eng, ce = D.testing_windows(short, feats, 30)
    assert X.shape == (1, 30, 1) and ce[0] == 20 and (X[0, 20:, 0] == 19).all()
    long = pd.DataFrame({"engine": 1, "time": range(1, 61), "f": np.arange(60.0)})
    X, _, ce = D.testing_windows(long, feats, 30)
    assert ce[0] == 59 and X[0, -1, 0] == 58  # ends one cycle before the last observation


def test_subsample_arm_checks(tiny_data: dict[str, Path]) -> None:
    with pytest.raises(RuntimeError):
        D.DataCache("FD001", 0, tiny_data["data_dir"], tiny_data["splits"], train_engines=[10])
    c = D.DataCache("FD001", 0, tiny_data["data_dir"], tiny_data["splits"], train_engines=[1, 2, 3, 4])
    assert c.tr_ids == [1, 2, 3, 4]


# ----------------------------------------------------------------------------- metrics (R0.6)
def test_score_constants_and_values() -> None:
    assert (EV.A_EARLY, EV.A_LATE) == (13.0, 10.0)
    s_late = EV.nasa_score([100.0], [110.0])["total"]
    s_early = EV.nasa_score([100.0], [90.0])["total"]
    assert s_late == pytest.approx(np.expm1(1.0)) and s_early == pytest.approx(np.expm1(10 / 13))
    assert s_late > s_early  # late errors are penalised more
    assert EV.nasa_score_code_constants([100.0], [110.0]) == pytest.approx(np.expm1(10 / 13))


def test_rmse_mae_r2() -> None:
    y, p = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0])
    assert EV.rmse(y, p) == pytest.approx(np.sqrt(4 / 3))
    assert EV.mae(y, p) == pytest.approx(2 / 3)
    assert EV.r2(y, p) == pytest.approx(1 - 4 / 2)
    with pytest.raises(ValueError):
        EV.rmse([1.0], [float("nan")])


# ----------------------------------------------------------------------------- guard
def test_guard_blocks_test_files_in_a_fresh_process() -> None:
    code = ("from saca_rul import guard; guard.install()\n"
            "import pandas as pd\n"
            "try:\n    pd.read_csv('x/test_FD001.txt')\nexcept guard.TestAccessError:\n    print('BLOCKED')\n"
            "try:\n    open('RUL_FD003.txt')\nexcept guard.TestAccessError:\n    print('BLOCKED')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                       env={"PYTHONPATH": str(ROOT / "src"), "SYSTEMROOT": "C:/Windows", "PATH": "",
                            "PYTHONDONTWRITEBYTECODE": "1"})
    assert r.stdout.split().count("BLOCKED") == 2, r.stderr


# ----------------------------------------------------------------------------- config
def test_expand_and_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    assert expand("${A:-x}/y", {}) == "x/y" and expand("${A:-x}", {"A": "z"}) == "z"
    with pytest.raises(KeyError):
        expand("${NOPE}", {})
    monkeypatch.setenv("SACA_DATA_DIR", str(ROOT))
    assert resolve_path("${SACA_DATA_DIR:-../workspace}") == ROOT


def test_all_configs_load_and_have_no_absolute_paths() -> None:
    files = sorted((ROOT / "configs").glob("exp_*.yaml"))
    assert len(files) >= 20
    for f in files:
        text = f.read_text(encoding="utf-8")
        assert ":\\" not in text and "C:/" not in text and "/Users/" not in text, f.name
        cfg = load_config(f)
        assert Path(cfg["protocol"]).exists() and Path(cfg["selection"]).exists(), f.name
        for v in (cfg.get("selection_by_subset") or {}).values():
            assert Path(v).exists()
        for k in ("sensor_order",):
            if cfg["arm"].get(k):
                assert Path(cfg["arm"][k]).exists()
        assert cfg["paths"]["test_log"].endswith("test_eval_log.csv")
        assert "workspace" not in cfg["paths"]["test_log"]  # the release never writes the project test log


# ----------------------------------------------------------------------------- smoke: full pipeline on a tiny subset
def _spec(tiny: dict[str, Path], protocol: Path, out: Path, family: str, params: dict[str, object], **kw: object):
    from saca_rul.training import RunSpec

    return RunSpec(experiment_id="smoke", family=family, subset="FD001", seed=0, params=dict(params),
                   protocol_path=protocol, data_dir=tiny["data_dir"], splits_path=tiny["splits"], out_dir=out,
                   device="cpu", max_epochs=2, **kw)  # type: ignore[arg-type]


@pytest.mark.parametrize("family", ["r3_D4", "dualdim"])
def test_smoke_training_checkpoint_reload(family: str, tiny_data: dict[str, Path], protocol_path: Path,
                                          tiny_d4_params: dict[str, object], tmp_path: Path) -> None:
    from saca_rul import checkpoint as CK
    from saca_rul.training import run_training

    meta = run_training(_spec(tiny_data, protocol_path, tmp_path, family, tiny_d4_params))
    run = tmp_path / "runs" / "FD001_seed0_smoke"
    for f in ("ckpt_rmse.pt", "history.csv", "preds_val.csv", "train.log", "train_meta.json"):
        assert (run / f).exists(), f
    assert meta["epochs_run"] == 2 and meta["smoke"] is True
    assert "strict deterministic" in (run / "train.log").read_text(encoding="utf-8")
    model, m2 = CK.load_for_eval(run / "ckpt_rmse.pt")
    w = D.DataCache("FD001", 0, tiny_data["data_dir"], tiny_data["splits"]).windows(30)
    pv = CK.predict(model, w["Xva"], 64)
    ref = pd.read_csv(run / "preds_val.csv")
    assert np.abs(pv - ref.y_pred.to_numpy()).max() < 1e-4
    assert m2["features"] == meta["features"]


def test_smoke_training_is_bitwise_repeatable_on_cpu(tiny_data: dict[str, Path], protocol_path: Path,
                                                     tiny_d4_params: dict[str, object], tmp_path: Path) -> None:
    from saca_rul.training import run_training, sha256_file

    for k in ("a", "b"):
        run_training(_spec(tiny_data, protocol_path, tmp_path / k, "r3_D4", tiny_d4_params))
    a = tmp_path / "a/runs/FD001_seed0_smoke/preds_val.csv"
    b = tmp_path / "b/runs/FD001_seed0_smoke/preds_val.csv"
    assert sha256_file(a) == sha256_file(b)


def test_smoke_lstm_ridge_and_sensor_order_arm(tiny_data: dict[str, Path], tmp_path: Path, protocol_path: Path,
                                               tiny_d4_params: dict[str, object]) -> None:
    from saca_rul.training import run_ridge, run_training

    r4 = ROOT / "configs" / "protocols" / "protocol_P2_r4.yaml"
    lstm = {"L": 30, "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 32, "hidden": 8, "layers": 1, "dropout": 0.1}
    assert run_training(_spec(tiny_data, r4, tmp_path / "l", "lstm", lstm))["epochs_run"] == 2
    rm = run_ridge(_spec(tiny_data, r4, tmp_path / "r", "ridge", {"L": 30, "alpha": 3.0}))
    assert rm["loader_check_maxdiff_vs_sklearn"] < 1e-6
    cache = D.DataCache("FD001", 0, tiny_data["data_dir"], tiny_data["splits"])
    perm = list(reversed(range(cache.C)))
    order = {"orders": {"FD001": {"0": {"split_seed": 0, "perm": perm, "original_features": cache.features,
                                        "permuted_features": [cache.features[i] for i in perm]}}}}
    of = tmp_path / "orders.json"
    of.write_text(json.dumps(order), encoding="utf-8")
    meta = run_training(_spec(tiny_data, protocol_path, tmp_path / "o", "r3_D4", tiny_d4_params,
                              arm={"sensor_order": str(of)}))
    assert meta["features"] == order["orders"]["FD001"]["0"]["permuted_features"]


def test_training_never_opens_the_test_file(tiny_data: dict[str, Path], protocol_path: Path,
                                            tiny_d4_params: dict[str, object], tmp_path: Path) -> None:
    from saca_rul import guard
    from saca_rul.training import run_training

    run_training(_spec(tiny_data, protocol_path, tmp_path, "r3_D4", tiny_d4_params))
    assert guard.is_installed() and guard.blocked_count() == 0
    with pytest.raises(guard.TestAccessError):
        open(tmp_path / "test_FD001.txt")


def test_evaluate_test_refuses_without_frozen_selection(tiny_data: dict[str, Path], tmp_path: Path) -> None:
    from saca_rul.evaluation import Refused, evaluate_test

    with pytest.raises(Refused):
        evaluate_test(tmp_path, "FD001", 0, tiny_data["data_dir"], tiny_data["splits"], tmp_path / "log.csv")
    assert not (tmp_path / "log.csv").exists()
