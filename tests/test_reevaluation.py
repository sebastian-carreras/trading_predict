"""Tests for lifecycle.reevaluation (fair re-backtest building blocks).

Covers:
* recompute_metrics_on_window returns registry-style keys and sane values.
* load_walkforward_predictions round-trips a CSV.
* (conditional) predict_series reproduces a champion's own stored walk-forward
  y_pred on overlapping dates — the key inference-correctness check. Skipped
  when torch or the model/data artifacts are unavailable in the environment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.lifecycle import reevaluation


# ---------------------------------------------------------------
# recompute_metrics_on_window
# ---------------------------------------------------------------
def _synthetic_ohlcv(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="B", tz="UTC")
    # Random-walk close prices, strictly positive.
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=n)))
    return pd.DataFrame({"close": close}, index=dates)


def test_recompute_metrics_keys_and_finite():
    ohlcv = _synthetic_ohlcv()
    dates = ohlcv.index[-60:]
    # Predicted returns correlated with a realised target (so IC is defined).
    y_true = np.linspace(-0.1, 0.1, len(dates))
    pred = y_true + np.random.default_rng(1).normal(0, 0.01, len(dates))
    params = {
        "tau_buy": 0.02, "tau_sell": 0.0, "round_trip_bps": 10.0,
        "holding_period_days": 20, "allow_short": False, "max_position": 1.0,
    }
    m = reevaluation.recompute_metrics_on_window(ohlcv, dates, pred, y_true, params)

    # Registry-style keys the composite score consumes.
    for key in ("bt_sharpe", "bt_calmar", "ml_ic", "ml_directional_accuracy"):
        assert key in m, f"missing {key}"
    assert np.isfinite(m["ml_directional_accuracy"])
    assert 0.0 <= m["ml_directional_accuracy"] <= 1.0
    # Strongly correlated preds → positive IC.
    assert m["ml_ic"] > 0.5


def test_backtest_params_from_config_defaults():
    cfg = {
        "strategies": {
            "e1_conservative": {
                "horizon_days": 90,
                "thresholds": {"tau_buy": 0.03, "tau_sell": 0.01},
                "backtest": {"holding_period_days": 90, "allow_short": True},
            }
        },
        "costs": {"daily_round_trip_bps": 12.0},
    }
    p = reevaluation.backtest_params_from_config(cfg, "e1_conservative")
    assert p["tau_buy"] == 0.03
    assert p["round_trip_bps"] == 12.0
    assert p["holding_period_days"] == 90
    assert p["allow_short"] is True


# ---------------------------------------------------------------
# load_walkforward_predictions
# ---------------------------------------------------------------
def test_load_walkforward_predictions_roundtrip(tmp_path: Path):
    ticker = "TEST.BA"
    idx = pd.date_range("2025-01-01", periods=5, freq="B", tz="UTC")
    df = pd.DataFrame({"fold": 1, "y_true": [0.1, 0.2, 0.3, 0.4, 0.5],
                       "y_pred": [0.09, 0.19, 0.31, 0.39, 0.52]}, index=idx)
    df.index.name = "timestamp"
    df.to_csv(tmp_path / f"{ticker}_walkforward_predictions.csv")

    loaded = reevaluation.load_walkforward_predictions(tmp_path, ticker)
    assert loaded is not None
    assert list(loaded.columns) == ["fold", "y_true", "y_pred"]
    assert len(loaded) == 5
    assert isinstance(loaded.index, pd.DatetimeIndex)


def test_load_walkforward_predictions_missing(tmp_path: Path):
    assert reevaluation.load_walkforward_predictions(tmp_path, "NOPE") is None


# ---------------------------------------------------------------
# predict_series inference-correctness (conditional / integration)
# ---------------------------------------------------------------
def _iter_champion_artifacts():
    """Yield (strategy, ticker, run_dir, clean_csv) for champions with artifacts."""
    root = Path(__file__).resolve().parents[1]
    registry_path = root / "models" / "registry.json"
    if not registry_path.exists():
        return
    import json
    reg = json.loads(registry_path.read_text())
    for strat, sdata in reg.get("strategies", {}).items():
        if strat not in ("e1", "e2"):
            continue
        for ticker, entry in sdata.get("tickers", {}).items():
            champ = entry.get("champion")
            if not champ:
                continue
            run_dir = root / champ["run_dir"]
            model_files = list(run_dir.glob(f"{ticker}*_model.pth"))
            preds = run_dir / f"{ticker}_walkforward_predictions.csv"
            clean = root / "data" / "clean" / f"{ticker}_daily.csv"
            if model_files and preds.exists() and clean.exists():
                yield strat, ticker, run_dir, clean


def _first_compatible_champion():
    """First champion whose feature set the CURRENT code can still build, or None.

    Old champions trained with a since-removed feature set are skipped — the fair
    path handles them via graceful fallback, but the reproduction tests need a
    champion that actually reconstructs.
    """
    from src.lifecycle.loader import ModelLoader
    from src.lifecycle.registry import ModelRegistry
    root = Path(__file__).resolve().parents[1]
    registry = ModelRegistry(root / "models" / "registry.json")
    for strategy, ticker, run_dir, clean_csv in _iter_champion_artifacts():
        payload = ModelLoader(registry, root).load_champion(strategy, ticker)
        if payload is None:
            continue
        if strategy == "e1":
            from src.e1.train_pipeline import load_ohlcv_csv
        else:
            from src.e2.train_pipeline import load_ohlcv_csv
        ohlcv = load_ohlcv_csv(clean_csv)
        try:
            if reevaluation.predict_latest(strategy, payload, ohlcv) is not None:
                return strategy, ticker, run_dir, clean_csv, payload, ohlcv
        except Exception:
            continue
    return None


def test_predict_latest_on_real_champion_matches_predict_series():
    """predict_latest (no target) should produce a finite forecast whose as_of
    is at least as fresh as predict_series' newest (target-lagged) date."""
    pytest.importorskip("torch")
    found = _first_compatible_champion()
    if found is None:
        pytest.skip("no feature-compatible champion to reconstruct")
    strategy, ticker, run_dir, clean_csv, payload, ohlcv = found

    out = reevaluation.predict_latest(strategy, payload, ohlcv)
    assert out is not None
    pred, as_of = out
    assert np.isfinite(pred)

    # predict_series' newest date must be <= predict_latest's as_of (latter has no target lag).
    series = reevaluation.predict_series(strategy, payload, ohlcv)
    assert pd.Timestamp(as_of) >= series.index.max()


def test_predict_series_reproduces_stored_walkforward():
    pytest.importorskip("torch")
    found = _first_compatible_champion()
    if found is None:
        pytest.skip("no feature-compatible champion to reconstruct")
    strategy, ticker, run_dir, clean_csv, payload, ohlcv = found

    preds = pd.read_csv(run_dir / f"{ticker}_walkforward_predictions.csv",
                        index_col=0, parse_dates=True)
    preds.index = pd.DatetimeIndex(preds.index)

    series = reevaluation.predict_series(strategy, payload, ohlcv)

    # Compare on the LAST fold's dates (produced by the saved last-fold model).
    last_fold = preds["fold"].max()
    last_dates = preds.index[preds["fold"] == last_fold]
    common = series.index.intersection(last_dates)
    # Require a reasonable overlap to make the assertion meaningful.
    if len(common) < 10:
        pytest.skip(f"insufficient overlap to validate ({len(common)} dates)")

    stored = preds.loc[common, "y_pred"].to_numpy()
    recomputed = series.loc[common].to_numpy()
    # Frozen model + same scalers + causal features → near-exact reproduction.
    assert np.allclose(recomputed, stored, atol=1e-3), (
        f"max abs diff = {np.max(np.abs(recomputed - stored)):.4g}"
    )
