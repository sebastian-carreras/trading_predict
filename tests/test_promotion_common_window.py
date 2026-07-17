"""Tests for the fair champion-vs-candidate comparison on a common OOS window.

The frozen-model inference (torch) is mocked so these run anywhere: we inject
champion predictions and the candidate's walk-forward predictions directly and
exercise the window logic, guardrails, identity guard, and recent_metrics
persistence in ``evaluate_and_promote`` / ``compare_on_common_window``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.lifecycle import reevaluation
from src.lifecycle.promotion import evaluate_and_promote
from src.lifecycle.registry import ModelRegistry


DATES = pd.date_range("2025-01-01", periods=100, freq="B", tz="UTC")
METRICS = {"bt_sharpe": 1.2, "ml_ic": 0.05, "ml_directional_accuracy": 0.55, "bt_calmar": 1.1}


@pytest.fixture
def full_config() -> dict:
    return {
        "strategies": {
            "e1_conservative": {
                "horizon_days": 90,
                "thresholds": {"tau_buy": 0.0, "tau_sell": 0.0},
                "backtest": {"holding_period_days": 20, "allow_short": False, "max_position": 1.0},
            }
        },
        "costs": {"daily_round_trip_bps": 10.0},
    }


@pytest.fixture
def promo_cfg() -> dict:
    return {
        "min_improvement": 0.0,
        "require_positive_sharpe": False,   # keep structural tests deterministic
        "fair_window": True,
        "min_eval_samples": 30,
        "scoring_weights": {"bt_sharpe": 0.35, "ml_ic": 0.25,
                            "ml_directional_accuracy": 0.20, "bt_calmar": 0.20},
    }


def _ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=len(DATES))))
    return pd.DataFrame({"close": close}, index=DATES)


def _make_registry(tmp_path: Path, champ_train_end: str | None,
                   cand_train_end: str = "2025-05-01",
                   cand_variant: str = "e1_conservative") -> ModelRegistry:
    reg = ModelRegistry(tmp_path / "registry.json")
    # Champion first (register + promote).
    reg.register_candidate(
        strategy="e1", ticker="AAA", run_dir="runs/champ", metrics=dict(METRICS),
        variant="e1_conservative", train_data_end=champ_train_end,
    )
    reg.promote_to_champion("e1", "AAA", reason="setup")
    # Fresh candidate.
    reg.register_candidate(
        strategy="e1", ticker="AAA", run_dir="runs/cand", metrics=dict(METRICS),
        variant=cand_variant, train_data_end=cand_train_end,
    )
    return reg


def _patch_predictions(monkeypatch, wf_index):
    """Inject champion + candidate walk-forward predictions over ``wf_index``."""
    n = len(wf_index)
    y_true = np.linspace(-0.05, 0.05, n)
    cand_wf = pd.DataFrame(
        {"fold": 1, "y_true": y_true, "y_pred": y_true},  # candidate = perfect
        index=wf_index,
    )
    champ_series = pd.Series(-y_true, index=wf_index, name="y_pred")  # champion = anti-skill

    monkeypatch.setattr(reevaluation, "champion_predictions",
                        lambda *a, **k: champ_series)
    monkeypatch.setattr(reevaluation, "load_walkforward_predictions",
                        lambda *a, **k: cand_wf)


def test_fair_window_runs_and_persists_recent_metrics(tmp_path, monkeypatch, promo_cfg, full_config):
    # Champion cutoff at index 50 → 39 OOS dates in DATES[:90] (> min_eval_samples).
    cutoff = str(DATES[50].date())
    reg = _make_registry(tmp_path, champ_train_end=cutoff)
    _patch_predictions(monkeypatch, DATES[:90])

    d = evaluate_and_promote(
        reg, "e1", "AAA", promo_cfg,
        ohlcv_loader=lambda t: _ohlcv(), full_config=full_config,
    )

    assert d.comparison_mode == "fair_window"
    expected_n = int((DATES[:90].tz_localize(None).normalize()
                      > pd.Timestamp(cutoff)).sum())
    assert d.eval_n_samples == expected_n
    assert d.eval_window_start is not None and d.eval_window_end is not None

    # Whoever is champion now carries recomputed recent_metrics on the same window.
    champ = reg.get_champion("e1", "AAA")
    assert "recent_metrics" in champ
    assert champ["recent_metrics_n_samples"] == expected_n


def test_identity_guard_skips_self_comparison(tmp_path, monkeypatch, promo_cfg, full_config):
    same = str(DATES[60].date())
    # Candidate identical to champion: same variant + same train_data_end.
    reg = _make_registry(tmp_path, champ_train_end=same, cand_train_end=same)
    _patch_predictions(monkeypatch, DATES[:90])

    d = evaluate_and_promote(
        reg, "e1", "AAA", promo_cfg,
        ohlcv_loader=lambda t: _ohlcv(), full_config=full_config,
    )
    assert d.comparison_mode == "identity_skip"
    assert d.promoted is False
    assert "recent_metrics" not in reg.get_champion("e1", "AAA")


def test_insufficient_evidence_keeps_champion(tmp_path, monkeypatch, promo_cfg, full_config):
    # Champion cutoff very recent → tiny OOS window (< min_eval_samples).
    cutoff = str(DATES[95].date())
    reg = _make_registry(tmp_path, champ_train_end=cutoff)
    _patch_predictions(monkeypatch, DATES[:90])

    d = evaluate_and_promote(
        reg, "e1", "AAA", promo_cfg,
        ohlcv_loader=lambda t: _ohlcv(), full_config=full_config,
    )
    assert d.comparison_mode == "insufficient_evidence"
    assert d.promoted is False


def test_champion_without_train_data_end_falls_back_to_stored(tmp_path, monkeypatch, promo_cfg, full_config):
    reg = _make_registry(tmp_path, champ_train_end=None)
    _patch_predictions(monkeypatch, DATES[:90])

    d = evaluate_and_promote(
        reg, "e1", "AAA", promo_cfg,
        ohlcv_loader=lambda t: _ohlcv(), full_config=full_config,
    )
    # Fair window unavailable → real stored comparison (not a forced keep).
    assert d.comparison_mode == "stored"
    assert d.candidate_score > 0  # stored metrics actually scored
