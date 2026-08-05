"""Tests for scripts.evaluation.leaderboard (live-signal leaderboard).

The regression these guard against: the leaderboard used to report each
champion's last WALK-FORWARD prediction, an artifact frozen at training time.
Rows were therefore dated anywhere from 30 to 185 days apart, which made
``--rank-by pred`` compare forecasts issued on different dates.
"""
from __future__ import annotations

import importlib

import pandas as pd
import pytest

leaderboard = importlib.import_module("scripts.evaluation.leaderboard")


CONFIG = {
    "strategies": {"e2_moderate": {"horizon_days": 20, "thresholds": {"tau_buy": 0.025}}},
    "reporting": {"max_signal_age_days": 5},
    "lifecycle": {"promotion": {"scoring_weights": {"bt_sharpe": 1.0}}},
}


class _FakeRegistry:
    def __init__(self, champions):
        self._champions = champions

    def get_champion(self, strategy, ticker):
        return self._champions.get((strategy, ticker))

    def list_all(self, strategy=None, stage=None):
        return [
            {"strategy": s, "ticker": t, "stage": "champion", **entry}
            for (s, t), entry in self._champions.items()
            if strategy is None or s == strategy
        ]


def _champ(sharpe: float, train_end: str) -> dict:
    return {
        "variant": "e2_moderate",
        "train_data_end": train_end,
        "run_dir": "runs/e2_moderate/20260226_101010",
        "metrics": {"bt_sharpe": sharpe, "ml_ic": 0.1},
    }


@pytest.fixture
def registry():
    return _FakeRegistry({
        ("e2", "AAA"): _champ(1.5, "2026-07-29"),
        ("e2", "BBB"): _champ(0.5, "2026-02-26"),
    })


@pytest.fixture
def live_signal(monkeypatch):
    """Todos los champions devuelven una señal fresca del mismo día."""
    def _signal(strategy, ticker, reg, config, *, root=None, today=None):
        return {
            "pred_return": 0.04 if ticker == "AAA" else 0.01,
            "pred_annualized": 0.5 if ticker == "AAA" else 0.12,
            "signal": "LONG" if ticker == "AAA" else "FLAT",
            "as_of": "2026-07-29",
            "data_end": "2026-07-29",
            "data_age_days": 1,
            "stale_data": False,
            "pred_mode": "live",
        }

    monkeypatch.setattr(leaderboard.signals, "champion_signal", _signal)


def test_leaderboard_never_reads_walkforward_csv(monkeypatch, registry, live_signal):
    """El CSV del run es una predicción de otra fecha: no debe tocarse."""
    def _boom(*a, **k):
        raise AssertionError("el leaderboard no debe leer el CSV walk-forward")

    monkeypatch.setattr(
        importlib.import_module("src.lifecycle.reevaluation"),
        "load_walkforward_predictions",
        _boom,
    )
    df = leaderboard.build_leaderboard(registry, CONFIG, ["e2"])
    assert len(df) == 2


def test_live_rows_share_as_of(registry, live_signal):
    """Lo que hace comparable el ranking: todas las señales vivas, del mismo día."""
    df = leaderboard.build_leaderboard(registry, CONFIG, ["e2"])
    active = df[df["signal"].notna()]
    assert len(active) == 2
    assert active["as_of"].nunique() == 1


def test_rank_by_pred_orders_by_annualized_forecast(registry, live_signal):
    df = leaderboard.build_leaderboard(registry, CONFIG, ["e2"], rank_by="pred")
    assert list(df["ticker"]) == ["AAA", "BBB"]
    assert list(df["rank"]) == [1, 2]


def test_rank_by_score_uses_training_quality(registry, live_signal):
    """SCORE sigue siendo calidad histórica: AAA gana por Sharpe, no por PRED%."""
    df = leaderboard.build_leaderboard(registry, CONFIG, ["e2"], rank_by="score")
    assert df.iloc[0]["ticker"] == "AAA"
    assert df.iloc[0]["bt_sharpe"] == 1.5


def test_model_staleness_and_data_staleness_are_separate(registry, monkeypatch):
    """BBB entrenó en febrero (modelo viejo) pero predice sobre datos de hoy."""
    def _signal(strategy, ticker, reg, config, *, root=None, today=None):
        return {
            "pred_return": 0.03, "pred_annualized": 0.38, "signal": "LONG",
            "as_of": "2026-07-29", "data_end": "2026-07-29", "data_age_days": 1,
            "stale_data": False, "pred_mode": "live",
        }

    monkeypatch.setattr(leaderboard.signals, "champion_signal", _signal)
    df = leaderboard.build_leaderboard(registry, CONFIG, ["e2"])
    bbb = df[df["ticker"] == "BBB"].iloc[0]
    assert bool(bbb["stale"]), "el modelo es viejo (entrenó en febrero)"
    assert not bool(bbb["stale_data"]), "pero los datos de la predicción son de hoy"


def test_unavailable_rows_carry_no_signal(registry, monkeypatch):
    def _signal(strategy, ticker, reg, config, *, root=None, today=None):
        return leaderboard.signals.unavailable_signal()

    monkeypatch.setattr(leaderboard.signals, "champion_signal", _signal)
    df = leaderboard.build_leaderboard(registry, CONFIG, ["e2"])
    assert df["signal"].isna().all()
    assert (df["pred_mode"] == "unavailable").all()
    # No debe romper el render.
    leaderboard.print_leaderboard(leaderboard.add_deltas(df), top=None)


def test_print_leaderboard_renders_stale_rows(registry, monkeypatch, capsys):
    def _signal(strategy, ticker, reg, config, *, root=None, today=None):
        return {
            "pred_return": 0.11, "pred_annualized": 1.4, "signal": None,
            "as_of": "2026-07-15", "data_end": "2026-07-15", "data_age_days": 15,
            "stale_data": True, "pred_mode": "live",
        }

    monkeypatch.setattr(leaderboard.signals, "champion_signal", _signal)
    df = leaderboard.add_deltas(leaderboard.build_leaderboard(registry, CONFIG, ["e2"]))
    leaderboard.print_leaderboard(df, top=None)
    out = capsys.readouterr().out
    assert "2026-07-15" in out, "AS_OF debe ser visible"
    assert "15d" in out, "la antigüedad de los datos debe ser visible"
    assert "max_signal_age_days" in out, "debe explicar por qué no hay señal"
