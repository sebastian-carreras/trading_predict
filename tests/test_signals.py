"""Tests for lifecycle.signals (live "what do I invest in today" track).

The point of this module is that a signal reflects TODAY's data, so these tests
focus on the freshness contract rather than on the numerics of the forecast
(which ``test_reevaluation.py`` already covers):

* stale data degrades the signal to None but still reports the prediction
* an unservable champion yields ``pred_mode="unavailable"``, never a fallback
  to the run's walk-forward CSV (that would be a prediction from another date)
* the buy threshold comes from the champion's own variant
* an explicit ``root`` reaches the exogenous-cache loader
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.lifecycle import signals


CONFIG = {
    "strategies": {
        "e2_moderate": {"horizon_days": 20, "thresholds": {"tau_buy": 0.025}},
        "e1_conservative": {"horizon_days": 90, "thresholds": {"tau_buy": 0.05}},
    },
    "reporting": {"max_signal_age_days": 5},
}


class _FakeRegistry:
    """Minimal stand-in for ModelRegistry: only what signals.py touches."""

    def __init__(self, champions: dict[tuple[str, str], dict]):
        self._champions = champions

    def get_champion(self, strategy: str, ticker: str):
        return self._champions.get((strategy, ticker))

    def list_all(self, strategy=None, stage=None):
        return [
            {"strategy": s, "ticker": t, "stage": "champion", **entry}
            for (s, t), entry in self._champions.items()
            if strategy is None or s == strategy
        ]


def _ohlcv(last: str, n: int = 10) -> pd.DataFrame:
    idx = pd.date_range(end=pd.Timestamp(last), periods=n, freq="B")
    return pd.DataFrame({"close": [100.0] * n}, index=idx)


@pytest.fixture
def registry():
    return _FakeRegistry({("e2", "AAA"): {"variant": "e2_moderate"}})


def _patch(monkeypatch, *, ohlcv, forecast):
    monkeypatch.setattr(signals.reevaluation, "load_clean_ohlcv", lambda t, root=None: ohlcv)
    monkeypatch.setattr(
        signals.reevaluation,
        "champion_latest_forecast",
        lambda s, t, r, o, root=None: forecast,
    )


def test_fresh_data_emits_signal(monkeypatch, registry):
    _patch(monkeypatch, ohlcv=_ohlcv("2026-07-29"), forecast=(0.04, pd.Timestamp("2026-07-29")))
    row = signals.champion_signal(
        "e2", "AAA", registry, CONFIG, today=date(2026, 7, 30)
    )
    assert row["pred_mode"] == "live"
    assert row["stale_data"] is False
    assert row["data_age_days"] == 1
    assert row["as_of"] == "2026-07-29"
    # 0.04 > tau_buy 0.025 → LONG; anualizado a 20 días = 0.04 * 252/20
    assert row["signal"] == "LONG"
    assert row["pred_annualized"] == pytest.approx(0.04 * 252 / 20, rel=1e-6)


def test_below_threshold_is_flat_not_long(monkeypatch, registry):
    _patch(monkeypatch, ohlcv=_ohlcv("2026-07-29"), forecast=(0.01, pd.Timestamp("2026-07-29")))
    row = signals.champion_signal("e2", "AAA", registry, CONFIG, today=date(2026, 7, 30))
    assert row["signal"] == "FLAT"


def test_threshold_comes_from_the_champion_variant(monkeypatch):
    """0.04 is a LONG under e2 (tau .025) but only FLAT under e1 (tau .05)."""
    reg = _FakeRegistry({("e1", "AAA"): {"variant": "e1_conservative"}})
    _patch(monkeypatch, ohlcv=_ohlcv("2026-07-29"), forecast=(0.04, pd.Timestamp("2026-07-29")))
    row = signals.champion_signal("e1", "AAA", reg, CONFIG, today=date(2026, 7, 30))
    assert row["signal"] == "FLAT"


def test_stale_data_degrades_signal_but_keeps_prediction(monkeypatch, registry):
    """Datos de 15 días: se muestra PRED% pero NO se recomienda comprar."""
    _patch(monkeypatch, ohlcv=_ohlcv("2026-07-15"), forecast=(0.11, pd.Timestamp("2026-07-15")))
    row = signals.champion_signal("e2", "AAA", registry, CONFIG, today=date(2026, 7, 30))
    assert row["data_age_days"] == 15
    assert row["stale_data"] is True
    assert row["signal"] is None, "no se emite señal sobre datos rancios"
    assert row["pred_return"] == 0.11, "la predicción se sigue reportando"
    assert row["pred_mode"] == "live"


def test_age_exactly_at_threshold_is_not_stale(monkeypatch, registry):
    _patch(monkeypatch, ohlcv=_ohlcv("2026-07-25"), forecast=(0.04, pd.Timestamp("2026-07-25")))
    row = signals.champion_signal("e2", "AAA", registry, CONFIG, today=date(2026, 7, 30))
    assert row["data_age_days"] == 5
    assert row["stale_data"] is False
    assert row["signal"] == "LONG"


def test_unservable_champion_never_falls_back_to_walkforward(monkeypatch, registry):
    """Sin inferencia no hay número: el CSV del run es de otra fecha, no de hoy."""
    _patch(monkeypatch, ohlcv=_ohlcv("2026-07-29"), forecast=None)
    row = signals.champion_signal("e2", "AAA", registry, CONFIG, today=date(2026, 7, 30))
    assert row["pred_mode"] == "unavailable"
    assert row["pred_return"] is None
    assert row["signal"] is None
    assert row["data_end"] == "2026-07-29", "se reporta qué datos había, aunque no se pudo inferir"


def test_missing_ohlcv_is_unavailable(monkeypatch, registry):
    _patch(monkeypatch, ohlcv=None, forecast=(0.04, pd.Timestamp("2026-07-29")))
    row = signals.champion_signal("e2", "AAA", registry, CONFIG, today=date(2026, 7, 30))
    assert row["pred_mode"] == "unavailable"


def test_no_champion_returns_none(registry):
    assert signals.champion_signal("e2", "NOPE", registry, CONFIG) is None


def test_build_signals_covers_every_champion(monkeypatch):
    reg = _FakeRegistry({
        ("e2", "AAA"): {"variant": "e2_moderate"},
        ("e2", "BBB"): {"variant": "e2_moderate"},
    })
    _patch(monkeypatch, ohlcv=_ohlcv("2026-07-29"), forecast=(0.04, pd.Timestamp("2026-07-29")))
    rows = signals.build_signals("e2", reg, CONFIG, today=date(2026, 7, 30))
    assert {r["ticker"] for r in rows} == {"AAA", "BBB"}
    # Todas las señales vivas comparten as_of: eso es lo que hace comparable el ranking.
    assert len({r["as_of"] for r in rows}) == 1


def test_root_reaches_the_exog_loader(monkeypatch, registry, tmp_path):
    """El root explícito debe gobernar también la carga del cache exógeno."""
    seen: dict[str, Path | None] = {}

    def _load(t, root=None):
        seen["ohlcv_root"] = root
        return _ohlcv("2026-07-29")

    monkeypatch.setattr(signals.reevaluation, "load_clean_ohlcv", _load)

    def _forecast(s, t, r, o, root=None):
        seen["forecast_root"] = root
        return (0.04, pd.Timestamp("2026-07-29"))

    monkeypatch.setattr(signals.reevaluation, "champion_latest_forecast", _forecast)
    signals.champion_signal("e2", "AAA", registry, CONFIG, root=tmp_path, today=date(2026, 7, 30))
    assert seen["ohlcv_root"] == tmp_path
    assert seen["forecast_root"] == tmp_path


def test_max_signal_age_days_falls_back_to_default():
    assert signals.max_signal_age_days({}) == signals.DEFAULT_MAX_SIGNAL_AGE_DAYS
    assert signals.max_signal_age_days({"reporting": {"max_signal_age_days": 2}}) == 2
