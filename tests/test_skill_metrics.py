"""Tests de las métricas neutrales a la deriva (src/metrics/skill.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.backtest_daily import backtest_daily_signals, summarize_backtest
from src.metrics.skill import (
    buy_and_hold_sharpe,
    directional_accuracy_edge,
    naive_up_rate,
    pesaran_timmermann,
    sharpe_excess,
)


@pytest.fixture
def drifting_prices() -> np.ndarray:
    """Serie con deriva positiva clara (el caso que rompe DirAcc cruda)."""
    rng = np.random.default_rng(0)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0006, 0.012, 500)))


# ---------------------------------------------------------------
# naive_up_rate / directional_accuracy_edge
# ---------------------------------------------------------------
def test_naive_up_rate_cuenta_positivos():
    assert naive_up_rate(np.array([1.0, 1.0, -1.0, -1.0])) == 0.5
    assert naive_up_rate(np.array([1.0, 2.0, 3.0])) == 1.0
    assert naive_up_rate(np.array([-1.0, -2.0])) == 0.0


def test_naive_up_rate_vacio_es_nan():
    assert np.isnan(naive_up_rate(np.array([])))


def test_edge_de_predictor_siempre_positivo_es_cero():
    """El caso central: un modelo que predice 'sube' siempre no aporta nada.

    Su accuracy direccional cruda iguala exactamente la tasa de subida, así que su
    edge es 0 por más alta que sea esa accuracy. Es lo que se observó en 38 de 41
    ventanas del backfill de promociones.
    """
    y_true = np.array([0.1, 0.2, -0.05, 0.3, 0.1, -0.02])
    y_pred = np.full_like(y_true, 0.5)  # siempre positivo
    assert directional_accuracy_edge(y_true, y_pred) == pytest.approx(0.0)


def test_edge_de_predictor_perfecto_es_positivo():
    y_true = np.array([0.1, -0.2, 0.3, -0.4])
    assert directional_accuracy_edge(y_true, y_true) == pytest.approx(1.0 - 0.5)


def test_edge_ignora_no_finitos():
    y_true = np.array([0.1, np.nan, -0.2, 0.3])
    y_pred = np.array([0.1, 0.5, -0.2, np.inf])
    # Solo quedan las posiciones 0 y 2, ambas acertadas; up_rate = 0.5
    assert directional_accuracy_edge(y_true, y_pred) == pytest.approx(0.5)


def test_edge_rechaza_largos_distintos():
    with pytest.raises(ValueError):
        directional_accuracy_edge(np.array([1.0, 2.0]), np.array([1.0]))


# ---------------------------------------------------------------
# sharpe_excess
# ---------------------------------------------------------------
def test_sharpe_excess_de_estrategia_siempre_comprada_es_cero(drifting_prices):
    """Invariante que define la métrica: comprar y mantener da exactamente 0.

    Se construye la estrategia siempre-comprada con el motor real (tau muy bajo,
    sin costos, holding 1) para que el resultado no dependa de replicar la
    convención de alineación temporal a mano.
    """
    n = len(drifting_prices)
    ts = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    bt = backtest_daily_signals(
        timestamps=ts,
        close_prices=drifting_prices,
        pred_returns=np.full(n, 1.0),
        tau_buy=0.001,
        tau_sell=0.001,
        round_trip_bps=0.0,
        holding_period_days=1,
        allow_short=False,
        max_position=1.0,
    )
    strategy_sharpe = summarize_backtest(bt)["sharpe"]
    # Tolerancia float32: el motor de backtest acumula en float32.
    assert sharpe_excess(strategy_sharpe, drifting_prices) == pytest.approx(0.0, abs=1e-6)


def test_sharpe_excess_positivo_si_supera_al_buy_and_hold(drifting_prices):
    bh = buy_and_hold_sharpe(drifting_prices)
    assert sharpe_excess(bh + 0.5, drifting_prices) == pytest.approx(0.5)
    assert sharpe_excess(bh - 0.5, drifting_prices) == pytest.approx(-0.5)


def test_sharpe_excess_no_finito_es_nan(drifting_prices):
    assert np.isnan(sharpe_excess(float("nan"), drifting_prices))
    assert np.isnan(sharpe_excess(1.0, np.array([100.0])))          # serie muy corta
    assert np.isnan(sharpe_excess(1.0, np.array([100.0, -5.0])))    # precio no positivo


# ---------------------------------------------------------------
# Pesaran-Timmermann
# ---------------------------------------------------------------
def test_pt_predicciones_perfectas_rechazan_la_nula():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 300)
    stat, p = pesaran_timmermann(y, y)
    assert stat > 0
    assert p < 1e-6


def test_pt_calibrado_bajo_la_nula():
    """Con predicciones independientes, el p-valor debe ser ~uniforme.

    Es lo que separa a este test de la accuracy direccional cruda: el contrafáctico
    no es 0.5, sino la tasa de aciertos que dan las frecuencias marginales.
    """
    rng = np.random.default_rng(2)
    pvals = []
    for _ in range(600):
        # Target con deriva positiva: la accuracy cruda sería engañosamente alta.
        y = rng.normal(0.3, 1.0, 250)
        p = rng.normal(0.0, 1.0, 250)
        _, pv = pesaran_timmermann(y, p)
        if np.isfinite(pv):
            pvals.append(pv)

    pvals = np.array(pvals)
    assert len(pvals) > 500
    assert 0.35 < np.median(pvals) < 0.65
    assert np.mean(pvals < 0.05) < 0.12      # tamaño nominal 5%, margen por muestreo


def test_pt_no_definido_con_signo_constante():
    """Sin varianza en el signo predicho el test no aplica: nan, no un número inventado."""
    y = np.array([0.1, -0.2, 0.3, 0.4, -0.1])
    stat, p = pesaran_timmermann(y, np.ones_like(y))
    assert np.isnan(stat) and np.isnan(p)


def test_pt_muestra_insuficiente():
    assert all(np.isnan(v) for v in pesaran_timmermann(np.array([]), np.array([])))
    assert all(np.isnan(v) for v in pesaran_timmermann(np.array([1.0]), np.array([1.0])))
