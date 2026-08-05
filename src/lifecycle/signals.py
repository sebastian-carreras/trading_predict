"""Track de señales: qué dice HOY el champion congelado sobre el mercado.

Este módulo responde una pregunta distinta a la del lifecycle. El lifecycle
(``promotion.py``) pregunta *"¿apareció un modelo mejor que el champion?"* y para
eso entrena un candidato y lo compara en una ventana OOS común. Acá se pregunta
*"¿qué predice el champion que YA tenemos, con los datos de hoy?"* — que es lo
que hace falta para decidir una inversión.

Por qué NO se re-entrena
------------------------
Re-entrenar sobre todos los datos hasta hoy produciría un modelo *distinto* del
que se evaluó: sus métricas OOS ya no aplicarían y dejaría de ser "el champion".
El split y el walk-forward existen para **evaluar**, no para predecir. Acá se
corre el modelo congelado hacia adelante sobre la ventana más reciente, que es
inferencia honesta y cuesta ~0.13 s por ticker.

La inferencia vive en ``reevaluation.predict_latest``; este módulo solo le agrega
la capa de decisión: anualización, umbral de señal y control de frescura.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils import get_nested
from . import reevaluation
from .registry import ModelRegistry

# Días de trading por año, para anualizar el retorno del horizonte.
TRADING_DAYS = 252
# Si los datos que sostienen la predicción superan esta edad, la señal se degrada.
DEFAULT_MAX_SIGNAL_AGE_DAYS = 5

# Campos de un registro sin predicción utilizable. ``pred_mode="unavailable"``
# distingue "no se pudo inferir" de "se infirió y dio FLAT".
_UNAVAILABLE: dict[str, Any] = {
    "pred_return": None,
    "pred_annualized": None,
    "signal": None,
    "as_of": None,
    "data_end": None,
    "data_age_days": None,
    "stale_data": True,
    "pred_mode": "unavailable",
}


def unavailable_signal() -> dict[str, Any]:
    """Registro vacío con ``pred_mode="unavailable"`` (mismas claves que una señal)."""
    return dict(_UNAVAILABLE)


def max_signal_age_days(config: dict[str, Any]) -> int:
    """Umbral de frescura (días corridos) desde ``reporting.max_signal_age_days``."""
    return int(
        get_nested(
            config,
            ["reporting", "max_signal_age_days"],
            default=DEFAULT_MAX_SIGNAL_AGE_DAYS,
        )
    )


def champion_signal(
    strategy: str,
    ticker: str,
    registry: ModelRegistry,
    config: dict[str, Any],
    *,
    root: Path | None = None,
    today: date | None = None,
) -> dict[str, Any] | None:
    """Señal de hoy del champion de ``(strategy, ticker)``, o None si no hay champion.

    El retorno predicho es al horizonte de la estrategia; ``pred_annualized`` lo
    escala a un año. La señal usa el mismo ``tau_buy`` que el backtest, para que
    lo que se muestra coincida con lo que se midió.

    Si los datos que sostienen la predicción son más viejos que
    ``max_signal_age_days``, ``signal`` se degrada a None y ``stale_data`` queda
    en True: la predicción se sigue mostrando, pero no se emite una recomendación
    sobre datos rancios. Nunca se cae al CSV walk-forward del run — esa es una
    predicción de otra fecha, no una señal de hoy.
    """
    champ = registry.get_champion(strategy, ticker)
    if champ is None:
        return None

    variant = champ.get("variant") or ""
    strat_cfg = config.get("strategies", {}).get(variant, {})
    horizon = int(strat_cfg.get("horizon_days", 90))
    tau_buy = float(strat_cfg.get("thresholds", {}).get("tau_buy", 0.02))

    ohlcv = reevaluation.load_clean_ohlcv(ticker, root=root)
    if ohlcv is None or ohlcv.empty:
        return unavailable_signal()

    data_end = pd.Timestamp(ohlcv.index[-1]).date()
    forecast = reevaluation.champion_latest_forecast(
        strategy, ticker, registry, ohlcv, root=root
    )
    if forecast is None:
        # Sin torch, sin model.pth, o drift de features: no hay señal servible.
        return {**_UNAVAILABLE, "data_end": str(data_end)}

    pred_return, as_of_ts = forecast
    as_of = pd.Timestamp(as_of_ts).date()
    today = today or datetime.now(timezone.utc).date()
    age = (today - as_of).days
    stale = age > max_signal_age_days(config)

    return {
        "pred_return": round(float(pred_return), 6),
        "pred_annualized": round(float(pred_return) * (TRADING_DAYS / horizon), 6),
        "signal": None if stale else ("LONG" if pred_return > tau_buy else "FLAT"),
        "as_of": str(as_of),
        "data_end": str(data_end),
        "data_age_days": age,
        "stale_data": stale,
        "pred_mode": "live",
    }


def build_signals(
    strategy: str,
    registry: ModelRegistry,
    config: dict[str, Any],
    *,
    tickers: list[str] | None = None,
    root: Path | None = None,
    today: date | None = None,
) -> list[dict[str, Any]]:
    """Señales de hoy para todos los champions de una estrategia.

    El universo sale del registry (``stage="champion"``), no de una lista en
    config: un champion recién promovido entra solo.
    """
    wanted = set(tickers) if tickers else None
    rows: list[dict[str, Any]] = []
    for champ in registry.list_all(strategy=strategy, stage="champion"):
        ticker = champ.get("ticker")
        if wanted is not None and ticker not in wanted:
            continue
        record = champion_signal(
            strategy, ticker, registry, config, root=root, today=today
        )
        if record is None:
            continue
        rows.append({
            "strategy": strategy,
            "ticker": ticker,
            "variant": champ.get("variant"),
            **record,
        })
    return rows
