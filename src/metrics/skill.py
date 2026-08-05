"""Métricas de habilidad **neutrales a la deriva**.

Motivación
----------
``ml_directional_accuracy`` y ``bt_sharpe`` crudos premian que el activo haya subido,
no que el modelo haya acertado. Con horizontes largos el efecto es grande: el retorno
forward a 90 días de un activo con deriva positiva es positivo el 70-85% de las veces,
así que un modelo que emite signo positivo siempre saca una accuracy direccional de
0.70-0.85 sin ninguna capacidad predictiva.

Caso real del repo (``runs/e1_conservative/20260731_093628``): GGAL.BA fold 3 obtuvo
``directional_accuracy = 1.000`` con ``ic = 0.000`` — acertó el signo el 100% de las
veces con capacidad de ordenamiento nula.

Este módulo provee los contrafácticos contra los que hay que medir:

- ``naive_up_rate``      — qué saca "predecir siempre que sube".
- ``directional_accuracy_edge`` — cuánto le gana el modelo a ese naive.
- ``buy_and_hold_sharpe`` — qué saca "comprar y no hacer nada".
- ``sharpe_excess``      — cuánto le gana la estrategia a comprar y no hacer nada.
- ``pesaran_timmermann`` — si el edge direccional es estadísticamente distinguible del azar.

Referencias
-----------
- Pesaran, M. H. & Timmermann, A. (1992). "A Simple Nonparametric Test of Predictive
  Performance". *Journal of Business & Economic Statistics*, 10(4), 461-465.
- Bailey, D. H. & López de Prado, M. (2014). "The Deflated Sharpe Ratio".
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, cap. 4.
"""

from __future__ import annotations

import math

import numpy as np

from ..backtest.backtest_daily import compute_sharpe_ratio

__all__ = [
    "naive_up_rate",
    "directional_accuracy_edge",
    "buy_and_hold_returns",
    "buy_and_hold_sharpe",
    "sharpe_excess",
    "pesaran_timmermann",
]

_NAN = float("nan")


def naive_up_rate(y_true: np.ndarray) -> float:
    """Fracción de observaciones con retorno futuro positivo.

    Es la accuracy direccional que obtiene el predictor trivial "siempre sube".
    Cualquier modelo debe superarla para tener valor direccional.
    """
    y = np.asarray(y_true, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return _NAN
    return float(np.mean(y > 0))


def directional_accuracy_edge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy direccional del modelo **menos** la del naive "siempre sube".

    Positivo = el modelo aporta información direccional. Cero o negativo = no le
    gana a asumir que el activo sube siempre, por más alta que sea su accuracy cruda.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.size != p.size:
        raise ValueError(f"y_true y y_pred deben tener el mismo largo ({y.size} vs {p.size})")

    mask = np.isfinite(y) & np.isfinite(p)
    if not mask.any():
        return _NAN

    y, p = y[mask], p[mask]
    dir_acc = float(np.mean(np.sign(y) == np.sign(p)))
    return dir_acc - float(np.mean(y > 0))


def buy_and_hold_returns(close_prices: np.ndarray) -> np.ndarray:
    """Retornos diarios de una posición comprada y mantenida.

    Replica exactamente la convención de ``backtest_daily_signals``: el retorno
    imputado al día ``i`` es el del día ``i+1`` (la posición de hoy captura el
    movimiento de mañana), y el último día queda en 0 por no tener siguiente.
    Sin costos: es un benchmark de referencia, no una estrategia operable.
    """
    close = np.asarray(close_prices, dtype=float)
    n = close.size
    if n < 2:
        return np.zeros(max(n, 0), dtype=float)

    daily = np.diff(np.log(close))          # largo n-1: retorno de i -> i+1
    out = np.zeros(n, dtype=float)
    out[:-1] = daily                        # posición en i cobra el retorno de i+1
    return out


def buy_and_hold_sharpe(close_prices: np.ndarray, periods_per_year: int = 252) -> float:
    """Sharpe anualizado de comprar y mantener sobre la misma ventana."""
    close = np.asarray(close_prices, dtype=float)
    if close.size < 2 or not np.all(np.isfinite(close)) or np.any(close <= 0):
        return _NAN
    return compute_sharpe_ratio(buy_and_hold_returns(close), periods_per_year)


def sharpe_excess(
    strategy_sharpe: float,
    close_prices: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """Sharpe de la estrategia **menos** el de comprar y mantener en la misma ventana.

    Es la métrica primaria de promoción: responde "¿le gana a comprar y no hacer nada?"
    en las mismas unidades que el Sharpe. Una estrategia siempre-comprada sin costos da
    exactamente 0.
    """
    if strategy_sharpe is None or not np.isfinite(strategy_sharpe):
        return _NAN
    bh = buy_and_hold_sharpe(close_prices, periods_per_year)
    if not np.isfinite(bh):
        return _NAN
    return float(strategy_sharpe) - bh


def pesaran_timmermann(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Test no paramétrico de precisión direccional (Pesaran & Timmermann, 1992).

    Contrasta la tasa de aciertos observada contra la que se obtendría si signo
    predicho y signo real fueran independientes — que **no** es 0.5 cuando el activo
    tiene deriva, sino ``Py·Pz + (1-Py)·(1-Pz)``. Ésa es justamente la corrección que
    le falta a la accuracy direccional cruda.

    Retorna
    -------
    (statistic, p_value)
        ``statistic`` es N(0,1) bajo la hipótesis nula de no-predictibilidad.
        ``p_value`` es de una cola (probabilidad de observar este acierto o más por azar).
        ``(nan, nan)`` cuando el test no está definido (muestra vacía, o signo constante
        en las predicciones o en el target, que anula la varianza).

    Advertencia
    -----------
    El test asume independencia serial. Con etiquetas solapadas —el caso de E1 (H=90) y
    E2 (H=20), donde muestras consecutivas comparten casi toda su ventana— queda
    **sobredimensionado**: rechaza la nula más de lo que debería. Se reporta como métrica
    informativa, nunca como compuerta de promoción.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.size != p.size:
        raise ValueError(f"y_true y y_pred deben tener el mismo largo ({y.size} vs {p.size})")

    mask = np.isfinite(y) & np.isfinite(p)
    n = int(mask.sum())
    if n < 2:
        return _NAN, _NAN

    y, p = y[mask], p[mask]

    up_true = (y > 0).astype(float)
    up_pred = (p > 0).astype(float)

    hit = float(np.mean(np.sign(y) == np.sign(p)))   # P
    py = float(np.mean(up_true))                      # proporción de subidas reales
    pz = float(np.mean(up_pred))                      # proporción de subidas predichas

    # Tasa de aciertos esperada bajo independencia entre signo real y predicho.
    hit_indep = py * pz + (1.0 - py) * (1.0 - pz)     # P*

    var_hit = hit_indep * (1.0 - hit_indep) / n
    var_indep = (
        ((2.0 * py - 1.0) ** 2) * pz * (1.0 - pz) / n
        + ((2.0 * pz - 1.0) ** 2) * py * (1.0 - py) / n
        + 4.0 * py * pz * (1.0 - py) * (1.0 - pz) / (n * n)
    )

    denom = var_hit - var_indep
    if denom <= 0 or not np.isfinite(denom):
        # Ocurre cuando el signo es constante (py o pz en {0,1}): el test no aplica.
        return _NAN, _NAN

    stat = (hit - hit_indep) / math.sqrt(denom)
    # p de una cola: P(Z >= stat) = 0.5 * erfc(stat / sqrt(2))
    p_value = 0.5 * math.erfc(stat / math.sqrt(2.0))
    return float(stat), float(p_value)
