"""
Construcción de features para E2 (Estrategia Moderada).

Horizonte: 20 días | Lookback: 60 días | 12 features

Categorías:
- Retornos (2): ret_1d, ret_20d
- Volatilidad (2): vol_20d, atr_14
- Volumen (1): vol_ratio_20d
- Tendencia (2): sma50_sma200_ratio, close_sma50_dist
- Momentum (2): macd_hist, rsi_14
- Bollinger (1): bb_pct_b
- Fuerza tendencia (1): adx_14
- Riesgo/asimetría (1): skew_ret_20d

Desactivados para entrenamientos nuevos tras EDA §2.2 (correlación) — ver
strategies.e2_moderate.features.active en src/config/base.yaml, que es la fuente de
verdad de qué se usa para entrenar HOY. Esta función sigue calculando el catálogo
completo (incluidas estas) porque algún champion vivo puede depender de ellas — ver
src/lifecycle/reevaluation.py. Nunca borrar el cálculo de una feature mientras exista
un modelo vivo (models/registry.json) que la use; sacarla solo de features.active:
- ret_5d, ret_10d: redundantes con ret_1d/ret_20d (r≈0.7-0.85)
- close_sma200_dist: redundante con sma50_sma200_ratio + close_sma50_dist (r≈0.78)
- bb_bandwidth: redundante con vol_20d (r≈0.85-0.95)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.macro import align_exog as _align_exog  # alineado exógeno PIT-safe (compartido)


def compute_e2_features(df: pd.DataFrame, exog: pd.DataFrame | None = None) -> pd.DataFrame:
    """Calcula las 12 features base de E2 (+ exógenas opcionales).

    Args:
        df: DataFrame con OHLCV diario (index=timestamp, cols: open/high/low/close/volume)
        exog: DataFrame exógeno opcional (macro/cross-asset) de src.data.macro.load_exog_for.
            Si se pasa, se agregan las features de los Bloques A/B/C/D (ver docs/FEATURES.md),
            alineadas por fecha con ffill (PIT-safe). Si es None, se devuelven solo las 12
            base — comportamiento idéntico al histórico (compatibilidad hacia atrás).

    Returns:
        DataFrame de features (mismo índice que df).
    """
    out = pd.DataFrame(index=df.index)

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    volume = df["volume"].astype("float64")

    # ── Retornos (2 features) ──────────────────────────────────────────

    log_close = np.log(close)

    # ret_1d: Retorno logarítmico diario
    # Para H=20, el retorno diario es informativo (a diferencia de H=90 donde es ruido)
    out["ret_1d"] = log_close.diff()

    # Desactivada para entrenamientos nuevos (ver strategies.e2_moderate.features.active en
    # src/config/base.yaml): redundante con ret_1d y ret_20d — ventanas solapadas generan
    # r≈0.7-0.85. Se sigue calculando acá porque algún champion vivo todavía puede depender
    # de ella — nunca borrar esta línea, solo sacarla de features.active en config.
    out["ret_5d"] = out["ret_1d"].rolling(5).sum()

    # Desactivada para entrenamientos nuevos (ver strategies.e2_moderate.features.active en
    # src/config/base.yaml): redundante con ret_1d y ret_20d — ventanas solapadas generan
    # r≈0.7-0.85. Se sigue calculando acá porque algún champion vivo todavía puede depender
    # de ella — nunca borrar esta línea, solo sacarla de features.active en config.
    out["ret_10d"] = out["ret_1d"].rolling(10).sum()

    # ret_20d: Momentum mensual — directamente alineado con target H=20
    out["ret_20d"] = out["ret_1d"].rolling(20).sum()

    # ── Volatilidad (2 features) ──────────────────────────────────────

    # vol_20d: Volatilidad realizada (std retornos 20d)
    # Mide riesgo de corto plazo, alineado con horizonte
    out["vol_20d"] = out["ret_1d"].rolling(20).std()

    # atr_14: Average True Range normalizado
    # Volatilidad "real" considerando gaps (más robusto que high-low)
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean() / close

    # ── Volumen (1 feature) ───────────────────────────────────────────

    # vol_ratio_20d: Ratio volumen actual / SMA(volume, 20)
    # >1 = volumen por encima del promedio, <1 = por debajo
    # Ventana 20d alineada con horizonte H=20 (más reactivo que z-score 60d)
    sma_vol_20 = volume.rolling(20).mean()
    out["vol_ratio_20d"] = volume / (sma_vol_20 + 1e-12)

    # ── Tendencia (2 features) ────────────────────────────────────────

    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()

    # sma50_sma200_ratio: Golden/Death Cross — fuerza de tendencia LP
    # >0 → Golden Cross (alcista), <0 → Death Cross (bajista)
    out["sma50_sma200_ratio"] = (sma_50 / sma_200) - 1.0

    # close_sma50_dist: Desviación del precio vs tendencia de medio plazo
    # Más reactiva que SMA(200), adecuada para H=20
    out["close_sma50_dist"] = (close / sma_50) - 1.0

    # Desactivada para entrenamientos nuevos (ver strategies.e2_moderate.features.active en
    # src/config/base.yaml): redundante con sma50_sma200_ratio + close_sma50_dist (r≈0.78).
    # Precedente: E1 tiene la misma feature, mismo motivo. Se sigue calculando acá porque
    # algún champion vivo todavía puede depender de ella — nunca borrar esta línea, solo
    # sacarla de features.active en config.
    out["close_sma200_dist"] = (close / sma_200) - 1.0

    # ── Momentum (2 features) ─────────────────────────────────────────

    # MACD Histograma — señal compacta de momentum
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = macd_line - macd_signal

    # RSI(14) — sobrecompra/sobreventa (diferenciador clave de E2 vs E1)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # ── Bollinger Bands (1 feature) ───────────────────────────────────

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20

    # bb_pct_b: Posición relativa del precio en las bandas (0-1 normal)
    # >1 → sobrecompra extrema, <0 → sobreventa extrema
    out["bb_pct_b"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-12)

    # Desactivada para entrenamientos nuevos (ver strategies.e2_moderate.features.active en
    # src/config/base.yaml): redundante con vol_20d — ambos miden dispersión 20d, r≈0.85-0.95.
    # Se sigue calculando acá porque algún champion vivo todavía puede depender de ella —
    # nunca borrar esta línea, solo sacarla de features.active en config.
    out["bb_bandwidth"] = (bb_upper - bb_lower) / sma20

    # ── Fuerza de Tendencia (1 feature) ───────────────────────────────

    # ADX(14) — fuerza de tendencia (no dirección)
    # <20 → sin tendencia, 20-40 → moderada, >40 → fuerte
    high_diff = high.diff()
    low_diff = -low.diff()

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    plus_dm[(high_diff > low_diff) & (high_diff > 0)] = high_diff
    minus_dm[(low_diff > high_diff) & (low_diff > 0)] = low_diff

    atr = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr + 1e-12))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr + 1e-12))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12))
    out["adx_14"] = dx.rolling(14).mean()

    # ── Riesgo / Asimetría (1 feature) ────────────────────────────────

    # skew_ret_20d: Asimetría de retornos en ventana de 20 días
    # Skewness = 0 — perfectamente simétrica, como la normal (aqui no pasa)
    # Skewness > 0 — cola más larga hacia la derecha, hay más valores extremos positivos (rallies grandes)
    # Skewness < 0 — cola más larga hacia la izquierda, hay más valores extremos negativos (crashes)

    out["skew_ret_20d"] = out["ret_1d"].rolling(20).skew()

    # ── Features exógenas (macro / cross-asset) — Fase 1, Bloques A/B/C/D ──────
    # Solo si se pasó `exog` (data.exog.enabled). Ver src/data/macro.py y docs/FEATURES.md.
    # Alineación por fecha con ffill (PIT-safe): cada fila usa el último valor exógeno
    # conocido en o antes de esa fecha (no mira al futuro).
    if exog is not None and len(getattr(exog, "columns", [])):
        from ..data.macro import EXOG_HELPER_COLS  # import perezoso: evita costo/ciclo

        aligned = _align_exog(exog, out.index)
        helpers = set(EXOG_HELPER_COLS)

        # Bloques A/C/D — passthrough de features globales (mismas para todo ticker)
        for col in aligned.columns:
            if col not in helpers:
                out[col] = aligned[col]

        # Bloque B — fuerza relativa sectorial (per-ticker), derivada de las helpers
        # + los retornos propios del ticker (ret_1d/ret_20d ya calculados arriba).
        if {"sector_ret20", "spy_ret20"}.issubset(aligned.columns):
            out["rel_strength_sector"] = out["ret_20d"] - aligned["sector_ret20"]
            out["sector_rotation"] = aligned["sector_ret20"] - aligned["spy_ret20"]
        if "spy_ret1d" in aligned.columns:
            cov = out["ret_1d"].rolling(60).cov(aligned["spy_ret1d"])
            var = aligned["spy_ret1d"].rolling(60).var()
            out["beta_60"] = cov / (var + 1e-12)

    return out


def make_target_e2(df: pd.DataFrame, horizon_days: int = 20) -> pd.Series:
    """Construye target: retorno logarítmico acumulado forward a H días.

    Args:
        df: DataFrame con columna 'close'
        horizon_days: Horizonte de predicción (20 días por defecto para E2)

    Returns:
        Series con target (log-retorno acumulado forward)
    """
    close = df["close"].astype("float64")
    fwd_close = close.shift(-horizon_days)
    target = np.log(fwd_close / close)
    target.name = f"target_ret_{horizon_days}d"
    return target
