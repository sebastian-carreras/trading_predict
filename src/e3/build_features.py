from __future__ import annotations

import numpy as np
import pandas as pd

# NOTA (aplicada hoy a E1/E2, ver src/e1/build_features.py y src/e2/build_features.py):
# E3 no llega hoy al camino de re-evaluación fair-window (evaluate_and_promote se llama
# sin ohlcv_loader/full_config en src/e3/train_pipeline.py y en el DAG de E3, y
# reevaluation._strategy_kit no soporta "e3"), así que este archivo NO tiene todavía el
# bug de feature-drift que rompía RECENT en E1/E2. Pero el mismo patrón de features
# comentadas/eliminadas está presente acá — si en el futuro se rehabilita E3 (hoy "needs
# rework", ver PORTFOLIO.md) y se conecta el fair-window, aplicar la misma disciplina:
# nunca borrar el cálculo de una feature mientras un modelo vivo la use en su
# feature_names; solo excluirla de un features.active en base.yaml (strategies.e3_intraday).


def compute_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute minimal intraday features from OHLCV 5m bars.

    Input: index must be timestamp; cols: open/high/low/close/volume
    Output: dataframe aligned with input index.
    """

    out = pd.DataFrame(index=df.index)

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    volume = df["volume"].astype("float64")

    # Returns
    log_close = pd.Series(np.log(close.to_numpy()), index=close.index)
    out["ret_1"] = log_close.diff()

    # Rolling stats
    out["ret_12m"] = out["ret_1"].rolling(12).sum()  # ~1h
    out["vol_24"] = out["ret_1"].rolling(24).std()  # ~2h
    # ELIMINADA: vol_96 — correlación con vol_24: r=0.778 (EDA, sección 3.2).
    # Ambas miden la misma dimensión de volatilidad realizada; vol_24 es más sensible
    # a cambios de régimen intradiario (2h vs 1 día). Se conserva como variable local
    # únicamente para calcular vol_ratio, que captura la relación entre ambas escalas.
    _vol_96 = out["ret_1"].rolling(96).std()  # variable intermedia para vol_ratio

    # Range / ATR proxy
    out["range_pct"] = (high - low) / close
    # ELIMINADA: atr_14 — correlaciones: vol_24↔atr_14 r=0.881, vol_96↔atr_14 r=0.740 (EDA, sección 3.2).
    # El ATR en escala intradiaria (14 barras ≈ 70 min) mide esencialmente la misma
    # dimensión que vol_24 y vol_96. Dado que range_pct ya captura el rango de la barra
    # actual, atr_14 resulta redundante. La correlación doble con ambas medidas de
    # volatilidad realizada lo convierte en el candidato más claro a eliminar.
    # tr = pd.concat(
    #     [
    #         (high - low),
    #         (high - close.shift(1)).abs(),
    #         (low - close.shift(1)).abs(),
    #     ],
    #     axis=1,
    # ).max(axis=1)
    # out["atr_14"] = tr.rolling(14).mean() / close

    # Volume z-score
    vol_roll = volume.rolling(96)
    out["vol_z_96"] = (volume - vol_roll.mean()) / (vol_roll.std() + 1e-12)

    # Time-of-day (cyclical)
    # Use UTC timestamps; if you want exchange-local time, convert before calling.
    idx = pd.DatetimeIndex(df.index)
    minutes = idx.hour * 60 + idx.minute
    out["tod_sin"] = np.sin(2 * np.pi * minutes / (24 * 60))
    out["tod_cos"] = np.cos(2 * np.pi * minutes / (24 * 60))

    # Day-of-week (cyclical) — captures Mon/Fri effects
    out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 5)
    out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 5)

    # Intraday VWAP deviation — (close - VWAP_daily) / VWAP_daily
    # VWAP is reset each calendar day using cumsum within the day
    typical_price = (high + low + close) / 3
    tp_vol = typical_price * volume
    dates = idx.normalize()
    cumvol = volume.groupby(dates).cumsum()
    cumtpvol = tp_vol.groupby(dates).cumsum()
    vwap = cumtpvol / (cumvol + 1e-12)
    out["vwap_dev"] = (close - vwap) / (vwap + 1e-12)

    # Volatility ratio: short-term vs long-term (regime detection)
    # Nota: usa _vol_96 local (no out["vol_96"]) porque esa feature fue eliminada.
    out["vol_ratio"] = out["vol_24"] / (_vol_96 + 1e-12)

    # RSI-14 (Wilder smoothing via EWM)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    alpha = 1.0 / 14
    avg_gain = gain.ewm(alpha=alpha, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    out["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # ELIMINADA: macd_hist — correlación con ret_12m: r=0.822 (EDA, sección 3.2).
    # MACD(12,26,9) sobre barras de 5-min produce EMA de 60/130/45 min, períodos que
    # solapan conceptualmente con el momentum acumulado de 1h (ret_12m). Además, MACD
    # fue diseñado para datos diarios: sus parámetros estándar pierden significado
    # económico en escala intradiaria. ret_12m es más directo e interpretable.
    # ema12 = close.ewm(span=12, adjust=False).mean()
    # ema26 = close.ewm(span=26, adjust=False).mean()
    # macd_line = ema12 - ema26
    # macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    # out["macd_hist"] = (macd_line - macd_signal) / (close + 1e-12)

    # ELIMINADA: bb_pct_b — correlación con rsi_14: r=0.853 (EDA, sección 3.2).
    # Tanto BB %B como RSI capturan sobrecompra/sobreventa: BB %B mide la posición del
    # precio relativa a bandas de desviación estándar, mientras RSI mide fuerza relativa
    # de ganancias vs pérdidas. En datos intradiarios ambos convergen hacia la misma
    # señal. RSI se conserva por ser más estándar y no depender de la elección de sigma.
    # sma20 = close.rolling(20).mean()
    # std20 = close.rolling(20).std()
    # bb_upper = sma20 + 2.0 * std20
    # bb_lower = sma20 - 2.0 * std20
    # out["bb_pct_b"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-12)

    # ELIMINADA: close_sma78_dist — correlación con vwap_dev: r=0.803 (EDA, sección 3.2).
    # Ambas miden la desviación del precio respecto de una referencia del día completo
    # (SMA de 78 barras ≈ 6.5h vs VWAP diario). vwap_dev es la referencia intraday
    # canónica porque pondera por volumen y se resetea naturalmente cada sesión,
    # reflejando mejor el precio de equilibrio real que una media simple.
    # sma78 = close.rolling(78).mean()
    # out["close_sma78_dist"] = (close / (sma78 + 1e-12)) - 1.0

    return out


def make_target_return(df: pd.DataFrame, horizon_bars: int) -> pd.Series:
    close = df["close"].astype("float64")
    fwd = close.shift(-horizon_bars)
    y = pd.Series(np.log((fwd / close).to_numpy(dtype="float64")), index=close.index)
    y.name = f"target_ret_fwd_{horizon_bars}"
    return y


def make_sequences(
    features: pd.DataFrame,
    target: pd.Series,
    lookback_bars: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    """Convert feature matrix to LSTM sequences.

    Returns:
      X: (N, lookback_bars, n_features)
      y: (N,)
      ts: timestamps for each sample (aligned to the end of the lookback window)
    """
    # Align and drop NaNs
    data = pd.concat([features, target], axis=1).dropna()

    feat_cols = list(features.columns)
    feat_values = data[feat_cols].to_numpy(dtype=np.float32)
    y_values = data[target.name].to_numpy(dtype=np.float32)

    n = len(data)
    if n <= lookback_bars:
        raise ValueError("Not enough rows after NaN drop for the requested lookback")

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    ts_list: list[pd.Timestamp] = []

    for end in range(lookback_bars, n):
        start = end - lookback_bars
        X_list.append(feat_values[start:end])
        y_list.append(float(y_values[end]))
        ts_list.append(data.index[end])

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.float32)
    ts = pd.DatetimeIndex(ts_list)

    return X, y, ts, feat_cols
