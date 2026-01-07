from __future__ import annotations

import numpy as np
import pandas as pd


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
    out["vol_96"] = out["ret_1"].rolling(96).std()  # ~1 day (approx trading day)

    # Range / ATR proxy
    out["range_pct"] = (high - low) / close
    tr = pd.concat(
        [
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean() / close

    # Volume z-score
    vol_roll = volume.rolling(96)
    out["vol_z_96"] = (volume - vol_roll.mean()) / (vol_roll.std() + 1e-12)

    # Time-of-day (cyclical)
    # Use UTC timestamps; if you want exchange-local time, convert before calling.
    idx = pd.DatetimeIndex(df.index)
    minutes = idx.hour * 60 + idx.minute
    out["tod_sin"] = np.sin(2 * np.pi * minutes / (24 * 60))
    out["tod_cos"] = np.cos(2 * np.pi * minutes / (24 * 60))

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
