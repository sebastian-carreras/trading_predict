"""
Construcción de secuencias (lookback windows) para modelos RNN (GRU/LSTM).

Para E1/E2: convierte features + target en formato (n_samples, lookback, n_features).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_sequences(
    features: pd.DataFrame,
    target: pd.Series,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    """Convierte features y target a secuencias para RNN.

    Args:
        features: DataFrame con features (index temporal)
        target: Series con target (mismo índice)
        lookback: Ventana de lookback (ej. 180 días para E1)

    Returns:
        X: (n_samples, lookback, n_features)
        y: (n_samples,)
        timestamps: índice temporal de cada muestra (fin de ventana)
        feature_names: nombres de features
    """
    # Alinear y eliminar NaNs
    data = pd.concat([features, target], axis=1).dropna()

    feat_cols = list(features.columns)
    feat_values = data[feat_cols].to_numpy(dtype=np.float32)
    y_values = data[target.name].to_numpy(dtype=np.float32)

    n = len(data)
    if n <= lookback:
        raise ValueError(
            f"Datos insuficientes después de dropna: {n} <= lookback={lookback}"
        )

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    ts_list: list[pd.Timestamp] = []

    for end in range(lookback, n):
        start = end - lookback
        X_list.append(feat_values[start:end])
        y_list.append(float(y_values[end]))
        ts_list.append(data.index[end])

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.float32)
    ts = pd.DatetimeIndex(ts_list)

    return X, y, ts, feat_cols


def time_split(
    n: int, train_frac: float = 0.7, val_frac: float = 0.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split temporal (sin shuffle) para series temporales.

    Args:
        n: Total de muestras
        train_frac: Fracción para train
        val_frac: Fracción para validación

    Returns:
        idx_train, idx_val, idx_test (arrays de índices)
    """
    if not (0 < train_frac < 1) or not (0 < val_frac < 1):
        raise ValueError("Fracciones inválidas")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac debe ser < 1.0")

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    idx_train = np.arange(0, train_end)
    idx_val = np.arange(train_end, val_end)
    idx_test = np.arange(val_end, n)

    return idx_train, idx_val, idx_test
