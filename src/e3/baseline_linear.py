"""
Modelo Baseline: Ridge Regression para E3 (Estrategia Intradía).

Sirve como piso de comparación contra el LSTM ensemble intradía.
Usa las mismas features (9 indicadores de 5-min) con modelo lineal regularizado.

Diferencias clave vs E2 RidgeBaseline:
- Alpha más alto (default 100.0): con 96 lookback × 9 features = 864 features
  aplanadas sobre retornos de 30 min, la regularización L2 necesita ser mucho
  mayor que en E2 (60 × 16 = 960 features sobre retornos de 20 días).
- Reporta n_features_effective (dimensionalidad post-flatten).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BaselineResult:
    """Resultado del entrenamiento del baseline."""
    train_score: float
    val_score: float
    n_features_effective: int


class IntradayRidgeBaseline:
    """Ridge Regression adaptado para predicción de retornos intradía (5-min).

    Aplana secuencias 3D (n_samples, lookback_bars, n_features) a 2D y entrena
    Ridge con regularización L2 alta para manejar la dimensionalidad resultante.
    """

    def __init__(self, alpha: float = 100.0, seed: int = 42) -> None:
        try:
            from sklearn.linear_model import Ridge
            self._Ridge = Ridge
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'scikit-learn'. Install with: pip install scikit-learn"
            ) from exc

        self.alpha = alpha
        np.random.seed(seed)
        self.model: Optional[object] = None
        self.input_shape_: Optional[tuple] = None
        self.n_features_effective_: int = 0

    def _flatten_sequences(self, X: np.ndarray) -> np.ndarray:
        """Convierte secuencias 3D a 2D para regresión lineal.

        Input:  (n_samples, lookback_bars, n_features)
        Output: (n_samples, lookback_bars * n_features)
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {X.shape}")
        n_samples, seq_len, n_features = X.shape
        return X.reshape(n_samples, seq_len * n_features)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> BaselineResult:
        """Entrena Ridge Regression.

        Args:
            X_train: (n_samples, lookback_bars, n_features)
            y_train: (n_samples,)
            X_val:   (n_samples_val, lookback_bars, n_features)
            y_val:   (n_samples_val,)

        Returns:
            BaselineResult con R² scores y dimensionalidad efectiva.
        """
        self.input_shape_ = X_train.shape[1:]

        X_train_flat = self._flatten_sequences(X_train)
        X_val_flat = self._flatten_sequences(X_val)
        self.n_features_effective_ = X_train_flat.shape[1]

        self.model = self._Ridge(alpha=self.alpha)
        self.model.fit(X_train_flat, y_train)

        train_score = self.model.score(X_train_flat, y_train)
        val_score = self.model.score(X_val_flat, y_val)

        return BaselineResult(
            train_score=float(train_score),
            val_score=float(val_score),
            n_features_effective=self.n_features_effective_,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Genera predicciones.

        Args:
            X: (n_samples, lookback_bars, n_features)

        Returns:
            predictions: (n_samples,)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        if X.shape[1:] != self.input_shape_:
            raise ValueError(
                f"Input shape mismatch. Expected {self.input_shape_}, got {X.shape[1:]}"
            )
        X_flat = self._flatten_sequences(X)
        return self.model.predict(X_flat)
