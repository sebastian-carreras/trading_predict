"""
Modelo Baseline: Regresión Lineal Simple para E1 (Estrategia Conservadora).

Este baseline sirve como punto de comparación contra el modelo GRU.
Usa las mismas features pero con un modelo lineal simple (sklearn).

Métricas de comparación:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Directional Accuracy (% predicciones con signo correcto)
- IC (Information Coefficient - correlación Spearman)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class BaselineResult:
    """Resultado del entrenamiento del baseline."""
    train_score: float
    val_score: float


class LinearRegressionBaseline:
    """
    Regresión Lineal simple para predicción de retornos a 90 días.

    Estrategia:
    - Aplana las secuencias 3D (n_samples, seq_len, features) a 2D
    - Entrena regresión lineal estándar
    - Reporta métricas comparables con GRU
    """

    def __init__(self, seed: int = 42) -> None:
        try:
            from sklearn.linear_model import LinearRegression
            self.LinearRegression = LinearRegression
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'scikit-learn'. Install with: pip install scikit-learn"
            ) from exc

        np.random.seed(seed)
        self.model: Optional[LinearRegression] = None
        self.input_shape_: Optional[tuple] = None

    def _flatten_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Convierte secuencias 3D a 2D para regresión lineal.

        Input: (n_samples, seq_length, n_features)
        Output: (n_samples, seq_length * n_features)
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
        """
        Entrena la regresión lineal.

        Args:
            X_train: (n_samples, seq_length, n_features)
            y_train: (n_samples,)
            X_val: (n_samples_val, seq_length, n_features)
            y_val: (n_samples_val,)

        Returns:
            BaselineResult con R² scores
        """
        # Guardar forma para validación posterior
        self.input_shape_ = X_train.shape[1:]

        # Aplanar secuencias
        X_train_flat = self._flatten_sequences(X_train)
        X_val_flat = self._flatten_sequences(X_val)

        # Entrenar modelo
        self.model = self.LinearRegression()
        self.model.fit(X_train_flat, y_train)

        # Calcular scores
        train_score = self.model.score(X_train_flat, y_train)
        val_score = self.model.score(X_val_flat, y_val)

        return BaselineResult(
            train_score=float(train_score),
            val_score=float(val_score),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Genera predicciones.

        Args:
            X: (n_samples, seq_length, n_features)

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

    def save(self, path: str) -> None:
        """Guarda el modelo entrenado."""
        import pickle

        if self.model is None:
            raise RuntimeError("No model to save. Train first.")

        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "input_shape": self.input_shape_,
            }, f)

    def load(self, path: str) -> None:
        """Carga un modelo previamente entrenado."""
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.input_shape_ = data["input_shape"]


def compute_baseline_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calcula métricas de evaluación para el baseline.

    Métricas:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - Directional Accuracy: % predicciones con signo correcto
    - IC: Information Coefficient (correlación Spearman)

    Args:
        y_true: valores reales
        y_pred: predicciones del modelo

    Returns:
        dict con métricas
    """
    from scipy.stats import spearmanr

    # Validar inputs
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    if len(y_true) == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "directional_accuracy": float("nan"),
            "ic": float("nan"),
        }

    # MAE
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # RMSE
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # Directional Accuracy
    correct_direction = np.sign(y_true) == np.sign(y_pred)
    directional_accuracy = float(np.mean(correct_direction))

    # IC (Information Coefficient - Spearman correlation)
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        ic, _ = spearmanr(y_true, y_pred)
        ic = float(ic) if np.isfinite(ic) else 0.0
    else:
        ic = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": directional_accuracy,
        "ic": ic,
    }
