"""
Modelo Baseline: Ridge Regression para E2 (Estrategia Moderada).

Sirve como punto de comparación contra el modelo LSTM.
Usa las mismas 16 features pero con un modelo lineal regularizado (sklearn Ridge).

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


class RidgeBaseline:
    """
    Ridge Regression para predicción de retornos a 20 días.
    
    Estrategia:
    - Aplana las secuencias 3D (n_samples, seq_len, features) a 2D
    - Entrena Ridge Regression (L2 regularización)
    - Reporta métricas comparables con LSTM
    
    Se usa Ridge en vez de LinearRegression porque con lookback=60 × 16 features
    = 960 features aplanadas, la regularización L2 previene overfitting.
    """

    def __init__(self, alpha: float = 1.0, seed: int = 42) -> None:
        try:
            from sklearn.linear_model import Ridge
            self.Ridge = Ridge
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'scikit-learn'. Install with: pip install scikit-learn"
            ) from exc

        self.alpha = alpha
        np.random.seed(seed)
        self.model: Optional[Ridge] = None
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
        Entrena la Ridge Regression.
        
        Args:
            X_train: (n_samples, seq_length, n_features)
            y_train: (n_samples,)
            X_val: (n_samples_val, seq_length, n_features)
            y_val: (n_samples_val,)
        
        Returns:
            BaselineResult con R² scores
        """
        self.input_shape_ = X_train.shape[1:]
        
        X_train_flat = self._flatten_sequences(X_train)
        X_val_flat = self._flatten_sequences(X_val)
        
        self.model = self.Ridge(alpha=self.alpha)
        self.model.fit(X_train_flat, y_train)
        
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
                "alpha": self.alpha,
            }, f)

    def load(self, path: str) -> None:
        """Carga un modelo previamente entrenado."""
        import pickle
        
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.model = data["model"]
        self.input_shape_ = data["input_shape"]
        self.alpha = data.get("alpha", 1.0)


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
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    if len(y_true) == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "directional_accuracy": float("nan"),
            "ic": float("nan"),
        }
    
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    correct_direction = np.sign(y_true) == np.sign(y_pred)
    directional_accuracy = float(np.mean(correct_direction))
    
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
