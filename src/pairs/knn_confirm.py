"""
k-NN confirmación para pairs trading.

Implementa:
- k-NN para predecir dirección de convergencia del spread
- Estado features: [spread, delta_spread, zscore, correlation, volume_ratio]
- Distancia: Euclidiana o Mahalanobis
- Predicción: promedio ponderado por 1/distancia
"""

from __future__ import annotations

import logging
from typing import Dict, Literal

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def build_knn_state_features(
    spread_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye espacio de estados para k-NN.
    
    Args:
        spread_features: DataFrame con features del spread
                        (debe incluir: spread, delta_spread, zscore)
    
    Returns:
        DataFrame con features de estado normalizados:
        - spread: Spread actual
        - delta_spread: Cambio en spread
        - zscore: Z-score del spread
        - correlation_30: Correlación rolling (si disponible)
        - volume_ratio: Ratio de volumen (si disponible)
    """
    
    required = ["spread", "zscore"]
    missing = [col for col in required if col not in spread_features.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Features básicos
    features = pd.DataFrame(index=spread_features.index)
    features["spread"] = spread_features["spread"]
    features["zscore"] = spread_features["zscore"]
    
    # Delta spread (si no existe, calcular)
    if "delta_spread" in spread_features.columns:
        features["delta_spread"] = spread_features["delta_spread"]
    else:
        features["delta_spread"] = spread_features["spread"].diff()
    
    # Features opcionales
    if "correlation_30" in spread_features.columns:
        features["correlation_30"] = spread_features["correlation_30"]
    
    if "volume_ratio" in spread_features.columns:
        features["volume_ratio"] = spread_features["volume_ratio"]
    
    return features


def create_knn_target(
    spread: pd.Series,
    horizon_days: int = 10,
) -> pd.Series:
    """
    Crea target para k-NN: cambio en spread a horizonte H.
    
    Args:
        spread: Serie temporal del spread
        horizon_days: Horizonte de predicción
    
    Returns:
        Serie con target: spread_{t+H} - spread_t
    """
    
    # Shift hacia atrás para obtener spread futuro
    spread_future = spread.shift(-horizon_days)
    
    # Target = cambio en spread
    target = spread_future - spread
    
    return target


def train_knn_model(
    state_features: pd.DataFrame,
    target: pd.Series,
    k: int = 10,
    distance: Literal["euclidean", "mahalanobis"] = "euclidean",
    test_size: float = 0.2,
) -> Dict:
    """
    Entrena modelo k-NN para predicción de convergencia.
    
    Args:
        state_features: Features de estado
        target: Target (cambio en spread)
        k: Número de vecinos
        distance: Métrica de distancia
        test_size: Proporción para validación
    
    Returns:
        Dict con:
        - model: Modelo entrenado (KNeighborsRegressor)
        - scaler: Scaler para normalizar features
        - train_score: R² en train
        - test_score: R² en test
    """
    
    # Alinear features y target
    df = state_features.copy()
    df["target"] = target
    df = df.dropna()
    
    if len(df) < k * 5:
        logger.warning(f"Insufficient data for k-NN: {len(df)} < {k*5}")
        return {
            "model": None,
            "scaler": None,
            "train_score": np.nan,
            "test_score": np.nan,
        }
    
    # Split temporal
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df.drop(columns=["target"]).values
    y_train = train_df["target"].values
    X_test = test_df.drop(columns=["target"]).values
    y_test = test_df["target"].values
    
    # Normalizar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Configurar métrica de distancia
    if distance == "euclidean":
        metric = "euclidean"
    elif distance == "mahalanobis":
        metric = "mahalanobis"
        # Para Mahalanobis necesitamos matriz de covarianza
        # (implementación simplificada)
        metric = "euclidean"  # Fallback
        logger.warning("Mahalanobis not fully implemented, using euclidean")
    else:
        metric = "euclidean"
    
    # Entrenar k-NN
    model = KNeighborsRegressor(
        n_neighbors=k,
        weights="distance",  # Ponderado por 1/distancia
        metric=metric,
        algorithm="auto",
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluar
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    logger.info(
        f"k-NN trained: k={k}, train_R²={train_score:.4f}, test_R²={test_score:.4f}"
    )
    
    return {
        "model": model,
        "scaler": scaler,
        "train_score": train_score,
        "test_score": test_score,
    }


def predict_convergence(
    model_dict: Dict,
    current_state: pd.Series | np.ndarray,
) -> Dict[str, float]:
    """
    Predice convergencia del spread usando k-NN.
    
    Args:
        model_dict: Dict con modelo y scaler (output de train_knn_model)
        current_state: Estado actual (features)
    
    Returns:
        Dict con predicción:
        - predicted_change: Cambio predicho en spread
        - convergence_direction: Dirección de convergencia (1/-1/0)
        - confidence: Confianza de la predicción (basado en std de vecinos)
    """
    
    model = model_dict["model"]
    scaler = model_dict["scaler"]
    
    if model is None or scaler is None:
        return {
            "predicted_change": 0.0,
            "convergence_direction": 0,
            "confidence": 0.0,
        }
    
    # Preparar estado
    if isinstance(current_state, pd.Series):
        X = current_state.values.reshape(1, -1)
    else:
        X = current_state.reshape(1, -1)
    
    # Normalizar
    X_scaled = scaler.transform(X)
    
    # Predecir
    predicted_change = model.predict(X_scaled)[0]
    
    # Dirección de convergencia
    if predicted_change > 0.01:  # Umbral pequeño para ruido
        direction = 1  # Spread sube
    elif predicted_change < -0.01:
        direction = -1  # Spread baja
    else:
        direction = 0  # Flat
    
    # Calcular confianza (basado en dispersión de vecinos)
    # Obtener k vecinos más cercanos
    distances, indices = model.kneighbors(X_scaled)
    
    # Usar inverso de distancia promedio como proxy de confianza
    avg_distance = np.mean(distances)
    confidence = 1.0 / (1.0 + avg_distance)  # Normalizado [0, 1]
    
    return {
        "predicted_change": predicted_change,
        "convergence_direction": direction,
        "confidence": confidence,
    }


def cross_validate_knn(
    state_features: pd.DataFrame,
    target: pd.Series,
    k_candidates: list[int] = [5, 10, 20],
    n_folds: int = 5,
) -> pd.DataFrame:
    """
    Cross-validation para seleccionar mejor k.
    
    Args:
        state_features: Features de estado
        target: Target (cambio en spread)
        k_candidates: Lista de valores k a probar
        n_folds: Número de folds para CV
    
    Returns:
        DataFrame con resultados de CV:
        - k: Valor de k
        - mean_score: R² promedio
        - std_score: Desviación estándar de R²
    """
    
    from sklearn.model_selection import TimeSeriesSplit
    
    # Alinear features y target
    df = state_features.copy()
    df["target"] = target
    df = df.dropna()
    
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    
    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=n_folds)
    
    results = []
    
    for k in k_candidates:
        scores = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = KNeighborsRegressor(
                n_neighbors=k,
                weights="distance",
                metric="euclidean",
            )
            
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores.append(score)
        
        results.append({
            "k": k,
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
        })
        
        logger.info(
            f"k={k}: mean_R²={np.mean(scores):.4f} ± {np.std(scores):.4f}"
        )
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("mean_score", ascending=False)
    
    return df_results


def generate_knn_signals(
    spread_features: pd.DataFrame,
    spread: pd.Series,
    k: int = 10,
    horizon_days: int = 10,
    threshold_change: float = 0.5,
) -> pd.DataFrame:
    """
    Genera señales de trading usando k-NN.
    
    Args:
        spread_features: DataFrame con features del spread
        spread: Serie del spread
        k: Número de vecinos
        horizon_days: Horizonte de predicción
        threshold_change: Cambio mínimo predicho para señal
    
    Returns:
        DataFrame con señales:
        - predicted_change: Cambio predicho
        - signal: 1 (long), -1 (short), 0 (flat)
        - confidence: Confianza de la señal
    """
    
    # Construir estado
    state_features = build_knn_state_features(spread_features)
    
    # Crear target
    target = create_knn_target(spread, horizon_days=horizon_days)
    
    # Entrenar modelo
    model_dict = train_knn_model(
        state_features, target, k=k, test_size=0.2
    )
    
    if model_dict["model"] is None:
        logger.warning("k-NN training failed, returning empty signals")
        return pd.DataFrame(index=spread.index)
    
    # Generar predicciones
    predictions = []
    
    for idx in state_features.index:
        if pd.isna(state_features.loc[idx]).any():
            predictions.append({
                "timestamp": idx,
                "predicted_change": 0.0,
                "signal": 0,
                "confidence": 0.0,
            })
            continue
        
        current_state = state_features.loc[idx]
        pred = predict_convergence(model_dict, current_state)
        
        # Generar señal
        if pred["predicted_change"] > threshold_change:
            signal = 1  # Espera que spread suba (long A, short B)
        elif pred["predicted_change"] < -threshold_change:
            signal = -1  # Espera que spread baje (short A, long B)
        else:
            signal = 0  # Flat
        
        predictions.append({
            "timestamp": idx,
            "predicted_change": pred["predicted_change"],
            "signal": signal,
            "confidence": pred["confidence"],
        })
    
    df_signals = pd.DataFrame(predictions)
    df_signals = df_signals.set_index("timestamp")
    
    return df_signals
