"""
Construcción de spreads para pairs trading.

Implementa:
- Cálculo de hedge ratio (β) via OLS rolling
- Construcción del spread: S = P_A - β * P_B
- Cálculo de z-score normalizado
- Validación de estabilidad del spread
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


def calculate_hedge_ratio(
    price_a: pd.Series,
    price_b: pd.Series,
    window_days: int | None = None,
) -> pd.Series | float:
    """
    Calcula el hedge ratio (β) para el par usando OLS.
    
    Regresión: P_A = α + β * P_B + ε
    
    Args:
        price_a: Serie de precios del activo A
        price_b: Serie de precios del activo B
        window_days: Si es None, calcula β estático para toda la serie.
                     Si es int, calcula β rolling en ventana móvil.
    
    Returns:
        Si window_days es None: float con β estático
        Si window_days es int: pd.Series con β rolling
    """
    
    # Alinear series
    df = pd.DataFrame({"a": price_a, "b": price_b}).dropna()
    
    if len(df) < 30:
        logger.warning(f"Insufficient data for hedge ratio: {len(df)} < 30")
        if window_days is None:
            return np.nan
        else:
            return pd.Series(np.nan, index=df.index)
    
    if window_days is None:
        # === Beta estático: usar TODOS los datos históricos ===
        # Útil para relaciones muy estables que no cambian en el tiempo
        # Ejemplo: KO vs PEP (Coca-Cola vs Pepsi)
        
        # Preparar datos para regresión OLS
        X = df["b"].values.reshape(-1, 1)  # Variable independiente (predictor)
        y = df["a"].values                  # Variable dependiente (target)
        
        # Regresión lineal: P_A = α + β·P_B + ε
        # fit_intercept=True incluye el término α (intercepto)
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        
        # β es la pendiente: cuánto cambia P_A por cada cambio en P_B
        beta = model.coef_[0]
        return beta
    
    else:
        # === Beta rolling: recalcular cada día con ventana móvil ===
        # Útil cuando la relación cambia en el tiempo
        # Ejemplo: Acciones que cambian su correlación por cambios fundamentales
        
        betas = []
        
        # Iterar sobre cada día
        for i in range(len(df)):
            # Primeros días: no hay suficiente historia → NaN
            if i < window_days:
                betas.append(np.nan)
                continue
            
            # Tomar ventana de los últimos N días (ej: 120 días)
            window = df.iloc[i-window_days:i]
            X = window["b"].values.reshape(-1, 1)
            y = window["a"].values
            
            # OLS en la ventana
            model = LinearRegression(fit_intercept=True)
            model.fit(X, y)
            
            # β calculado con los últimos 120 días
            beta = model.coef_[0]
            betas.append(beta)
        
        # Retornar serie temporal de betas (uno por día)
        return pd.Series(betas, index=df.index)


def build_spread(
    price_a: pd.Series,
    price_b: pd.Series,
    hedge_ratio: pd.Series | float | None = None,
    window_days: int = 120,
) -> pd.Series:
    """
    Construye el spread entre dos activos.
    
    Spread: S_t = P_A,t - β * P_B,t
    
    Args:
        price_a: Serie de precios del activo A
        price_b: Serie de precios del activo B
        hedge_ratio: Hedge ratio (β). Si None, se calcula rolling.
                     Puede ser float (estático) o pd.Series (rolling)
        window_days: Ventana para calcular β rolling (si hedge_ratio es None)
    
    Returns:
        Serie temporal del spread
    """
    
    # Alinear series
    df = pd.DataFrame({"a": price_a, "b": price_b}).dropna()
    
    if len(df) < 30:
        logger.warning(f"Insufficient data for spread: {len(df)} < 30")
        return pd.Series(np.nan, index=df.index)
    
    # === Obtener hedge ratio (β) ===
    # El β determina cuántas unidades de B necesitas por cada unidad de A
    # para crear un spread market-neutral
    
    if hedge_ratio is None:
        # Opción 1: Calcular β dinámicamente (ventana móvil)
        # Se recalcula cada día → se adapta a cambios en la relación
        beta = calculate_hedge_ratio(df["a"], df["b"], window_days=window_days)
    elif isinstance(hedge_ratio, (int, float)):
        # Opción 2: β fijo (un solo número para todo el período)
        # Más simple pero asume relación constante
        beta = hedge_ratio
    else:
        # Opción 3: β pre-calculado (serie temporal)
        # Permite usar β calculado externamente
        beta = hedge_ratio.reindex(df.index)
    
    # === Construir spread ===
    # Spread = P_A - β·P_B
    # 
    # Ejemplo: Si GGAL=$100, BMA=$50, β=2.0
    #   Spread = 100 - 2.0*50 = 0
    # 
    # Si GGAL sube a $105 pero BMA se queda en $50:
    #   Spread = 105 - 2.0*50 = 5 → ¡Oportunidad de short!
    #   (Esperas que GGAL baje o BMA suba para cerrar en 0)
    
    if isinstance(beta, pd.Series):
        # Beta cambia cada día → spread también es dinámico
        spread = df["a"] - beta * df["b"]
    else:
        # Beta constante → spread simple
        spread = df["a"] - beta * df["b"]
    
    return spread


def calculate_zscore(
    spread: pd.Series,
    window_days: int = 60,
    min_periods: int = 30,
) -> pd.Series:
    """
    Calcula z-score del spread en ventana rolling.
    
    Z-score: (S_t - μ_S) / σ_S
    
    Args:
        spread: Serie temporal del spread
        window_days: Ventana para calcular μ y σ
        min_periods: Períodos mínimos para calcular estadísticas
    
    Returns:
        Serie temporal del z-score
    """
    
    # === Calcular estadísticas móviles del spread ===
    # Usamos ventana de 60 días para capturar el comportamiento "normal" reciente
    rolling = spread.rolling(window=window_days, min_periods=min_periods)
    
    mean = rolling.mean()  # Media del spread en los últimos 60 días
    std = rolling.std()    # Desviación estándar en los últimos 60 días
    
    # === Z-score: normalización del spread ===
    # Z-score = (valor actual - media) / desviación estándar
    # 
    # Interpretación:
    #   z = 0:  Spread en su valor promedio → No hacer nada
    #   z = +2: Spread está 2 desviaciones por encima → Anormalmente ALTO
    #           → Short el spread (apostar a que baje)
    #   z = -2: Spread está 2 desviaciones por debajo → Anormalmente BAJO
    #           → Long el spread (apostar a que suba)
    # 
    # El +1e-12 evita división por cero cuando std=0
    zscore = (spread - mean) / (std + 1e-12)
    
    return zscore


def calculate_spread_statistics(
    spread: pd.Series,
    window_days: int = 60,
) -> pd.DataFrame:
    """
    Calcula estadísticas del spread en ventana rolling.
    
    Args:
        spread: Serie temporal del spread
        window_days: Ventana para estadísticas rolling
    
    Returns:
        DataFrame con estadísticas:
        - mean: Media del spread
        - std: Desviación estándar
        - zscore: Z-score actual
        - min: Mínimo en ventana
        - max: Máximo en ventana
    """
    
    rolling = spread.rolling(window=window_days, min_periods=30)
    
    df = pd.DataFrame(index=spread.index)
    df["spread"] = spread
    df["mean"] = rolling.mean()
    df["std"] = rolling.std()
    df["zscore"] = (spread - df["mean"]) / (df["std"] + 1e-12)
    df["min"] = rolling.min()
    df["max"] = rolling.max()
    
    # Percentiles
    df["q25"] = rolling.quantile(0.25)
    df["q75"] = rolling.quantile(0.75)
    
    return df


def validate_spread_stability(
    spread: pd.Series,
    beta: pd.Series | float,
    window_days: int = 60,
) -> Dict[str, float]:
    """
    Valida estabilidad del spread y del hedge ratio.
    
    Args:
        spread: Serie temporal del spread
        beta: Hedge ratio (puede ser serie rolling o float estático)
        window_days: Ventana para calcular métricas
    
    Returns:
        Dict con métricas de estabilidad:
        - spread_cv: Coeficiente de variación del spread
        - beta_cv: Coeficiente de variación de β (si es rolling)
        - beta_mean: β promedio
        - beta_std: Desviación estándar de β
        - spread_mean: Spread promedio
        - spread_std: Desviación estándar del spread
    """
    
    spread_clean = spread.dropna()
    
    if len(spread_clean) < window_days:
        return {
            "spread_cv": np.nan,
            "beta_cv": np.nan,
            "beta_mean": np.nan,
            "beta_std": np.nan,
            "spread_mean": np.nan,
            "spread_std": np.nan,
        }
    
    # Estadísticas del spread
    spread_mean = spread_clean.mean()
    spread_std = spread_clean.std()
    spread_cv = spread_std / (abs(spread_mean) + 1e-12)
    
    # Estadísticas de β
    if isinstance(beta, pd.Series):
        beta_clean = beta.dropna()
        beta_mean = beta_clean.mean()
        beta_std = beta_clean.std()
        beta_cv = beta_std / (abs(beta_mean) + 1e-12)
    else:
        beta_mean = beta
        beta_std = 0.0
        beta_cv = 0.0
    
    return {
        "spread_cv": spread_cv,
        "beta_cv": beta_cv,
        "beta_mean": beta_mean,
        "beta_std": beta_std,
        "spread_mean": spread_mean,
        "spread_std": spread_std,
    }


def build_pair_features(
    price_a: pd.Series,
    price_b: pd.Series,
    volume_a: pd.Series | None = None,
    volume_b: pd.Series | None = None,
    beta_window: int = 120,
    zscore_window: int = 60,
) -> pd.DataFrame:
    """
    Construye features completos para un par de activos.
    
    Args:
        price_a: Precios del activo A
        price_b: Precios del activo B
        volume_a: Volumen del activo A (opcional)
        volume_b: Volumen del activo B (opcional)
        beta_window: Ventana para calcular β rolling
        zscore_window: Ventana para calcular z-score
    
    Returns:
        DataFrame con features:
        - spread: Spread S_t = P_A - β * P_B
        - zscore: Z-score del spread
        - delta_spread: Cambio en spread (ΔS_t)
        - beta_rolling: Hedge ratio rolling
        - volume_ratio: Vol_A / Vol_B (si volúmenes disponibles)
        - correlation_30: Correlación rolling 30 días
        - price_ratio: P_A / P_B
    """
    
    # Alinear series
    df = pd.DataFrame({"price_a": price_a, "price_b": price_b}).dropna()
    
    # Calcular beta rolling
    beta_rolling = calculate_hedge_ratio(
        df["price_a"], df["price_b"], window_days=beta_window
    )
    
    # Construir spread
    spread = build_spread(
        df["price_a"], df["price_b"], hedge_ratio=beta_rolling
    )
    
    # Z-score
    zscore = calculate_zscore(spread, window_days=zscore_window)
    
    # Features adicionales
    features = pd.DataFrame(index=spread.index)
    features["spread"] = spread
    features["zscore"] = zscore
    features["delta_spread"] = spread.diff()
    features["beta_rolling"] = beta_rolling
    features["price_ratio"] = df["price_a"] / df["price_b"]
    
    # Correlación rolling
    features["correlation_30"] = (
        df["price_a"].rolling(30).corr(df["price_b"])
    )
    
    # Volumen ratio (si disponible)
    if volume_a is not None and volume_b is not None:
        vol_df = pd.DataFrame({"vol_a": volume_a, "vol_b": volume_b}).dropna()
        vol_ratio = vol_df["vol_a"] / (vol_df["vol_b"] + 1e-12)
        features["volume_ratio"] = vol_ratio.reindex(features.index)
    
    return features
