"""
Modelado de procesos Ornstein-Uhlenbeck para spreads de pairs trading.

Implementa:
- Estimación de parámetros OU (θ, μ, σ) vía MLE
- Cálculo de half-life (tiempo de reversión a la media)
- Validación de estacionariedad del spread
- Predicción de spread futuro usando modelo OU
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


def estimate_ou_parameters(
    spread: pd.Series,
    dt: float = 1.0,
) -> Dict[str, float]:
    """
    Estima parámetros del proceso Ornstein-Uhlenbeck via MLE.
    
    El proceso OU se modela como:
        dS_t = θ(μ - S_t)dt + σ dW_t
    
    Donde:
        θ: velocidad de reversión a la media
        μ: nivel medio del spread
        σ: volatilidad del proceso
    
    Args:
        spread: Serie temporal del spread
        dt: Intervalo de tiempo (1.0 para diario)
    
    Returns:
        Dict con parámetros estimados:
        - theta: velocidad de reversión
        - mu: nivel medio
        - sigma: volatilidad
        - half_life: tiempo característico de reversión (días)
        - log_likelihood: log-verosimilitud del modelo
    """
    
    # Eliminar NaNs
    spread_clean = spread.dropna()
    
    if len(spread_clean) < 30:
        logger.warning(f"Insufficient data for OU estimation: {len(spread_clean)} < 30")
        return {
            "theta": np.nan,
            "mu": np.nan,
            "sigma": np.nan,
            "half_life": np.nan,
            "log_likelihood": np.nan,
        }
    
    # Valores del spread
    s = spread_clean.values
    
    # Método discreto (Euler-Maruyama)
    # ΔS_t ≈ θ(μ - S_t)Δt + σ√Δt ε_t
    # donde ε_t ~ N(0, 1)
    
    # Primera estimación: μ = media del spread
    mu_init = np.mean(s)
    
    # Regresión AR(1) para estimar θ
    # S_{t+1} - S_t = θμΔt - θS_t Δt + σ√Δt ε_t
    # Reordenando: S_{t+1} = (1 - θΔt)S_t + θμΔt + σ√Δt ε_t
    #              S_{t+1} = α + β S_t + ε_t
    # donde β = (1 - θΔt), α = θμΔt
    
    s_lagged = s[:-1]
    s_next = s[1:]
    
    # OLS para estimar α, β
    # β = cov(S_t, S_{t+1}) / var(S_t)
    # α = mean(S_{t+1}) - β * mean(S_t)
    
    mean_s = np.mean(s_lagged)
    mean_s_next = np.mean(s_next)
    
    cov = np.mean((s_lagged - mean_s) * (s_next - mean_s_next))
    var_s = np.var(s_lagged, ddof=1)
    
    if var_s < 1e-12:
        logger.warning("Spread has near-zero variance")
        return {
            "theta": np.nan,
            "mu": mu_init,
            "sigma": np.nan,
            "half_life": np.nan,
            "log_likelihood": np.nan,
        }
    
    beta = cov / var_s
    alpha = mean_s_next - beta * mean_s
    
    # Recuperar θ y μ desde α y β
    # θ = (1 - β) / Δt
    # μ = α / (θ Δt) = α / (1 - β)
    
    theta = (1 - beta) / dt
    
    if abs(theta) < 1e-6:
        logger.warning("Theta near zero - no mean reversion")
        mu = mu_init
    else:
        mu = alpha / (theta * dt)
    
    # Estimar σ desde residuales
    residuals = s_next - (alpha + beta * s_lagged)
    sigma = np.std(residuals, ddof=1) / np.sqrt(dt)
    
    # Calcular half-life
    if theta > 1e-6:
        half_life = np.log(2) / theta
    else:
        half_life = np.inf
    
    # Log-likelihood (Gaussian)
    # L = -n/2 log(2π) - n/2 log(σ²Δt) - 1/(2σ²Δt) Σ(residuals²)
    n = len(residuals)
    var_res = sigma**2 * dt
    log_likelihood = (
        -n/2 * np.log(2 * np.pi)
        - n/2 * np.log(var_res)
        - np.sum(residuals**2) / (2 * var_res)
    )
    
    return {
        "theta": theta,
        "mu": mu,
        "sigma": sigma,
        "half_life": half_life,
        "log_likelihood": log_likelihood,
    }


def calculate_half_life(spread: pd.Series) -> float:
    """
    Calcula half-life del spread usando método AR(1) simplificado.
    
    Args:
        spread: Serie temporal del spread
    
    Returns:
        Half-life en unidades de tiempo (días si spread es diario)
    """
    
    ou_params = estimate_ou_parameters(spread)
    return ou_params["half_life"]


def test_stationarity(spread: pd.Series) -> Dict[str, float]:
    """
    Test de estacionariedad (Augmented Dickey-Fuller).
    
    Args:
        spread: Serie temporal del spread
    
    Returns:
        Dict con resultados del test ADF:
        - adf_statistic: Estadístico ADF
        - pvalue: p-value (< 0.05 indica estacionariedad)
        - is_stationary: bool indicando si es estacionario
    """
    
    spread_clean = spread.dropna()
    
    if len(spread_clean) < 30:
        return {
            "adf_statistic": np.nan,
            "pvalue": 1.0,
            "is_stationary": False,
        }
    
    # Test ADF
    result = adfuller(spread_clean.values, autolag="AIC")
    
    adf_stat = result[0]
    pvalue = result[1]
    
    # p-value < 0.05 indica rechazo de H0 (no estacionariedad)
    # es decir, evidencia de estacionariedad
    is_stationary = pvalue < 0.05
    
    return {
        "adf_statistic": adf_stat,
        "pvalue": pvalue,
        "is_stationary": is_stationary,
    }


def predict_ou_convergence(
    current_spread: float,
    ou_params: Dict[str, float],
    horizon_days: int = 10,
    dt: float = 1.0,
) -> Dict[str, float]:
    """
    Predice convergencia del spread usando modelo OU.
    
    Solución analítica del proceso OU:
        E[S_t | S_0] = μ + (S_0 - μ) exp(-θt)
        Var[S_t | S_0] = σ²/(2θ) * (1 - exp(-2θt))
    
    Args:
        current_spread: Valor actual del spread
        ou_params: Parámetros OU (theta, mu, sigma)
        horizon_days: Horizonte de predicción (días)
        dt: Intervalo de tiempo
    
    Returns:
        Dict con predicción:
        - expected_spread: Spread esperado en t+horizon
        - expected_return: Retorno esperado hacia la media
        - variance: Varianza de la predicción
        - probability_convergence: P(spread se mueve hacia μ)
    """
    
    theta = ou_params["theta"]
    mu = ou_params["mu"]
    sigma = ou_params["sigma"]
    
    if np.isnan(theta) or np.isnan(mu):
        return {
            "expected_spread": np.nan,
            "expected_return": np.nan,
            "variance": np.nan,
            "probability_convergence": 0.5,
        }
    
    t = horizon_days * dt
    
    # Spread esperado
    expected_spread = mu + (current_spread - mu) * np.exp(-theta * t)
    
    # Retorno esperado hacia la media
    expected_return = (expected_spread - current_spread) / abs(current_spread + 1e-12)
    
    # Varianza del spread en t
    if theta > 1e-6:
        variance = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * t))
    else:
        variance = sigma**2 * t
    
    # Probabilidad de convergencia
    # Si spread actual > μ: P(spread baja)
    # Si spread actual < μ: P(spread sube)
    # Usar distribución normal para aproximar
    
    if abs(current_spread - mu) < 1e-6:
        prob_convergence = 0.5
    else:
        # Dirección de convergencia
        target_direction = np.sign(mu - current_spread)
        
        # Movimiento esperado
        expected_move = expected_spread - current_spread
        actual_direction = np.sign(expected_move)
        
        # Si van en la misma dirección, alta probabilidad
        # (simplificación: en realidad depende de la distribución completa)
        if target_direction == actual_direction:
            # Aproximación: usar magnitud del movimiento esperado vs varianza
            z_score = abs(expected_move) / (np.sqrt(variance) + 1e-12)
            # Convertir a probabilidad (CDF normal)
            # Aproximación usando error function
            # CDF(z) ≈ 0.5 * (1 + erf(z / sqrt(2)))
            # Para simplificar, usar aproximación lineal para z pequeño
            if z_score < 1:
                prob_convergence = 0.5 + 0.34 * z_score  # Aproximación lineal
            elif z_score < 2:
                prob_convergence = 0.84 + 0.08 * (z_score - 1)
            else:
                prob_convergence = 0.95
        else:
            # Movimiento contrario
            z_score = abs(expected_move) / (np.sqrt(variance) + 1e-12)
            if z_score < 1:
                prob_convergence = 0.5 - 0.34 * z_score
            else:
                prob_convergence = 0.16
    
    return {
        "expected_spread": expected_spread,
        "expected_return": expected_return,
        "variance": variance,
        "probability_convergence": prob_convergence,
    }


def rolling_ou_estimation(
    spread: pd.Series,
    window_days: int = 120,
) -> pd.DataFrame:
    """
    Estima parámetros OU en ventanas rolling.
    
    Args:
        spread: Serie temporal del spread
        window_days: Tamaño de la ventana rolling
    
    Returns:
        DataFrame con parámetros OU estimados en cada punto:
        - theta, mu, sigma, half_life
    """
    
    results = []
    
    for i in range(window_days, len(spread)):
        window = spread.iloc[i-window_days:i]
        ou_params = estimate_ou_parameters(window)
        
        results.append({
            "timestamp": spread.index[i],
            "theta": ou_params["theta"],
            "mu": ou_params["mu"],
            "sigma": ou_params["sigma"],
            "half_life": ou_params["half_life"],
        })
    
    df = pd.DataFrame(results)
    df = df.set_index("timestamp")
    
    return df
