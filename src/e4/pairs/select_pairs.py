"""
Selección de pares cointegrados para trading de pares.

Implementa:
- Test de cointegración Engle-Granger
- Test de cointegración Johansen (multivariado)
- Filtros de similitud fundamental
- Validación de estabilidad de cointegración
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

logger = logging.getLogger(__name__)


def test_pair_cointegration(
    price_a: pd.Series,
    price_b: pd.Series,
    method: str = "engle-granger",
    significance: float = 0.05,
) -> Dict[str, float]:
    """
    Testea cointegración entre dos series de precios.

    Args:
        price_a: Serie de precios del activo A
        price_b: Serie de precios del activo B
        method: Método de test ("engle-granger" o "johansen")
        significance: Nivel de significancia para el test (default 0.05)

    Returns:
        Dict con resultados del test:
        - pvalue: p-value del test
        - is_cointegrated: bool indicando si están cointegrados
        - test_statistic: estadístico del test
        - critical_value: valor crítico para el nivel de significancia
    """

    # Alinear series (dropna y mismo índice)
    df = pd.DataFrame({"a": price_a, "b": price_b}).dropna()

    if len(df) < 50:
        logger.warning(f"Insufficient data points: {len(df)} < 50")
        return {
            "pvalue": 1.0,
            "is_cointegrated": False,
            "test_statistic": np.nan,
            "critical_value": np.nan,
        }

    if method == "engle-granger":
        # Test de Engle-Granger (2-step method)
        # ==========================================
        # Paso 1: Regresión OLS (Ordinary Least Squares)
        #   - Ajusta: P_A = α + β·P_B + residuos
        #   - Encuentra β (hedge ratio) que minimiza varianza de residuos
        #
        # Paso 2: Test ADF (Augmented Dickey-Fuller) sobre residuos
        #   - H0: Los residuos NO son estacionarios (tienen raíz unitaria)
        #   - H1: Los residuos SÍ son estacionarios (revierten a la media)
        #   - Si p-value < 0.05: rechazamos H0 → hay cointegración ✓

        # statsmodels.coint() ejecuta ambos pasos automáticamente
        score, pvalue, crit_value = coint(df["a"].values, df["b"].values)

        return {
            "pvalue": pvalue,  # Probabilidad de observar estos datos si NO hay cointegración
            "is_cointegrated": pvalue < significance,  # True si p < 0.05 (cointegrados)
            "test_statistic": score,  # Estadístico ADF (más negativo = más evidencia)
            "critical_value": crit_value[1],  # Valor crítico al 5% de significancia
        }

    elif method == "johansen":
        # Test de Johansen (multivariado)
        # ==================================
        # Diferencia con Engle-Granger:
        #   - E-G: Solo detecta 1 relación de cointegración
        #   - Johansen: Puede detectar múltiples relaciones (útil para >2 series)
        #
        # Para 2 series (pares), ambos métodos son equivalentes
        # Johansen es más robusto pero más complejo de interpretar

        # det_order=0: No incluir tendencia determinística en el modelo
        # k_ar_diff=1: Usar 1 lag en diferencias (modelo VAR(1))
        result = coint_johansen(df[["a", "b"]].values, det_order=0, k_ar_diff=1)

        # Usar trace statistic (estadístico de traza - más común que max eigenvalue)
        trace_stat = result.lr1[0]  # Primera estadística de traza
        crit_val_5pct = result.cvt[0, 1]  # Valor crítico al 5% de significancia

        # Aproximar p-value basado en comparación con valor crítico
        # (Johansen no retorna p-value directamente, solo estadísticos)
        is_coint = trace_stat > crit_val_5pct  # Si supera umbral → cointegrados
        approx_pvalue = 0.01 if is_coint else 0.10  # Aproximación conservadora

        return {
            "pvalue": approx_pvalue,
            "is_cointegrated": is_coint,
            "test_statistic": trace_stat,
            "critical_value": crit_val_5pct,
        }

    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_correlation_stability(
    price_a: pd.Series,
    price_b: pd.Series,
    window_days: int = 60,
) -> float:
    """
    Calcula estabilidad de correlación entre dos series.

    Args:
        price_a: Serie de precios del activo A
        price_b: Serie de precios del activo B
        window_days: Ventana para correlación rolling

    Returns:
        Coeficiente de variación de la correlación rolling
        (menor = más estable)
    """
    # Alinear series temporalmente (eliminar fechas faltantes)
    df = pd.DataFrame({"a": price_a, "b": price_b}).dropna()

    # Necesitamos al menos 2 ventanas de datos para calcular estabilidad
    if len(df) < window_days * 2:
        return np.nan

    # Calcular correlación en ventana móvil de 60 días
    # Esto muestra cómo cambia la relación entre activos en el tiempo
    rolling_corr = df["a"].rolling(window_days).corr(df["b"])

    # Coeficiente de variación (CV) = desviación estándar / media
    # Mide qué tan estable es la correlación:
    #   - CV bajo (< 0.3): correlación estable → buen par
    #   - CV alto (> 0.5): correlación errática → mal par
    mean_corr = rolling_corr.mean()  # Correlación promedio
    std_corr = rolling_corr.std()    # Volatilidad de la correlación

    # Evitar división por cero
    if abs(mean_corr) < 1e-6:
        return np.nan

    # CV = qué porcentaje de la media es la desviación estándar
    cv = std_corr / abs(mean_corr)
    return cv


def find_cointegrated_pairs(
    price_data: Dict[str, pd.Series],
    pair_candidates: List[Tuple[str, str]],
    *,
    method: str = "engle-granger",
    min_pvalue: float = 0.05,
    min_correlation: float = 0.7,
    max_corr_cv: float = 0.3,
    min_data_points: int = 252,
) -> pd.DataFrame:
    """
    Encuentra pares cointegrados de una lista de candidatos.

    Args:
        price_data: Dict de ticker -> pd.Series con precios
        pair_candidates: Lista de tuplas (ticker_a, ticker_b)
        method: Método de cointegración ("engle-granger" o "johansen")
        min_pvalue: p-value máximo para considerar cointegración
        min_correlation: Correlación mínima requerida
        max_corr_cv: CV de correlación máximo (estabilidad)
        min_data_points: Puntos mínimos de datos requeridos

    Returns:
        DataFrame con pares cointegrados y sus estadísticas:
        - ticker_a, ticker_b: Nombres de los activos
        - pvalue: p-value del test de cointegración
        - correlation: Correlación Pearson promedio
        - corr_stability: CV de correlación rolling
        - n_points: Número de puntos de datos
    """

    results = []

    for ticker_a, ticker_b in pair_candidates:
        # Validar que ambos tickers existen
        if ticker_a not in price_data or ticker_b not in price_data:
            logger.warning(f"Missing data for pair {ticker_a}-{ticker_b}")
            continue

        price_a = price_data[ticker_a]
        price_b = price_data[ticker_b]

        # Alinear series
        df = pd.DataFrame({"a": price_a, "b": price_b}).dropna()

        if len(df) < min_data_points:
            logger.info(
                f"Insufficient data for {ticker_a}-{ticker_b}: "
                f"{len(df)} < {min_data_points}"
            )
            continue

        # Calcular correlación
        corr = df["a"].corr(df["b"])

        if corr < min_correlation:
            logger.info(
                f"Low correlation for {ticker_a}-{ticker_b}: {corr:.3f} < {min_correlation}"
            )
            continue

        # Test de cointegración
        coint_result = test_pair_cointegration(
            df["a"], df["b"], method=method, significance=min_pvalue
        )

        if not coint_result["is_cointegrated"]:
            logger.info(
                f"Not cointegrated: {ticker_a}-{ticker_b}, "
                f"pvalue={coint_result['pvalue']:.4f}"
            )
            continue

        # Calcular estabilidad de correlación
        corr_cv = calculate_correlation_stability(df["a"], df["b"])

        if corr_cv > max_corr_cv:
            logger.info(
                f"Unstable correlation for {ticker_a}-{ticker_b}: "
                f"CV={corr_cv:.3f} > {max_corr_cv}"
            )
            continue

        # Par válido - agregar a resultados
        results.append({
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "pvalue": coint_result["pvalue"],
            "test_statistic": coint_result["test_statistic"],
            "correlation": corr,
            "corr_stability": corr_cv,
            "n_points": len(df),
        })

        logger.info(
            f"✓ Cointegrated pair: {ticker_a}-{ticker_b}, "
            f"pvalue={coint_result['pvalue']:.4f}, corr={corr:.3f}"
        )

    if not results:
        logger.warning("No cointegrated pairs found!")
        return pd.DataFrame()

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("pvalue")

    return df_results


def load_pair_prices(
    data_dir: Path,
    ticker_a: str,
    ticker_b: str,
) -> Tuple[pd.Series, pd.Series]:
    """
    Carga datos de precios para un par de activos.

    Args:
        data_dir: Directorio con archivos CSV (e.g., data/clean/)
        ticker_a: Ticker del activo A
        ticker_b: Ticker del activo B

    Returns:
        Tupla (price_a, price_b) como pd.Series con timestamp index
    """

    # Construir rutas de archivos
    file_a = data_dir / f"{ticker_a}_daily.csv"
    file_b = data_dir / f"{ticker_b}_daily.csv"

    if not file_a.exists():
        raise FileNotFoundError(f"Data file not found: {file_a}")
    if not file_b.exists():
        raise FileNotFoundError(f"Data file not found: {file_b}")

    # Cargar datos
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)

    # Preparar series de precios
    if "timestamp" not in df_a.columns or "timestamp" not in df_b.columns:
        raise ValueError("Missing 'timestamp' column")

    df_a["timestamp"] = pd.to_datetime(df_a["timestamp"], format='ISO8601', utc=True)
    df_b["timestamp"] = pd.to_datetime(df_b["timestamp"], format='ISO8601', utc=True)

    df_a = df_a.set_index("timestamp").sort_index()
    df_b = df_b.set_index("timestamp").sort_index()

    if "close" not in df_a.columns or "close" not in df_b.columns:
        raise ValueError("Missing 'close' column")

    price_a = df_a["close"]
    price_b = df_b["close"]

    return price_a, price_b
