"""
Descubrimiento automático de pares cointegrados.

A partir de un universo de tickers, encuentra todos los pares
que están cointegrados y son candidatos para pairs trading.
"""

from __future__ import annotations

import logging
import hashlib
from datetime import datetime
from pathlib import Path
from itertools import combinations
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

from .select_pairs import test_pair_cointegration, load_pair_prices, calculate_correlation_stability
from .ou_process import estimate_ou_parameters

logger = logging.getLogger(__name__)


def _get_cache_key(
    tickers: List[str],
    pvalue_max: float,
    half_life_min: float,
    half_life_max: float,
    correlation_min: float,
    method: str,
) -> str:
    """
    Genera una clave única para los parámetros de discovery.

    Combina los tickers ordenados y parámetros para crear un hash.
    """
    # Ordenar tickers para que el orden no afecte el hash
    sorted_tickers = sorted(tickers)

    # Crear string con todos los parámetros
    param_str = (
        f"tickers={'_'.join(sorted_tickers)}_"
        f"pv={pvalue_max}_"
        f"hlmin={half_life_min}_"
        f"hlmax={half_life_max}_"
        f"corr={correlation_min}_"
        f"method={method}"
    )

    # Generar hash corto
    return hashlib.md5(param_str.encode()).hexdigest()[:12]


def _get_cache_path(data_dir: Path, cache_key: str, date: str) -> Path:
    """Obtiene la ruta del archivo de caché para una fecha y configuración."""
    cache_dir = data_dir.parent / "cache" / "pair_discovery"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"discovery_{date}_{cache_key}.parquet"


def _load_cached_results(cache_path: Path, verbose: bool = True) -> Optional[pd.DataFrame]:
    """Carga resultados cacheados si existen."""
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            if verbose:
                print(f"✓ Usando resultados cacheados: {cache_path.name}")
                print(f"  Cache creado: {datetime.fromtimestamp(cache_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  Pares en cache: {len(df)}")
            return df
        except Exception as e:
            logger.warning(f"Error cargando cache {cache_path}: {e}")
            return None
    return None


def _save_cache_results(df: pd.DataFrame, cache_path: Path, verbose: bool = True):
    """Guarda resultados en cache."""
    try:
        df.to_parquet(cache_path, index=False, compression='snappy')
        if verbose:
            print(f"✓ Resultados guardados en cache: {cache_path.name}")
    except Exception as e:
        logger.warning(f"Error guardando cache {cache_path}: {e}")


def discover_cointegrated_pairs(
    tickers: List[str],
    data_dir: Path,
    *,
    pvalue_max: float = 0.05,
    half_life_min: float = 5.0,
    half_life_max: float = 360.0,
    correlation_min: float = 0.5,
    method: str = "engle-granger",
    verbose: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Descubre todos los pares cointegrados en un universo de tickers.

    Implementa sistema de caché por día: si ya se ejecutó el discovery
    con los mismos parámetros en el día actual, usa resultados guardados.

    Args:
        tickers: Lista de tickers a analizar
        data_dir: Directorio con archivos de precios limpios
        pvalue_max: P-value máximo para aceptar cointegración (default: 0.05)
        half_life_min: Half-life mínimo en días (default: 5)
        half_life_max: Half-life máximo en días (default: 360)
        correlation_min: Correlación mínima requerida (default: 0.5)
        method: Método de cointegración ('engle-granger' o 'johansen')
        verbose: Mostrar progreso
        use_cache: Si True, usa cache del día si existe (default: True)

    Returns:
        DataFrame con pares cointegrados y sus métricas:
        - ticker_a, ticker_b: Tickers del par
        - pvalue: P-value del test de cointegración
        - test_statistic: Estadístico del test
        - correlation: Correlación de precios
        - correlation_stability: Estabilidad de correlación (CV)
        - ou_half_life: Half-life del proceso OU
        - ou_theta, ou_mu, ou_sigma: Parámetros OU
        - is_stationary: Si el spread es estacionario
    """

    # Verificar cache
    if use_cache:
        today = datetime.now().strftime('%Y%m%d')
        cache_key = _get_cache_key(
            tickers, pvalue_max, half_life_min,
            half_life_max, correlation_min, method
        )
        cache_path = _get_cache_path(data_dir, cache_key, today)

        # Intentar cargar desde cache
        cached_results = _load_cached_results(cache_path, verbose)
        if cached_results is not None:
            return cached_results

    if verbose:
        print(f"Descubriendo pares cointegrados de {len(tickers)} tickers...")
        print(f"Parámetros:")
        print(f"  - P-value max: {pvalue_max}")
        print(f"  - Half-life range: [{half_life_min}, {half_life_max}] días")
        print(f"  - Correlación min: {correlation_min}")
        print()

    # Generar todas las combinaciones de pares
    all_pairs = list(combinations(tickers, 2))
    n_pairs = len(all_pairs)

    logger.info(f"Testing {n_pairs} pairs from {len(tickers)} tickers...")

    results = []

    # Iterar sobre todos los pares
    iterator = tqdm(all_pairs, desc="Discovering pairs") if verbose else all_pairs

    for ticker_a, ticker_b in iterator:
        try:
            # Cargar datos
            price_a, price_b = load_pair_prices(data_dir, ticker_a, ticker_b)

            if len(price_a) < 252:  # Mínimo 1 año de datos
                if verbose:
                    print(f"✗ {ticker_a}-{ticker_b}: insufficient data ({len(price_a)} days)")
                continue

            # Test de cointegración
            coint_result = test_pair_cointegration(
                price_a, price_b, method=method
            )

            if verbose:
                print(f"Testing {ticker_a}-{ticker_b}: pvalue={coint_result['pvalue']:.4f}, cointegrated={coint_result['is_cointegrated']}")

            # Filtro 1: Cointegración
            if not coint_result['is_cointegrated'] or coint_result['pvalue'] > pvalue_max:
                if verbose:
                    print(f"  ✗ Rejected: pvalue {coint_result['pvalue']:.4f} > {pvalue_max}")
                continue

            # Calcular correlación
            correlation = price_a.corr(price_b)

            # Filtro 2: Correlación mínima
            if abs(correlation) < correlation_min:
                if verbose:
                    print(f"  ✗ Rejected: correlation {correlation:.3f} < {correlation_min}")
                continue

            # Estabilidad de correlación
            corr_stability = calculate_correlation_stability(
                price_a, price_b, window_days=60
            )

            # Construir spread usando la misma metodología que train_e4_pipeline
            # Esto garantiza que los parámetros OU sean consistentes
            from .build_spread import build_pair_features

            beta_window = 120  # Mismo valor que en train_e4_pipeline
            spread_features = build_pair_features(
                price_a, price_b,
                volume_a=None, volume_b=None,
                beta_window=beta_window,
                zscore_window=60
            )

            spread = spread_features["spread"]

            # Estimar parámetros OU
            ou_params = estimate_ou_parameters(spread)

            if verbose:
                print(f"  OU: half_life={ou_params['half_life']:.1f}d, theta={ou_params['theta']:.4f}")

            # Filtro 3: Half-life razonable
            if ou_params['half_life'] < half_life_min or ou_params['half_life'] > half_life_max:
                if verbose:
                    print(f"  ✗ Rejected: half_life {ou_params['half_life']:.1f} outside [{half_life_min}, {half_life_max}]")
                continue

            # Par válido - guardar resultados
            results.append({
                'ticker_a': ticker_a,
                'ticker_b': ticker_b,
                'pvalue': coint_result['pvalue'],
                'test_statistic': coint_result['test_statistic'],
                'correlation': correlation,
                'correlation_stability': corr_stability,
                'ou_half_life': ou_params['half_life'],
                'ou_theta': ou_params['theta'],
                'ou_mu': ou_params['mu'],
                'ou_sigma': ou_params['sigma'],
                'is_stationary': ou_params.get('is_stationary', False),
            })

            if verbose:
                print(
                    f"  ✓ ACCEPTED: {ticker_a}-{ticker_b} | "
                    f"p={coint_result['pvalue']:.4f}, "
                    f"hl={ou_params['half_life']:.1f}d, "
                    f"corr={correlation:.3f}"
                )

        except Exception as e:
            if verbose:
                print(f"✗ {ticker_a}-{ticker_b}: ERROR - {e}")
            continue

    # Crear DataFrame con resultados
    if results:
        df_pairs = pd.DataFrame(results)

        # Ordenar por calidad (p-value bajo, half-life razonable)
        df_pairs['quality_score'] = (
            (1 - df_pairs['pvalue']) * 0.4 +  # 40% peso en cointegración
            (1 - df_pairs['correlation_stability']) * 0.3 +  # 30% en estabilidad
            np.clip(30 / df_pairs['ou_half_life'], 0, 1) * 0.3  # 30% en half-life (ideal ~30 días)
        )

        df_pairs = df_pairs.sort_values('quality_score', ascending=False)

        # Guardar en cache si está habilitado
        if use_cache:
            _save_cache_results(df_pairs, cache_path, verbose)

        logger.info(f"Discovered {len(df_pairs)} cointegrated pairs")
        return df_pairs
    else:
        logger.warning("No cointegrated pairs found")
        # Guardar cache vacío para evitar recálculo
        if use_cache:
            empty_df = pd.DataFrame()
            _save_cache_results(empty_df, cache_path, verbose)
        return pd.DataFrame()


def filter_best_pairs(
    pairs_df: pd.DataFrame,
    *,
    max_pairs: int | None = None,
    min_quality_score: float = 0.5,
    max_half_life: float = 60.0,
) -> pd.DataFrame:
    """
    Filtra los mejores pares según criterios de calidad.

    Args:
        pairs_df: DataFrame con pares descubiertos
        max_pairs: Número máximo de pares a retornar
        min_quality_score: Score mínimo de calidad
        max_half_life: Half-life máximo permitido

    Returns:
        DataFrame filtrado con los mejores pares
    """

    if pairs_df.empty:
        return pairs_df

    # Filtrar por quality score
    filtered = pairs_df[pairs_df['quality_score'] >= min_quality_score].copy()

    # Filtrar por half-life
    filtered = filtered[filtered['ou_half_life'] <= max_half_life]

    # Limitar cantidad
    if max_pairs is not None:
        filtered = filtered.head(max_pairs)

    logger.info(
        f"Filtered to {len(filtered)} pairs "
        f"(quality>={min_quality_score}, hl<={max_half_life}d)"
    )

    return filtered


def pairs_to_list(pairs_df: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Convierte DataFrame de pares a lista de tuplas.

    Args:
        pairs_df: DataFrame con columnas 'ticker_a' y 'ticker_b'

    Returns:
        Lista de tuplas [(ticker_a, ticker_b), ...]
    """
    return list(zip(pairs_df['ticker_a'], pairs_df['ticker_b']))
