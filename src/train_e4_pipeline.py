"""
Pipeline completo E4 (Estrategia Pairs Trading).

Flujo:
1. Cargar datos de pares
2. Validar cointegración
3. Construir spreads y calcular z-scores
4. Estimar parámetros Ornstein-Uhlenbeck
5. Entrenar k-NN (opcional)
6. Generar señales de trading
7. Backtest con gestión dollar-neutral
8. Evaluar y guardar resultados
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .pairs.select_pairs import (
    find_cointegrated_pairs,
    load_pair_prices,
    test_pair_cointegration,
)
from .pairs.build_spread import (
    build_pair_features,
    calculate_hedge_ratio,
    validate_spread_stability,
)
from .pairs.ou_process import (
    estimate_ou_parameters,
    test_stationarity,
)
from .pairs.knn_confirm import (
    train_knn_model,
    build_knn_state_features,
    create_knn_target,
    cross_validate_knn,
)
from .backtest.rules_e4 import (
    generate_pair_signals,
    backtest_pair_strategy,
    summarize_pair_backtest,
)
from .utils import ensure_dir, load_yaml, project_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Carga configuración desde base.yaml."""
    config_path = project_root() / "src" / "config" / "base.yaml"
    return load_yaml(config_path)


def process_pair(
    ticker_a: str,
    ticker_b: str,
    data_dir: Path,
    config: dict,
    output_dir: Path,
) -> Dict | None:
    """
    Procesa un par completo: validación, spread, OU, k-NN, backtest.
    
    Args:
        ticker_a: Ticker del activo A
        ticker_b: Ticker del activo B
        data_dir: Directorio con datos limpios
        config: Configuración de la estrategia E4
        output_dir: Directorio de salida
    
    Returns:
        Dict con resultados del par, o None si el par no es válido
    """
    
    pair_name = f"{ticker_a}_{ticker_b}"
    logger.info(f"Processing pair: {pair_name}")
    
    # Crear directorio para el par
    pair_dir = output_dir / pair_name
    ensure_dir(pair_dir)
    
    # 1. Cargar datos
    try:
        price_a, price_b = load_pair_prices(data_dir, ticker_a, ticker_b)
    except FileNotFoundError as e:
        logger.error(f"Data not found for {pair_name}: {e}")
        return None
    
    # Cargar también volumen (opcional)
    try:
        file_a = data_dir / f"{ticker_a}_daily.csv"
        file_b = data_dir / f"{ticker_b}_daily.csv"
        df_a = pd.read_csv(file_a)
        df_b = pd.read_csv(file_b)
        df_a["timestamp"] = pd.to_datetime(df_a["timestamp"], format='ISO8601', utc=True)
        df_b["timestamp"] = pd.to_datetime(df_b["timestamp"], format='ISO8601', utc=True)
        df_a = df_a.set_index("timestamp")
        df_b = df_b.set_index("timestamp")
        volume_a = df_a.get("volume")
        volume_b = df_b.get("volume")
    except Exception:
        volume_a = None
        volume_b = None
    
    # 2. Test de cointegración
    coint_config = config.get("filters", {})
    max_pvalue = coint_config.get("cointegration_pvalue_max", 0.05)
    
    coint_result = test_pair_cointegration(price_a, price_b, method="engle-granger")
    
    if not coint_result["is_cointegrated"]:
        logger.warning(
            f"{pair_name} not cointegrated: pvalue={coint_result['pvalue']:.4f}"
        )
        return None
    
    logger.info(
        f"{pair_name} cointegrated: pvalue={coint_result['pvalue']:.4f}"
    )
    
    # 3. Construir spread y features
    beta_window = config.get("beta_lookback_days", 120)
    zscore_window = 60
    
    spread_features = build_pair_features(
        price_a,
        price_b,
        volume_a=volume_a,
        volume_b=volume_b,
        beta_window=beta_window,
        zscore_window=zscore_window,
    )
    
    spread = spread_features["spread"]
    zscore = spread_features["zscore"]
    
    # Guardar spread timeseries
    spread_df = pd.DataFrame({
        "spread": spread,
        "zscore": zscore,
        "beta": spread_features.get("beta_rolling", np.nan),
    })
    spread_csv = pair_dir / "spread_timeseries.csv"
    spread_df.to_csv(spread_csv)
    logger.info(f"Saved spread to {spread_csv}")
    
    # 4. Estimar parámetros OU
    ou_params = estimate_ou_parameters(spread)
    
    # Validar half-life
    max_half_life = coint_config.get("half_life_days_max", 20)
    if ou_params["half_life"] > max_half_life:
        logger.warning(
            f"{pair_name} half-life too long: {ou_params['half_life']:.1f} > {max_half_life}"
        )
        # No rechazar automáticamente, pero advertir
    
    logger.info(
        f"{pair_name} OU params: theta={ou_params['theta']:.4f}, "
        f"mu={ou_params['mu']:.4f}, sigma={ou_params['sigma']:.4f}, "
        f"half_life={ou_params['half_life']:.1f} days"
    )
    
    # Guardar parámetros OU
    ou_json = pair_dir / "ou_params.json"
    with open(ou_json, "w") as f:
        json.dump(ou_params, f, indent=2)
    
    # Test de estacionariedad
    stationarity = test_stationarity(spread)
    logger.info(
        f"{pair_name} stationarity test: pvalue={stationarity['pvalue']:.4f}, "
        f"stationary={stationarity['is_stationary']}"
    )
    
    # 5. k-NN confirmación (opcional)
    knn_config = config.get("knn", {})
    use_knn = knn_config.get("enabled", True)
    knn_signals = None
    
    if use_knn:
        logger.info(f"Training k-NN for {pair_name}")
        
        # Construir estado
        state_features = build_knn_state_features(spread_features)
        
        # Target: cambio en spread
        horizon = config.get("horizon_days", 10)
        target = create_knn_target(spread, horizon_days=horizon)
        
        # Cross-validation para seleccionar k
        k_candidates = knn_config.get("k_candidates", [5, 10, 20])
        cv_results = cross_validate_knn(
            state_features, target, k_candidates=k_candidates, n_folds=3
        )
        
        # Mejor k
        best_k = cv_results.iloc[0]["k"]
        logger.info(f"Best k for {pair_name}: {best_k}")
        
        # Entrenar modelo final
        knn_model = train_knn_model(
            state_features, target, k=int(best_k), test_size=0.2
        )
        
        # Generar señales k-NN
        from .pairs.knn_confirm import generate_knn_signals
        knn_signals = generate_knn_signals(
            spread_features, spread, k=int(best_k), horizon_days=horizon
        )
    
    # 6. Generar señales de trading
    entry_exit_config = config.get("entry_exit", {})
    entry_z = entry_exit_config.get("entry_z", 2.0)
    exit_z = entry_exit_config.get("exit_z", 0.25)
    stop_z = entry_exit_config.get("stop_z", 3.0)
    time_stop_days = entry_exit_config.get("time_stop_days", 20)
    
    signals = generate_pair_signals(
        zscore,
        spread,
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z,
        time_stop_days=time_stop_days,
        use_knn=use_knn,
        knn_signals=knn_signals,
    )
    
    # 7. Backtest
    round_trip_bps = 20.0  # 2 activos x 10 bps
    
    # Obtener hedge ratio (beta) para backtest dollar-neutral
    # spread_features es un DataFrame con columna "beta_rolling"
    beta = spread_features["beta_rolling"]
    
    backtest_results = backtest_pair_strategy(
        price_a,
        price_b,
        signals,
        hedge_ratio=beta,
        round_trip_bps=round_trip_bps,
        initial_capital=100000.0,
        position_size=0.25,
    )
    
    # Guardar trades
    trades_csv = pair_dir / "trades.csv"
    trades_df = pd.concat([signals, backtest_results], axis=1)
    trades_df.to_csv(trades_csv)
    logger.info(f"Saved trades to {trades_csv}")
    
    # 8. Resumir resultados
    summary = summarize_pair_backtest(
        backtest_results, signals, ticker_a, ticker_b
    )
    
    # Agregar info adicional
    summary.update({
        "cointegration_pvalue": coint_result["pvalue"],
        "ou_half_life": ou_params["half_life"],
        "ou_theta": ou_params["theta"],
        "stationarity_pvalue": stationarity["pvalue"],
    })
    
    # Guardar summary
    summary_json = pair_dir / "backtest_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(
        f"{pair_name} summary: return={summary['total_return']:.2%}, "
        f"sharpe={summary['sharpe']:.2f}, num_trades={summary['num_trades']}"
    )
    
    return summary


def main(
    pairs: List[Tuple[str, str]] | None = None,
    output_tag: str | None = None,
):
    """
    Pipeline principal E4.
    
    Args:
        pairs: Lista de tuplas (ticker_a, ticker_b). Si None, usar de config.
        output_tag: Tag para directorio de salida. Si None, usar timestamp.
    """
    
    logger.info("=" * 80)
    logger.info("E4 Pairs Trading Pipeline")
    logger.info("=" * 80)
    
    # Cargar configuración
    config = load_config()
    e4_config = config.get("strategies", {}).get("e4_pairs", {})
    
    if not e4_config.get("enabled", True):
        logger.error("E4 strategy is not enabled in config")
        return
    
    # Obtener lista de pares
    if pairs is None:
        pairs_config = e4_config.get("pairs", [])
        if not pairs_config:
            logger.error("No pairs defined in config")
            return
        # Convertir de lista de listas a tuplas
        pairs = [(p[0], p[1]) for p in pairs_config]
    
    logger.info(f"Processing {len(pairs)} pairs")
    
    # Directorios
    data_dir = project_root() / "data" / "clean"
    
    if output_tag is None:
        output_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = project_root() / "runs" / "e4_pairs" / output_tag
    ensure_dir(output_dir)
    
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Guardar configuración usada
    config_used = output_dir / "config_used.yaml"
    import yaml
    with open(config_used, "w") as f:
        yaml.dump({"e4_pairs": e4_config}, f, default_flow_style=False)
    
    # Procesar cada par
    all_results = []
    
    for ticker_a, ticker_b in pairs:
        try:
            result = process_pair(
                ticker_a, ticker_b, data_dir, e4_config, output_dir
            )
            if result is not None:
                all_results.append(result)
        except Exception as e:
            logger.error(f"Error processing {ticker_a}-{ticker_b}: {e}", exc_info=True)
            continue
    
    # Guardar resumen agregado
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_csv = output_dir / "summary_all_pairs.csv"
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f"Saved summary to {summary_csv}")
        
        # Estadísticas agregadas
        logger.info("=" * 80)
        logger.info("Aggregate Statistics")
        logger.info("=" * 80)
        logger.info(f"Pairs processed: {len(all_results)}")
        logger.info(f"Average return: {summary_df['total_return'].mean():.2%}")
        logger.info(f"Average Sharpe: {summary_df['sharpe'].mean():.2f}")
        logger.info(f"Average win rate: {summary_df['win_rate'].mean():.2%}")
        logger.info(f"Total trades: {summary_df['num_trades'].sum()}")
        
        # Mejor y peor par
        best_idx = summary_df['sharpe'].idxmax()
        worst_idx = summary_df['sharpe'].idxmin()
        
        logger.info(f"\nBest pair: {summary_df.loc[best_idx, 'pair']} "
                   f"(Sharpe={summary_df.loc[best_idx, 'sharpe']:.2f})")
        logger.info(f"Worst pair: {summary_df.loc[worst_idx, 'pair']} "
                   f"(Sharpe={summary_df.loc[worst_idx, 'sharpe']:.2f})")
    else:
        logger.warning("No pairs successfully processed")
    
    logger.info("=" * 80)
    logger.info("Pipeline complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="E4 Pairs Trading Pipeline"
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        help="Pairs to process (format: TICKER_A,TICKER_B). "
             "Example: --pairs GGAL.BA,BMA.BA YPFD.BA,PAMP.BA",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        help="Tag for output directory (default: timestamp)",
    )
    
    args = parser.parse_args()
    
    # Parse pairs
    pairs = None
    if args.pairs:
        pairs = []
        for pair_str in args.pairs:
            parts = pair_str.split(",")
            if len(parts) != 2:
                logger.error(f"Invalid pair format: {pair_str}")
                continue
            pairs.append((parts[0], parts[1]))
    
    main(pairs=pairs, output_tag=args.output_tag)
