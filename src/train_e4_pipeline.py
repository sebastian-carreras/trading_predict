"""
Pipeline completo E4 (Estrategia Pairs Trading).

Characteristics:
- Identifica pares de activos cointegrados (spread estacionario)
- Estima parámetros Ornstein-Uhlenbeck del spread
- Entrena modelo k-NN para confirmar señales
- Genera señales cuando spread desviado de equilibrio
- Backtest con gestión dollar-neutral (beta hedge)
- Evalúa métricas de trading

Flujo:
1. Cargar datos de pares (precios históricos)
2. Validar cointegración (ADF test)
3. Construir spreads y calcular z-scores
4. Estimar parámetros Ornstein-Uhlenbeck
5. Test de estacionariedad del spread
6. Entrenar k-NN (opcional) para confirmar señales
7. Generar señales de trading (entrada/salida)
8. Backtest con gestión dollar-neutral
9. Evaluar y guardar resultados
"""

from __future__ import annotations  # Forward refs

import argparse  # CLI args
from datetime import datetime  # Timestamp
import json  # JSON
import logging  # Logging
import os  # Env vars
from pathlib import Path  # Paths
from typing import Dict, List, Tuple  # Typing

import numpy as np  # NumPy
import pandas as pd  # Pandas

from .pairs.select_pairs import (  # Selección de pares
    find_cointegrated_pairs,        # Identifica pares cointegrados
    load_pair_prices,               # Carga precios de dos activos
    test_pair_cointegration,        # Test de cointegración ADF
)  # Fin import select_pairs
from .pairs.build_spread import (  # Construcción de spread
    build_pair_features,            # Construye features del spread
    calculate_hedge_ratio,          # Calcula ratio de hedge
    validate_spread_stability,      # Valida estabilidad del spread
)  # Fin import build_spread
from .pairs.ou_process import (  # OU process
    estimate_ou_parameters,         # Estima parámetros OU
    test_stationarity,              # Test ADF de estacionariedad
)  # Fin import ou_process
from .pairs.knn_confirm import (  # k-NN confirm
    train_knn_model,                # Entrena modelo k-NN
    build_knn_state_features,       # Features para k-NN
    create_knn_target,              # Target para k-NN
    cross_validate_knn,             # Validación cruzada k-NN
)  # Fin import knn_confirm
from .backtest.backtest_rules_e4 import (  # Backtest E4
    generate_pair_signals,          # Genera señales de entrada/salida
    backtest_pair_strategy,         # Simula ejecución de trades
    summarize_pair_backtest,        # Resumen de resultados
)  # Fin import backtest
from .utils import ensure_dir, load_yaml, project_root  # Utils

logging.basicConfig(  # Config logging
    level=logging.INFO,  # Nivel
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",  # Formato
)  # Fin config logging
logger = logging.getLogger(__name__)  # Logger


def load_config() -> dict:  # Cargar config
    """Carga configuración desde archivo base.yaml.
    
    Retorna:
        dict: Configuración con estrategias, modelos, splits, etc
    """
    config_path = project_root() / "src" / "config" / "base.yaml"  # Path config
    return load_yaml(config_path)  # Cargar YAML


def process_pair(  # Procesar par
    ticker_a: str,  # Ticker A
    ticker_b: str,  # Ticker B
    data_dir: Path,  # Dir datos
    config: dict,  # Config
    output_dir: Path,  # Dir salida
) -> Dict | None:  # Retorna dict o None
    """
    Procesa un par completo: validación, spread, OU, k-NN, backtest.
    
    Procedimiento:
    1. Carga precios de ambos activos
    2. Valida cointegración (pvalue < 0.05)
    3. Construye spread y features (beta, zscore)
    4. Estima parámetros del proceso OU
    5. Valida estacionariedad del spread
    6. Entrena modelo k-NN (opcional) para confirmar señales
    7. Genera señales de trading
    8. Ejecuta backtest con rebalanceo diario
    9. Guarda resultados
    
    Args:
        ticker_a: Ticker del activo A (ej: "AAPL")
        ticker_b: Ticker del activo B (ej: "MSFT")
        data_dir: Directorio con datos limpios (CSV)
        config: Configuración de la estrategia E4
        output_dir: Directorio para salidas
    
    Returns:
        Dict con resultados del par, o None si el par no es válido
    """
    
    pair_name = f"{ticker_a}_{ticker_b}"  # Nombre par
    logger.info(f"Processing pair: {pair_name}")  # Log
    
    # Crear directorio para el par
    pair_dir = output_dir / pair_name  # Dir par
    ensure_dir(pair_dir)  # Crear dir
    
    # 1. CARGAR DATOS
    # Carga series de precios de cierre para ambos activos
    try:  # Cargar precios
        price_a, price_b = load_pair_prices(data_dir, ticker_a, ticker_b)  # Precios
    except FileNotFoundError as e:  # Error
        logger.error(f"Data not found for {pair_name}: {e}")  # Log error
        return None  # Abortar
    
    # Cargar también volumen (opcional, para filtros)
    try:  # Cargar volumen
        file_a = data_dir / f"{ticker_a}_daily.csv"  # Path A
        file_b = data_dir / f"{ticker_b}_daily.csv"  # Path B
        df_a = pd.read_csv(file_a)  # CSV A
        df_b = pd.read_csv(file_b)  # CSV B
        df_a["timestamp"] = pd.to_datetime(df_a["timestamp"], format='ISO8601', utc=True)  # Timestamp A
        df_b["timestamp"] = pd.to_datetime(df_b["timestamp"], format='ISO8601', utc=True)  # Timestamp B
        df_a = df_a.set_index("timestamp")  # Index A
        df_b = df_b.set_index("timestamp")  # Index B
        volume_a = df_a.get("volume")  # Volumen A
        volume_b = df_b.get("volume")  # Volumen B
    except Exception:  # Fallback
        volume_a = None  # Sin volumen A
        volume_b = None  # Sin volumen B
    
    # 2. TEST DE COINTEGRACIÓN
    # Verifica que el spread sea estacionario (condición para pairs trading)
    coint_config = config.get("filters", {})  # Config filtros
    max_pvalue = coint_config.get("cointegration_pvalue_max", 0.05)  # Pvalue máx
    
    coint_result = test_pair_cointegration(price_a, price_b, method="engle-granger")  # Test
    
    if not coint_result["is_cointegrated"]:  # No cointegrado
        logger.warning(  # Warning
            f"{pair_name} not cointegrated: pvalue={coint_result['pvalue']:.4f}"  # Msg
        )  # Fin warning
        return None  # Abortar
    
    logger.info(  # Log
        f"{pair_name} cointegrated: pvalue={coint_result['pvalue']:.4f}"  # Msg
    )  # Fin log
    
    # 3. CONSTRUIR SPREAD Y FEATURES
    # Calcula el spread (diferencia hedgeada entre los precios)
    # Incluye: zscore (desviaciones estándar del spread), beta, volumen
    beta_window = config.get("beta_lookback_days", 120)  # Ventana beta
    zscore_window = 60  # Ventana zscore
    
    spread_features = build_pair_features(  # Features spread
        price_a,  # Precio A
        price_b,  # Precio B
        volume_a=volume_a,  # Volumen A
        volume_b=volume_b,  # Volumen B
        beta_window=beta_window,  # Ventana beta
        zscore_window=zscore_window,  # Ventana zscore
    )  # Fin build
    
    spread = spread_features["spread"]  # Spread
    zscore = spread_features["zscore"]  # Zscore
    
    # Guardar spread timeseries para análisis
    spread_df = pd.DataFrame({  # DF spread
        "spread": spread,  # Spread
        "zscore": zscore,  # Zscore
        "beta": spread_features.get("beta_rolling", np.nan),  # Beta
    })  # Fin DF
    spread_csv = pair_dir / "spread_timeseries.csv"  # Path CSV
    spread_df.to_csv(spread_csv)  # Guardar CSV
    logger.info(f"Saved spread to {spread_csv}")  # Log
    
    # 4. ESTIMAR PARÁMETROS OU (Ornstein-Uhlenbeck)
    # Modela el spread como proceso de reversión a la media:
    # d(spread) = theta * (mu - spread) * dt + sigma * dW
    # theta: velocidad de reversión (mayor = más rápida reversión)
    # mu: media de largo plazo
    # sigma: volatilidad instantánea
    ou_params = estimate_ou_parameters(spread)  # Params OU
    
    # Validar half-life (tiempo para revertir 50% a la media)
    max_half_life = coint_config.get("half_life_days_max", 20)  # Half-life máx
    if ou_params["half_life"] > max_half_life:  # Validar half-life
        logger.warning(  # Warning
            f"{pair_name} half-life too long: {ou_params['half_life']:.1f} > {max_half_life}"  # Msg
        )  # Fin warning
        # No rechazar automáticamente, pero advertir
    
    logger.info(  # Log OU
        f"{pair_name} OU params: theta={ou_params['theta']:.4f}, "  # Theta
        f"mu={ou_params['mu']:.4f}, sigma={ou_params['sigma']:.4f}, "  # Mu/Sigma
        f"half_life={ou_params['half_life']:.1f} days"  # Half-life
    )  # Fin log OU
    
    # Guardar parámetros OU para futura inferencia
    ou_json = pair_dir / "ou_params.json"  # Path OU
    with open(ou_json, "w") as f:  # Abrir
        json.dump(ou_params, f, indent=2)  # Guardar
    
    # 5. TEST DE ESTACIONARIEDAD
    stationarity = test_stationarity(spread)  # Test
    logger.info(  # Log
        f"{pair_name} stationarity test: pvalue={stationarity['pvalue']:.4f}, "  # Pvalue
        f"stationary={stationarity['is_stationary']}"  # Flag
    )  # Fin log stationarity
    
    # 6. k-NN CONFIRMACIÓN (OPCIONAL)
    # Entrena modelo k-NN para confirmar/refinar señales del spread zscore
    knn_config = config.get("knn", {})  # Config kNN
    use_knn = knn_config.get("enabled", True)  # Flag
    knn_signals = None  # Señales kNN
    
    if use_knn:  # Si usa kNN
        logger.info(f"Training k-NN for {pair_name}")  # Log
        
        # Construir estado
        state_features = build_knn_state_features(spread_features)  # Features
        
        # Target: cambio en spread
        horizon = config.get("horizon_days", 10)  # Horizon
        target = create_knn_target(spread, horizon_days=horizon)  # Target
        
        # Cross-validation para seleccionar k
        k_candidates = knn_config.get("k_candidates", [5, 10, 20])  # K candidatos
        cv_results = cross_validate_knn(  # Cross-validate
            state_features, target, k_candidates=k_candidates, n_folds=3  # Args
        )  # Fin cross-validate
        
        # Mejor k
        best_k = cv_results.iloc[0]["k"]  # Best k
        logger.info(f"Best k for {pair_name}: {best_k}")  # Log
        
        # Entrenar modelo final
        knn_model = train_knn_model(  # Entrenar
            state_features, target, k=int(best_k), test_size=0.2  # Args
        )  # Fin train
        
        # Generar señales k-NN
        from .pairs.knn_confirm import generate_knn_signals  # Import
        knn_signals = generate_knn_signals(  # Generar
            spread_features, spread, k=int(best_k), horizon_days=horizon  # Args
        )  # Fin generate
    
    # 6. Generar señales de trading
    entry_exit_config = config.get("entry_exit", {})  # Config entry/exit
    entry_z = entry_exit_config.get("entry_z", 2.0)  # Entry z
    exit_z = entry_exit_config.get("exit_z", 0.25)  # Exit z
    stop_z = entry_exit_config.get("stop_z", 3.0)  # Stop z
    time_stop_days = entry_exit_config.get("time_stop_days", 20)  # Time stop
    
    signals = generate_pair_signals(  # Señales
        zscore,  # Zscore
        spread,  # Spread
        entry_z=entry_z,  # Entry
        exit_z=exit_z,  # Exit
        stop_z=stop_z,  # Stop
        time_stop_days=time_stop_days,  # Time stop
        use_knn=use_knn,  # Use kNN
        knn_signals=knn_signals,  # Señales kNN
    )  # Fin señales
    
    # 7. Backtest
    round_trip_bps = 20.0  # 2 activos x 10 bps
    
    # Obtener hedge ratio (beta) para backtest dollar-neutral
    # spread_features es un DataFrame con columna "beta_rolling"
    beta = spread_features["beta_rolling"]  # Beta rolling
    
    backtest_results = backtest_pair_strategy(  # Backtest
        price_a,  # Precio A
        price_b,  # Precio B
        signals,  # Señales
        hedge_ratio=beta,  # Hedge ratio
        round_trip_bps=round_trip_bps,  # Costos
        initial_capital=100000.0,  # Capital
        position_size=0.25,  # Tamaño posición
    )  # Fin backtest
    
    # Guardar trades
    trades_csv = pair_dir / "trades.csv"  # Path trades
    trades_df = pd.concat([signals, backtest_results], axis=1)  # DF trades
    trades_df.to_csv(trades_csv)  # Guardar
    logger.info(f"Saved trades to {trades_csv}")  # Log
    
    # 8. Resumir resultados
    summary = summarize_pair_backtest(  # Resumen
        backtest_results, signals, ticker_a, ticker_b  # Args
    )  # Fin resumen
    
    # Agregar info adicional
    summary.update({  # Update summary
        "cointegration_pvalue": coint_result["pvalue"],  # Pvalue
        "ou_half_life": ou_params["half_life"],  # Half-life
        "ou_theta": ou_params["theta"],  # Theta
        "stationarity_pvalue": stationarity["pvalue"],  # Stationarity pvalue
    })  # Fin update
    
    # Guardar summary
    summary_json = pair_dir / "backtest_summary.json"  # Path summary
    with open(summary_json, "w") as f:  # Abrir
        json.dump(summary, f, indent=2)  # Guardar
    
    logger.info(  # Log
        f"{pair_name} summary: return={summary['total_return']:.2%}, "  # Return
        f"sharpe={summary['sharpe']:.2f}, num_trades={summary['num_trades']}"  # Sharpe/trades
    )  # Fin log summary
    
    return summary  # Retornar


def main(  # Main
    pairs: List[Tuple[str, str]] | None = None,  # Pairs
    output_tag: str | None = None,  # Tag salida
):  # Fin firma
    """
    Pipeline principal E4.
    
    Args:
        pairs: Lista de tuplas (ticker_a, ticker_b). Si None, usar de config.
        output_tag: Tag para directorio de salida. Si None, usar timestamp.
    """
    
    logger.info("=" * 80)  # Separador
    logger.info("E4 Pairs Trading Pipeline")  # Log
    logger.info("=" * 80)  # Separador
    
    # Cargar configuración
    config = load_config()  # Cargar config
    e4_config = config.get("strategies", {}).get("e4_pairs", {})  # Config E4
    
    if not e4_config.get("enabled", True):  # Validar enabled
        logger.error("E4 strategy is not enabled in config")  # Error
        return  # Salir
    
    # Obtener lista de pares
    if pairs is None:  # Sin pares explícitos
        pairs_config = e4_config.get("pairs", [])  # Config pares
        if not pairs_config:  # Validar
            logger.error("No pairs defined in config")  # Error
            return  # Salir
        # Convertir de lista de listas a tuplas
        pairs = [(p[0], p[1]) for p in pairs_config]  # Convertir
    
    logger.info(f"Processing {len(pairs)} pairs")  # Log
    
    # Directorios
    data_dir = project_root() / "data" / "clean"  # Dir data
    
    if output_tag is None:  # Tag default
        output_tag = datetime.now().strftime("%Y%m%d_%H%M%S")  # Timestamp
    
    output_dir = project_root() / "runs" / "e4_pairs" / output_tag  # Dir salida
    ensure_dir(output_dir)  # Crear dir
    
    logger.info(f"Data directory: {data_dir}")  # Log
    logger.info(f"Output directory: {output_dir}")  # Log
    
    # Guardar configuración usada
    config_used = output_dir / "config_used.yaml"  # Path config
    import yaml  # Import yaml
    with open(config_used, "w") as f:  # Abrir
        yaml.dump({"e4_pairs": e4_config}, f, default_flow_style=False)  # Guardar

    # MLflow setup (default: local tracking if installed)
    mlflow_enabled = False
    mlflow = None
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "E4_Pairs")
    try:
        import mlflow as _mlflow  # type: ignore

        if tracking_uri:
            _mlflow.set_tracking_uri(tracking_uri)
        _mlflow.set_experiment(experiment_name)
        mlflow = _mlflow
        mlflow_enabled = True
        if tracking_uri:
            logger.info(f"✓ MLflow habilitado: {tracking_uri} (experiment={experiment_name})")
        else:
            logger.info(f"✓ MLflow habilitado (tracking local, experiment={experiment_name})")
    except Exception as exc:
        logger.warning(f"MLflow no disponible, continuando sin tracking: {exc}")
        mlflow_enabled = False
    
    # Procesar cada par
    all_results = []  # Resultados
    
    for ticker_a, ticker_b in pairs:  # Loop pares
        pair_name = f"{ticker_a}_{ticker_b}"  # Nombre par
        pair_dir = output_dir / pair_name  # Dir par
        try:  # Procesar par
            if mlflow_enabled and mlflow is not None:
                with mlflow.start_run(run_name=f"E4_{pair_name}_{output_tag}"):
                    entry_exit = e4_config.get("entry_exit", {})
                    knn_cfg = e4_config.get("knn", {})
                    mlflow.log_param("strategy", "e4_pairs")
                    mlflow.log_param("pair", pair_name)
                    mlflow.log_param("output_tag", output_tag)
                    mlflow.log_params(
                        {
                            "beta_lookback_days": int(e4_config.get("beta_lookback_days", 120)),
                            "horizon_days": int(e4_config.get("horizon_days", 10)),
                            "entry_z": float(entry_exit.get("entry_z", 2.0)),
                            "exit_z": float(entry_exit.get("exit_z", 0.25)),
                            "stop_z": float(entry_exit.get("stop_z", 3.0)),
                            "time_stop_days": int(entry_exit.get("time_stop_days", 20)),
                            "knn_enabled": bool(knn_cfg.get("enabled", True)),
                            "knn_k_candidates": str(knn_cfg.get("k_candidates", [5, 10, 20])),
                        }
                    )

                    result = process_pair(  # Ejecutar
                        ticker_a, ticker_b, data_dir, e4_config, output_dir  # Args
                    )  # Fin process_pair

                    if result is not None:
                        metrics: dict[str, float] = {}
                        for k, v in result.items():
                            if isinstance(v, (int, float)):
                                metrics[k] = float(v)
                        if metrics:
                            mlflow.log_metrics(metrics)

                        for fname in [
                            "spread_timeseries.csv",
                            "trades.csv",
                            "backtest_summary.json",
                            "ou_params.json",
                        ]:
                            p = pair_dir / fname
                            if p.exists():
                                if "spread" in fname:
                                    artifact_path = "spread"
                                elif "trades" in fname:
                                    artifact_path = "trades"
                                elif "ou_params" in fname:
                                    artifact_path = "params"
                                else:
                                    artifact_path = "artifacts"
                                mlflow.log_artifact(str(p), artifact_path=artifact_path)
            else:
                result = process_pair(  # Ejecutar
                    ticker_a, ticker_b, data_dir, e4_config, output_dir  # Args
                )  # Fin process_pair

            if result is not None:  # Validar
                all_results.append(result)  # Append
        except Exception as e:  # Error par
            logger.error(f"Error processing {ticker_a}-{ticker_b}: {e}", exc_info=True)  # Log
            continue  # Siguiente
    
    # Guardar resumen agregado
    if all_results:  # Hay resultados
        summary_df = pd.DataFrame(all_results)  # DF summary
        summary_csv = output_dir / "summary_all_pairs.csv"  # Path summary
        summary_df.to_csv(summary_csv, index=False)  # Guardar
        logger.info(f"Saved summary to {summary_csv}")  # Log

        if mlflow_enabled and mlflow is not None:
            try:
                with mlflow.start_run(run_name=f"E4_Aggregate_{output_tag}"):
                    mlflow.log_param("strategy", "e4_pairs")
                    mlflow.log_param("output_tag", output_tag)
                    mlflow.log_metric("pairs_processed", float(len(all_results)))
                    mlflow.log_metric("avg_return", float(summary_df['total_return'].mean()))
                    mlflow.log_metric("avg_sharpe", float(summary_df['sharpe'].mean()))
                    mlflow.log_metric("avg_win_rate", float(summary_df['win_rate'].mean()))
                    mlflow.log_metric("total_trades", float(summary_df['num_trades'].sum()))
                    if summary_csv.exists():
                        mlflow.log_artifact(str(summary_csv), artifact_path="summaries")
            except Exception as exc:
                logger.warning(f"No se pudo loguear resumen agregado en MLflow: {exc}")
        
        # Estadísticas agregadas
        logger.info("=" * 80)  # Separador
        logger.info("Aggregate Statistics")  # Log
        logger.info("=" * 80)  # Separador
        logger.info(f"Pairs processed: {len(all_results)}")  # Log
        logger.info(f"Average return: {summary_df['total_return'].mean():.2%}")  # Avg return
        logger.info(f"Average Sharpe: {summary_df['sharpe'].mean():.2f}")  # Avg Sharpe
        logger.info(f"Average win rate: {summary_df['win_rate'].mean():.2%}")  # Avg win rate
        logger.info(f"Total trades: {summary_df['num_trades'].sum()}")  # Total trades
        
        # Mejor y peor par
        best_idx = summary_df['sharpe'].idxmax()  # Mejor
        worst_idx = summary_df['sharpe'].idxmin()  # Peor
        
        logger.info(f"\nBest pair: {summary_df.loc[best_idx, 'pair']} "  # Log best
               f"(Sharpe={summary_df.loc[best_idx, 'sharpe']:.2f})")  # Sharpe best
        logger.info(f"Worst pair: {summary_df.loc[worst_idx, 'pair']} "  # Log worst
               f"(Sharpe={summary_df.loc[worst_idx, 'sharpe']:.2f})")  # Sharpe worst
    else:  # Sin resultados
        logger.warning("No pairs successfully processed")  # Warning
    
    logger.info("=" * 80)  # Separador
    logger.info("Pipeline complete")  # Log
    logger.info("=" * 80)  # Separador


if __name__ == "__main__":  # Entry point
    parser = argparse.ArgumentParser(  # Parser
        description="E4 Pairs Trading Pipeline"  # Description
    )  # Fin parser
    parser.add_argument(  # Arg pairs
        "--pairs",  # Flag
        nargs="+",  # Nargs
        help="Pairs to process (format: TICKER_A,TICKER_B). "  # Help
             "Example: --pairs GGAL.BA,BMA.BA YPFD.BA,PAMP.BA",  # Example
    )  # Fin arg pairs
    parser.add_argument(  # Arg output-tag
        "--output-tag",  # Flag
        type=str,  # Tipo
        help="Tag for output directory (default: timestamp)",  # Help
    )  # Fin arg output-tag
    
    args = parser.parse_args()  # Parse args
    
    # Parse pairs
    pairs = None  # Inicializar
    if args.pairs:  # Si hay pares
        pairs = []  # Lista
        for pair_str in args.pairs:  # Loop
            parts = pair_str.split(",")  # Split
            if len(parts) != 2:  # Validar
                logger.error(f"Invalid pair format: {pair_str}")  # Error
                continue  # Saltar
            pairs.append((parts[0], parts[1]))  # Append
    
    main(pairs=pairs, output_tag=args.output_tag)  # Ejecutar main
