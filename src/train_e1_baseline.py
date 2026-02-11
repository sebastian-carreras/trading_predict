"""
Pipeline de entrenamiento para Baseline E1 (Regresión Lineal Simple).

Este script ejecuta el mismo pipeline que train_e1_pipeline.py pero usando
Regresión Lineal en lugar de GRU, para establecer un baseline de comparación.

Uso:
    python -m src.train_e1_baseline --tickers AAPL GOOGL
    python -m src.train_e1_baseline --tickers AAPL --compare-with-gru
"""

from __future__ import annotations  # Permite forward references en type hints

import argparse  # Parsing de argumentos CLI
from datetime import datetime  # Timestamp de ejecución
from pathlib import Path  # Manejo de rutas
import os  # Variables de entorno y paths

import numpy as np  # Cálculo numérico
import pandas as pd  # DataFrames
from sklearn.model_selection import TimeSeriesSplit  # Split temporal sin leakage

from src.features.build_features_e1 import compute_e1_features, make_target_e1  # Features y target E1
from src.features.build_sequences_e1e2 import make_sequences, temporal_train_val_split  # Secuencias y split train/val
from src.models.e1_baseline_linear import (  # Import baseline linear
    LinearRegressionBaseline,  # Modelo baseline de regresión lineal
    compute_baseline_metrics,  # Métricas del baseline
)  # Fin import baseline linear
from src.backtest.backtest_daily import backtest_daily_signals, summarize_backtest  # Backtesting diario
from src.utils import ensure_dir, load_yaml, project_root  # Utilidades comunes


def run_baseline_for_ticker(  # Ejecutar baseline por ticker
    config: dict,  # Configuración global (YAML cargado)
    ticker: str,  # Símbolo del activo
    raw_dir: Path,  # Directorio con CSVs diarios
    out_dir: Path,  # Directorio de salida por ticker
    benchmark_df: pd.DataFrame | None = None,  # Benchmark opcional
) -> dict:  # Retorna resumen con métricas
    """
    Ejecuta pipeline completo de baseline para un ticker.
    
    Usa Regresión Lineal en lugar de GRU para establecer punto de comparación.
    
    Argumentos:
        config: Diccionario de configuración con estrategias, modelos, splits
        ticker: Símbolo del activo (ej: "AAPL")
        raw_dir: Directorio con datos CSV limpios
        out_dir: Directorio para guardar outputs (predicciones, métricas, backtest)
        benchmark_df: DataFrame opcional con benchmarks
    
    Retorna:
        dict: Resumen con métricas ML y trading del modelo entrenado
    """
    ensure_dir(out_dir)  # Asegurar directorio de salida
    
    print(f"\n{'='*60}")  # Separador visual
    print(f"BASELINE E1 - {ticker}")  # Encabezado del ticker
    print(f"{'='*60}")  # Separador visual
    
    # 1. CARGAR DATOS OHLCV
    # Lee archivo CSV con datos de precios OHLC y volumen históricos
    csv_path = raw_dir / f"{ticker}_daily.csv"  # Path del CSV diario
    if not csv_path.exists():  # Validar existencia del CSV
        raise FileNotFoundError(f"No existe {csv_path}")  # Error si no está
    
    # Convertir a DataFrame, con timestamp como índice
    df = pd.read_csv(csv_path)  # Leer CSV
    df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601', utc=True)  # Parsear timestamp
    df = df.sort_values("timestamp").set_index("timestamp")  # Ordenar e indexar
    
    # Guardar una copia del DataFrame original para backtesting
    # (antes de cualquier transformación o dropna)
    ohlcv = df.copy()  # Copia para backtest sin alterar
    
    # DEBUG: Imprimir columnas antes de cualquier transformación
    print(f"DEBUG - Columnas después de cargar CSV: {list(df.columns)}")  # Debug columnas
    
    # 2. CALCULAR FEATURES E1
    # Extrae indicadores técnicos (momentum, volatilidad, tendencia, etc.)
    print("Calculando features...")  # Log features
    df_feat = compute_e1_features(df)  # Calcular features E1
    
    # 3. CREAR TARGET
    # Define el objetivo de predicción: retorno esperado en N días
    strats = config.get("strategies", {})  # Sección estrategias
    e1_cfg = strats.get("e1_conservative", {})  # Config E1
    horizon_days = int(e1_cfg.get("horizon_days", 90))  # Horizonte futuro
    
    print(f"Creando target (horizonte={horizon_days} días)...")  # Log target
    target_series = make_target_e1(df, horizon_days=horizon_days)  # Target futuro
    
    # 4. CREAR SECUENCIAS
    # Transforma datos en ventanas de tiempo (lookback) para redes recurrentes
    lookback_days = int(e1_cfg.get("lookback_days", 360))  # Lookback histórico
    
    print(f"Creando secuencias (lookback={lookback_days})...")  # Log secuencias
    
    # Combinar features y target en un solo DataFrame
    # Necesario para sincronizar índices en make_sequences
    df_combined = df_feat.copy()  # Copia de features
    df_combined["target"] = target_series  # Adjuntar target
    
    # Separar features y target
    feature_cols = [c for c in df_combined.columns if c != "target"]  # Columnas features
    features_df = df_combined[feature_cols]  # DataFrame de features
    target_series_clean = df_combined["target"]  # Serie target limpia
    
    # Crear secuencias: [n_samples, lookback_days, n_features]
    # Cada muestra contiene lookback_days de datos históricos
    X, y, timestamps, feature_names = make_sequences(  # Construir secuencias
        features=features_df,  # Features
        target=target_series_clean,  # Target
        lookback=lookback_days,  # Lookback
    )  # Crear secuencias
    
    print(f"Forma de datos: X={X.shape}, y={y.shape}")  # Log shapes
    
    # 5. VALIDACIÓN WALK-FORWARD
    # Simula entrenamientos progresivos con ventanas de tiempo no superpuestas
    # Previene data leakage y valida robustez del modelo
    splits_cfg = config.get("splits", {})  # Config de splits
    n_folds = int(splits_cfg.get("folds", 5))  # Número de folds
    test_size = max(1, len(X) // (n_folds + 1))  # Tamaño test por fold
    gap_samples = int(splits_cfg.get("embargo_days", {}).get("e1", 0))  # Embargo en muestras
    
    # TimeSeriesSplit mantiene orden temporal
    splitter = TimeSeriesSplit(n_splits=n_folds, test_size=test_size, gap=gap_samples)  # Splitter temporal
    
    val_fraction = float(splits_cfg.get("internal_val_fraction", 0.15))  # Fracción de validación
    
    fold_results = []  # Resultados por fold
    all_predictions = []  # Predicciones acumuladas
    
    print(f"\nEjecutando walk-forward ({n_folds} folds)...")  # Log inicio WF
    
    for fold_idx, (train_full_idx, test_idx) in enumerate(  # Loop de folds
        splitter.split(np.arange(len(X))), start=1  # Iterar splits temporales
    ):  # Fin cabecera loop
        if len(test_idx) == 0 or len(train_full_idx) == 0:  # Validar tamaños
            continue  # Saltar folds vacíos
        
        print(f"\n  Fold {fold_idx}/{n_folds}")  # Log fold
        
        # Split interno train/val
        try:  # Intentar split train/val
            train_idx, val_idx = temporal_train_val_split(  # Split train/val
                train_full_idx,  # Índices completos de train
                val_fraction=val_fraction,  # Fracción de validación
            )  # Aplicar split temporal
        except ValueError:  # Fallback si falla
            split_point = max(1, int(len(train_full_idx) * 0.85))  # Fallback 85/15
            train_idx = train_full_idx[:split_point]  # Índices train
            val_idx = train_full_idx[split_point:]  # Índices val
        
        # ESTANDARIZAR FEATURES
        # Fit scaler solo en datos de entrenamiento (prevenir data leakage)
        from sklearn.preprocessing import StandardScaler  # Import local para scaler
        scaler = StandardScaler()  # Inicializar scaler
        
        # Reshape para scaler: (n_samples * seq_len, n_features)
        X_train_2d = X[train_idx].reshape(-1, X.shape[-1])  # Flatten train
        scaler.fit(X_train_2d)  # Ajustar scaler con train
        
        # Aplicar transformación a todos los sets
        X_train_scaled = scaler.transform(X[train_idx].reshape(-1, X.shape[-1])).reshape(X[train_idx].shape)  # X train
        X_val_scaled = scaler.transform(X[val_idx].reshape(-1, X.shape[-1])).reshape(X[val_idx].shape)  # X val
        X_test_scaled = scaler.transform(X[test_idx].reshape(-1, X.shape[-1])).reshape(X[test_idx].shape)  # X test
        
        y_train = y[train_idx]  # y train
        y_val = y[val_idx]  # y val
        y_test = y[test_idx]  # y test
        
        # ENTRENAR REGRESIÓN LINEAL
        # Baseline simple: modela relación lineal entre features y target
        print(f"    Entrenando regresión lineal...")  # Log entrenamiento
        model = LinearRegressionBaseline(seed=42)  # Modelo baseline
        
        train_result = model.fit(  # Entrenar modelo
            X_train_scaled,  # Features de entrenamiento (normalizados)
            y_train,  # Targets de entrenamiento
            X_val_scaled,  # Features de validación
            y_val,  # Targets de validación
        )  # Fin entrenamiento
        
        print(f"    Train R²={train_result.train_score:.4f}, Val R²={train_result.val_score:.4f}")  # Log scores
        
        # PREDICCIONES EN TEST
        y_pred_test = model.predict(X_test_scaled)  # Predicciones test
        
        # MÉTRICAS ML
        test_metrics = compute_baseline_metrics(y_test, y_pred_test)  # Métricas ML
        
          print(f"    Test: MAE={test_metrics['mae']:.4f}, RMSE={test_metrics['rmse']:.4f}, "  # Log métricas
              f"Dir.Acc={test_metrics['directional_accuracy']:.3f}, IC={test_metrics['ic']:.3f}")  # Log métricas
        
        # Guardar predicciones de este fold
        test_timestamps = timestamps[test_idx]  # Timestamps test
        pred_df = pd.DataFrame({  # DataFrame de predicciones
            "timestamp": test_timestamps,  # Timestamp
            "y_true": y_test,  # Target real
            "y_pred": y_pred_test,  # Predicción
            "fold": fold_idx,  # Fold
        })  # Fin DataFrame
        all_predictions.append(pred_df)  # Acumular predicciones
        
        fold_results.append({  # Agregar resultados
            "fold": fold_idx,  # ID fold
            "train_r2": train_result.train_score,  # R2 train
            "val_r2": train_result.val_score,  # R2 val
            **{f"test_{k}": v for k, v in test_metrics.items()},  # Métricas test
        })  # Fin dict
    
    # Consolidar predicciones de todos los folds
    df_all_preds = pd.concat(all_predictions, ignore_index=True)  # Consolidar preds
    df_all_preds.to_csv(out_dir / f"{ticker}_baseline_predictions.csv", index=False)  # Guardar CSV
    
    # MÉTRICAS AGREGADAS (todos los folds)
    y_true_all = df_all_preds["y_true"].values  # Targets globales
    y_pred_all = df_all_preds["y_pred"].values  # Predicciones globales
    
    overall_metrics = compute_baseline_metrics(y_true_all, y_pred_all)  # Métricas globales
    
    print(f"\n  Métricas agregadas (todos los folds):")  # Log métricas globales
    print(f"    MAE={overall_metrics['mae']:.4f}")  # MAE
    print(f"    RMSE={overall_metrics['rmse']:.4f}")  # RMSE
    print(f"    Directional Accuracy={overall_metrics['directional_accuracy']:.3f}")  # Accuracy direccional
    print(f"    IC={overall_metrics['ic']:.3f}")  # IC
    
    # Backtesting - Usar el mismo enfoque que train_e1_pipeline.py
    thresholds = e1_cfg.get("thresholds", {})  # Thresholds config
    tau_buy = float(thresholds.get("tau_buy", 0.06))  # Umbral compra
    tau_sell = float(thresholds.get("tau_sell", 0.00))  # Umbral venta
    
    costs = config.get("costs", {})  # Config costos
    round_trip_bps = float(costs.get("round_trip_bps_daily", 10.0))  # Costos diarios
    holding_period = int(e1_cfg.get("horizon_days", 90))  # Holding period
    
    print(f"\n  Ejecutando backtest (tau_buy={tau_buy}, tau_sell={tau_sell})...")  # Log backtest
    
    # Obtener timestamps y precios de cierre de las predicciones
    # Similar a como lo hace run_e1_walk_forward(): ohlcv.loc[ts_test, "close"]
    test_timestamps = pd.DatetimeIndex(df_all_preds["timestamp"].values)  # Timestamps preds
    
    # CRITICAL: Asegurar que los timestamps tengan el mismo timezone que ohlcv
    # El problema era: ohlcv tiene timestamps con timezone (+00:00) pero
    # df_all_preds los tiene sin timezone
    if test_timestamps.tz is None and ohlcv.index.tz is not None:  # Ajustar a UTC
        # Localizar a UTC
        test_timestamps = test_timestamps.tz_localize('UTC')  # Ajuste a UTC
    elif test_timestamps.tz is not None and ohlcv.index.tz is None:  # Quitar tz
        # Remover timezone
        test_timestamps = test_timestamps.tz_localize(None)  # Quitar tz
    
    # Usar el DataFrame OHLCV original (antes de dropna en make_sequences)
    # para obtener los close prices
    close_prices = ohlcv.loc[test_timestamps, "close"].to_numpy()  # Close prices
    
    backtest_df = backtest_daily_signals(  # Ejecutar backtest
        timestamps=test_timestamps,  # Timestamps
        close_prices=close_prices,  # Close prices
        pred_returns=y_pred_all,  # predicciones de retorno
        tau_buy=tau_buy,  # Umbral compra
        tau_sell=tau_sell,  # Umbral venta
        round_trip_bps=round_trip_bps,  # Costos
        holding_period_days=holding_period,  # Holding
    )  # Fin backtest
    
    backtest_df.to_csv(out_dir / f"{ticker}_baseline_backtest.csv", index=False)  # Guardar backtest
    
    bt_summary = summarize_backtest(backtest_df)  # Métricas backtest
    
    print(f"\n  Backtest Summary:")  # Log summary
    print(f"    CAGR: {bt_summary.get('cagr', 0):.2%}")  # CAGR
    print(f"    Sharpe: {bt_summary.get('sharpe', 0):.3f}")  # Sharpe
    print(f"    Max DD: {bt_summary.get('max_drawdown', 0):.2%}")  # Max drawdown
    
    # Summary completo
    summary = {  # Resumen final
        "ticker": ticker,  # Ticker
        "model": "LinearRegression_Baseline",  # Modelo
        **{f"ml_{k}": v for k, v in overall_metrics.items()},  # Métricas ML
        **{f"bt_{k}": v for k, v in bt_summary.items()},  # Métricas backtest
        "n_folds": len(fold_results),  # Folds
        "lookback_days": lookback_days,  # Lookback
        "horizon_days": horizon_days,  # Horizon
    }  # Fin resumen
    
    # Guardar fold results
    pd.DataFrame(fold_results).to_csv(  # Guardar folds
        out_dir / f"{ticker}_baseline_folds.csv", index=False  # Guardar folds
    )  # Fin guardado folds
    
    # Guardar summary
    pd.DataFrame([summary]).to_csv(  # Guardar summary
        out_dir / f"{ticker}_baseline_summary.csv", index=False  # Guardar summary
    )  # Fin guardado summary
    
    return summary  # Retornar resumen


def main() -> None:  # Entry-point principal
    parser = argparse.ArgumentParser(description="E1 Baseline - Linear Regression")  # Parser CLI
    parser.add_argument(  # Argumento tickers
        "--tickers",  # Lista de tickers
        nargs="+",  # Múltiples valores
        default=["AAPL"],  # Default
        help="Lista de tickers a procesar",  # Help
    )  # Fin argumento tickers
    parser.add_argument(  # Argumento config
        "--config",  # Path config
        default="src/config/base.yaml",  # Default config
        help="Archivo de configuración",  # Help
    )  # Fin argumento config
    parser.add_argument(  # Argumento raw dir
        "--raw-dir",  # Path raw dir
        default="data/clean",  # Default raw dir
        help="Directorio con datos limpios",  # Help
    )  # Fin argumento raw dir
    parser.add_argument(  # Argumento output dir
        "--output-dir",  # Output dir override
        default=None,  # Default None
        help="Directorio de salida (auto si None)",  # Help
    )  # Fin argumento output dir
    parser.add_argument(  # Argumento compare
        "--compare-with-gru",  # Flag comparación
        action="store_true",  # True si se pasa
        help="Cargar y comparar con resultados del GRU",  # Help
    )  # Fin argumento compare
    
    args = parser.parse_args()  # Parsear args
    
    # Cargar config
    root = project_root()  # Root del proyecto
    config_path = root / args.config  # Path config
    config = load_yaml(config_path)  # Cargar YAML
    
    # Setup directorios
    raw_dir = root / args.raw_dir  # Directorio raw
    
    if args.output_dir:  # Si se especifica output
        out_base = Path(args.output_dir)  # Output custom
    else:  # Output auto con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Timestamp
        out_base = root / "runs" / "e1_baseline" / timestamp  # Output auto
    
    ensure_dir(out_base)  # Crear output
    
    # Guardar config usada
    import yaml  # Import local
    with open(out_base / "config_used.yaml", "w") as f:  # Abrir YAML
        yaml.dump(config, f)  # Dump YAML
    
    # Benchmark (opcional)
    benchmark_df = None  # Sin benchmark
    
    # Procesar tickers
    summaries = []  # Resúmenes
    
    for ticker in args.tickers:  # Iterar tickers
        try:  # Manejo de errores por ticker
            summary = run_baseline_for_ticker(  # Ejecutar baseline
                config=config,  # Config
                ticker=ticker,  # Ticker
                raw_dir=raw_dir,  # Raw dir
                out_dir=out_base / ticker,  # Output por ticker
                benchmark_df=benchmark_df,  # Benchmark
            )  # Fin ejecución ticker
            summaries.append(summary)  # Guardar resumen
            print(f"✓ {ticker} completado\n")  # Log OK
        except Exception as exc:  # Captura errores
            print(f"✗ Error en {ticker}: {exc}\n")  # Log error
    
    # Summary consolidado
    df_summary = pd.DataFrame(summaries)  # DataFrame summary
    df_summary.to_csv(out_base / "baseline_summary_all.csv", index=False)  # Guardar CSV
    
    print(f"\n{'='*60}")  # Separador
    print(f"RESULTADOS BASELINE")  # Título
    print(f"{'='*60}")  # Separador
    print(df_summary.to_string(index=False))  # Mostrar tabla
    print(f"\n✓ Guardado en: {out_base}")  # Log final
    
    # Comparación con GRU (opcional)
    if args.compare_with_gru:  # Si se pidió comparación
        print(f"\n{'='*60}")  # Separador
        print("COMPARACIÓN: Baseline vs GRU")  # Título
        print(f"{'='*60}")  # Separador
        
        # Buscar último run de GRU
        gru_runs_dir = root / "runs" / "e1_conservative"  # Directorio runs GRU
        if gru_runs_dir.exists():  # Validar existencia
            gru_runs = sorted([d for d in gru_runs_dir.iterdir() if d.is_dir()])  # Listar runs
            if gru_runs:  # Si hay runs
                latest_gru = gru_runs[-1]  # Último run
                gru_summary_path = latest_gru / "summary_all.csv"  # Path summary GRU
                
                if gru_summary_path.exists():  # Validar summary
                    df_gru = pd.read_csv(gru_summary_path)  # Cargar summary
                    
                    print("\nMétricas promedio:")  # Encabezado
                    print(f"{'Métrica':<25} {'Baseline':<15} {'GRU':<15} {'Diferencia':<15}")  # Tabla header
                    print("-" * 70)  # Separador
                    
                    metrics_to_compare = [  # Lista de métricas
                        "ml_mae", "ml_rmse", "ml_directional_accuracy", "ml_ic",  # Métricas ML
                        "bt_sharpe", "bt_cagr", "bt_max_drawdown"  # Métricas BT
                    ]  # Fin lista
                    
                    for metric in metrics_to_compare:  # Iterar métricas
                        if metric in df_summary.columns and metric in df_gru.columns:  # Validar columnas
                            baseline_val = df_summary[metric].mean()  # Promedio baseline
                            gru_val = df_gru[metric].mean()  # Promedio GRU
                            diff = gru_val - baseline_val  # Diferencia
                            
                            # Para max_drawdown, negativo es mejor
                            is_better = "✓" if (diff > 0 and "drawdown" not in metric) or (diff < 0 and "drawdown" in metric) else "✗"  # Mejor
                            
                            print(f"{metric:<25} {baseline_val:>14.4f} {gru_val:>14.4f} {diff:>+14.4f} {is_better}")  # Imprimir fila
                else:  # Si no existe summary GRU
                    print("⚠️  No se encontró summary del GRU")  # Warning GRU summary
        else:  # Si no hay directorio de runs
            print("⚠️  No hay runs previos de GRU para comparar")  # Warning no runs


if __name__ == "__main__":  # Entry-point script
    main()  # Ejecutar main
