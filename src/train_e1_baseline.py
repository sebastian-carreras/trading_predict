"""
Pipeline de entrenamiento para Baseline E1 (Regresión Lineal Simple).

Este script ejecuta el mismo pipeline que train_e1_pipeline.py pero usando
Regresión Lineal en lugar de GRU, para establecer un baseline de comparación.

Uso:
    python -m src.train_e1_baseline --tickers AAPL GOOGL
    python -m src.train_e1_baseline --tickers AAPL --compare-with-gru
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.features.build_features_e1 import compute_e1_features, make_target_e1
from src.features.build_sequences import make_sequences, temporal_train_val_split
from src.models.e1_baseline_linear import (
    LinearRegressionBaseline,
    compute_baseline_metrics,
)
from src.backtest.daily import backtest_daily_signals, summarize_backtest
from src.utils import ensure_dir, load_yaml, project_root


def run_baseline_for_ticker(
    config: dict,
    ticker: str,
    raw_dir: Path,
    out_dir: Path,
    benchmark_df: pd.DataFrame | None = None,
) -> dict:
    """
    Ejecuta pipeline completo de baseline para un ticker.
    
    Retorna dict con métricas para comparación.
    """
    ensure_dir(out_dir)
    
    print(f"\n{'='*60}")
    print(f"BASELINE E1 - {ticker}")
    print(f"{'='*60}")
    
    # 1. Cargar datos
    csv_path = raw_dir / f"{ticker}_daily.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe {csv_path}")
    
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601', utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    
    # Guardar una copia del DataFrame original para backtesting
    # (antes de cualquier transformación o dropna)
    ohlcv = df.copy()
    
    # DEBUG: Imprimir columnas antes de cualquier transformación
    print(f"DEBUG - Columnas después de cargar CSV: {list(df.columns)}")
    
    # 2. Calcular features E1
    print("Calculando features...")
    df_feat = compute_e1_features(df)
    
    # 3. Crear target
    strats = config.get("strategies", {})
    e1_cfg = strats.get("e1_conservative", {})
    horizon_days = int(e1_cfg.get("horizon_days", 90))
    
    print(f"Creando target (horizonte={horizon_days} días)...")
    target_series = make_target_e1(df, horizon_days=horizon_days)  # Pasar df (con 'close'), no df_feat
    
    # 4. Crear secuencias
    lookback_days = int(e1_cfg.get("lookback_days", 180))
    
    print(f"Creando secuencias (lookback={lookback_days})...")
    
    # Combinar features y target en un solo DataFrame
    df_combined = df_feat.copy()
    df_combined["target"] = target_series
    
    # Separar features y target
    feature_cols = [c for c in df_combined.columns if c != "target"]
    features_df = df_combined[feature_cols]
    target_series_clean = df_combined["target"]
    
    X, y, timestamps, feature_names = make_sequences(
        features=features_df,
        target=target_series_clean,
        lookback=lookback_days,
    )
    
    print(f"Forma de datos: X={X.shape}, y={y.shape}")
    
    # 5. Walk-forward validation
    splits_cfg = config.get("splits", {})
    n_folds = int(splits_cfg.get("folds", 5))
    test_size = max(1, len(X) // (n_folds + 1))
    gap_samples = int(splits_cfg.get("embargo_days", {}).get("e1", 0))
    
    splitter = TimeSeriesSplit(n_splits=n_folds, test_size=test_size, gap=gap_samples)
    
    val_fraction = float(splits_cfg.get("internal_val_fraction", 0.15))
    
    fold_results = []
    all_predictions = []
    
    print(f"\nEjecutando walk-forward ({n_folds} folds)...")
    
    for fold_idx, (train_full_idx, test_idx) in enumerate(
        splitter.split(np.arange(len(X))), start=1
    ):
        if len(test_idx) == 0 or len(train_full_idx) == 0:
            continue
        
        print(f"\n  Fold {fold_idx}/{n_folds}")
        
        # Split interno train/val
        try:
            train_idx, val_idx = temporal_train_val_split(
                train_full_idx,
                val_fraction=val_fraction,
            )
        except ValueError:
            split_point = max(1, int(len(train_full_idx) * 0.85))
            train_idx = train_full_idx[:split_point]
            val_idx = train_full_idx[split_point:]
        
        # Normalizar features (fit solo en train)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        # Reshape para scaler: (n_samples * seq_len, n_features)
        X_train_2d = X[train_idx].reshape(-1, X.shape[-1])
        scaler.fit(X_train_2d)
        
        # Aplicar a todos los sets
        X_train_scaled = scaler.transform(X[train_idx].reshape(-1, X.shape[-1])).reshape(X[train_idx].shape)
        X_val_scaled = scaler.transform(X[val_idx].reshape(-1, X.shape[-1])).reshape(X[val_idx].shape)
        X_test_scaled = scaler.transform(X[test_idx].reshape(-1, X.shape[-1])).reshape(X[test_idx].shape)
        
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]
        
        # Entrenar baseline
        print(f"    Entrenando regresión lineal...")
        model = LinearRegressionBaseline(seed=42)
        
        train_result = model.fit(
            X_train_scaled,
            y_train,
            X_val_scaled,
            y_val,
        )
        
        print(f"    Train R²={train_result.train_score:.4f}, Val R²={train_result.val_score:.4f}")
        
        # Predicciones en test
        y_pred_test = model.predict(X_test_scaled)
        
        # Métricas ML
        test_metrics = compute_baseline_metrics(y_test, y_pred_test)
        
        print(f"    Test: MAE={test_metrics['mae']:.4f}, RMSE={test_metrics['rmse']:.4f}, "
              f"Dir.Acc={test_metrics['directional_accuracy']:.3f}, IC={test_metrics['ic']:.3f}")
        
        # Guardar predicciones
        test_timestamps = timestamps[test_idx]
        pred_df = pd.DataFrame({
            "timestamp": test_timestamps,
            "y_true": y_test,
            "y_pred": y_pred_test,
            "fold": fold_idx,
        })
        all_predictions.append(pred_df)
        
        fold_results.append({
            "fold": fold_idx,
            "train_r2": train_result.train_score,
            "val_r2": train_result.val_score,
            **{f"test_{k}": v for k, v in test_metrics.items()},
        })
    
    # Consolidar predicciones
    df_all_preds = pd.concat(all_predictions, ignore_index=True)
    df_all_preds.to_csv(out_dir / f"{ticker}_baseline_predictions.csv", index=False)
    
    # Métricas agregadas
    y_true_all = df_all_preds["y_true"].values
    y_pred_all = df_all_preds["y_pred"].values
    
    overall_metrics = compute_baseline_metrics(y_true_all, y_pred_all)
    
    print(f"\n  Métricas agregadas (todos los folds):")
    print(f"    MAE={overall_metrics['mae']:.4f}")
    print(f"    RMSE={overall_metrics['rmse']:.4f}")
    print(f"    Directional Accuracy={overall_metrics['directional_accuracy']:.3f}")
    print(f"    IC={overall_metrics['ic']:.3f}")
    
    # Backtesting - Usar el mismo enfoque que train_e1_pipeline.py
    thresholds = e1_cfg.get("thresholds", {})
    tau_buy = float(thresholds.get("tau_buy", 0.06))
    tau_sell = float(thresholds.get("tau_sell", 0.00))
    
    costs = config.get("costs", {})
    round_trip_bps = float(costs.get("round_trip_bps_daily", 10.0))
    holding_period = int(e1_cfg.get("horizon_days", 90))
    
    print(f"\n  Ejecutando backtest (tau_buy={tau_buy}, tau_sell={tau_sell})...")
    
    # Obtener timestamps y precios de cierre de las predicciones
    # Similar a como lo hace run_e1_walk_forward(): ohlcv.loc[ts_test, "close"]
    test_timestamps = pd.DatetimeIndex(df_all_preds["timestamp"].values)
    
    # CRITICAL: Asegurar que los timestamps tengan el mismo timezone que ohlcv
    # El problema era: ohlcv tiene timestamps con timezone (+00:00) pero
    # df_all_preds los tiene sin timezone
    if test_timestamps.tz is None and ohlcv.index.tz is not None:
        # Localizar a UTC
        test_timestamps = test_timestamps.tz_localize('UTC')
    elif test_timestamps.tz is not None and ohlcv.index.tz is None:
        # Remover timezone
        test_timestamps = test_timestamps.tz_localize(None)
    
    # Usar el DataFrame OHLCV original (antes de dropna en make_sequences)
    # para obtener los close prices
    close_prices = ohlcv.loc[test_timestamps, "close"].to_numpy()
    
    backtest_df = backtest_daily_signals(
        timestamps=test_timestamps,
        close_prices=close_prices,
        pred_returns=y_pred_all,  # predicciones de retorno
        tau_buy=tau_buy,
        tau_sell=tau_sell,
        round_trip_bps=round_trip_bps,
        holding_period_days=holding_period,
    )
    
    backtest_df.to_csv(out_dir / f"{ticker}_baseline_backtest.csv", index=False)
    
    bt_summary = summarize_backtest(backtest_df)
    
    print(f"\n  Backtest Summary:")
    print(f"    CAGR: {bt_summary.get('cagr', 0):.2%}")
    print(f"    Sharpe: {bt_summary.get('sharpe', 0):.3f}")
    print(f"    Max DD: {bt_summary.get('max_drawdown', 0):.2%}")
    
    # Summary completo
    summary = {
        "ticker": ticker,
        "model": "LinearRegression_Baseline",
        **{f"ml_{k}": v for k, v in overall_metrics.items()},
        **{f"bt_{k}": v for k, v in bt_summary.items()},
        "n_folds": len(fold_results),
        "lookback_days": lookback_days,
        "horizon_days": horizon_days,
    }
    
    # Guardar fold results
    pd.DataFrame(fold_results).to_csv(
        out_dir / f"{ticker}_baseline_folds.csv", index=False
    )
    
    # Guardar summary
    pd.DataFrame([summary]).to_csv(
        out_dir / f"{ticker}_baseline_summary.csv", index=False
    )
    
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 Baseline - Linear Regression")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL"],
        help="Lista de tickers a procesar",
    )
    parser.add_argument(
        "--config",
        default="src/config/base.yaml",
        help="Archivo de configuración",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/clean",
        help="Directorio con datos limpios",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directorio de salida (auto si None)",
    )
    parser.add_argument(
        "--compare-with-gru",
        action="store_true",
        help="Cargar y comparar con resultados del GRU",
    )
    
    args = parser.parse_args()
    
    # Cargar config
    root = project_root()
    config_path = root / args.config
    config = load_yaml(config_path)
    
    # Setup directorios
    raw_dir = root / args.raw_dir
    
    if args.output_dir:
        out_base = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_base = root / "runs" / "e1_baseline" / timestamp
    
    ensure_dir(out_base)
    
    # Guardar config usada
    import yaml
    with open(out_base / "config_used.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Benchmark (opcional)
    benchmark_df = None
    
    # Procesar tickers
    summaries = []
    
    for ticker in args.tickers:
        try:
            summary = run_baseline_for_ticker(
                config=config,
                ticker=ticker,
                raw_dir=raw_dir,
                out_dir=out_base / ticker,
                benchmark_df=benchmark_df,
            )
            summaries.append(summary)
            print(f"✓ {ticker} completado\n")
        except Exception as exc:
            print(f"✗ Error en {ticker}: {exc}\n")
    
    # Summary consolidado
    df_summary = pd.DataFrame(summaries)
    df_summary.to_csv(out_base / "baseline_summary_all.csv", index=False)
    
    print(f"\n{'='*60}")
    print(f"RESULTADOS BASELINE")
    print(f"{'='*60}")
    print(df_summary.to_string(index=False))
    print(f"\n✓ Guardado en: {out_base}")
    
    # Comparación con GRU (opcional)
    if args.compare_with_gru:
        print(f"\n{'='*60}")
        print("COMPARACIÓN: Baseline vs GRU")
        print(f"{'='*60}")
        
        # Buscar último run de GRU
        gru_runs_dir = root / "runs" / "e1_conservative"
        if gru_runs_dir.exists():
            gru_runs = sorted([d for d in gru_runs_dir.iterdir() if d.is_dir()])
            if gru_runs:
                latest_gru = gru_runs[-1]
                gru_summary_path = latest_gru / "summary_all.csv"
                
                if gru_summary_path.exists():
                    df_gru = pd.read_csv(gru_summary_path)
                    
                    print("\nMétricas promedio:")
                    print(f"{'Métrica':<25} {'Baseline':<15} {'GRU':<15} {'Diferencia':<15}")
                    print("-" * 70)
                    
                    metrics_to_compare = [
                        "ml_mae", "ml_rmse", "ml_directional_accuracy", "ml_ic",
                        "bt_sharpe", "bt_cagr", "bt_max_drawdown"
                    ]
                    
                    for metric in metrics_to_compare:
                        if metric in df_summary.columns and metric in df_gru.columns:
                            baseline_val = df_summary[metric].mean()
                            gru_val = df_gru[metric].mean()
                            diff = gru_val - baseline_val
                            
                            # Para max_drawdown, negativo es mejor
                            is_better = "✓" if (diff > 0 and "drawdown" not in metric) or (diff < 0 and "drawdown" in metric) else "✗"
                            
                            print(f"{metric:<25} {baseline_val:>14.4f} {gru_val:>14.4f} {diff:>+14.4f} {is_better}")
                else:
                    print("⚠️  No se encontró summary del GRU")
        else:
            print("⚠️  No hay runs previos de GRU para comparar")


if __name__ == "__main__":
    main()
