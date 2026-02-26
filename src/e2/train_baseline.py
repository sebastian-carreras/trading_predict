"""
Pipeline de entrenamiento para Baseline E2 (Ridge Regression).

Ejecuta el mismo pipeline walk-forward que se usará para LSTM, pero usando
Ridge Regression para establecer un baseline de comparación.

Uso:
    python -m src.e2.train_baseline
    python -m src.e2.train_baseline --tickers NVDA,GOOGL,AMZN
    python -m src.e2.train_baseline --tickers AAPL --compare-with-lstm
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import os
import socket
import time
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .build_features import compute_e2_features, make_target_e2
from ..features.build_sequences_e1e2 import make_sequences, temporal_train_val_split
from .baseline_linear import RidgeBaseline, compute_baseline_metrics
from ..backtest.backtest_daily import backtest_daily_signals, summarize_backtest
from ..utils import ensure_dir, load_yaml, project_root


def run_baseline_for_ticker(
    config: dict,
    ticker: str,
    raw_dir: Path,
    out_dir: Path,
    mlflow_enabled: bool = False,
    mlflow=None,
    timestamp: str | None = None,
) -> dict:
    """
    Ejecuta pipeline completo de baseline E2 para un ticker.
    
    Usa Ridge Regression en lugar de LSTM para establecer punto de comparación.
    """
    start_time = time.perf_counter()
    ensure_dir(out_dir)
    
    print(f"\n{'='*60}")
    print(f"BASELINE E2 - {ticker}")
    print(f"{'='*60}")
    
    # 1. CARGAR DATOS OHLCV
    clean_csv_path = project_root() / "data" / "clean" / f"{ticker}_daily.csv"
    if clean_csv_path.exists():
        csv_path = clean_csv_path
        print(f"  ✓ Usando datos limpios: {csv_path.name}")
    else:
        csv_path = raw_dir / f"{ticker}_daily.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"No existe {csv_path}")
        print(f"  ⚠️  Usando datos raw (limpieza no ejecutada): {csv_path.name}")
    
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601', utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    
    # Copia del DataFrame original para backtesting (antes de transformaciones)
    ohlcv = df.copy()

    # 2. CALCULAR FEATURES E2 (16 features)
    print("Calculando features (16 indicadores)...")
    df_feat = compute_e2_features(df)
    
    # 3. CREAR TARGET
    strats = config.get("strategies", {})
    e2_cfg = strats.get("e2_moderate", {})
    horizon_days = int(e2_cfg.get("horizon_days", 20))
    
    print(f"Creando target (horizonte={horizon_days} días)...")
    target_series = make_target_e2(df, horizon_days=horizon_days)
    
    # 4. CREAR SECUENCIAS
    lookback_days = int(e2_cfg.get("lookback_days", 60))
    
    print(f"Creando secuencias (lookback={lookback_days})...")
    
    df_combined = df_feat.copy()
    df_combined["target"] = target_series
    
    feature_cols = [c for c in df_combined.columns if c != "target"]
    features_df = df_combined[feature_cols]
    target_series_clean = df_combined["target"]
    
    X, y, timestamps, feature_names = make_sequences(
        features=features_df,
        target=target_series_clean,
        lookback=lookback_days,
    )
    
    print(f"Forma de datos: X={X.shape}, y={y.shape}")
    print(f"Features ({len(feature_names)}): {', '.join(feature_names)}")
    
    # 5. VALIDACIÓN WALK-FORWARD
    splits_cfg = config.get("splits", {})
    n_folds = int(splits_cfg.get("folds", 5))
    test_size = max(1, len(X) // (n_folds + 1))
    gap_samples = int(splits_cfg.get("embargo_days", {}).get("e2", 0))
    
    splitter = TimeSeriesSplit(n_splits=n_folds, test_size=test_size, gap=gap_samples)
    
    val_fraction = float(splits_cfg.get("internal_val_fraction", 0.15))
    
    fold_results = []
    all_predictions = []
    
    print(f"\nEjecutando walk-forward ({n_folds} folds, embargo={gap_samples})...")
    
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
        
        # ESTANDARIZAR FEATURES
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        X_train_2d = X[train_idx].reshape(-1, X.shape[-1])
        scaler.fit(X_train_2d)
        
        X_train_scaled = scaler.transform(X[train_idx].reshape(-1, X.shape[-1])).reshape(X[train_idx].shape)
        X_val_scaled = scaler.transform(X[val_idx].reshape(-1, X.shape[-1])).reshape(X[val_idx].shape)
        X_test_scaled = scaler.transform(X[test_idx].reshape(-1, X.shape[-1])).reshape(X[test_idx].shape)
        
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]

        # Z-SCORE DEL TARGET (consistente con futuro LSTM)
        mean_y = float(y_train.mean())
        std_y = float(y_train.std()) + 1e-12

        y_train_s = (y_train - mean_y) / std_y
        y_val_s = (y_val - mean_y) / std_y

        # ENTRENAR RIDGE REGRESSION
        print(f"    Entrenando Ridge Regression (alpha=1.0)...")
        model = RidgeBaseline(alpha=1.0, seed=42)

        train_result = model.fit(
            X_train_scaled, y_train_s,
            X_val_scaled, y_val_s,
        )

        print(f"    Train R²={train_result.train_score:.4f}, Val R²={train_result.val_score:.4f}")

        # PREDICCIONES EN TEST (des-escalar a escala original)
        y_pred_test_s = model.predict(X_test_scaled)
        y_pred_test = y_pred_test_s * std_y + mean_y

        # MÉTRICAS ML
        test_metrics = compute_baseline_metrics(y_test, y_pred_test)
        print(
            f"    Test: MAE={test_metrics['mae']:.4f}, RMSE={test_metrics['rmse']:.4f}, "
            f"Dir.Acc={test_metrics['directional_accuracy']:.3f}, IC={test_metrics['ic']:.3f}"
        )
        
        # Guardar predicciones de este fold
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
    
    # Consolidar predicciones de todos los folds
    df_all_preds = pd.concat(all_predictions, ignore_index=True)
    df_all_preds.to_csv(out_dir / f"{ticker}_baseline_predictions.csv", index=False)
    
    # MÉTRICAS AGREGADAS (todos los folds)
    y_true_all = df_all_preds["y_true"].values
    y_pred_all = df_all_preds["y_pred"].values
    
    overall_metrics = compute_baseline_metrics(y_true_all, y_pred_all)
    
    print(f"\n  Métricas agregadas (todos los folds):")
    print(f"    MAE={overall_metrics['mae']:.4f}")
    print(f"    RMSE={overall_metrics['rmse']:.4f}")
    print(f"    Directional Accuracy={overall_metrics['directional_accuracy']:.3f}")
    print(f"    IC={overall_metrics['ic']:.3f}")
    
    # BACKTEST
    thresholds = e2_cfg.get("thresholds", {})
    tau_buy = float(thresholds.get("tau_buy", 0.025))
    tau_sell = float(thresholds.get("tau_sell", 0.00))
    
    costs = config.get("costs", {})
    round_trip_bps = float(costs.get("daily_round_trip_bps", costs.get("round_trip_bps_daily", 10.0)))
    holding_period = int(e2_cfg.get("horizon_days", 20))
    
    print(f"\n  Ejecutando backtest (tau_buy={tau_buy}, tau_sell={tau_sell})...")
    
    test_timestamps = pd.DatetimeIndex(df_all_preds["timestamp"].values)
    
    # Asegurar timezone consistency
    if test_timestamps.tz is None and ohlcv.index.tz is not None:
        test_timestamps = test_timestamps.tz_localize('UTC')
    elif test_timestamps.tz is not None and ohlcv.index.tz is None:
        test_timestamps = test_timestamps.tz_localize(None)
    
    close_prices = ohlcv.loc[test_timestamps, "close"].to_numpy()
    
    backtest_df = backtest_daily_signals(
        timestamps=test_timestamps,
        close_prices=close_prices,
        pred_returns=y_pred_all,
        tau_buy=tau_buy,
        tau_sell=tau_sell,
        round_trip_bps=round_trip_bps,
        holding_period_days=holding_period,
    )
    
    backtest_df.to_csv(out_dir / f"{ticker}_baseline_backtest.csv", index=False)
    
    bt_summary = summarize_backtest(backtest_df)
    if "max_drawdown" not in bt_summary and "max_dd" in bt_summary:
        bt_summary["max_drawdown"] = bt_summary["max_dd"]
    
    print(f"\n  Backtest Summary:")
    print(f"    CAGR: {bt_summary.get('cagr', 0):.2%}")
    print(f"    Sharpe: {bt_summary.get('sharpe', 0):.3f}")
    print(f"    Max DD: {bt_summary.get('max_drawdown', 0):.2%}")
    
    # Summary completo
    summary = {
        "ticker": ticker,
        "model": "Ridge_Baseline",
        **{f"ml_{k}": v for k, v in overall_metrics.items()},
        **{f"bt_{k}": v for k, v in bt_summary.items()},
        "n_folds": len(fold_results),
        "lookback_days": lookback_days,
        "horizon_days": horizon_days,
        "n_features": len(feature_names),
        "timing_train_seconds": round(time.perf_counter() - start_time, 2),
    }

    print(f"  Duración total: {summary['timing_train_seconds']:.2f}s")
    
    # Guardar fold results y summary
    pd.DataFrame(fold_results).to_csv(
        out_dir / f"{ticker}_baseline_folds.csv", index=False
    )
    pd.DataFrame([summary]).to_csv(
        out_dir / f"{ticker}_baseline_summary.csv", index=False
    )

    # Register as baseline in model lifecycle registry
    try:
        from ..lifecycle.registry import ModelRegistry
        from ..lifecycle.guardrails import log_candidate_metrics
        root = project_root()
        registry_path = root / "models" / "registry.json"

        log_candidate_metrics(
            metrics=summary, strategy="e2", ticker=ticker,
            run_dir=out_dir, variant="e2_baseline",
            log_path=root / "models" / "metrics_log.jsonl",
        )

        registry = ModelRegistry(registry_path)
        registry.register_baseline(
            strategy="e2", ticker=ticker,
            run_dir=str(out_dir.relative_to(root)),
            metrics=summary, variant="e2_baseline",
        )
        print(f"  Registered {ticker} as baseline in lifecycle registry")
    except Exception as exc:
        print(f"  Lifecycle registration skipped: {exc}")

    # MLFLOW TRACKING
    if mlflow_enabled and mlflow is not None:
        try:
            run_name = f"E2Baseline_{ticker}_{timestamp}" if timestamp else f"E2Baseline_{ticker}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("strategy", "e2_baseline")
                mlflow.log_param("ticker", ticker)
                mlflow.log_param("model_type", "Ridge")
                if timestamp:
                    mlflow.log_param("timestamp", timestamp)
                
                mlflow.log_params({
                    "lookback_days": lookback_days,
                    "horizon_days": horizon_days,
                    "n_folds": len(fold_results),
                    "n_features": len(feature_names),
                    "ridge_alpha": 1.0,
                    "tau_buy": tau_buy,
                    "tau_sell": tau_sell,
                    "round_trip_bps": round_trip_bps,
                })
                
                metrics_to_log = {
                    k: float(v) for k, v in summary.items()
                    if k not in ["ticker", "model"] and isinstance(v, (int, float, np.number))
                }
                mlflow.log_metrics(metrics_to_log)
                
                artifacts_to_log = [
                    (out_dir / f"{ticker}_baseline_predictions.csv", "predictions"),
                    (out_dir / f"{ticker}_baseline_backtest.csv", "backtest"),
                    (out_dir / f"{ticker}_baseline_folds.csv", "folds"),
                    (out_dir / f"{ticker}_baseline_summary.csv", "summary"),
                ]
                
                for artifact_path, artifact_folder in artifacts_to_log:
                    if artifact_path.exists():
                        mlflow.log_artifact(str(artifact_path), artifact_path=artifact_folder)
                
            print(f"  ✓ Run guardado en MLflow: {run_name}")
        except Exception as exc:
            print(f"  ⚠️  Error guardando en MLflow: {exc}")
    
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="E2 Baseline - Ridge Regression")
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Tickers separados por coma (ej: NVDA,GOOGL). Si se omite, usa los del config.",
    )
    parser.add_argument(
        "--config",
        default="src/config/base.yaml",
        help="Archivo de configuración",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Omite la descarga de datos (usa datos existentes)",
    )
    parser.add_argument(
        "--skip-cleaning",
        action="store_true",
        help="Omite la limpieza de datos (usa datos raw)",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/daily",
        help="Directorio con datos raw descargados",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directorio de salida (auto si None)",
    )
    parser.add_argument(
        "--compare-with-lstm",
        action="store_true",
        help="Cargar y comparar con resultados del LSTM",
    )
    
    args = parser.parse_args()
    
    # Cargar config
    root = project_root()
    config_path = root / args.config
    config = load_yaml(config_path)
    
    # Determinar tickers
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
    else:
        tickers = config.get("universe", {}).get("tickers_by_strategy", {}).get("e2_moderate", [])
    
    if not tickers:
        raise ValueError("No se especificaron tickers ni en args ni en config (universe.tickers_by_strategy.e2_moderate)")
    
    print(f"Entrenando baseline E2 para {len(tickers)} tickers: {', '.join(tickers)}")
    
    # Setup directorios
    raw_dir = root / args.raw_dir
    clean_dir = root / "data" / "clean"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.output_dir:
        out_base = Path(args.output_dir)
    else:
        out_base = root / "runs" / "e2_baseline" / timestamp
    
    ensure_dir(out_base)
    
    # Guardar config usada
    import yaml
    with open(out_base / "config_used.yaml", "w") as f:
        yaml.dump(config, f)
    
    # MLFLOW SETUP
    mlflow_enabled = False
    mlflow = None
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    remote_check_timeout_seconds = float(os.getenv("MLFLOW_REMOTE_CHECK_TIMEOUT_SECONDS", "1.5"))
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "E2_Baseline")
    local_sqlite_dir = root / "runs" / "mlflow_local"
    ensure_dir(local_sqlite_dir)
    local_sqlite_db = local_sqlite_dir / "mlflow.db"
    local_artifacts_dir = local_sqlite_dir / "artifacts"
    ensure_dir(local_artifacts_dir)
    local_sqlite_uri = f"sqlite:///{local_sqlite_db}"
    
    fallback_local_tracking_uri = os.getenv("MLFLOW_LOCAL_TRACKING_URI", "").strip()
    if fallback_local_tracking_uri:
        local_sqlite_uri = fallback_local_tracking_uri
    
    def _activate_mlflow(
        _mlflow_module,
        uri: str,
        label: str,
        experiment_artifact_dir: Path | None = None,
    ) -> tuple[bool, str | None]:
        """Activa MLflow con el URI especificado."""
        try:
            _mlflow_module.set_tracking_uri(uri)
            if experiment_artifact_dir is not None:
                exp = _mlflow_module.get_experiment_by_name(experiment_name)
                if exp is None:
                    _mlflow_module.create_experiment(
                        experiment_name,
                        artifact_location=experiment_artifact_dir.resolve().as_uri(),
                    )
                _mlflow_module.set_experiment(experiment_name)
            else:
                _mlflow_module.set_experiment(experiment_name)
            return True, None
        except Exception as exc:
            return False, str(exc)
    
    def _is_tracking_uri_reachable(uri: str, timeout_seconds: float) -> tuple[bool, str | None]:
        """Verifica si el tracking URI es alcanzable vía socket."""
        parsed = urlparse(uri)
        if parsed.scheme not in {"http", "https"}:
            return True, None
        
        host = parsed.hostname
        if not host:
            return True, None
        
        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme == "https" else 80
        
        try:
            with socket.create_connection((host, port), timeout=timeout_seconds):
                return True, None
        except OSError as exc:
            return False, str(exc)
    
    # Intentar activar MLflow
    try:
        import mlflow as _mlflow  # type: ignore
        
        if tracking_uri:
            remote_reachable, remote_err_msg = _is_tracking_uri_reachable(
                tracking_uri, remote_check_timeout_seconds,
            )
            if remote_reachable:
                ok_remote, remote_err = _activate_mlflow(_mlflow, tracking_uri, "remoto")
                if ok_remote:
                    mlflow = _mlflow
                    mlflow_enabled = True
                    print(f"✓ MLflow habilitado: {tracking_uri} (experiment={experiment_name})")
                else:
                    print(f"⚠️  MLflow servidor no disponible ({remote_err})")
            else:
                print(
                    f"⚠️  MLflow remoto no accesible "
                    f"({tracking_uri}, timeout={remote_check_timeout_seconds}s): {remote_err_msg}"
                )
        
        if not mlflow_enabled:
            ok_local, local_err = _activate_mlflow(
                _mlflow, local_sqlite_uri, "local",
                experiment_artifact_dir=local_artifacts_dir,
            )
            if ok_local:
                mlflow = _mlflow
                mlflow_enabled = True
                mode = "local fallback" if tracking_uri else "tracking local"
                print(f"✓ MLflow habilitado ({mode}, experiment={experiment_name})")
                print(f"  → Store URI: {local_sqlite_db}")
                print(f"  → Para visualizar: mlflow ui --backend-store-uri {local_sqlite_uri}")
            else:
                print(f"⚠️  Falló MLflow local: {local_err}")
    except Exception as exc:
        print(f"⚠️  MLflow no disponible, continuando sin tracking: {exc}")
        mlflow_enabled = False
    
    # Paso 1: Descargar datos
    if not args.skip_download:
        print("Paso 1/3: Descargando datos...")
        print("-" * 60)
        from src.data.download_daily import download_daily_ohlcv

        try:
            written = download_daily_ohlcv(
                tickers,
                out_dir=raw_dir,
                period="10y",
                skip_existing=True,
                min_days_fresh=1,
            )
            print(f"✓ Descargados/actualizados {len(written)} archivos\n")
        except Exception as exc:
            print(f"⚠️  Error en descarga: {exc}")
            print("Continuando con datos existentes...\n")
    else:
        print("Paso 1/3: Descarga omitida (usando datos existentes)\n")

    # Paso 2: Limpiar datos
    if not args.skip_cleaning:
        print("Paso 2/3: Limpiando datos...")
        print("-" * 60)
        from src.data.clean_daily import process_daily_data_with_cleaning

        try:
            tickers_to_clean = list(dict.fromkeys(tickers))
            reports = process_daily_data_with_cleaning(
                raw_dir=raw_dir,
                clean_dir=clean_dir,
                strategy="forward_fill",
                min_days=252,
                remove_zero_volume=True,
                verbose=False,
                tickers=tickers_to_clean,
            )
            cleaned = sum(1 for r in reports.values() if r.get("status") == "cleaned")
            rejected = sum(1 for r in reports.values() if r.get("status") == "rejected")
            print(f"✓ Limpiados: {cleaned} | Rechazados: {rejected}\n")
        except Exception as exc:
            print(f"⚠️  Error en limpieza: {exc}")
            print("Continuando con datos raw...\n")
    else:
        print("Paso 2/3: Limpieza omitida (usando datos existentes)\n")

    # Paso 3: Entrenar modelos
    print("Paso 3/3: Entrenando modelos baseline E2...")
    print("-" * 60)

    summaries = []
    
    for ticker in tickers:
        try:
            summary = run_baseline_for_ticker(
                config=config,
                ticker=ticker,
                raw_dir=raw_dir,
                out_dir=out_base / ticker,
                mlflow_enabled=mlflow_enabled,
                mlflow=mlflow,
                timestamp=timestamp,
            )
            summaries.append(summary)
            print(f"✓ {ticker} completado\n")
        except Exception as exc:
            print(f"✗ Error en {ticker}: {exc}\n")
    
    # Summary consolidado
    df_summary = pd.DataFrame(summaries)
    df_summary.to_csv(out_base / "baseline_summary_all.csv", index=False)
    df_summary.to_csv(out_base / "summary_all.csv", index=False)
    
    print(f"\n{'='*60}")
    print(f"RESULTADOS BASELINE E2")
    print(f"{'='*60}")
    print(df_summary.to_string(index=False))
    print(f"\n✓ Guardado en: {out_base}")
    
    # Comparación con LSTM (opcional)
    if args.compare_with_lstm:
        print(f"\n{'='*60}")
        print("COMPARACIÓN: Baseline vs LSTM")
        print(f"{'='*60}")
        
        lstm_runs_dir = root / "runs" / "e2_moderate"
        if lstm_runs_dir.exists():
            lstm_runs = sorted([d for d in lstm_runs_dir.iterdir() if d.is_dir()])
            if lstm_runs:
                latest_lstm = lstm_runs[-1]
                lstm_summary_path = latest_lstm / "summary_all.csv"
                
                if lstm_summary_path.exists():
                    df_lstm = pd.read_csv(lstm_summary_path)
                    
                    print("\nMétricas promedio:")
                    print(f"{'Métrica':<25} {'Baseline':<15} {'LSTM':<15} {'Diferencia':<15}")
                    print("-" * 70)
                    
                    metrics_to_compare = [
                        "ml_mae", "ml_rmse", "ml_directional_accuracy", "ml_ic",
                        "bt_sharpe", "bt_cagr", "bt_max_drawdown"
                    ]
                    
                    for metric in metrics_to_compare:
                        if metric in df_summary.columns and metric in df_lstm.columns:
                            baseline_val = df_summary[metric].mean()
                            lstm_val = df_lstm[metric].mean()
                            diff = lstm_val - baseline_val
                            
                            is_better = "✓" if (diff > 0 and "drawdown" not in metric) or (diff < 0 and "drawdown" in metric) else "✗"
                            
                            print(f"{metric:<25} {baseline_val:>14.4f} {lstm_val:>14.4f} {diff:>+14.4f} {is_better}")
                else:
                    print("⚠️  No se encontró summary del LSTM")
        else:
            print("⚠️  No hay runs previos de LSTM para comparar")


if __name__ == "__main__":
    main()
