"""
Pipeline simplificado E2 - Sin walk-forward, decision score simple.

Simplificaciones:
1. Decision score con solo 5 métricas: IC, directional_accuracy, sharpe, MAE, RMSE
2. Sin walk-forward validation (solo un split temporal train/val/test)
3. Arquitectura LSTM simplificada (1 capa en lugar de 2)

Flujo:
1. Descargar datos (opcional, con --skip-download)
2. Limpiar datos (opcional, con --skip-cleaning)
3. Calcular features E2 (momentum-focused: RSI, MACD, Stochastic, OBV)
4. Crear target y secuencias
5. Split temporal simple (train/val/test)
6. Entrenar LSTM simplificado
7. Evaluar y guardar
"""

from __future__ import annotations  # Permite forward references

import argparse  # CLI args
import copy  # Copias defensivas
from datetime import datetime, timezone  # Timestamp
import os  # Env vars
from pathlib import Path  # Rutas
import time

import numpy as np  # NumPy
import pandas as pd  # Pandas

from .features.build_features_e2 import compute_e2_features, make_target_e2  # Features/target E2
from .features.build_sequences_e1e2 import make_sequences, time_split  # Secuencias y split
from .models.e2_lstm import LSTMRegressor  # Modelo LSTM
from .backtest.backtest_daily import backtest_daily_signals, summarize_backtest  # Backtest
from .utils import ensure_dir, load_yaml, project_root, log_timing_event  # Utils


def load_ohlcv_csv(path: Path) -> pd.DataFrame:  # Cargar OHLCV
    """Carga CSV OHLCV y lo prepara.
    
    - Lee datos OHLCV desde un archivo CSV
    - Convierte la columna 'timestamp' a datetime con timezone UTC
    - Ordena datos cronológicamente y los indexa por timestamp
    - Valida que todas las columnas requeridas (OHLCV) estén presentes
    - Retorna DataFrame indexado por timestamp (timezone-aware)
    """
    # Cargar datos desde CSV
    df = pd.read_csv(path)  # Leer CSV
    
    # Validar que existe la columna timestamp
    if "timestamp" not in df.columns:  # Validar timestamp
        raise ValueError(f"Missing 'timestamp' column in {path}")  # Error

    # Convertir timestamp a datetime con timezone UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601', utc=True)  # A datetime
    
    # Ordenar por timestamp (cronológicamente ascendente)
    df = df.sort_values("timestamp")  # Ordenar
    
    # Usar timestamp como índice (más eficiente para acceso temporal)
    df = df.set_index("timestamp")  # Indexar

    # Validar que todas las columnas OHLCV estén presentes
    required = {"open", "high", "low", "close", "volume"}  # Columnas requeridas
    missing = required.difference(df.columns)  # Faltantes
    if missing:  # Si hay faltantes
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")  # Error

    return df  # Retornar df


def compute_information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # IC
    """Calcula el Information Coefficient (correlación de Spearman).
    
    El IC mide qué tan bien las predicciones están correlacionadas con valores reales:
    - IC > 0.05 se considera significativo en finanzas
    - IC < 0 indica overfitting o falta de capacidad predictiva
    - IC ≈ 0 indica predicciones aleatorias
    """
    # Si hay menos de 2 muestras, no se puede calcular correlación
    if len(y_true) <= 1:  # Validar tamaño
        return float("nan")  # NaN

    # Si alguno de los valores es constante (sin variación), la correlación es indefinida
    if np.std(y_true) == 0 or np.std(y_pred) == 0:  # Sin variación
        return float("nan")  # NaN

    try:  # Intentar Spearman
        # Usar Spearman (correlación de rangos) que es más robusta a outliers
        from scipy.stats import spearmanr  # Import Spearman
        ic, _ = spearmanr(y_true, y_pred)  # Calcular IC
        # Asegurar que devolvemos un número finito
        return float(ic) if np.isfinite(ic) else 0.0  # Retornar IC
    except Exception:  # Fallback
        # Fallback a correlación de Pearson si scipy no está disponible
        return float(np.corrcoef(y_true, y_pred)[0, 1])  # Pearson


def run_e2_simple_for_ticker(  # Ejecutar E2 simple
    config: dict,  # Config
    ticker: str,  # Ticker
    raw_dir: Path,  # Dir raw
    out_dir: Path,  # Dir salida
    benchmark_df: pd.DataFrame | None  # Benchmark
) -> dict:  # Retorna resumen
    """Entrena y evalúa E2 simple para un ticker (sin walk-forward)."""
    
    config = copy.deepcopy(config)  # Copia defensiva
    
    out_dir = Path(out_dir)  # A Path
    if out_dir.name != ticker:  # Ajustar subdir
        out_dir = out_dir / ticker  # Subdir ticker
    ensure_dir(out_dir)  # Crear dir
    
    # Parámetros de la estrategia E2 simple
    e2_simple = config.get("strategies", {}).get("e2_simple", {})  # Config E2 simple
    lookback_days = int(e2_simple.get("lookback_days", 60))  # Lookback
    horizon_days = int(e2_simple.get("horizon_days", 20))  # Horizon
    
    # Modelo (arquitectura simplificada)
    model_cfg = e2_simple.get("model", {})  # Config modelo
    lstm_units = model_cfg.get("lstm_units", [128, 64])  # 2 capas LSTM por defecto
    dropout = float(model_cfg.get("dropout", 0.2))  # Dropout
    dense_units = int(model_cfg.get("dense_units", 16))  # Dense
    lr = float(model_cfg.get("learning_rate", 0.001))  # LR
    batch_size = int(model_cfg.get("batch_size", 64))  # Batch
    max_epochs = int(model_cfg.get("max_epochs", 100))  # Epochs
    patience = int(model_cfg.get("early_stopping_patience", 10))  # Patience
    loss = str(model_cfg.get("loss", "huber"))  # Loss
    huber_delta = float(model_cfg.get("huber_delta", 1.0))  # Delta
    
    seed = int(config.get("project", {}).get("seed", 42))  # Seed
    
    # Cargar datos
    clean_dir = raw_dir.parent.parent / "clean"  # Dir clean
    clean_csv_path = clean_dir / f"{ticker}_daily.csv"  # Path clean
    
    if clean_csv_path.exists():  # Usar limpio
        csv_path = clean_csv_path  # Path CSV
        print(f"  ✓ Usando datos limpios: {csv_path.name}")  # Log
    else:  # Usar raw
        csv_path = raw_dir / f"{ticker}_daily.csv"  # Path raw
        if not csv_path.exists():  # Validar
            raise FileNotFoundError(f"Missing daily CSV: {csv_path}")  # Error
        print(f"  ⚠️  Usando datos raw: {csv_path.name}")  # Log
    
    ohlcv = load_ohlcv_csv(csv_path)  # Cargar OHLCV

    # Features y target
    feat_df = compute_e2_features(ohlcv, benchmark_df=benchmark_df)  # Features E2
    target = make_target_e2(ohlcv, horizon_days=horizon_days)  # Target E2

    # Secuencias
    X, y, ts, feat_names = make_sequences(feat_df, target, lookback=lookback_days)  # Secuencias

    # Split temporal
    idx_train, idx_val, idx_test = time_split(len(X))  # Split train/val/test
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    ts_test = ts[idx_test]

    # Normalizar X por feature
    mean_X = X_train.mean(axis=(0, 1))
    std_X = X_train.std(axis=(0, 1)) + 1e-8

    def scale_X(data: np.ndarray) -> np.ndarray:
        return ((data - mean_X) / std_X).astype(np.float32)

    # Normalizar y
    mean_y = float(y_train.mean())
    std_y = float(y_train.std()) + 1e-8

    def scale_y(data: np.ndarray) -> np.ndarray:
        return ((data - mean_y) / std_y).astype(np.float32)

    def unscale_y(data: np.ndarray) -> np.ndarray:
        return (data * std_y + mean_y).astype(np.float32)

    X_train_s = scale_X(X_train)
    X_val_s = scale_X(X_val)
    X_test_s = scale_X(X_test)
    y_train_s = scale_y(y_train)
    y_val_s = scale_y(y_val)
    
    # Entrenar modelo LSTM simplificado
    print(f"  Entrenando LSTM {lstm_units}...")  # Log entrenamiento
    model = LSTMRegressor(  # Instanciar LSTM
        input_size=X_train.shape[-1],  # Input size
        hidden_sizes=list(lstm_units),  # Units
        dropout=dropout,  # Dropout
        dense_units=dense_units,  # Dense
        seed=seed,  # Seed
    )  # Fin init
    
    train_started_at = datetime.now(timezone.utc).isoformat()
    train_start = time.perf_counter()
    res = model.fit(  # Entrenar
        X_train_s,  # X train
        y_train_s,  # y train
        X_val_s,  # X val
        y_val_s,  # y val
        learning_rate=lr,  # LR
        batch_size=batch_size,  # Batch
        max_epochs=max_epochs,  # Epochs
        early_stopping_patience=patience,  # Patience
        loss=loss,  # Loss
        huber_delta=huber_delta,  # Delta
    )  # Fin fit
    train_end = time.perf_counter()
    train_ended_at = datetime.now(timezone.utc).isoformat()
    log_timing_event(
        strategy="e2_simple",
        phase="train",
        duration_seconds=train_end - train_start,
        started_at=train_started_at,
        ended_at=train_ended_at,
        ticker=ticker,
        run_dir=out_dir,
        extra={
            "split": "time_split",
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
        },
    )
    
    # Predicciones
    pred_started_at = datetime.now(timezone.utc).isoformat()
    pred_start = time.perf_counter()
    y_pred_s = model.predict(X_test_s)  # Preds escaladas
    pred_end = time.perf_counter()
    pred_ended_at = datetime.now(timezone.utc).isoformat()
    log_timing_event(
        strategy="e2_simple",
        phase="predict",
        duration_seconds=pred_end - pred_start,
        started_at=pred_started_at,
        ended_at=pred_ended_at,
        ticker=ticker,
        run_dir=out_dir,
        extra={
            "split": "time_split",
            "n_test": int(len(X_test)),
        },
    )
    y_pred = unscale_y(y_pred_s)  # Preds reales
    
    # Métricas ML
    mae = float(np.mean(np.abs(y_test - y_pred)))  # MAE
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))  # RMSE
    dir_acc = float(np.mean(np.sign(y_test) == np.sign(y_pred)))  # Accuracy
    ic = compute_information_coefficient(y_test, y_pred)  # IC
    
    # Sanitizar valores NaN para IC
    if not np.isfinite(ic):  # Validar IC
        ic = 0.0  # Fallback
    
    # Backtest
    thresholds = e2_simple.get("thresholds", {})  # Thresholds
    tau_buy = float(thresholds.get("tau_buy", 0.025))  # Tau buy
    tau_sell = float(thresholds.get("tau_sell", 0.00))  # Tau sell
    
    costs_cfg = config.get("costs", {})  # Costos
    round_trip_bps = float(costs_cfg.get("daily_round_trip_bps", 10))  # BPS
    
    bt_cfg = e2_simple.get("backtest", {})  # Config backtest
    holding_period = int(bt_cfg.get("holding_period_days", horizon_days))  # Holding
    allow_short = bool(bt_cfg.get("allow_short", False))  # Shorts
    max_position = float(bt_cfg.get("max_position", 1.0))  # Max posición
    
    close_prices = ohlcv.loc[ts_test, "close"].to_numpy()  # Close prices
    bt = backtest_daily_signals(  # Backtest
        timestamps=ts_test,  # Timestamps
        close_prices=close_prices,  # Close
        pred_returns=y_pred,  # Pred returns
        tau_buy=tau_buy,  # Tau buy
        tau_sell=tau_sell,  # Tau sell
        round_trip_bps=round_trip_bps,  # BPS
        holding_period_days=holding_period,  # Holding
        allow_short=allow_short,  # Shorts
        max_position=max_position,  # Max pos
    )  # Fin backtest
    trading_metrics = summarize_backtest(bt)  # Resumen
    
    sharpe = float(trading_metrics.get("sharpe", float("nan")))  # Sharpe
    
    # Print resultados
    ic_str = "nan" if np.isnan(ic) else f"{ic:.3f}"  # IC string
    sharpe_str = "nan" if np.isnan(sharpe) else f"{sharpe:.2f}"  # Sharpe string

    print(f"  ✓ Epochs: {res.epochs_ran}/{max_epochs} | Val Loss: {res.best_val_loss:.6f}")  # Log epochs
    print(f"  ✓ MAE={mae:.4f} RMSE={rmse:.4f} IC={ic_str} Dir={dir_acc:.1%} Sharpe={sharpe_str}")  # Log métricas
    
    # Guardar resultados
    pred_df = pd.DataFrame(  # DataFrame preds
        {'y_true': y_test, 'y_pred': y_pred},  # Dict preds
        index=ts_test  # Index ts
    )  # Fin DataFrame
    pred_df.index.name = 'timestamp'  # Nombre index
    pred_df.to_csv(out_dir / f"{ticker}_predictions.csv")  # Guardar preds
    
    bt.to_csv(out_dir / f"{ticker}_backtest.csv")  # Guardar backtest
    
    scaler_df = pd.DataFrame({  # DF scaler
        'feature': list(feat_names),  # Features
        'mean': mean_X,  # Media
        'std': std_X,  # Std
    })  # Fin DF scaler
    scaler_df.to_csv(out_dir / f"{ticker}_scaler.csv", index=False)  # Guardar scaler
    
    # Guardar modelo
    model_payload = {  # Payload modelo
        "ticker": ticker,  # Ticker
        "strategy": "e2_simple",  # Estrategia
        "created_at": datetime.now(timezone.utc).isoformat(),  # Timestamp
        "model_class": "LSTMRegressor",  # Clase
        "model_kwargs": {  # Args modelo
            "input_size": int(X_train.shape[-1]),  # Input size
            "hidden_sizes": list(lstm_units),  # Units
            "dropout": float(dropout),  # Dropout
            "dense_units": int(dense_units),  # Dense
            "seed": int(seed),  # Seed
        },  # Fin model_kwargs
        "lookback_days": int(lookback_days),  # Lookback
        "horizon_days": int(horizon_days),  # Horizon
        "feature_names": list(feat_names),  # Features
        "scaler_X": {"mean": mean_X.tolist(), "std": std_X.tolist()},  # Scaler X
        "scaler_y": {"mean": float(mean_y), "std": float(std_y)},  # Scaler y
        "train_result": {  # Resultados
            "epochs_ran": int(res.epochs_ran),  # Epochs
            "best_val_loss": float(res.best_val_loss)  # Val loss
        },  # Fin train_result
        "state_dict": {k: v.detach().cpu() for k, v in model.model.state_dict().items()},  # Pesos
    }  # Fin payload
    
    model_path = out_dir / f"{ticker}_model.pth"  # Path modelo
    try:  # Guardar modelo
        model.torch.save(model_payload, model_path)  # Guardar
        print(f"  ✓ Modelo guardado: {model_path.name}")  # Log
    except Exception as exc:  # Error guardado
        print(f"  ⚠️  No se pudo guardar modelo: {exc}")  # Warning
    
    # Summary
    root = project_root()  # Root
    def as_relative(path: Path) -> str:  # Relativizar
        try:  # Intentar
            return str(path.relative_to(root))  # Relativo
        except ValueError:  # Fallback
            return str(path)  # Absoluto
    
    summary = {  # Summary
        "ticker": ticker,  # Ticker
        "n_samples": int(len(X)),  # Samples
        "n_train": int(len(idx_train)),  # N train
        "n_val": int(len(idx_val)),  # N val
        "n_test": int(len(idx_test)),  # N test
        "lookback_days": int(lookback_days),  # Lookback
        "horizon_days": int(horizon_days),  # Horizon
        "split_method": "time_split",  # Split
        "epochs_ran": int(res.epochs_ran),  # Epochs
        "val_loss": float(res.best_val_loss),  # Val loss
        "ml_mae": mae,  # MAE
        "ml_rmse": rmse,  # RMSE
        "ml_directional_accuracy": dir_acc,  # Accuracy
        "ml_ic": ic,  # IC
        **{f"bt_{k}": float(v) for k, v in trading_metrics.items()},  # Métricas BT
        "predictions_file": as_relative(out_dir / f"{ticker}_predictions.csv"),  # Preds file
        "backtest_file": as_relative(out_dir / f"{ticker}_backtest.csv"),  # Backtest file
        "scaler_file": as_relative(out_dir / f"{ticker}_scaler.csv"),  # Scaler file
        "model_file": as_relative(model_path),  # Model file
    }  # Fin summary
    
    summary_df = pd.DataFrame([summary])  # DF summary
    summary_df.to_csv(out_dir / f"{ticker}_summary.csv", index=False)  # Guardar summary
    
    return summary  # Retornar


def main():  # Main
    """Ejecuta el pipeline E2 simple para uno o más tickers."""
    parser = argparse.ArgumentParser(description="Pipeline E2 Simple (sin walk-forward, LSTM simplificado)")  # Parser
    parser.add_argument(  # Arg tickers
        "--tickers",  # Flag
        type=str,  # Tipo
        help="Tickers separados por coma (ej: NVDA,GOOGL,AMZN). Si se omite, usa los del config."  # Help
    )  # Fin arg tickers
    parser.add_argument(  # Arg config
        "--config",  # Flag
        type=str,  # Tipo
        default="src/config/base.yaml",  # Default
        help="Ruta al archivo de configuración"  # Help
    )  # Fin arg config
    parser.add_argument(  # Arg skip-download
        "--skip-download",  # Flag
        action="store_true",  # Action
        help="Omite la descarga de datos (usa datos existentes)"  # Help
    )  # Fin arg skip-download
    parser.add_argument(  # Arg skip-cleaning
        "--skip-cleaning",  # Flag
        action="store_true",  # Action
        help="Omite la limpieza de datos (usa datos raw)"  # Help
    )  # Fin arg skip-cleaning
    args = parser.parse_args()  # Parse args
    
    root = project_root()  # Root
    config_path = root / args.config  # Path config
    
    if not config_path.exists():  # Validar config
        raise FileNotFoundError(f"Config no encontrado: {config_path}")  # Error
    
    config = load_yaml(config_path)  # Cargar config
    
    # Determinar tickers
    if args.tickers:  # Tickers por args
        tickers = [t.strip() for t in args.tickers.split(",")]  # Parse tickers
    else:  # Tickers por config
        tickers = config.get("universe", {}).get("tickers_by_strategy", {}).get("e2_simple", [])  # Config
        if not tickers:  # Validar
            raise ValueError("No se especificaron tickers ni en args ni en config para e2_simple")  # Error
    
    if not tickers:  # Validar
        raise ValueError("No se especificaron tickers ni en args ni en config")  # Error
    
    # Benchmark: deshabilitado (no descargar/limpiar/cargar benchmark)
    all_tickers = tickers  # Universe
    
    # Directorios
    raw_dir = root / "data" / "raw" / "daily"  # Dir raw
    clean_dir = root / "data" / "clean"  # Dir clean
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Timestamp
    out_dir = root / "runs" / "e2_simple" / timestamp  # Dir salida
    ensure_dir(out_dir)  # Crear dir
    
    # Guardar config usado
    config_used_path = out_dir / "config_used.yaml"  # Path config
    import yaml  # Import yaml
    with open(config_used_path, 'w') as f:  # Abrir archivo
        yaml.dump(config, f, default_flow_style=False)  # Escribir config
    
    print(f"\n{'='*60}")  # Separador
    print(f"Pipeline E2 Simple - {len(tickers)} tickers")  # Header
    print(f"Output: {out_dir.relative_to(root)}")  # Output
    print(f"{'='*60}\n")  # Separador

    # MLflow setup (default: local tracking if installed)
    mlflow_enabled = False
    mlflow = None
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "E2_Simple")
    try:
        import mlflow as _mlflow  # type: ignore

        if tracking_uri:
            _mlflow.set_tracking_uri(tracking_uri)
        _mlflow.set_experiment(experiment_name)
        mlflow = _mlflow
        mlflow_enabled = True
        if tracking_uri:
            print(f"✓ MLflow habilitado: {tracking_uri} (experiment={experiment_name})")
        else:
            print(f"✓ MLflow habilitado (tracking local, experiment={experiment_name})")
    except Exception as exc:
        print(f"⚠️  MLflow no disponible, continuando sin tracking: {exc}")
        mlflow_enabled = False
    
    # Paso 1: Descargar datos
    if not args.skip_download:  # Si no omite
        print("Paso 1/3: Descargando datos...")  # Log
        print("-" * 60)  # Separador
        from .data.download_daily import download_daily_ohlcv  # Import download
        
        try:  # Descargar
            written = download_daily_ohlcv(  # Ejecutar descarga
                all_tickers,  # Tickers
                out_dir=raw_dir,  # Output
                period="10y",  # Periodo
                skip_existing=True,  # Skip
                min_days_fresh=1,  # Frescura
            )  # Fin download
            print(f"✓ Descargados/actualizados {len(written)} archivos\n")  # Log
        except Exception as exc:  # Error descarga
            print(f"⚠️  Error en descarga: {exc}")  # Warning
            print("Continuando con datos existentes...\n")  # Fallback
    else:  # Descarga omitida
        print("Paso 1/3: Descarga omitida (usando datos existentes)\n")  # Log
    
    # Paso 2: Limpiar datos
    if not args.skip_cleaning:  # Si no omite
        print("Paso 2/3: Limpiando datos...")  # Log
        print("-" * 60)  # Separador
        from .data.clean_daily import process_daily_data_with_cleaning  # Import cleaning
        
        try:  # Limpiar
            tickers_to_clean = list(dict.fromkeys(tickers))  # Tickers a limpiar
            reports = process_daily_data_with_cleaning(  # Ejecutar limpieza
                raw_dir=raw_dir,  # Dir raw
                clean_dir=clean_dir,  # Dir clean
                strategy="forward_fill",  # Estrategia
                min_days=252,  # Min días
                remove_zero_volume=True,  # Quitar cero volumen
                verbose=False,  # Verbose
                tickers=tickers_to_clean,  # Filtrar tickers
            )  # Fin limpieza
            cleaned = sum(1 for r in reports.values() if r.get("status") == "cleaned")  # Contar clean
            rejected = sum(1 for r in reports.values() if r.get("status") == "rejected")  # Contar rechazados
            print(f"✓ Limpiados: {cleaned} | Rechazados: {rejected}\n")  # Log
        except Exception as exc:  # Error limpieza
            print(f"⚠️  Error en limpieza: {exc}")  # Warning
            print("Continuando con datos raw...\n")  # Fallback
    else:  # Limpieza omitida
        print("Paso 2/3: Limpieza omitida (usando datos raw)\n")  # Log
    
    # Paso 3: Entrenar modelos
    print("Paso 3/3: Entrenando modelos...")  # Log
    print("-" * 60)  # Separador
    
    # Benchmark deshabilitado
    benchmark_df = None
    
    # Entrenar cada ticker
    summaries = []  # Resúmenes
    for i, ticker in enumerate(tickers, 1):  # Loop tickers
        print(f"[{i}/{len(tickers)}] {ticker}")  # Log ticker
        try:  # Ejecutar ticker
            if mlflow_enabled and mlflow is not None:
                timestamp = out_dir.name
                e2_simple_cfg = config.get("strategies", {}).get("e2_simple", {})
                model_cfg = e2_simple_cfg.get("model", {})
                thresholds = e2_simple_cfg.get("thresholds", {})
                costs_cfg = config.get("costs", {})
                bt_cfg = e2_simple_cfg.get("backtest", {})
                seed = int(config.get("project", {}).get("seed", 42))

                with mlflow.start_run(run_name=f"E2Simple_{ticker}_{timestamp}"):
                    mlflow.log_param("strategy", "e2_simple")
                    mlflow.log_param("ticker", ticker)
                    mlflow.log_param("timestamp", timestamp)

                    mlflow.log_params(
                        {
                            "lookback_days": int(e2_simple_cfg.get("lookback_days", 120)),
                            "horizon_days": int(e2_simple_cfg.get("horizon_days", 30)),
                            "lstm_units": str(model_cfg.get("lstm_units", [64])),
                            "dropout": float(model_cfg.get("dropout", 0.2)),
                            "dense_units": int(model_cfg.get("dense_units", 16)),
                            "learning_rate": float(model_cfg.get("learning_rate", 0.001)),
                            "batch_size": int(model_cfg.get("batch_size", 64)),
                            "max_epochs": int(model_cfg.get("max_epochs", 100)),
                            "early_stopping_patience": int(model_cfg.get("early_stopping_patience", 10)),
                            "loss": str(model_cfg.get("loss", "huber")),
                            "huber_delta": float(model_cfg.get("huber_delta", 1.0)),
                            "tau_buy": float(thresholds.get("tau_buy", 0.04)),
                            "tau_sell": float(thresholds.get("tau_sell", 0.00)),
                            "round_trip_bps": float(costs_cfg.get("daily_round_trip_bps", 10)),
                            "holding_period_days": int(bt_cfg.get("holding_period_days", e2_simple_cfg.get("horizon_days", 30))),
                            "allow_short": bool(bt_cfg.get("allow_short", False)),
                            "max_position": float(bt_cfg.get("max_position", 1.0)),
                            "seed": seed,
                        }
                    )

                    summary = run_e2_simple_for_ticker(  # Ejecutar
                        config=config,  # Config
                        ticker=ticker,  # Ticker
                        raw_dir=raw_dir,  # Dir raw
                        out_dir=out_dir,  # Dir salida
                        benchmark_df=benchmark_df,  # Benchmark
                    )  # Fin run

                    metrics: dict[str, float] = {}
                    for k, v in summary.items():
                        if not isinstance(v, (int, float)):
                            continue
                        if k.startswith("ml_") or k.startswith("bt_"):
                            metrics[k] = float(v)
                    if metrics:
                        mlflow.log_metrics(metrics)

                    ticker_out = out_dir / ticker
                    for fname in [
                        f"{ticker}_predictions.csv",
                        f"{ticker}_backtest.csv",
                        f"{ticker}_summary.csv",
                        f"{ticker}_scaler.csv",
                        f"{ticker}_model.pth",
                    ]:
                        p = ticker_out / fname
                        if p.exists():
                            if "pred" in fname:
                                artifact_path = "predictions"
                            elif "backtest" in fname:
                                artifact_path = "backtest"
                            elif "model" in fname:
                                artifact_path = "models"
                            elif "scaler" in fname:
                                artifact_path = "scalers"
                            else:
                                artifact_path = "artifacts"
                            mlflow.log_artifact(str(p), artifact_path=artifact_path)
            else:
                summary = run_e2_simple_for_ticker(  # Ejecutar
                    config=config,  # Config
                    ticker=ticker,  # Ticker
                    raw_dir=raw_dir,  # Dir raw
                    out_dir=out_dir,  # Dir salida
                    benchmark_df=benchmark_df,  # Benchmark
                )  # Fin run
            summaries.append(summary)  # Append
        except Exception as exc:  # Error
            print(f"  ❌ Error: {exc}\n")  # Log error
            continue  # Siguiente
        print()  # Línea en blanco
    
    # Guardar summary agregado
    if summaries:  # Hay resultados
        summary_all = pd.DataFrame(summaries)  # DF summary
        summary_all.to_csv(out_dir / "summary_all.csv", index=False)  # Guardar
        
        print(f"\n{'='*60}")  # Separador
        print(f"✓ Completado: {len(summaries)}/{len(tickers)} tickers")  # Log
        print(f"  Resultados en: {out_dir.relative_to(root)}/")  # Output
        print(f"{'='*60}\n")  # Separador
    else:  # Sin resultados
        print("\n❌ No se completó ningún ticker exitosamente\n")  # Log


if __name__ == "__main__":  # Entry point
    main()  # Ejecutar main
