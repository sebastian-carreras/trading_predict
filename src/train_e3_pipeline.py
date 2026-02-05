"""E3 Pipeline: Intraday Trading con LSTM Ensemble.

Características:
- Datos de 5 minutos (intraday)
- Modelo: LSTM ensemble (múltiples miembros para robustez)
- Estrategia: Predicción de retorno intradiario
- Backtest: Con ejecución de órdenes y seguimiento de equity

Flujo:
1. Cargar datos OHLCV de 5 minutos
2. Calcular features intraday
3. Crear target: retorno forward (N barras)
4. Crear secuencias temporales
5. Split temporal train/val/test
6. Entrenar ensemble de LSTM
7. Combinar predicciones (promedio)
8. Evaluar ML metrics (IC, MAE, RMSE, etc)
9. Ejecutar backtest con reglas de trading
10. Guardar resultados y análisis
"""

from __future__ import annotations  # Forward refs

import argparse  # CLI args
import os  # Env vars
from datetime import datetime  # Timestamps
from pathlib import Path  # Paths

import numpy as np  # NumPy
import pandas as pd  # Pandas

from .backtest.backtest_intraday import (  # Backtest intraday
    backtest_intraday_signals,  # Simula ejecución de órdenes
    compute_max_drawdown,        # Calcula máxima caída del equity
    compute_profit_factor,       # Calcula profit factor (ganancias/pérdidas)
)  # Fin import backtest
from .data.intraday_yfinance import download_ohlcv_5m, load_ohlcv_csv  # Data intraday
from .features.build_features_e3 import compute_intraday_features, make_sequences, make_target_return  # Features/target
from .models.e3_lstm import LSTMRegressor  # Modelo LSTM
from .reporting.intraday_metrics import directional_accuracy, information_coefficient, mae, rmse  # Métricas
from .utils import ensure_dir, get_nested, load_yaml, project_root  # Utils


def _time_split(n: int, train_frac: float = 0.7, val_frac: float = 0.15):  # Split temporal
    """Divide datos en train/val/test respetando orden temporal.
    
    Args:
        n: Total de muestras
        train_frac: Fracción de datos para entrenamiento (default 70%)
        val_frac: Fracción de datos para validación (default 15%)
                  El resto (15%) se usa para test
    
    Returns:
        (train_idx, val_idx, test_idx): Arrays de índices para cada split
    """
    # Validar que las fracciones son válidas
    if not (0 < train_frac < 1) or not (0 < val_frac < 1) or train_frac + val_frac >= 1:  # Validar
        raise ValueError("Invalid split fractions")  # Error

    # Calcular puntos de corte
    train_end = int(n * train_frac)  # Corte train
    val_end = int(n * (train_frac + val_frac))  # Corte val
    
    # Crear arrays de índices (mantiene orden temporal)
    idx_train = np.arange(0, train_end)  # Índices train
    idx_val = np.arange(train_end, val_end)  # Índices val
    idx_test = np.arange(val_end, n)  # Índices test
    
    return idx_train, idx_val, idx_test  # Retornar splits


def run_for_ticker(config: dict, ticker: str, raw_dir: Path, out_dir: Path) -> dict:  # Pipeline E3
    """Ejecuta pipeline E3 completo para un ticker.
    
    Procedimiento:
    1. Carga datos OHLCV de 5 minutos
    2. Extrae features intraday (momentum, volatilidad, etc)
    3. Crea target: retorno forward en N barras
    4. Divide en train/val/test manteniendo orden temporal
    5. Normaliza features (Z-score)
    6. Entrena ensemble de modelos LSTM
    7. Evalúa métricas ML (IC, MAE, RMSE, Directional Accuracy)
    8. Ejecuta backtest con señales de trading
    9. Calcula métricas de trading (Sharpe, Max DD, Profit Factor)
    10. Guarda todos los artefactos
    
    Args:
        config: Diccionario de configuración
        ticker: Símbolo del activo (ej: "AAPL")
        raw_dir: Directorio con datos CSV intraday
        out_dir: Directorio para salidas
    
    Returns:
        dict: Resumen con todas las métricas
    """
    print(f"  Cargando configuración para {ticker}...")  # Log config
    e3 = get_nested(config, ["strategies", "e3_intraday"], {})  # Config E3
    
    print(f"  Extrayendo parámetros...")  # Log params
    # Parámetros de secuencias
    lookback_bars = int(e3.get("lookback_bars", 96))      # Histórico: 96 barras (8 horas)
    horizon_bars = int(e3.get("horizon_bars", 6))         # Forward: 6 barras (30 minutos)

    # Parámetros de señales de trading
    thresholds = e3.get("thresholds", {})  # Thresholds
    print(f"  Thresholds type: {type(thresholds)}, value: {thresholds}")  # Debug
    tau_buy = float(thresholds.get("tau_buy", 0.001))      # Umbral para compra
    tau_sell = float(thresholds.get("tau_sell", 0.001))    # Umbral para venta

    # Costos de transacción
    costs = config.get("costs", {})  # Costos
    round_trip_bps = float(costs.get("intraday_round_trip_bps", 20))  # Costo intraday (basis points)

    # Configuración del modelo ensemble
    model_cfg = e3.get("model", {})  # Config modelo
    ensemble_members = int(model_cfg.get("ensemble_members", 3))    # Número de miembros en ensemble
    hidden_size = int(model_cfg.get("lstm_hidden_size", 64))        # Unidades LSTM
    num_layers = int(model_cfg.get("lstm_num_layers", 2))           # Capas LSTM
    dropout = float(model_cfg.get("dropout", 0.2))                  # Regularización
    lr = float(model_cfg.get("learning_rate", 1e-3))                # Velocidad de aprendizaje
    batch_size = int(model_cfg.get("batch_size", 256))              # Tamaño de batch
    max_epochs = int(model_cfg.get("max_epochs", 30))               # Épocas máx
    patience = int(model_cfg.get("early_stopping_patience", 5))     # Early stopping
    loss = str(model_cfg.get("loss", "huber"))                      # Función de pérdida
    huber_delta = float(model_cfg.get("huber_delta", 1.0))          # Parámetro delta

    # Configuración de backtest
    bt_cfg = e3.get("backtest", {})  # Config backtest
    execution_delay_bars = int(bt_cfg.get("execution_delay_bars", 1))  # Delay en ejecución
    allow_short = bool(bt_cfg.get("allow_short", True))                 # Permite ventas cortas
    max_position = float(bt_cfg.get("max_position", 1.0))               # Posición máxima

    # Cargar datos OHLCV intraday (5 minutos)
    csv_path = raw_dir / f"{ticker}_5m.csv"  # Path CSV
    if not csv_path.exists():  # Validar CSV
        raise FileNotFoundError(f"Missing intraday CSV: {csv_path}")  # Error

    ohlcv = load_ohlcv_csv(csv_path)  # Cargar OHLCV
    
    # Extraer features técnicos intraday
    features = compute_intraday_features(ohlcv)  # Features
    
    # Crear target: retorno forward (siguiente N barras)
    target = make_target_return(ohlcv, horizon_bars=horizon_bars)  # Target

    # Crear secuencias: [n_samples, lookback_bars, n_features]
    X, y, ts, feat_names = make_sequences(features, target, lookback_bars=lookback_bars)  # Secuencias

    # Split temporal: 70% train, 15% val, 15% test
    idx_train, idx_val, idx_test = _time_split(len(X))  # Split
    X_train, y_train = X[idx_train], y[idx_train]  # Train
    X_val, y_val = X[idx_val], y[idx_val]  # Val
    X_test, y_test = X[idx_test], y[idx_test]  # Test
    ts_test = ts[idx_test]  # Timestamps test

    # ESTANDARIZAR FEATURES (Z-score normalization)
    # Compute statistics only on training data to prevent data leakage
    Xtr2d = X_train.reshape(-1, X_train.shape[-1])  # Flatten
    mean = Xtr2d.mean(axis=0)  # Media
    std = Xtr2d.std(axis=0) + 1e-12  # Std

    def scale(Xa: np.ndarray) -> np.ndarray:  # Escalar
        """Aplica normalización Z-score a los datos."""
        return ((Xa - mean) / std).astype(np.float32)  # Z-score

    X_train_s = scale(X_train)  # Train scaled
    X_val_s = scale(X_val)  # Val scaled
    X_test_s = scale(X_test)  # Test scaled

    # ENTRENAR ENSEMBLE DE LSTM
    # Entrenar múltiples modelos con diferentes seeds para mayor robustez
    preds_members: list[np.ndarray] = []  # Preds por miembro
    val_losses: list[float] = []  # Losses val

    base_seed = int(get_nested(config, ["project", "seed"], 42))  # Seed base

    print(f"\nEntrenando ensemble de {ensemble_members} modelos LSTM para {ticker}...")  # Log
    for m in range(ensemble_members):  # Loop miembros
        print(f"\n  Modelo {m+1}/{ensemble_members} (seed={base_seed + 1000 * m}):")  # Log miembro
        seed = base_seed + 1000 * m  # Seed miembro
        
        # Crear modelo LSTM
        model = LSTMRegressor(  # Instanciar LSTM
            input_size=X_train_s.shape[-1],  # Input size
            hidden_size=hidden_size,  # Hidden
            num_layers=num_layers,  # Capas
            dropout=dropout,  # Dropout
            seed=seed,  # Seed
        )  # Fin init
        
        # Entrenar modelo
        res = model.fit(  # Entrenar
            X_train_s,  # X train
            y_train,  # y train
            X_val_s,  # X val
            y_val,  # y val
            learning_rate=lr,  # LR
            batch_size=batch_size,  # Batch
            max_epochs=max_epochs,  # Epochs
            early_stopping_patience=patience,  # Patience
            loss=loss,  # Loss
            huber_delta=huber_delta,  # Delta
            verbose=True,  # Verbose
        )  # Fin fit
        
        val_losses.append(res.best_val_loss)  # Guardar val loss
        preds_members.append(model.predict(X_test_s))  # Guardar preds
        print(f"  Modelo {m+1} final: val_loss={res.best_val_loss:.6f}, epochs={res.epochs_ran}")  # Log

    print(f"\nEnsemble completado. Promediando {len(preds_members)} predicciones...")  # Log

    # COMBINAR PREDICCIONES DEL ENSEMBLE
    # Promediamos las predicciones de todos los miembros para robustez
    y_pred = np.mean(np.stack(preds_members, axis=0), axis=0)  # Promedio

    # MÉTRICAS ML
    ml = {  # Métricas ML
        "mae": mae(y_test, y_pred),  # MAE
        "rmse": rmse(y_test, y_pred),  # RMSE
        "ic": information_coefficient(y_test, y_pred),  # IC
        "directional_accuracy": directional_accuracy(y_test, y_pred),  # Accuracy
        "val_loss_mean": float(np.mean(val_losses)),  # Val loss mean
    }  # Fin métricas

    # BACKTEST
    # Calcular retornos de 1 barra (log returns alineados con timestamps)
    close_series = ohlcv["close"].astype("float64")  # Close series
    log_close = pd.Series(np.log(close_series.to_numpy()), index=close_series.index)  # Log close
    bar_ret = log_close.diff().reindex(ts_test).fillna(0.0).to_numpy(dtype=np.float32)  # Retorno barra

    # Ejecutar backtest con señales predichas
    bt = backtest_intraday_signals(  # Backtest
        timestamps=ts_test,  # Timestamps
        bar_returns=bar_ret,  # Retornos barra
        pred_forward_returns=y_pred.astype(np.float32),  # Pred forward
        tau_buy=tau_buy,  # Tau buy
        tau_sell=tau_sell,  # Tau sell
        round_trip_bps=round_trip_bps,  # Costos
        execution_delay_bars=execution_delay_bars,  # Delay
        allow_short=allow_short,  # Shorts
        max_position=max_position,  # Max posición
    )  # Fin backtest

    # MÉTRICAS DE TRADING
    trading = {  # Métricas trading
        "profit_factor": compute_profit_factor(bt["net_ret"].to_numpy(dtype=np.float32)),  # Profit factor
        "max_drawdown": compute_max_drawdown(bt["equity"].to_numpy(dtype=np.float32)),  # Max DD
        "turnover_mean": float(bt["turnover"].mean()),  # Turnover
        "time_in_market": float((bt["pos"].abs() > 0).mean()),  # Time in market
    }  # Fin métricas trading

    # GUARDAR ARTEFACTOS
    ensure_dir(out_dir)  # Crear dir
    
    # Guardar backtest completo
    bt.to_csv(out_dir / f"{ticker}_backtest.csv")  # Guardar backtest

    # Guardar predicciones
    preds_df = pd.DataFrame(  # DF preds
        {"y_true": y_test, "y_pred": y_pred},  # Dict preds
        index=ts_test,  # Index ts
    )  # Fin DF
    preds_df.to_csv(out_dir / f"{ticker}_predictions.csv")  # Guardar preds

    # Guardar metadatos y resumen
    meta = {  # Metadata
        "ticker": ticker,  # Ticker
        "n_samples": int(len(X)),  # Samples
        "n_test": int(len(X_test)),  # N test
        "lookback_bars": lookback_bars,  # Lookback
        "horizon_bars": horizon_bars,  # Horizon
        "tau_buy": tau_buy,  # Tau buy
        "tau_sell": tau_sell,  # Tau sell
        "round_trip_bps": round_trip_bps,  # Costos
        "ensemble_members": ensemble_members,  # Ensemble size
        "feature_count": int(len(feat_names)),  # Feature count
    }  # Fin meta

    summary = {**meta, **{f"ml_{k}": v for k, v in ml.items()}, **{f"tr_{k}": v for k, v in trading.items()}}  # Summary
    (pd.Series(summary)).to_csv(out_dir / f"{ticker}_summary.csv")  # Guardar summary

    return summary  # Retornar


def main() -> None:  # Main
    parser = argparse.ArgumentParser(description="E3 intraday pipeline (5m, LSTM ensemble)")  # Parser
    parser.add_argument(  # Arg config
        "--config",  # Flag
        type=str,  # Tipo
        default="src/config/base.yaml",  # Default
        help="Path to YAML config (relative to trading_predict root is allowed)",  # Help
    )  # Fin arg config
    parser.add_argument(  # Arg mode
        "--mode",  # Flag
        type=str,  # Tipo
        choices=["download", "run"],  # Choices
        default="run",  # Default
        help="download: fetch CSVs; run: train+backtest using existing CSVs",  # Help
    )  # Fin arg mode
    parser.add_argument(  # Arg tickers
        "--tickers",  # Flag
        type=str,  # Tipo
        default="",  # Default
        help="Comma-separated tickers override (otherwise uses universe.tickers_by_strategy.e3_intraday)",  # Help
    )  # Fin arg tickers
    args = parser.parse_args()  # Parse args

    root = project_root()  # Root
    cfg_path = Path(args.config)  # Path config
    if not cfg_path.is_absolute():  # Resolver relativo
        cfg_path = root / cfg_path  # Absoluto

    config = load_yaml(cfg_path)  # Cargar config

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]  # Parse tickers
    if not tickers:  # Fallback config
        tickers = list(get_nested(config, ["universe", "tickers_by_strategy", "e3_intraday"], []))  # Config

    if not tickers:  # Validar tickers
        raise ValueError("No tickers provided and config has no universe.tickers_by_strategy.e3_intraday")  # Error

    e3 = get_nested(config, ["strategies", "e3_intraday"], {})  # Config E3
    data_cfg = e3.get("data", {})  # Config data
    thresholds = e3.get("thresholds", {})  # Thresholds
    costs = config.get("costs", {})  # Costos

    raw_dir = root / "data" / "raw" / "intraday"  # Dir raw
    out_base = root / "runs" / "e3_intraday" / datetime.now().strftime("%Y%m%d_%H%M%S")  # Dir salida
    ensure_dir(out_base)  # Crear dir

    if args.mode == "download":  # Modo download
        period = str(data_cfg.get("period", "60d"))  # Period
        interval = str(data_cfg.get("interval", "5m"))  # Interval
        written = download_ohlcv_5m(tickers, out_dir=raw_dir, period=period, interval=interval)  # Descargar
        print(f"Downloaded {len(written)} files to {raw_dir}")  # Log
        return  # Salir

    # MLflow setup (default: local tracking if installed)
    mlflow_enabled = False  # Flag MLflow
    mlflow = None  # Ref MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()  # Tracking URI
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "E3_Intraday")  # Experimento
    try:  # Import MLflow
        import mlflow as _mlflow  # type: ignore  # Import

        if tracking_uri:
            _mlflow.set_tracking_uri(tracking_uri)  # Set URI
        _mlflow.set_experiment(experiment_name)  # Set experimento
        mlflow = _mlflow  # Guardar ref
        mlflow_enabled = True  # Flag on
        if tracking_uri:
            print(f"✓ MLflow habilitado: {tracking_uri} (experiment={experiment_name})")  # Log
        else:
            print(f"✓ MLflow habilitado (tracking local, experiment={experiment_name})")  # Log
    except Exception as exc:  # Error MLflow
        print(f"⚠️  MLflow no disponible, continuando sin tracking: {exc}")  # Warning
        mlflow_enabled = False  # Flag off

    summaries: list[dict] = []  # Resúmenes
    for ticker in tickers:  # Loop tickers
        ticker_out = out_base / ticker  # Dir ticker
        try:  # Ejecutar ticker
            if mlflow_enabled and mlflow is not None:  # Con MLflow
                timestamp = out_base.name  # Timestamp
                with mlflow.start_run(run_name=f"E3_{ticker}_{timestamp}"):  # Run MLflow
                    mlflow.log_param("strategy", "e3_intraday")  # Param estrategia
                    mlflow.log_param("ticker", ticker)  # Param ticker
                    mlflow.log_param("model_type", "LSTM_Ensemble")  # Param modelo
                    mlflow.log_param("timestamp", timestamp)  # Param timestamp

                    model_cfg = e3.get("model", {})  # Config modelo
                    mlflow.log_params(  # Log params
                        {  # Dict params
                            "lookback_bars": int(e3.get("lookback_bars", 96)),  # Lookback
                            "horizon_bars": int(e3.get("horizon_bars", 6)),  # Horizon
                            "ensemble_members": int(model_cfg.get("ensemble_members", 3)),  # Ensemble
                            "lstm_hidden_size": int(model_cfg.get("lstm_hidden_size", 64)),  # Hidden
                            "lstm_num_layers": int(model_cfg.get("lstm_num_layers", 2)),  # Layers
                            "dropout": float(model_cfg.get("dropout", 0.2)),  # Dropout
                            "learning_rate": float(model_cfg.get("learning_rate", 1e-3)),  # LR
                            "batch_size": int(model_cfg.get("batch_size", 256)),  # Batch
                            "max_epochs": int(model_cfg.get("max_epochs", 30)),  # Epochs
                            "early_stopping_patience": int(model_cfg.get("early_stopping_patience", 5)),  # Patience
                            "loss": str(model_cfg.get("loss", "huber")),  # Loss
                            "huber_delta": float(model_cfg.get("huber_delta", 1.0)),  # Delta
                            "tau_buy": float(thresholds.get("tau_buy", 0.001)),  # Tau buy
                            "tau_sell": float(thresholds.get("tau_sell", 0.001)),  # Tau sell
                            "round_trip_bps": float(costs.get("intraday_round_trip_bps", 20)),  # Costos
                            "seed": int(config.get("project", {}).get("seed", 42)),  # Seed
                        }  # Fin params
                    )  # Fin log_params

                    summary = run_for_ticker(config, ticker=ticker, raw_dir=raw_dir, out_dir=ticker_out)  # Ejecutar

                    metrics: dict[str, float] = {}  # Métricas
                    for k, v in summary.items():  # Iterar summary
                        if not (k.startswith("ml_") or k.startswith("tr_")):  # Filtrar
                            continue  # Saltar
                        if isinstance(v, (int, float)):  # Numéricas
                            metrics[k] = float(v)  # Convertir
                    if metrics:  # Log metrics
                        mlflow.log_metrics(metrics)  # Log

                    # Log artifacts
                    for fname in [  # Archivos
                        f"{ticker}_predictions.csv",  # Preds
                        f"{ticker}_backtest.csv",  # Backtest
                        f"{ticker}_summary.csv",  # Summary
                    ]:  # Fin lista
                        p = ticker_out / fname  # Path
                        if p.exists():  # Existe
                            if "pred" in fname:  # Preds
                                artifact_path = "predictions"  # Carpeta
                            elif "backtest" in fname:  # Backtest
                                artifact_path = "backtest"  # Carpeta
                            else:  # Otros
                                artifact_path = "artifacts"  # Carpeta
                            mlflow.log_artifact(str(p), artifact_path=artifact_path)  # Log
            else:  # Sin MLflow
                summary = run_for_ticker(config, ticker=ticker, raw_dir=raw_dir, out_dir=ticker_out)  # Ejecutar
            summaries.append(summary)  # Append
            print(f"Done {ticker}: PF={summary['tr_profit_factor']:.3f} DD={summary['tr_max_drawdown']:.3%} IC={summary['ml_ic']:.3f}")  # Log
        except Exception as exc:  # Error ticker
            print(f"✗ Error en {ticker}: {exc}")  # Log error

    pd.DataFrame(summaries).to_csv(out_base / "summary_all.csv", index=False)  # Guardar summary
    print(f"Wrote run outputs to {out_base}")  # Log salida


if __name__ == "__main__":  # Entry point
    main()  # Ejecutar main
