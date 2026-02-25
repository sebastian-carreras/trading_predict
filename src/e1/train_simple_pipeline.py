"""
Pipeline simplificado E1 - Sin walk-forward.

Simplificaciones:
1. Sin walk-forward validation (solo un split temporal train/val/test)
2. Arquitectura GRU más simple (1 capa)

Flujo:
1. Descargar datos (opcional, con --skip-download)
2. Limpiar datos (opcional, con --skip-cleaning)
3. Calcular features
4. Crear target y secuencias
5. Split temporal simple (train/val/test)
6. Entrenar GRU
7. Evaluar y guardar
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import socket
import time
from urllib.parse import urlparse

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()  # Cargar .env (MLflow, MinIO, etc.)
except ImportError:
    pass

from .build_features import compute_e1_features, make_target_e1
from ..features.build_sequences_e1e2 import make_sequences, time_split
from .gru import GRURegressor
from ..backtest.backtest_daily import backtest_daily_signals, summarize_backtest
from ..utils import ensure_dir, load_yaml, project_root, log_timing_event
from ..dashboard import (
    log_dashboard_tags,
    log_dashboard_timing,
    log_metric_alerts,
)


def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    """Carga CSV OHLCV y lo prepara."""
    df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        raise ValueError(f"Missing 'timestamp' column in {path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")

    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")

    return df


def compute_information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula el Information Coefficient (correlación de Spearman)."""
    if len(y_true) <= 1:
        return float("nan")

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")

    try:
        from scipy.stats import spearmanr

        res = spearmanr(y_true, y_pred)
        ic_raw = getattr(res, "correlation", res[0])
        ic_val = float(np.asarray(ic_raw, dtype=float).ravel()[0])
        return ic_val if np.isfinite(ic_val) else 0.0
    except Exception:
        return float(np.corrcoef(y_true, y_pred)[0, 1])


def run_e1_simple_for_ticker(
    config: dict,
    ticker: str,
    raw_dir: Path,
    out_dir: Path,
    benchmark_df: pd.DataFrame | None = None,
) -> dict:
    """Entrena y evalúa E1 simple para un ticker (sin walk-forward)."""

    config = copy.deepcopy(config)

    out_dir = Path(out_dir)
    if out_dir.name != ticker:
        out_dir = out_dir / ticker
    ensure_dir(out_dir)

    e1_simple = config.get("strategies", {}).get("e1_simple", {})
    lookback_days = int(e1_simple.get("lookback_days", 360))
    horizon_days = int(e1_simple.get("horizon_days", 90))

    model_cfg = e1_simple.get("model", {})
    gru_units = model_cfg.get("gru_units", [64])
    dropout = float(model_cfg.get("dropout", 0.2))
    dense_units = int(model_cfg.get("dense_units", 16))
    lr = float(model_cfg.get("learning_rate", 0.001))
    batch_size = int(model_cfg.get("batch_size", 64))
    max_epochs = int(model_cfg.get("max_epochs", 100))
    patience = int(model_cfg.get("early_stopping_patience", 10))
    loss = str(model_cfg.get("loss", "huber"))
    huber_delta = float(model_cfg.get("huber_delta", 1.0))

    seed = int(config.get("project", {}).get("seed", 42))

    root = project_root()
    clean_csv_path = root / "data" / "clean" / f"{ticker}_daily.csv"
    if clean_csv_path.exists():
        csv_path = clean_csv_path
        print(f"  ✓ Usando datos limpios: {csv_path.name}")
    else:
        csv_path = raw_dir / f"{ticker}_daily.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing daily CSV: {csv_path}")
        print(f"  ⚠️  Usando datos raw: {csv_path.name}")

    # Cargar CSV y preparar (timestamp como índice, timezone UTC)
    ohlcv = load_ohlcv_csv(csv_path)
    
    # CALCULAR FEATURES E1
    # Extrae indicadores técnicos: momentum, volatilidad, tendencia, etc.
    features = compute_e1_features(ohlcv)  # Retorna DataFrame [time, features]
    # CREAR TARGET
    # Define qué predecir: retorno esperado en N días futuros
    target = make_target_e1(ohlcv, horizon_days=horizon_days)  # Retorna Series con retornos
    
    # CREAR SECUENCIAS TEMPORALES
    # Transforma datos planos en ventanas de tiempo para redes recurrentes
    # X: [n_samples, lookback_days, n_features], y: [n_samples]
    X, y, ts, feat_names = make_sequences(features, target, lookback=lookback_days)
    
    # Validar que hay suficientes datos después de crear secuencias
    if len(X) == 0:
        raise ValueError(f"No hay suficientes datos para {ticker}")
    
    # Mostrar información sobre dataset
    print(f"  Samples: {len(X)} | Features: {X.shape[-1]} | Lookback: {lookback_days}d")
    
    # SPLIT TEMPORAL - División train/val/test
    # Mantiene orden temporal para validación realista (sin data leakage)
    idx_train, idx_val, idx_test = time_split(len(X))  # Default: 70%/15%/15%
    
    # Asignar datos de entrenamiento
    X_train, y_train = X[idx_train], y[idx_train]  # Features y targets de train
    # Asignar datos de validación (para early stopping)
    X_val, y_val = X[idx_val], y[idx_val]          # Features y targets de val
    # Asignar datos de test (evaluación final)
    X_test, y_test = X[idx_test], y[idx_test]      # Features y targets de test
    # Guardar timestamps de test para backtesting posterior
    ts_test = ts[idx_test]
    
    # Mostrar tamaño de cada split
    print(f"  Split: train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")
    
    # ESTANDARIZACIÓN DE FEATURES (Z-score normalization)
    # Importante: fit SOLO en datos de train (previene data leakage)
    # Reshape: [n_samples, lookback_days, features] → [n_samples*lookback_days, features]
    Xtr2d = X_train.reshape(-1, X_train.shape[-1])
    # Calcular media por feature (para centrar: media=0)
    mean_X = Xtr2d.mean(axis=0)
    # Calcular std por feature (para escalar: std=1)
    # +1e-12 previene división por cero si feature es constante
    std_X = Xtr2d.std(axis=0) + 1e-12
    
    def scale_X(data: np.ndarray) -> np.ndarray:
        """Aplica Z-score: (x - media) / std. Retorna float32 para GPU."""
        return ((data - mean_X) / std_X).astype(np.float32)
    
    # ESTANDARIZACIÓN DE TARGETS
    # Estabiliza entrenamiento: reduce bias inicial del modelo
    mean_y = float(y_train.mean())  # Media de targets
    # +1e-12 previene división por cero
    std_y = float(y_train.std()) + 1e-12  # Std de targets
    
    def scale_y(data: np.ndarray) -> np.ndarray:
        """Aplica Z-score a targets para entrenamiento."""
        return ((data - mean_y) / std_y).astype(np.float32)
    
    def unscale_y(data: np.ndarray) -> np.ndarray:
        """Deshace normalización: vuelve a escala original para métricas."""
        return (data * std_y + mean_y).astype(np.float32)
    
    # Aplicar normalización a features (usa media/std de train)
    X_train_s = scale_X(X_train)  # Features normalizados para train
    X_val_s = scale_X(X_val)      # Aplica MISMA transformación a val
    X_test_s = scale_X(X_test)    # Aplica MISMA transformación a test
    
    # Aplicar normalización a targets (usa media/std de train)
    y_train_s = scale_y(y_train)  # Targets normalizados para train
    y_val_s = scale_y(y_val)      # Targets normalizados para val
    
    # CREAR Y ENTRENAR MODELO GRU
    print(f"  Entrenando GRU {gru_units}...")
    # Inicializar modelo GRU (Gated Recurrent Unit)
    model = GRURegressor(
        input_size=X_train.shape[-1],      # Número de features por timestep
        hidden_sizes=list(gru_units),     # Unidades por capa GRU
        dropout=dropout,                   # Regularización: desactivar unidades al azar
        dense_units=dense_units,           # Neuronas en capa densa final
        seed=seed,                         # Para reproducibilidad
    )
    
    # AJUSTAR MODELO
    train_started_at = datetime.now(timezone.utc).isoformat()
    train_start = time.perf_counter()
    res = model.fit(
        X_train_s,                         # Features de train (normalizados)
        y_train_s,                         # Targets de train (normalizados)
        X_val_s,                           # Features de val (para early stopping)
        y_val_s,                           # Targets de val
        learning_rate=lr,                  # Velocidad de aprendizaje
        batch_size=batch_size,             # Muestras por actualización
        max_epochs=max_epochs,             # Máximo número de épocas
        early_stopping_patience=patience,  # Parar si no mejora en N épocas
        loss=loss,                         # Función de pérdida (huber o mse)
        huber_delta=huber_delta,           # Parámetro delta para Huber
    )
    train_end = time.perf_counter()
    train_ended_at = datetime.now(timezone.utc).isoformat()
    log_timing_event(
        strategy="e1_simple",
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
    
    # REALIZAR PREDICCIONES EN TEST
    # Predicciones en escala normalizada
    pred_started_at = datetime.now(timezone.utc).isoformat()
    pred_start = time.perf_counter()
    y_pred_s = model.predict(X_test_s)  # Output normalizado del modelo
    pred_end = time.perf_counter()
    pred_ended_at = datetime.now(timezone.utc).isoformat()
    log_timing_event(
        strategy="e1_simple",
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
    # Desnormalizar predicciones: volver a escala original
    y_pred = unscale_y(y_pred_s)        # Predicciones en escala real
    
    # CALCULAR MÉTRICAS DE MACHINE LEARNING
    # MAE: Mean Absolute Error (error absoluto promedio)
    mae = float(np.mean(np.abs(y_test - y_pred)))
    # RMSE: Root Mean Squared Error (raíz del error cuadrático medio)
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    # Directional Accuracy: porcentaje de predicciones con dirección correcta
    # np.sign: obtiene signo (+1, 0, -1) de cada valor
    dir_acc = float(np.mean(np.sign(y_test) == np.sign(y_pred)))
    # IC: Information Coefficient (correlación de Spearman)
    ic = compute_information_coefficient(y_test, y_pred)
    
    # Sanitizar IC: si es NaN, usar 0 (no hay correlación)
    if not np.isfinite(ic):
        ic = 0.0
    
    # CONFIGURACIÓN DE BACKTEST
    # Extrae parámetros de umbral de señales
    thresholds = e1_simple.get("thresholds", {})
    # tau_buy: umbral para generar señal de compra (si pred_return > tau_buy)
    tau_buy = float(thresholds.get("tau_buy", 0.06))
    # tau_sell: umbral para generar señal de venta (si pred_return < tau_sell)
    tau_sell = float(thresholds.get("tau_sell", 0.00))
    
    # Costos de transacción
    costs_cfg = config.get("costs", {})
    # round_trip_bps: costo ida y vuelta en basis points (0.1% = 10 bps)
    round_trip_bps = float(costs_cfg.get("daily_round_trip_bps", 10))
    
    # Configuración específica del backtest
    bt_cfg = e1_simple.get("backtest", {})
    # Período de holding: cuántos días mantener la posición
    holding_period = int(bt_cfg.get("holding_period_days", horizon_days))
    # Permitir posiciones cortas (venta en corto)
    allow_short = bool(bt_cfg.get("allow_short", False))
    # Máxima posición permitida (1.0 = 100%, 0.5 = 50% del capital)
    max_position = float(bt_cfg.get("max_position", 1.0))
    
    # Extraer precios de cierre para el período de test
    close_prices = ohlcv.loc[ts_test, "close"].to_numpy()  # Array de precios de cierre
    # EJECUTAR BACKTEST
    # Simula trading basado en señales predichas
    bt = backtest_daily_signals(
        timestamps=ts_test,                 # Fechas del período de test
        close_prices=close_prices,          # Precios de cierre reales
        pred_returns=y_pred,                # Predicciones del modelo
        tau_buy=tau_buy,                    # Umbral de compra
        tau_sell=tau_sell,                  # Umbral de venta
        round_trip_bps=round_trip_bps,      # Costos de transacción
        holding_period_days=holding_period, # Días para mantener posición
        allow_short=allow_short,            # Permitir posiciones cortas
        max_position=max_position,          # Tamaño máximo de posición
    )
    # Calcular métricas agregadas del backtest
    trading_metrics = summarize_backtest(bt)  # Retorna dict con Sharpe, CAGR, Max DD, etc.
    
    # Extraer Sharpe ratio (métrica de riesgo-retorno)
    # Si no está disponible, usar NaN
    sharpe = float(trading_metrics.get("sharpe", float("nan")))
    
    # IMPRIMIR RESULTADOS DE ENTRENAMIENTO
    # Convertir NaN a string para impresión limpia
    ic_str = "nan" if np.isnan(ic) else f"{ic:.3f}"
    sharpe_str = "nan" if np.isnan(sharpe) else f"{sharpe:.2f}"
    
    # Resumen de entrenamiento
    print(f"  ✓ Epochs: {res.epochs_ran}/{max_epochs} | Val Loss: {res.best_val_loss:.6f}")
    # Métricas ML
    print(f"  ✓ MAE={mae:.4f} RMSE={rmse:.4f} IC={ic_str} Dir={dir_acc:.1%} Sharpe={sharpe_str}")
    
    
    # GUARDAR OUTPUTS
    # 1. Guardar predicciones vs valores reales
    pred_df = pd.DataFrame(
        {'y_true': y_test, 'y_pred': y_pred},  # Valores reales vs predichos
        index=ts_test  # Índice: timestamps
    )
    pred_df.index.name = 'timestamp'
    pred_df.to_csv(out_dir / f"{ticker}_predictions.csv")  # CSV con predicciones
    
    # 2. Guardar resultado del backtest (trades, equity, etc.)
    bt.to_csv(out_dir / f"{ticker}_backtest.csv")
    
    # 3. Guardar scaler (necesario para inferencia futura)
    # Contiene media y std de cada feature para normalizar datos nuevos
    scaler_df = pd.DataFrame({
        'feature': list(feat_names),  # Nombres de features
        'mean': mean_X,               # Media de cada feature
        'std': std_X,                 # Std de cada feature
    })
    scaler_df.to_csv(out_dir / f"{ticker}_scaler.csv", index=False)
    
    # GUARDAR MODELO ENTRENADO
    # Prepara payload con todo lo necesario para inferencia futura
    model_payload = {
        "ticker": ticker,                          # Ticker del activo
        "strategy": "e1_simple",                   # Estrategia usada
        "created_at": datetime.now(timezone.utc).isoformat(),  # Timestamp de creación
        "model_class": "GRURegressor",             # Clase del modelo
        "model_kwargs": {                          # Parámetros de arquitectura
            "input_size": int(X_train.shape[-1]),   # Número de features
            "hidden_sizes": list(gru_units),        # Unidades GRU
            "dropout": float(dropout),              # Dropout
            "dense_units": int(dense_units),        # Neuronas dense
            "seed": int(seed),                      # Seed
        },
        "lookback_days": int(lookback_days),       # Histórico usado
        "horizon_days": int(horizon_days),         # Forward usado
        "feature_names": list(feat_names),         # Nombres ordenados de features
        "scaler_X": {"mean": mean_X.tolist(), "std": std_X.tolist()},  # Scaler features
        "scaler_y": {"mean": float(mean_y), "std": float(std_y)},      # Scaler target
        "train_result": {                          # Resultado del entrenamiento
            "epochs_ran": int(res.epochs_ran),      # Épocas ejecutadas
            "best_val_loss": float(res.best_val_loss)  # Mejor val loss
        },
        "state_dict": {k: v.detach().cpu() for k, v in model.model.state_dict().items()},  # Pesos
    }
    
    # Ruta donde guardar el modelo
    model_path = out_dir / f"{ticker}_model.pth"
    try:
        # Guardar usando torch.save
        model.torch.save(model_payload, model_path)
        print(f"  ✓ Modelo guardado: {model_path.name}")
    except Exception as exc:
        # Si falla, avisar pero continuar
        print(f"  ⚠️  No se pudo guardar modelo: {exc}")
    
    # CREAR SUMMARY CON TODAS LAS MÉTRICAS
    root = project_root()
    def as_relative(path: Path) -> str:
        """Convierte path absoluto a relativo respecto a project root."""
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)
    
    # Diccionario con todas las métricas y metadatos
    summary = {
        "ticker": ticker,
        "n_samples": int(len(X)),
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "lookback_days": int(lookback_days),
        "horizon_days": int(horizon_days),
        "split_method": "time_split",
        "epochs_ran": int(res.epochs_ran),
        "val_loss": float(res.best_val_loss),
        "ml_mae": mae,
        "ml_rmse": rmse,
        "ml_directional_accuracy": dir_acc,
        "ml_ic": ic,
        "tau_buy": float(tau_buy),
        "tau_sell": float(tau_sell),
        **{f"bt_{k}": float(v) for k, v in trading_metrics.items()},
        "timing_train_seconds": round(train_end - train_start, 2),
        "timing_predict_seconds": round(pred_end - pred_start, 4),
        "predictions_file": as_relative(out_dir / f"{ticker}_predictions.csv"),
        "backtest_file": as_relative(out_dir / f"{ticker}_backtest.csv"),
        "scaler_file": as_relative(out_dir / f"{ticker}_scaler.csv"),
        "model_file": as_relative(model_path),
    }
    
    # Convertir diccionario a DataFrame (1 fila)
    summary_df = pd.DataFrame([summary])
    # Guardar a CSV para fácil lectura
    summary_df.to_csv(out_dir / f"{ticker}_summary.csv", index=False)
    
    # Retornar summary para consolidación agregada
    return summary


def main():
    """Ejecuta el pipeline E1 simple para uno o más tickers."""
    parser = argparse.ArgumentParser(description="Pipeline E1 Simple (sin walk-forward)")
    parser.add_argument(
        "--tickers",
        type=str,
        help="Tickers separados por coma (ej: AAPL,MSFT). Si se omite, usa los del config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/base.yaml",
        help="Ruta al archivo de configuración"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Omite la descarga de datos (usa datos existentes)"
    )
    parser.add_argument(
        "--skip-cleaning",
        action="store_true",
        help="Omite la limpieza de datos (usa datos raw)"
    )
    args = parser.parse_args()
    
    root = project_root()
    config_path = root / args.config
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config no encontrado: {config_path}")
    
    config = load_yaml(config_path)
    
    # Determinar tickers
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
    else:
        tickers = config.get("universe", {}).get("tickers_by_strategy", {}).get("e1_simple", [])
        if not tickers:
            # Fallback a e1_conservative si no hay e1_simple definido
            tickers = config.get("universe", {}).get("tickers_by_strategy", {}).get("e1_conservative", [])
    
    if not tickers:
        raise ValueError("No se especificaron tickers ni en args ni en config")
    
    # Directorios
    raw_dir = root / "data" / "raw" / "daily"
    clean_dir = root / "data" / "clean"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "runs" / "e1_simple" / timestamp
    ensure_dir(out_dir)
    
    # Guardar config usado
    config_used_path = out_dir / "config_used.yaml"
    import yaml
    with open(config_used_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n{'='*60}")
    print(f"Pipeline E1 Simple - {len(tickers)} tickers")
    print(f"Output: {out_dir.relative_to(root)}")
    print(f"{'='*60}\n")

    # MLflow setup: intenta servidor remoto; fallback local robusto
    mlflow_enabled = False
    mlflow = None
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    remote_check_timeout_seconds = float(os.getenv("MLFLOW_REMOTE_CHECK_TIMEOUT_SECONDS", "1.5"))
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "E1_Simple")
    local_sqlite_dir = root / "runs" / "mlflow_local"
    ensure_dir(local_sqlite_dir)
    local_sqlite_db = local_sqlite_dir / "mlflow.db"
    local_artifacts_dir = local_sqlite_dir / "artifacts"
    ensure_dir(local_artifacts_dir)
    local_sqlite_uri = f"sqlite:///{local_sqlite_db}"

    fallback_local_tracking_uri = os.getenv("MLFLOW_LOCAL_TRACKING_URI", "").strip()
    if fallback_local_tracking_uri:
        local_sqlite_uri = fallback_local_tracking_uri

    local_mlruns = str(root / "mlruns")
    isolated_mlruns = str(root / "runs" / "e1_simple" / "mlflow_store")
    ensure_dir(Path(isolated_mlruns))

    base_mlruns_path = Path(local_mlruns)
    has_malformed_mlruns = False
    if base_mlruns_path.exists():
        for exp_dir in base_mlruns_path.iterdir():
            if not exp_dir.is_dir() or not exp_dir.name.isdigit():
                continue
            if not (exp_dir / "meta.yaml").exists():
                has_malformed_mlruns = True
                break

    def _activate_mlflow(
        _mlflow_module,
        uri: str,
        label: str,
        experiment_artifact_dir: Path | None = None,
    ) -> tuple[bool, str | None]:
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
        parsed = urlparse(uri)
        if parsed.scheme not in {"http", "https"}:
            return True, None

        host = parsed.hostname
        if not host:
            return True, None

        if parsed.port is not None:
            port = parsed.port
        elif parsed.scheme == "https":
            port = 443
        else:
            port = 80

        try:
            with socket.create_connection((host, port), timeout=timeout_seconds):
                return True, None
        except OSError as exc:
            return False, str(exc)

    local_uris = [
        {
            "uri": local_sqlite_uri,
            "store_path": str(local_sqlite_db),
            "artifact_dir": local_artifacts_dir,
        },
        {
            "uri": f"file://{isolated_mlruns}",
            "store_path": isolated_mlruns,
            "artifact_dir": None,
        },
        {
            "uri": f"file://{local_mlruns}",
            "store_path": local_mlruns,
            "artifact_dir": None,
        },
    ] if has_malformed_mlruns else [
        {
            "uri": local_sqlite_uri,
            "store_path": str(local_sqlite_db),
            "artifact_dir": local_artifacts_dir,
        },
        {
            "uri": f"file://{local_mlruns}",
            "store_path": local_mlruns,
            "artifact_dir": None,
        },
        {
            "uri": f"file://{isolated_mlruns}",
            "store_path": isolated_mlruns,
            "artifact_dir": None,
        },
    ]
    try:
        import mlflow as _mlflow  # type: ignore

        if tracking_uri:
            remote_reachable, remote_reachability_err = _is_tracking_uri_reachable(
                tracking_uri,
                remote_check_timeout_seconds,
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
                    "⚠️  MLflow remoto no accesible "
                    f"({tracking_uri}, timeout={remote_check_timeout_seconds}s): {remote_reachability_err}"
                )

        if not mlflow_enabled:
            for local_cfg in local_uris:
                uri = local_cfg["uri"]
                store_path = local_cfg["store_path"]
                artifact_dir = local_cfg["artifact_dir"]
                ok_local, local_err = _activate_mlflow(
                    _mlflow,
                    uri,
                    "local",
                    experiment_artifact_dir=artifact_dir,
                )
                if ok_local:
                    mlflow = _mlflow
                    mlflow_enabled = True
                    mode = "local fallback" if tracking_uri else "tracking local"
                    print(f"✓ MLflow habilitado ({mode}, experiment={experiment_name})")
                    print(f"  → Store URI: {store_path}")
                    print(f"  → Para visualizar: mlflow ui --backend-store-uri {store_path}")
                    break
                print(f"⚠️  Falló MLflow local en {store_path}: {local_err}")
    except Exception as exc:
        print(f"⚠️  MLflow no disponible, continuando sin tracking: {exc}")
        mlflow_enabled = False
    
    # Paso 1: Descargar datos
    if not args.skip_download:
        print("Paso 1/3: Descargando datos...")
        print("-" * 60)
        from ..data.download_daily import download_daily_ohlcv
        
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
        from ..data.clean_daily import process_daily_data_with_cleaning
        
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
        print("Paso 2/3: Limpieza omitida (usando datos raw)\n")
    
    # PASO 3: ENTRENAR MODELOS
    print("Paso 3/3: Entrenando modelos...")
    print("-" * 60)
    
    # ENTRENAR CADA TICKER
    summaries = []  # Acumular summaries para consolidar
    for i, ticker in enumerate(tickers, 1):  # Enumerar desde 1
        print(f"[{i}/{len(tickers)}] {ticker}")  # Mostrar progreso
        try:
            # Ejecutar pipeline E1 simple para este ticker
            if mlflow_enabled and mlflow is not None:
                timestamp = out_dir.name
                e1_simple_cfg = config.get("strategies", {}).get("e1_simple", {})
                model_cfg = e1_simple_cfg.get("model", {})
                thresholds = e1_simple_cfg.get("thresholds", {})
                costs_cfg = config.get("costs", {})
                bt_cfg = e1_simple_cfg.get("backtest", {})
                seed = int(config.get("project", {}).get("seed", 42))

                with mlflow.start_run(run_name=f"E1Simple_{ticker}_{timestamp}"):
                    mlflow.log_param("strategy", "e1_simple")
                    mlflow.log_param("ticker", ticker)
                    mlflow.log_param("timestamp", timestamp)

                    # Dashboard tags for filtering in MLflow UI
                    log_dashboard_tags(
                        strategy="e1_simple",
                        ticker=ticker,
                        run_type="train",
                    )

                    mlflow.log_params(
                        {
                            "lookback_days": int(e1_simple_cfg.get("lookback_days", 360)),
                            "horizon_days": int(e1_simple_cfg.get("horizon_days", 90)),
                            "gru_units": str(model_cfg.get("gru_units", [64])),
                            "dropout": float(model_cfg.get("dropout", 0.2)),
                            "dense_units": int(model_cfg.get("dense_units", 16)),
                            "learning_rate": float(model_cfg.get("learning_rate", 0.001)),
                            "batch_size": int(model_cfg.get("batch_size", 64)),
                            "max_epochs": int(model_cfg.get("max_epochs", 100)),
                            "early_stopping_patience": int(model_cfg.get("early_stopping_patience", 10)),
                            "loss": str(model_cfg.get("loss", "huber")),
                            "huber_delta": float(model_cfg.get("huber_delta", 1.0)),
                            "tau_buy": float(thresholds.get("tau_buy", 0.06)),
                            "tau_sell": float(thresholds.get("tau_sell", 0.00)),
                            "round_trip_bps": float(costs_cfg.get("daily_round_trip_bps", 10)),
                            "holding_period_days": int(bt_cfg.get("holding_period_days", e1_simple_cfg.get("horizon_days", 90))),
                            "allow_short": bool(bt_cfg.get("allow_short", False)),
                            "max_position": float(bt_cfg.get("max_position", 1.0)),
                            "seed": seed,
                        }
                    )

                    summary = run_e1_simple_for_ticker(
                        config=config,
                        ticker=ticker,
                        raw_dir=raw_dir,
                        out_dir=out_dir,
                    )

                    metrics: dict[str, float] = {}
                    for k, v in summary.items():
                        if not isinstance(v, (int, float)):
                            continue
                        if k.startswith("ml_") or k.startswith("bt_"):
                            metrics[k] = float(v)
                    if metrics:
                        mlflow.log_metrics(metrics)

                    # Dashboard: log timing + alert colors
                    train_t = summary.get("timing_train_seconds")
                    pred_t = summary.get("timing_predict_seconds")
                    log_dashboard_timing(
                        train_seconds=train_t,
                        predict_seconds=pred_t,
                    )
                    log_metric_alerts("e1_simple", metrics)

                    try:
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
                    except Exception as art_exc:
                        print(f"  ⚠️  MLflow artifacts no guardados: {art_exc}")
            else:
                summary = run_e1_simple_for_ticker(
                    config=config,
                    ticker=ticker,
                    raw_dir=raw_dir,
                    out_dir=out_dir,
                )
            summaries.append(summary)  # Guardar summary
        except Exception as exc:
            # Si falla, loguear error pero continuar con siguiente ticker
            print(f"  ❌ Error: {exc}\n")
            continue
        print()  # Línea en blanco para separación
    
    # CONSOLIDAR RESULTADOS
    if summaries:
        # Convertir lista de dicts a DataFrame (1 fila por ticker)
        summary_all = pd.DataFrame(summaries)
        # Guardar consolidado
        summary_all.to_csv(out_dir / "summary_all.csv", index=False)
        
        print(f"\n{'='*60}")
        print(f"✓ Completado: {len(summaries)}/{len(tickers)} tickers")
        print(f"  Resultados en: {out_dir.relative_to(root)}/")
        print(f"{'='*60}\n")
    else:
        # Si ningún ticker se completó exitosamente
        print("\n❌ No se completó ningún ticker exitosamente\n")


if __name__ == "__main__":
    # Ejecutar la función main si se llama directamente (no como módulo)
    # python -m src.train_e1_simple_pipeline [--options]
    main()
