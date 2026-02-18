"""
Pipeline completo E2 (Estrategia Moderada).

Flujo:
1. Cargar datos raw
2. Calcular features E2 (momentum-focused)
3. Crear target y secuencias
4. Split temporal
5. Estandarizar features
6. Entrenar LSTM
7. Evaluar y guardar resultados

Diferencias vs E1:
- Features: build_features_e2 (RSI, Stochastic, ROC, OBV)
- Modelo: LSTM en lugar de GRU
- Lookback: 60 días (vs 360 de E1)
- Horizon: 20 días (vs 90 de E1)
- Early stopping patience: 12 (vs 15 de E1)
- Max epochs: 150 (vs 200 de E1)
"""

from __future__ import annotations  # Permite forward references en type hints

import argparse  # Parsing de argumentos CLI
from datetime import datetime, timezone  # Timestamp de ejecución
import os  # Variables de entorno y paths
from pathlib import Path  # Manejo de rutas
from functools import lru_cache  # Cache para YAMLs tuning
import copy  # Copias defensivas
import time

import numpy as np  # Cálculo numérico
import pandas as pd  # DataFrames
from sklearn.model_selection import TimeSeriesSplit  # Split temporal sin leakage

from .build_features import compute_e2_features, make_target_e2  # Features y target E2
from ..features.build_sequences_e1e2 import make_sequences, time_split, temporal_train_val_split  # Secuencias y splits
from .lstm import LSTMRegressor  # Modelo LSTM
from ..backtest.backtest_daily import backtest_daily_signals, summarize_backtest  # Backtesting diario
from ..utils import ensure_dir, load_yaml, project_root, log_timing_event  # Utilidades comunes


@lru_cache(maxsize=8)  # Cachear lectura de YAMLs
def _load_tuned_params(path_str: str) -> dict:  # Cargar parámetros tuneados
    path = Path(path_str)  # Convertir string a Path
    if not path.exists():  # Validar existencia
        return {}  # Si no existe archivo, no hay overrides
    data = load_yaml(path)  # Cargar YAML
    if not isinstance(data, dict):  # Validar tipo
        return {}  # Validar tipo
    return data  # Retornar dict con parámetros


def _apply_tuned_overrides(*, config: dict, strategy_key: str, ticker: str) -> dict:  # Overrides tuning
    """Aplica hiperparámetros optimizados para un ticker específico.
    
    Lee el archivo YAML de parámetros tuneados (generado por optimize_e2_hyperparameters.py)
    y aplica los overrides para thresholds, model y filters.
    
    Variable de entorno:
        E2_TUNED_PARAMS_PATH o TUNED_PARAMS_PATH: Path al YAML con parámetros por ticker
    
    Formato del YAML:
        TICKER:
          tau_buy: 0.025
          tau_sell: 0.005
          lstm_units_1: 128
          lstm_units_2: 64
          dropout: 0.25
          learning_rate: 0.0008
          batch_size: 64
          rsi14_min: 30
          rsi14_max: 70
    """
    tuned_path = (  # Path tuning
        os.getenv("E2_TUNED_PARAMS_PATH", "").strip()  # Path específico E2
        or os.getenv("TUNED_PARAMS_PATH", "").strip()  # Path genérico
    )  # Fin path tuning
    if not tuned_path:  # Si no hay path
        return config  # Si no hay path, no aplicar overrides

    all_tuned = _load_tuned_params(tuned_path)  # Cargar YAML
    per_ticker = all_tuned.get(ticker)  # Overrides por ticker
    if not isinstance(per_ticker, dict):  # Validar dict ticker
        return config  # Si no hay dict, no aplicar

    strat = config.setdefault("strategies", {}).setdefault(strategy_key, {})  # Sección estrategia

    # Aplicar thresholds (tau_buy, tau_sell)
    if "tau_buy" in per_ticker or "tau_sell" in per_ticker:  # Overrides thresholds
        thresholds = strat.setdefault("thresholds", {})  # Sección thresholds
        if "tau_buy" in per_ticker:  # Override tau_buy
            thresholds["tau_buy"] = per_ticker["tau_buy"]  # Override tau_buy
        if "tau_sell" in per_ticker:  # Override tau_sell
            thresholds["tau_sell"] = per_ticker["tau_sell"]  # Override tau_sell

    # Aplicar parámetros de modelo (lstm_units, dropout, learning_rate, batch_size)
    model = strat.setdefault("model", {})  # Sección modelo
    
    if "lstm_units_1" in per_ticker and "lstm_units_2" in per_ticker:  # Units LSTM
        model["lstm_units"] = [per_ticker["lstm_units_1"], per_ticker["lstm_units_2"]]  # Units LSTM
    
    if "dropout" in per_ticker:  # Override dropout
        model["dropout"] = per_ticker["dropout"]  # Override dropout
    
    if "learning_rate" in per_ticker:  # Override LR
        model["learning_rate"] = per_ticker["learning_rate"]  # Override LR
    
    if "batch_size" in per_ticker:  # Override batch size
        model["batch_size"] = per_ticker["batch_size"]  # Override batch size

    # Aplicar filtros (rsi14_min, rsi14_max)
    if "rsi14_min" in per_ticker or "rsi14_max" in per_ticker:  # Overrides filtros RSI
        filters = strat.setdefault("filters", {})  # Sección filtros
        if "rsi14_min" in per_ticker:  # Override RSI min
            filters["rsi14_min"] = per_ticker["rsi14_min"]  # Override RSI min
        if "rsi14_max" in per_ticker:  # Override RSI max
            filters["rsi14_max"] = per_ticker["rsi14_max"]  # Override RSI max
    
    # Aplicar parámetros walk-forward (n_folds, internal_val_fraction)
    if "n_folds" in per_ticker or "internal_val_fraction" in per_ticker:  # Overrides WF
        splits = config.setdefault("splits", {})  # Sección splits
        if "n_folds" in per_ticker:  # Override folds
            splits["folds"] = int(per_ticker["n_folds"])  # Override folds
        if "internal_val_fraction" in per_ticker:  # Override val fraction
            splits["internal_val_fraction"] = float(per_ticker["internal_val_fraction"])  # Override val fraction

    return config  # Retornar config actualizado


def load_ohlcv_csv(path: Path) -> pd.DataFrame:  # Cargar CSV OHLCV
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
        raise ValueError(f"Missing 'timestamp' column in {path}")  # Validar timestamp

    # Convertir timestamp a datetime con timezone UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601', utc=True)  # Parse timestamp
    
    # Ordenar por timestamp (cronológicamente ascendente)
    df = df.sort_values("timestamp")  # Ordenar cronológicamente
    
    # Usar timestamp como índice (más eficiente para acceso temporal)
    df = df.set_index("timestamp")  # Indexar por timestamp

    # Validar que todas las columnas OHLCV estén presentes
    required = {"open", "high", "low", "close", "volume"}  # Columnas requeridas
    missing = required.difference(df.columns)  # Detectar faltantes
    if missing:  # Si faltan columnas
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")  # Error si faltan

    return df  # Retornar OHLCV limpio


def compute_information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # IC Pearson
    """Calcula el Information Coefficient (correlación) de forma robusta.
    
    El IC mide qué tan bien las predicciones están correlacionadas con valores reales:
    - IC > 0.05 se considera significativo en finanzas
    - IC < 0 indica overfitting o falta de capacidad predictiva
    - IC ≈ 0 indica predicciones aleatorias
    """
    # Si hay menos de 2 muestras, no se puede calcular correlación
    if len(y_true) <= 1:  # Validar tamaño mínimo
        return float("nan")  # No hay suficientes muestras

    # Si alguno de los valores es constante (sin variación), la correlación es indefinida
    if np.std(y_true) == 0 or np.std(y_pred) == 0:  # Variación nula
        return float("nan")  # Correlación indefinida si no hay variación

    # Calcular correlación de Pearson
    return float(np.corrcoef(y_true, y_pred)[0, 1])  # Pearson


def run_e2_walk_forward(  # Walk-forward E2
    *,  # Solo kwargs
    X: np.ndarray,  # Matriz de features
    y: np.ndarray,  # Targets
    ts: pd.DatetimeIndex,  # Timestamps
    feat_names: list[str] | pd.Index,  # Nombres de features
    ticker: str,  # Ticker
    out_dir: Path,  # Directorio salida
    ohlcv: pd.DataFrame,  # OHLCV completo
    splits_cfg: dict,  # Config splits
    lstm_units: list[int],  # Arquitectura LSTM
    dropout: float,  # Dropout
    dense_units: int,  # Unidades dense
    learning_rate: float,  # Learning rate
    batch_size: int,  # Batch size
    max_epochs: int,  # Max epochs
    patience: int,  # Patience
    loss: str,  # Tipo de loss
    huber_delta: float,  # Delta Huber
    tau_buy: float,  # Umbral compra
    tau_sell: float,  # Umbral venta
    round_trip_bps: float,  # Costos en bps
    holding_period: int,  # Holding period
    allow_short: bool,  # Permitir cortos
    max_position: float,  # Máx exposición
    seed: int,  # Seed
    lookback_days: int,  # Lookback
    horizon_days: int,  # Horizon
    rsi14_min: float | None = None,  # Filtro RSI min
    rsi14_max: float | None = None,  # Filtro RSI max
) -> dict:  # Retorna resumen
    """Ejecuta validación walk-forward para la estrategia E2 (LSTM)."""  # Docstring

    require_model_save = os.getenv("REQUIRE_MODEL_SAVE", "1").strip().lower() not in {  # Flag guardado
        "0",  # Deshabilitar
        "false",  # Deshabilitar
        "no",  # Deshabilitar
    }  # Fin set REQUIRE_MODEL_SAVE

    ensure_dir(out_dir)  # Crear directorio salida

    n_samples = len(X)  # Total de muestras
    if n_samples == 0:  # Validar muestras
        raise ValueError("No hay muestras disponibles para walk-forward")  # Error si vacío

    folds = int(splits_cfg.get("folds", 5))  # Número de folds
    if folds < 1:  # Validar folds
        raise ValueError("'folds' debe ser >= 1 para walk-forward")  # Validar folds

    embargo_cfg = splits_cfg.get("embargo_days", {})  # Config embargo
    if isinstance(embargo_cfg, dict):  # Embargo por estrategia
        embargo_days = int(embargo_cfg.get("e2", 0))  # Embargo específico E2
    else:  # Embargo genérico
        embargo_days = int(embargo_cfg or 0)  # Embargo genérico

    gap_samples = max(0, int(embargo_days))  # Gap en muestras

    default_test_size = max(1, n_samples // (folds + 1))  # Test size default
    test_size = int(splits_cfg.get("test_size", default_test_size))  # Test size
    if test_size <= gap_samples:  # Validar tamaño test
        test_size = gap_samples + 1  # Ajuste para respetar gap

    splitter = TimeSeriesSplit(n_splits=folds, test_size=test_size, gap=gap_samples)  # Splitter

    fold_summaries: list[dict] = []  # Resúmenes por fold
    pred_frames: list[pd.DataFrame] = []  # Predicciones por fold

    last_model_payload: dict | None = None  # Payload último fold
    last_torch = None  # Referencia a torch

    root = project_root()  # Root del proyecto

    def as_relative(path: Path) -> str:  # Path relativo helper
        try:  # Intentar relativizar
            return str(path.relative_to(root))  # Relativo al root
        except ValueError:  # Si no es relativo
            return str(path)  # Fallback a absoluto

    raw_val_fraction = splits_cfg.get(  # Fracción val raw
        "internal_val_fraction", splits_cfg.get("val_fraction", 0.15)  # Fallback legacy
    )  # Fin get
    try:  # Convertir a float
        val_fraction_cfg = float(raw_val_fraction)  # Normalizar a float
    except (TypeError, ValueError):  # Error conversión
        val_fraction_cfg = 0.15  # Fallback

    if not 0 < val_fraction_cfg < 1:  # Validar rango
        val_fraction_cfg = 0.15  # Enforce rango válido

    for fold_idx, (train_full_idx, test_idx) in enumerate(  # Iterar folds
        splitter.split(np.arange(n_samples)), start=1  # Iterar splits
    ):  # Fin enumerate
        if len(test_idx) == 0 or len(train_full_idx) == 0:  # Folds vacíos
            continue  # Saltar folds vacíos

        # Split interno train/val dentro del bloque de entrenamiento
        try:  # Split train/val
            train_idx, val_idx = temporal_train_val_split(  # Split train/val
                train_full_idx,  # Índices completos de train
                val_fraction=val_fraction_cfg,  # Fracción val
            )  # Fin split
        except ValueError:  # Fallback split
            split_point = max(1, int(len(train_full_idx) * 0.8))  # Fallback 80/20
            train_idx = train_full_idx[:split_point]  # Índices train
            val_idx = train_full_idx[split_point:]  # Índices val

        X_train = X[train_idx]  # X train
        X_val = X[val_idx]  # X val
        X_test = X[test_idx]  # X test

        # Normalizar X por features (eje 0=samples, 1=timesteps, 2=features)
        mean_X = X_train.mean(axis=(0, 1))  # Media por feature
        std_X = X_train.std(axis=(0, 1)) + 1e-8  # Std por feature

        def scale_X(data: np.ndarray) -> np.ndarray:  # Escalado Z-score
            return ((data - mean_X) / std_X).astype(np.float32)  # Z-score X

        y_train = y[train_idx]  # y train
        y_val = y[val_idx]  # y val
        y_test = y[test_idx]  # y test

        mean_y = float(y_train.mean())  # Media y
        std_y = float(y_train.std()) + 1e-8  # Std y

        def scale_y(data: np.ndarray) -> np.ndarray:  # Escalado y
            return ((data - mean_y) / std_y).astype(np.float32)  # Z-score y

        def unscale_y(data: np.ndarray) -> np.ndarray:  # Des-escalado y
            return (data * std_y + mean_y).astype(np.float32)  # Volver a escala original

        X_train_s = scale_X(X_train)  # X train escalado
        X_val_s = scale_X(X_val)  # X val escalado
        X_test_s = scale_X(X_test)  # X test escalado

        y_train_s = scale_y(y_train)  # y train escalado
        y_val_s = scale_y(y_val)  # y val escalado

        model = LSTMRegressor(  # Instanciar LSTM
            input_size=X_train.shape[-1],  # N features
            hidden_sizes=list(lstm_units),  # Arquitectura LSTM
            dropout=dropout,  # Dropout
            dense_units=dense_units,  # Dense units
            seed=seed,  # Seed
        )  # Fin init modelo

        train_started_at = datetime.now(timezone.utc).isoformat()
        train_start = time.perf_counter()
        res = model.fit(  # Entrenar
            X_train_s,  # X train
            y_train_s,  # y train
            X_val_s,  # X val
            y_val_s,  # y val
            learning_rate=learning_rate,  # LR
            batch_size=batch_size,  # Batch
            max_epochs=max_epochs,  # Epochs
            early_stopping_patience=patience,  # Patience
            loss=loss,  # Loss
            huber_delta=huber_delta,  # Delta
        )  # Fin fit
        train_end = time.perf_counter()
        train_ended_at = datetime.now(timezone.utc).isoformat()
        log_timing_event(
            strategy="e2_moderate",
            phase="train",
            duration_seconds=train_end - train_start,
            started_at=train_started_at,
            ended_at=train_ended_at,
            ticker=ticker,
            run_dir=out_dir,
            extra={
                "split": "walk_forward",
                "fold": int(fold_idx),
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
            },
        )

        pred_started_at = datetime.now(timezone.utc).isoformat()
        pred_start = time.perf_counter()
        y_pred_s = model.predict(X_test_s)  # Predicciones escaladas
        pred_end = time.perf_counter()
        pred_ended_at = datetime.now(timezone.utc).isoformat()
        log_timing_event(
            strategy="e2_moderate",
            phase="predict",
            duration_seconds=pred_end - pred_start,
            started_at=pred_started_at,
            ended_at=pred_ended_at,
            ticker=ticker,
            run_dir=out_dir,
            extra={
                "split": "walk_forward",
                "fold": int(fold_idx),
                "n_test": int(len(test_idx)),
            },
        )
        y_pred = unscale_y(y_pred_s)  # Predicciones en escala original

        last_torch = model.torch  # Referencia torch

        ts_test = ts[test_idx]  # Timestamps test
        window_label = f"{ts_test[0].date()} -> {ts_test[-1].date()}"  # Etiqueta ventana

        # Guardar payload del último fold
        last_model_payload = {  # Payload modelo
            "ticker": ticker,  # Ticker
            "strategy": "e2_moderate",  # Estrategia
            "created_at": datetime.now(timezone.utc).isoformat(),  # Timestamp
            "model_class": "LSTMRegressor",  # Clase modelo
            "model_kwargs": {  # Args modelo
                "input_size": int(X_train.shape[-1]),  # Input size
                "hidden_sizes": list(lstm_units),  # LSTM units
                "dropout": float(dropout),  # Dropout
                "dense_units": int(dense_units),  # Dense units
                "seed": int(seed),  # Seed
            },  # Fin model_kwargs
            "lookback_days": int(lookback_days),  # Lookback
            "horizon_days": int(horizon_days),  # Horizon
            "feature_names": list(feat_names),  # Feature names
            "scaler_X": {"mean": mean_X.tolist(), "std": std_X.tolist()},  # Scaler X
            "scaler_y": {"mean": float(mean_y), "std": float(std_y)},  # Scaler y
            "train_result": {"epochs_ran": int(res.epochs_ran), "best_val_loss": float(res.best_val_loss)},  # Resultados
            "walkforward": {  # Info walk-forward
                "fold": int(fold_idx),  # Fold
                "window": window_label,  # Ventana
                "test_size": int(test_size),  # Test size
                "gap_samples": int(gap_samples),  # Gap
            },  # Fin walkforward
            "state_dict": {k: v.detach().cpu() for k, v in model.model.state_dict().items()},  # Pesos
        }  # Fin payload

        mae = float(np.mean(np.abs(y_test - y_pred)))  # MAE
        rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))  # RMSE
        dir_acc = float(np.mean(np.sign(y_test) == np.sign(y_pred)))  # Accuracy direccional
        ic = compute_information_coefficient(y_test, y_pred)  # IC

        close_prices = ohlcv.loc[ts_test, "close"].to_numpy()  # Close prices
        bt = backtest_daily_signals(  # Backtest fold
            timestamps=ts_test,  # Timestamps
            close_prices=close_prices,  # Close prices
            pred_returns=y_pred,  # Predicciones
            tau_buy=tau_buy,  # Umbral compra
            tau_sell=tau_sell,  # Umbral venta
            round_trip_bps=round_trip_bps,  # Costos
            holding_period_days=holding_period,  # Holding
            allow_short=allow_short,  # Shorts
            max_position=max_position,  # Max posición
        )  # Fin backtest fold
        trading_metrics = summarize_backtest(bt)  # Métricas backtest

        sharpe = trading_metrics.get("sharpe", float("nan"))  # Sharpe

        ic_str = "nan" if np.isnan(ic) else f"{ic:.3f}"  # IC formateado
        sharpe_str = "nan" if np.isnan(sharpe) else f"{sharpe:.2f}"  # Sharpe formateado
        print(  # Log fold
            f"    Fold {fold_idx}: {window_label} | MAE={mae:.4f} IC={ic_str} Sharpe={sharpe_str}"  # Mensaje
        )  # Fin print

        bt.to_csv(out_dir / f"{ticker}_fold{fold_idx}_backtest.csv")  # Guardar backtest

        fold_summaries.append(  # Agregar resumen fold
            {  # Dict resumen
                "fold": fold_idx,  # Fold
                "window": window_label,  # Ventana
                "test_start": ts_test[0],  # Inicio test
                "test_end": ts_test[-1],  # Fin test
                "n_train": int(len(train_idx)),  # N train
                "n_val": int(len(val_idx)),  # N val
                "n_test": int(len(test_idx)),  # N test
                "epochs_ran": int(res.epochs_ran),  # Epochs
                "val_loss": float(res.best_val_loss),  # Val loss
                "ml_mae": mae,  # MAE
                "ml_rmse": rmse,  # RMSE
                "ml_directional_accuracy": dir_acc,  # Accuracy
                "ml_ic": ic,  # IC
                **{f"bt_{k}": float(v) for k, v in trading_metrics.items()},  # Métricas BT
            }  # Fin dict
        )  # Fin append

        preds_df = pd.DataFrame(  # DataFrame preds
            {  # Dict preds
                "fold": fold_idx,  # Fold
                "y_true": y_test,  # Real
                "y_pred": y_pred,  # Pred
            },  # Fin dict
            index=ts_test,  # Index ts
        )  # Fin DataFrame
        preds_df.index.name = "timestamp"  # Nombre index
        pred_frames.append(preds_df)  # Guardar preds

    if not fold_summaries:  # Validar folds
        raise ValueError("No se generaron folds válidos para walk-forward")  # Error

    fold_df = pd.DataFrame(fold_summaries)  # DataFrame folds
    fold_df.to_csv(out_dir / f"{ticker}_walkforward_folds.csv", index=False)  # Guardar folds

    combined_preds = pd.concat(pred_frames).sort_index()  # Preds combinadas
    combined_preds.to_csv(out_dir / f"{ticker}_walkforward_predictions.csv")  # Guardar preds

    y_true_all = combined_preds["y_true"].to_numpy()  # y true
    y_pred_all = combined_preds["y_pred"].to_numpy()  # y pred

    mae_all = float(np.mean(np.abs(y_true_all - y_pred_all)))  # MAE
    rmse_all = float(np.sqrt(np.mean((y_true_all - y_pred_all) ** 2)))  # RMSE
    dir_acc_all = float(np.mean(np.sign(y_true_all) == np.sign(y_pred_all)))  # Accuracy
    ic_all = compute_information_coefficient(y_true_all, y_pred_all)  # IC

    bt_all = backtest_daily_signals(  # Backtest agregado
        timestamps=pd.DatetimeIndex(combined_preds.index),  # Timestamps
        close_prices=ohlcv.loc[combined_preds.index, "close"].to_numpy(),  # Close
        pred_returns=y_pred_all,  # Pred returns
        tau_buy=tau_buy,  # Tau buy
        tau_sell=tau_sell,  # Tau sell
        round_trip_bps=round_trip_bps,  # Costos
        holding_period_days=holding_period,  # Holding
        allow_short=allow_short,  # Shorts
        max_position=max_position,  # Max posición
    )  # Fin backtest agregado
    trading_metrics_all = summarize_backtest(bt_all)  # Resumen backtest
    bt_all.to_csv(out_dir / f"{ticker}_walkforward_backtest.csv")  # Guardar BT

    # Guardar modelo del último fold
    if last_model_payload is not None and last_torch is not None:  # Guardar último modelo
        model_path = out_dir / f"{ticker}_model.pth"  # Path modelo
        try:  # Guardar modelo
            last_torch.save(last_model_payload, model_path)  # Guardar
        except Exception as exc:  # Error guardado
            print(f"⚠️  No se pudo guardar el modelo walk-forward para {ticker}: {exc}")  # Warning

        if require_model_save and not model_path.exists():  # Validar guardado
            raise RuntimeError(  # Error
                f"El pipeline terminó pero NO se guardó el modelo walk-forward en {model_path}. "  # Msg 1
                "Si querés permitir continuar sin guardar, setear REQUIRE_MODEL_SAVE=0."  # Msg 2
            )  # Fin raise

    summary = {  # Resumen final
        "ticker": ticker,  # Ticker
        "n_samples": int(n_samples),  # Muestras
        "n_test": int(len(combined_preds)),  # N test
        "folds": int(len(fold_df)),  # Folds
        "lookback_days": int(lookback_days),  # Lookback
        "horizon_days": int(horizon_days),  # Horizon
        "split_method": "walk_forward",  # Split method
        "walkforward_test_size": int(test_size),  # Test size
        "walkforward_gap": int(gap_samples),  # Gap
        "ml_mae": mae_all,  # MAE
        "ml_rmse": rmse_all,  # RMSE
        "ml_directional_accuracy": dir_acc_all,  # Accuracy
        "ml_ic": ic_all,  # IC
        **{f"bt_{k}": float(v) for k, v in trading_metrics_all.items()},  # Métricas BT
        "folds_file": as_relative(out_dir / f"{ticker}_walkforward_folds.csv"),  # Folds file
        "predictions_file": as_relative(out_dir / f"{ticker}_walkforward_predictions.csv"),  # Preds file
        "backtest_file": as_relative(out_dir / f"{ticker}_walkforward_backtest.csv"),  # Backtest file
        # Target columns for reference (E2 defaults)
        "ic_target_min": 0.05,  # IC target
        "sharpe_target_min": 1.0,  # Sharpe target
        "mae_target_max": 0.03,  # MAE target
        "rmse_target_max": 0.05,  # RMSE target
    }  # Fin resumen

    return summary  # Retornar resumen


def run_e2_for_ticker(  # Ejecutar E2 por ticker
    config: dict,  # Configuración global
    ticker: str,  # Ticker
    raw_dir: Path,  # Directorio raw
    out_dir: Path,  # Directorio salida
    benchmark_df: pd.DataFrame | None = None,  # Benchmark opcional
) -> dict:  # Retorna resumen
    """Ejecuta pipeline E2 para un ticker."""  # Docstring
    config = copy.deepcopy(config)  # Copia defensiva
    config = _apply_tuned_overrides(config=config, strategy_key="e2_moderate", ticker=ticker)  # Overrides tuning
    ensure_dir(out_dir)  # Crear directorio salida

    require_model_save = os.getenv("REQUIRE_MODEL_SAVE", "1").strip().lower() not in {  # Flag guardado
        "0",  # Deshabilitar
        "false",  # Deshabilitar
        "no",  # Deshabilitar
    }  # Fin set REQUIRE_MODEL_SAVE

    # 1. Cargar datos
    raw_path = raw_dir / f"{ticker}_daily.csv"  # Path CSV
    if not raw_path.exists():  # Validar existencia CSV
        raise FileNotFoundError(f"No existe {raw_path}")  # Error si falta

    df_raw = load_ohlcv_csv(raw_path)  # Cargar OHLCV
    print(f"\n{'='*60}")  # Separador
    print(f"Procesando {ticker} (E2 - Moderada)")  # Encabezado
    print(f"{'='*60}")  # Separador
    print(f"Período: {df_raw.index[0].date()} a {df_raw.index[-1].date()} ({len(df_raw)} días)")  # Log período

    # 2. Calcular features E2
    feat_df = compute_e2_features(df_raw, benchmark_df=benchmark_df)  # Features E2

    # 3. Crear target
    e2_cfg = config.get("strategies", {}).get("e2_moderate", {})  # Config E2
    horizon_days = e2_cfg.get("horizon_days", 20)  # Horizon
    target_series = make_target_e2(df_raw, horizon_days=horizon_days)  # Target futuro

    # Combinar
    df_full = feat_df.join(target_series, how="inner")  # Combinar features + target
    df_full = df_full.dropna()  # Eliminar NaNs

    if df_full.empty:  # Validar datos post-dropna
        raise ValueError(f"{ticker}: Sin datos después de dropna")  # Error si vacío

    print(f"Features calculadas: {len(feat_df.columns)}")  # Log features
    print(f"Datos válidos: {len(df_full)}")  # Log filas válidas

    # 4. Separar X, y
    feat_cols = [c for c in df_full.columns if c != target_series.name]  # Columnas features
    lookback_days = e2_cfg.get("lookback_days", 60)  # Lookback
    
    print(f"Lookback: {lookback_days} días, Horizon: {horizon_days} días")  # Log ventanas

    # Adaptación para usar make_sequences de build_sequences_e1e2.py
    # La firma retornada es: X, y, ts, feat_cols
    # PERO, make_sequences en train_e2_pipeline invocado recibe 5 args en el código actual: X_raw, y_raw, timestamps, lookback, dropna
    # Y build_sequences_e1e2.make_sequences espera (features_df, target_series, lookback)
    
    # CORRECCIÓN: Usar build_sequences correctamente
    # Primero necesitamos DataFrame y Series
    # X_raw proviene de df_full[feat_cols]
    # y_raw proviene de df_full[target_series.name]
    
    X_seq, y_seq, ts_seq, _ = make_sequences(  # Construir secuencias
        df_full[feat_cols], df_full[target_series.name], lookback=lookback_days  # Crear secuencias
    )  # Fin make_sequences

    print(f"Secuencias: {len(X_seq)} (shape: {X_seq.shape})")  # Log shapes

    # 6. Detectar método de split (walk-forward vs time_split)
    splits_cfg = config.get("splits", {})  # Config splits
    split_method = splits_cfg.get("method", "time_split")  # Método split

    if split_method == "walk_forward":  # Walk-forward habilitado
        print(f"  ▶ Ejecutando walk-forward ({splits_cfg.get('folds', 5)} folds, test={splits_cfg.get('test_size', 'auto')})")  # Log
        
        summary = run_e2_walk_forward(  # Ejecutar walk-forward
            X=X_seq,  # Features
            y=y_seq,  # Targets
            ts=ts_seq,  # Timestamps
            feat_names=feat_cols,  # Features
            ticker=ticker,  # Ticker
            out_dir=out_dir,  # Output
            ohlcv=df_raw,  # OHLCV
            splits_cfg=splits_cfg,  # Splits
            lstm_units=e2_cfg.get("model", {}).get("lstm_units", [128, 64]),  # Units
            dropout=e2_cfg.get("model", {}).get("dropout", 0.2),  # Dropout
            dense_units=e2_cfg.get("model", {}).get("dense_units", 32),  # Dense
            learning_rate=e2_cfg.get("model", {}).get("learning_rate", 1e-3),  # LR
            batch_size=e2_cfg.get("model", {}).get("batch_size", 64),  # Batch
            max_epochs=e2_cfg.get("model", {}).get("max_epochs", 150),  # Epochs
            patience=e2_cfg.get("model", {}).get("early_stopping_patience", 12),  # Patience
            loss=e2_cfg.get("model", {}).get("loss", "huber"),  # Loss
            huber_delta=e2_cfg.get("model", {}).get("huber_delta", 1.0),  # Huber delta
            tau_buy=e2_cfg.get("thresholds", {}).get("tau_buy", 0.02),  # Tau buy
            tau_sell=e2_cfg.get("thresholds", {}).get("tau_sell", 0.0),  # Tau sell
            round_trip_bps=config.get("backtest", {}).get("round_trip_bps", 10),  # Costos
            holding_period=horizon_days,  # Holding
            allow_short=e2_cfg.get("backtest", {}).get("allow_short", True),  # Shorts
            max_position=e2_cfg.get("backtest", {}).get("max_position", 1.0),  # Max posición
            seed=config.get("project", {}).get("seed", 42),  # Seed
            lookback_days=lookback_days,  # Lookback
            horizon_days=horizon_days,  # Horizon
            rsi14_min=e2_cfg.get("filters", {}).get("rsi14_min"),  # RSI min
            rsi14_max=e2_cfg.get("filters", {}).get("rsi14_max"),  # RSI max
        )  # Fin ejecución walk-forward
        
        return summary  # Retornar summary
    
    # 6. Split temporal simple (fallback)
    train_ratio = 0.7  # Fracción train
    val_ratio = 0.15  # Fracción val
    
    idx_train, idx_val, idx_test = time_split(  # Split temporal
        len(X_seq), train_frac=train_ratio, val_frac=val_ratio  # Split temporal
    )  # Índices de split

    X_train, y_train, ts_train = X_seq[idx_train], y_seq[idx_train], ts_seq[idx_train]  # Train
    X_val, y_val, ts_val = X_seq[idx_val], y_seq[idx_val], ts_seq[idx_val]  # Val
    X_test, y_test, ts_test = X_seq[idx_test], y_seq[idx_test], ts_seq[idx_test]  # Test

    print(f"\nSplit temporal:")  # Log split
    print(f"  Train: {len(X_train)} ({ts_train[0]} → {ts_train[-1]})")  # Train window
    print(f"  Val:   {len(X_val)} ({ts_val[0]} → {ts_val[-1]})")  # Val window
    print(f"  Test:  {len(X_test)} ({ts_test[0]} → {ts_test[-1]})")  # Test window

    # 7. Estandarizar features (solo sobre train)
    mean_X = X_train.mean(axis=(0, 1))  # Media X
    std_X = X_train.std(axis=(0, 1)) + 1e-8  # Std X

    X_train_norm = (X_train - mean_X) / std_X  # X train norm
    X_val_norm = (X_val - mean_X) / std_X  # X val norm
    X_test_norm = (X_test - mean_X) / std_X  # X test norm

    # Estandarizar target
    mean_y = y_train.mean()  # Media y
    std_y = y_train.std() + 1e-8  # Std y

    y_train_norm = (y_train - mean_y) / std_y  # y train norm
    y_val_norm = (y_val - mean_y) / std_y  # y val norm

    print(f"\nTarget statistics (train):")  # Log stats y
    print(f"  Mean: {mean_y:.4f}, Std: {std_y:.4f}")  # Mean/Std

    # 8. Entrenar LSTM
    print("\nEntrenando modelo LSTM...")  # Log entrenamiento
    model_cfg = e2_cfg.get("model", {})  # Config modelo

    model = LSTMRegressor(  # Instanciar LSTM
        input_size=X_train_norm.shape[2],  # Input size
        hidden_sizes=model_cfg.get("lstm_units", [128, 64]),  # LSTM units
        dropout=model_cfg.get("dropout", 0.2),  # Dropout
        dense_units=model_cfg.get("dense_units", 32),  # Dense
        seed=config.get("project", {}).get("seed", 42),  # Seed
    )  # Fin init modelo

    res = model.fit(  # Entrenar
        X_train_norm,  # X train
        y_train_norm,  # y train
        X_val_norm,  # X val
        y_val_norm,  # y val
        learning_rate=model_cfg.get("learning_rate", 1e-3),  # LR
        batch_size=model_cfg.get("batch_size", 64),  # Batch
        max_epochs=model_cfg.get("max_epochs", 150),  # Epochs
        early_stopping_patience=model_cfg.get("early_stopping_patience", 12),  # Patience
        loss=model_cfg.get("loss", "huber"),  # Loss
        huber_delta=model_cfg.get("huber_delta", 1.0),  # Huber delta
    )  # Fin fit

    print(f"✓ Entrenamiento completado:")  # Log entrenamiento
    print(f"  Epochs: {res.epochs_ran}")  # Epochs
    print(f"  Val loss: {res.best_val_loss:.6f}")  # Val loss

    # 8.5 Guardar modelo entrenado (por ticker)
    # Guardamos state_dict + metadata mínima para inferencia/reproducibilidad.
    model_path = out_dir / f"{ticker}_model.pth"  # Path modelo
    try:  # Guardar modelo
        torch = model.torch  # Referencia torch
        payload = {  # Payload modelo
            "ticker": ticker,  # Ticker
            "strategy": "e2_moderate",  # Estrategia
            "created_at": datetime.now(timezone.utc).isoformat(),  # Timestamp
            "model_class": "LSTMRegressor",  # Clase modelo
            "model_kwargs": {  # Args modelo
                "input_size": int(X_train_norm.shape[2]),  # Input size
                "hidden_sizes": list(model_cfg.get("lstm_units", [128, 64])),  # LSTM units
                "dropout": float(model_cfg.get("dropout", 0.2)),  # Dropout
                "dense_units": int(model_cfg.get("dense_units", 32)),  # Dense
                "seed": int(config.get("project", {}).get("seed", 42)),  # Seed
            },  # Fin model_kwargs
            "lookback_days": int(lookback_days),  # Lookback
            "horizon_days": int(horizon_days),  # Horizon
            "feature_names": list(feat_cols),  # Features
            "scaler_X": {"mean": mean_X.tolist(), "std": std_X.tolist()},  # Scaler X
            "scaler_y": {"mean": float(mean_y), "std": float(std_y)},  # Scaler y
            "train_result": {"epochs_ran": int(res.epochs_ran), "best_val_loss": float(res.best_val_loss)},  # Resultados
            "state_dict": {k: v.detach().cpu() for k, v in model.model.state_dict().items()},  # Pesos
        }  # Fin payload
        torch.save(payload, model_path)  # Guardar modelo
        print(f"✓ Modelo guardado: {model_path}")  # Log guardado
    except Exception as exc:  # Error guardado
        print(f"⚠️  No se pudo guardar el modelo para {ticker}: {exc}")  # Warning

    if require_model_save and not model_path.exists():  # Validar guardado
        raise RuntimeError(  # Error
            f"El pipeline terminó pero NO se guardó el modelo en {model_path}. "  # Msg 1
            "Revisar el bloque de guardado y permisos. "  # Msg 2
            "Si querés permitir continuar sin guardar, setear REQUIRE_MODEL_SAVE=0."  # Msg 3
        )  # Fin raise

    # 9. Predicciones en test (desnormalizar)
    pred_started_at = datetime.now(timezone.utc).isoformat()
    pred_start = time.perf_counter()
    y_pred_norm = model.predict(X_test_norm)  # Predicciones norm
    pred_end = time.perf_counter()
    pred_ended_at = datetime.now(timezone.utc).isoformat()
    log_timing_event(
        strategy="e2_moderate",
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
    y_pred = y_pred_norm * std_y + mean_y  # Predicciones desnorm

    # 10. Métricas ML
    mae = np.abs(y_test - y_pred).mean()  # MAE
    rmse = np.sqrt(((y_test - y_pred) ** 2).mean())  # RMSE
    ic = compute_information_coefficient(y_test, y_pred)  # IC

    # Directional accuracy
    dir_acc = float(np.mean((np.sign(y_test) == np.sign(y_pred))))  # Accuracy direccional

    ml_metrics = {  # Métricas ML
        "mae": mae,  # MAE
        "rmse": rmse,  # RMSE
        "ic": ic,  # IC
        "directional_accuracy": dir_acc,  # Accuracy
    }  # Fin métricas ML

    print(f"\nMétricas ML (test):")  # Log métricas
    print(f"  MAE:  {mae:.4f}")  # MAE
    print(f"  RMSE: {rmse:.4f}")  # RMSE
    print(f"  IC:   {ic:.3f}")  # IC
    print(f"  Directional Accuracy: {dir_acc:.2%}")  # Accuracy

    # 11. Backtest simple
    pred_df = pd.DataFrame(  # DataFrame predicciones
        {  # Dict predicciones
            "timestamp": ts_test,  # Timestamps
            "y_true": y_test,  # Target real
            "y_pred": y_pred,  # Target predicho
        }  # Fin dict
    )  # Fin DataFrame
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])  # A datetime
    pred_df = pred_df.set_index("timestamp")  # Index por timestamp

    # Obtener precios de test
    df_test = df_raw.loc[pred_df.index]  # Precios test

    # Señales básicas E2 (Vectorizado simplificado para reporte rápido)
    # Nota: Para backtest riguroso con reglas complejas (TP/SL/TimeStop/Filtros),
    # se debería usar el módulo src.backtest con loop evento a evento.
    # Aquí hacemos una aproximación vectorizada para validar el modelo.
    
    tau_buy = e2_cfg.get("thresholds", {}).get("tau_buy", 0.025)  # Tau buy
    tau_sell = e2_cfg.get("thresholds", {}).get("tau_sell", 0.00)  # Tau sell

    signals = pd.Series(0, index=pred_df.index)  # Serie señales
    signals[pred_df["y_pred"] > tau_buy] = 1  # Long
    signals[pred_df["y_pred"] < tau_sell] = -1  # Short/Exit
    
    # Aplicar filtros simples vectores si existen columnas (RSI, MACD) en df_test
    # Esto es una aproximación de rules_e2.py
    if "rsi_14" in df_test.columns:  # RSI disponible
        rsi_mask = (df_test["rsi_14"] >= 35) & (df_test["rsi_14"] <= 70)  # Mascara RSI
        # Solo permitimos entrada (1) si RSI ok. Salidas (-1) siempre permitidas.
        signals[(signals == 1) & (~rsi_mask)] = 0  # Neutralizar entradas
         
    if "macd_hist" in df_test.columns:  # MACD disponible
        macd_mask = df_test["macd_hist"] > 0  # Mascara MACD
        signals[(signals == 1) & (~macd_mask)] = 0  # Neutralizar entradas

    # backtest_daily_signals necesita timestamps, close, pred_returns, tau_buy, tau_sell
    # Usar wrapper simplificado
    bt_results = backtest_daily_signals(  # Backtest
        timestamps=pd.DatetimeIndex(pred_df.index),  # Timestamps
        close_prices=df_test["close"].to_numpy(),  # Precios close
        pred_returns=pred_df["y_pred"].to_numpy(),  # Retornos pred
        tau_buy=tau_buy,  # Umbral buy
        tau_sell=tau_sell,  # Umbral sell
    )  # Fin backtest
    trading_metrics = summarize_backtest(bt_results)  # Resumen

    print(f"\nMétricas Trading (test):")  # Log trading
    print(f"  Total Return: {trading_metrics['total_return']:.2%}")  # Total return
    print(f"  Sharpe:       {trading_metrics['sharpe']:.2f}")  # Sharpe
    print(f"  Max Drawdown: {trading_metrics['max_drawdown']:.2%}")  # Max drawdown

    # 12. Guardar resultados
    pred_df.to_csv(out_dir / f"{ticker}_predictions.csv")  # Guardar predicciones

    # Guardar scaler de features
    feat_names = feat_cols  # Nombres features
    scaler_X_df = pd.DataFrame({"mean": mean_X, "std": std_X}, index=feat_names)  # DF scaler X
    scaler_X_df.to_csv(out_dir / f"{ticker}_scaler.csv")  # Guardar scaler X

    # Guardar scaler de target
    scaler_y_df = pd.DataFrame({  # DF scaler y
        "mean_y": [mean_y],  # Media y
        "std_y": [std_y],  # Std y
    })  # Fin DF scaler y
    scaler_y_df.to_csv(out_dir / f"{ticker}_target_scaler.csv", index=False)  # Guardar scaler y

    meta = {  # Metadata
        "ticker": ticker,  # Ticker
        "n_samples": int(len(X_seq)),  # Samples
        "n_test": int(len(X_test)),  # Test size
        "lookback_days": lookback_days,  # Lookback
        "horizon_days": horizon_days,  # Horizon
        "split_method": splits_cfg.get("method", "time_split"),  # Split method
        "epochs_ran": res.epochs_ran,  # Epochs
        "val_loss": res.best_val_loss,  # Val loss
    }  # Fin metadata

    summary = {  # Summary
        **meta,  # Metadata
        **{f"ml_{k}": v for k, v in ml_metrics.items()},  # Métricas ML
        **{f"bt_{k}": v for k, v in trading_metrics.items()},  # Métricas backtest
    }  # Fin summary
    pd.Series(summary).to_csv(out_dir / f"{ticker}_summary.csv")  # Guardar resumen

    return summary  # Retornar resumen


def main() -> None:  # Main entry
    parser = argparse.ArgumentParser(description="Pipeline E2 (LSTM moderado)")  # Parser
    parser.add_argument(  # Arg config
        "--config",  # Flag
        type=str,  # Tipo
        default="src/config/base.yaml",  # Default
        help="Path to config YAML",  # Help
    )  # Fin arg config
    parser.add_argument(  # Arg tickers
        "--tickers",  # Flag
        type=str,  # Tipo
        default="",  # Default
        help="Comma-separated tickers (or uses universe.tickers_by_strategy.e2_moderate)",  # Help
    )  # Fin arg tickers
    args = parser.parse_args()  # Parse args

    root = project_root()  # Root proyecto
    cfg_path = Path(args.config)  # Path config
    if not cfg_path.is_absolute():  # Resolver relativo
        cfg_path = root / cfg_path  # Absoluto

    config = load_yaml(cfg_path)  # Cargar config

    # Tickers E2
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]  # Parse tickers
    if not tickers:  # Fallback config
        tickers = list(  # Lista tickers
            config.get("universe", {})  # Universe
            .get("tickers_by_strategy", {})  # Por estrategia
            .get("e2_moderate", [])  # E2
        )  # Fin list

    if not tickers:  # Sin tickers
        raise ValueError("No tickers for E2")  # Error

    # Benchmark (E2): deshabilitado por configuración del proyecto.
    # Nota: algunas features E2 pueden aceptar benchmark_df, pero aquí forzamos None.
    benchmark_df: pd.DataFrame | None = None

    raw_dir = root / "data" / "raw" / "daily"  # Dir raw

    out_base = root / "runs" / "e2_moderate" / datetime.now().strftime("%Y%m%d_%H%M%S")  # Dir salida
    ensure_dir(out_base)  # Crear dir

    # Guardar config usado
    import shutil  # Import shutil
    shutil.copy(cfg_path, out_base / "config_used.yaml")  # Copiar config

    # MLflow (default: local tracking if installed)
    mlflow_enabled = False  # Flag MLflow
    mlflow = None  # Referencia MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()  # Tracking URI
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "E2_Moderate")  # Nombre experimento
    try:  # Inicializar MLflow
        import mlflow as _mlflow  # type: ignore  # Import MLflow

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

    summaries: list[dict] = []  # Lista resúmenes
    for ticker in tickers:  # Iterar tickers
        ticker_out = out_base / ticker  # Dir ticker
        try:  # Ejecutar ticker
            if mlflow_enabled and mlflow is not None:  # Con MLflow
                timestamp = out_base.name  # Timestamp run
                with mlflow.start_run(run_name=f"E2_{ticker}_{timestamp}"):  # Run MLflow
                    mlflow.log_param("strategy", "e2_moderate")  # Param estrategia
                    mlflow.log_param("ticker", ticker)  # Param ticker
                    mlflow.log_param("model_type", "LSTM")  # Param modelo
                    mlflow.log_param("timestamp", timestamp)  # Param timestamp

                    e2_cfg = config.get("strategies", {}).get("e2_moderate", {})  # Config E2
                    model_cfg = e2_cfg.get("model", {})  # Config modelo
                    mlflow.log_params(  # Log params
                        {  # Dict params
                            "lookback_days": int(e2_cfg.get("lookback_days", 60)),  # Lookback
                            "horizon_days": int(e2_cfg.get("horizon_days", 20)),  # Horizon
                            "lstm_units": str(model_cfg.get("lstm_units", [128, 64])),  # LSTM units
                            "dropout": float(model_cfg.get("dropout", 0.2)),  # Dropout
                            "dense_units": int(model_cfg.get("dense_units", 32)),  # Dense
                            "learning_rate": float(model_cfg.get("learning_rate", 1e-3)),  # LR
                            "batch_size": int(model_cfg.get("batch_size", 64)),  # Batch
                            "max_epochs": int(model_cfg.get("max_epochs", 150)),  # Epochs
                            "early_stopping_patience": int(model_cfg.get("early_stopping_patience", 12)),  # Patience
                            "loss": str(model_cfg.get("loss", "huber")),  # Loss
                            "huber_delta": float(model_cfg.get("huber_delta", 1.0)),  # Delta
                            "split_method": str(config.get("splits", {}).get("method", "time_split")),  # Split
                            "seed": int(config.get("project", {}).get("seed", 42)),  # Seed
                        }  # Fin dict params
                    )  # Fin log_params

                    summary = run_e2_for_ticker(  # Ejecutar pipeline
                        config, ticker=ticker, raw_dir=raw_dir, out_dir=ticker_out, benchmark_df=benchmark_df  # Args
                    )  # Fin run_e2_for_ticker

                    metrics: dict[str, float] = {}  # Métricas MLflow
                    for k, v in summary.items():  # Iterar summary
                        if not (k.startswith("ml_") or k.startswith("bt_")):  # Filtrar
                            continue  # Saltar
                        if isinstance(v, (int, float)):  # Solo numéricas
                            metrics[k] = float(v)  # Convertir
                    if metrics:  # Log metrics
                        mlflow.log_metrics(metrics)  # Log

                    config_used = out_base / "config_used.yaml"  # Config usada
                    if config_used.exists():  # Existe config
                        mlflow.log_artifact(str(config_used), artifact_path="config")  # Log artifact

                    for fname in [  # Archivos artefactos
                        f"{ticker}_model.pth",  # Modelo
                        f"{ticker}_predictions.csv",  # Preds
                        f"{ticker}_summary.csv",  # Summary
                        f"{ticker}_scaler.csv",  # Scaler X
                        f"{ticker}_target_scaler.csv",  # Scaler y
                    ]:  # Fin lista artefactos
                        p = ticker_out / fname  # Path archivo
                        if p.exists():  # Existe
                            if fname.endswith(".pth"):  # Modelo
                                artifact_path = "models"  # Carpeta
                            elif "pred" in fname:  # Preds
                                artifact_path = "predictions"  # Carpeta
                            elif "scaler" in fname:  # Scalers
                                artifact_path = "scalers"  # Carpeta
                            else:  # Otros
                                artifact_path = "artifacts"  # Carpeta
                            mlflow.log_artifact(str(p), artifact_path=artifact_path)  # Log artifact
            else:  # Sin MLflow
                summary = run_e2_for_ticker(  # Ejecutar sin MLflow
                    config, ticker=ticker, raw_dir=raw_dir, out_dir=ticker_out, benchmark_df=benchmark_df  # Args
                )  # Fin run_e2_for_ticker
            summaries.append(summary)  # Append summary
            print(f"✓ {ticker}: MAE={summary['ml_mae']:.4f} IC={summary['ml_ic']:.3f}\n")  # Log ticker
        except Exception as exc:  # Error ticker
            print(f"✗ Error en {ticker}: {exc}\n")  # Error ticker

    pd.DataFrame(summaries).to_csv(out_base / "summary_all.csv", index=False)  # Guardar summary

    # Run agregado (opcional) para el summary de todos los tickers.
    if mlflow_enabled and mlflow is not None:  # Run agregado MLflow
        timestamp = out_base.name  # Timestamp
        try:  # Run summary
            with mlflow.start_run(run_name=f"E2_Summary_{timestamp}"):  # Run summary
                summary_path = out_base / "summary_all.csv"  # Path summary
                if summary_path.exists():  # Existe
                    mlflow.log_artifact(str(summary_path), artifact_path="reports")  # Log artifact
                mlflow.log_metric("total_tickers", float(len(tickers)))  # Log total
                mlflow.log_metric("successful_tickers", float(len(summaries)))  # Log exitosos
        except Exception as exc:  # Error summary MLflow
            print(f"⚠️  No se pudo loguear el summary en MLflow: {exc}")  # Warning

    print(f"\n✓ Resultados guardados en {out_base}")  # Log final


if __name__ == "__main__":  # Entry point
    main()  # Ejecutar main
