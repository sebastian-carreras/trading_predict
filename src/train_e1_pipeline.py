"""
Pipeline completo E1 (Estrategia Conservadora).

Flujo:
1. Cargar datos raw
2. Calcular features
3. Crear target y secuencias
4. Split temporal
5. Estandarizar features
6. Entrenar GRU
7. Evaluar y guardar resultados
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import os
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .features.build_features_e1 import compute_e1_features, make_target_e1
from .features.build_sequences_e1e2 import make_sequences, time_split, temporal_train_val_split
from .models.e1_gru import GRURegressor
from .backtest.backtest_daily import backtest_daily_signals, summarize_backtest
from .utils import ensure_dir, load_yaml, project_root


@lru_cache(maxsize=8)
def _load_tuned_params(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        return {}
    data = load_yaml(path)
    if not isinstance(data, dict):
        return {}
    return data


def _apply_tuned_overrides(*, config: dict, strategy_key: str, ticker: str) -> dict:
    tuned_path = (
        os.getenv("E1_TUNED_PARAMS_PATH", "").strip()
        or os.getenv("TUNED_PARAMS_PATH", "").strip()
    )
    if not tuned_path:
        return config

    all_tuned = _load_tuned_params(tuned_path)
    per_ticker = all_tuned.get(ticker)
    if not isinstance(per_ticker, dict):
        return config

    strat = config.setdefault("strategies", {}).setdefault(strategy_key, {})

    thresholds = per_ticker.get("thresholds")
    if isinstance(thresholds, dict):
        strat.setdefault("thresholds", {}).update(thresholds)

    model = per_ticker.get("model")
    if isinstance(model, dict):
        strat.setdefault("model", {}).update(model)

    if "n_folds" in per_ticker or "internal_val_fraction" in per_ticker:
        splits = config.setdefault("splits", {})
        if "n_folds" in per_ticker:
            splits["folds"] = int(per_ticker["n_folds"])
        if "internal_val_fraction" in per_ticker:
            splits["internal_val_fraction"] = float(per_ticker["internal_val_fraction"])

    return config


def load_ohlcv_csv(path: Path) -> pd.DataFrame:
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
    if len(y_true) <= 1:
        return float("nan")

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")

    try:
        from scipy.stats import spearmanr

        ic, _ = spearmanr(y_true, y_pred)
        return float(ic) if np.isfinite(ic) else 0.0
    except Exception:
        return float(np.corrcoef(y_true, y_pred)[0, 1])


def save_walkforward_plot(fold_df: pd.DataFrame, ticker: str, out_path: Path) -> None:
    """Guarda grafico de IC y Sharpe por fold."""
    if fold_df.empty:  # Sin datos
        return  # No hay datos para graficar

    try:  # Import matplotlib
        import matplotlib  # Import matplotlib

        matplotlib.use("Agg")  # Backend no interactivo
        import matplotlib.pyplot as plt  # Pyplot
    except Exception as exc:  # pragma: no cover - fallback si matplotlib no está
        print(f"⚠️  No se pudo generar el gráfico walk-forward para {ticker}: {exc}")  # Warning
        return  # Salir si no hay matplotlib

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)  # Crear figura con 2 subplots

    axes[0].plot(fold_df["fold"], fold_df["ml_ic"], marker="o")  # Serie IC
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)  # Línea base
    axes[0].set_ylabel("IC")  # Label eje Y
    axes[0].set_title(f"{ticker} - Walk-Forward Validation")  # Título

    axes[1].plot(fold_df["fold"], fold_df["bt_sharpe"], marker="o", color="#2ca02c")  # Serie Sharpe
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)  # Línea base
    axes[1].set_ylabel("Sharpe")  # Label eje Y
    axes[1].set_xlabel("Fold")  # Label eje X

    if "window" in fold_df.columns:  # Si hay etiquetas de ventana
        axes[1].set_xticks(fold_df["fold"].to_numpy())  # Ticks por fold
        axes[1].set_xticklabels(fold_df["window"].to_list(), rotation=35, ha="right")  # Labels ventanas

    for ax in axes:  # Iterar ejes
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)  # Grilla suave

    fig.tight_layout()  # Ajustar layout
    fig.savefig(str(out_path), dpi=150)  # Guardar PNG
    plt.close(fig)  # Cerrar figura


def run_e1_walk_forward(  # Walk-forward completo
    *,  # Solo kwargs
    X: np.ndarray,  # Matriz de features (n_samples, n_features)
    y: np.ndarray,  # Target de retornos futuros alineado a X
    ts: pd.DatetimeIndex,  # Timestamps utilizados para indexar las muestras
    feat_names: list[str] | pd.Index,  # Nombres de columnas de features
    ticker: str,  # Símbolo del activo en entrenamiento
    out_dir: Path,  # Carpeta de salida para predicciones/backtests
    ohlcv: pd.DataFrame,  # Historial OHLCV completo para cálculos de backtest
    splits_cfg: dict,  # Configuración de folds, embargo y tamaños
    gru_units: list[int],  # Tamaños de capas ocultas del GRU
    dropout: float,  # Dropout entre capas recurrentes
    dense_units: int,  # Dimensión de la capa densa final
    learning_rate: float,  # Learning rate del optimizador
    batch_size: int,  # Tamaño del batch para cada step
    max_epochs: int,  # Máximo epochs permitidos
    patience: int,  # Paciencia para early stopping
    loss: str,  # Tipo de loss («huber» o «mse»)
    huber_delta: float,  # Delta para Huber loss
    tau_buy: float,  # Umbral de activación para comprar
    tau_sell: float,  # Umbral para cerrar compras o abrir cortos
    round_trip_bps: float,  # Costos de ida y vuelta en bps
    holding_period: int,  # Días a mantener abiertas las posiciones
    allow_short: bool,  # Permitir señales de venta en corto
    max_position: float,  # Tamaño máximo de exposición por operación
    seed: int,  # Semilla para inicialización aleatoria
    lookback_days: int,  # Ventana histórica usada para features
    horizon_days: int,  # Horizonte de predicción siguiente
) -> dict:  # Retorna resumen
    """Ejecuta el walk-forward completo y resume resultados"""

    # Esta flag permite abortar si el script no llegó a guardar el modelo final
    require_model_save = os.getenv("REQUIRE_MODEL_SAVE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }  # Fin flags REQUIRE_MODEL_SAVE

    # Garantizamos que exista el directorio de salida donde se almacenarán los artefactos
    ensure_dir(out_dir)

    # Validamos que tengamos muestras suficientes para entrenar
    n_samples = len(X)
    if n_samples == 0:
        raise ValueError("No hay muestras disponibles para walk-forward")

    # Número de folds de validación (folds = cantidad de reentrenamientos temporales)
    folds = int(splits_cfg.get("folds", 5))
    if folds < 1:
        raise ValueError("'folds' debe ser >= 1 para walk-forward")

    # Embargo entre train y test para evitar look-ahead bias
    embargo_cfg = splits_cfg.get("embargo_days", {})
    if isinstance(embargo_cfg, dict):
        embargo_days = int(embargo_cfg.get("e1", 0))
    else:
        embargo_days = int(embargo_cfg or 0)

    # Convertimos los días de embargo a muestras (approx. 1 muestra ≈ 1 día hábil)
    gap_samples = max(0, int(embargo_days))

    # Definimos cuántas muestras atrevesará cada test set
    default_test_size = max(1, n_samples // (folds + 1))
    test_size = int(splits_cfg.get("test_size", default_test_size))
    if test_size <= gap_samples:
        test_size = gap_samples + 1

    # Inicializamos splitter que preserva orden temporal y respeta gap
    splitter = TimeSeriesSplit(n_splits=folds, test_size=test_size, gap=gap_samples)

    fold_summaries: list[dict] = []  # Lista de resúmenes por fold
    pred_frames: list[pd.DataFrame] = []  # Lista de predicciones por fold

    last_model_payload: dict | None = None  # Payload del último fold (para guardar)
    last_torch = None  # Referencia a torch para serializar el modelo

    root = project_root()  # Directorio raíz del proyecto (para paths relativos)

    def as_relative(path: Path) -> str:
        try:
            return str(path.relative_to(root))  # Convertir a path relativo si es posible
        except ValueError:
            return str(path)  # Fallback: devolver path absoluto si no está dentro del root

    # Determinar el tamaño de validación interna dentro del bloque de entrenamiento
    raw_val_fraction = splits_cfg.get(
        "internal_val_fraction", splits_cfg.get("val_fraction", 0.15)
    )
    try:
        val_fraction_cfg = float(raw_val_fraction)  # Normalizar a float
    except (TypeError, ValueError):
        val_fraction_cfg = 0.15  # Valor por defecto si no se puede convertir

    if not 0 < val_fraction_cfg < 1:
        val_fraction_cfg = 0.15  # Enforce rango válido (0,1)


    for fold_idx, (train_full_idx, test_idx) in enumerate(
        splitter.split(np.arange(n_samples)), start=1  # Generar splits temporales de índices
    ):
        if len(test_idx) == 0 or len(train_full_idx) == 0:
            continue  # Si algún bloque queda vacío, se salta este fold

        # Split interno train/val dentro del bloque de entrenamiento
        try:
            train_idx, val_idx = temporal_train_val_split(
                train_full_idx,  # Índices del bloque de entrenamiento completo
                val_fraction=val_fraction_cfg,  # Porcentaje destinado a validación
            )
        except ValueError:
            split_point = max(1, int(len(train_full_idx) * 0.8))  # Fallback 80/20
            train_idx = train_full_idx[:split_point]  # Índices train
            val_idx = train_full_idx[split_point:]  # Índices val

        X_train = X[train_idx]  # Features de entrenamiento
        X_val = X[val_idx]  # Features de validación
        X_test = X[test_idx]  # Features de test

        X_tr_2d = X_train.reshape(-1, X_train.shape[-1])  # Flatten para calcular stats
        mean_X = X_tr_2d.mean(axis=0)  # Media por feature (solo train)
        std_X = X_tr_2d.std(axis=0) + 1e-12  # Desvío por feature (evita div 0)

        def scale_X(data: np.ndarray) -> np.ndarray:
            return ((data - mean_X) / std_X).astype(np.float32)  # Z-score en X

        y_train = y[train_idx]  # Targets train
        y_val = y[val_idx]  # Targets val
        y_test = y[test_idx]  # Targets test

        mean_y = float(y_train.mean())  # Media del target (train)
        std_y = float(y_train.std()) + 1e-12  # Desvío del target (train)

        def scale_y(data: np.ndarray) -> np.ndarray:
            return ((data - mean_y) / std_y).astype(np.float32)  # Z-score en y

        def unscale_y(data: np.ndarray) -> np.ndarray:
            return (data * std_y + mean_y).astype(np.float32)  # Deshacer Z-score

        X_train_s = scale_X(X_train)  # X train escalado
        X_val_s = scale_X(X_val)  # X val escalado
        X_test_s = scale_X(X_test)  # X test escalado

        y_train_s = scale_y(y_train)  # y train escalado
        y_val_s = scale_y(y_val)  # y val escalado

        model = GRURegressor(
            input_size=X_train.shape[-1],  # Número de features
            hidden_sizes=list(gru_units),  # Arquitectura GRU
            dropout=dropout,  # Dropout entre capas
            dense_units=dense_units,  # Tamaño de la capa densa
            seed=seed,  # Seed para reproducibilidad
        )

        res = model.fit(
            X_train_s,  # X train
            y_train_s,  # y train
            X_val_s,  # X val
            y_val_s,  # y val
            learning_rate=learning_rate,  # LR
            batch_size=batch_size,  # Batch size
            max_epochs=max_epochs,  # Máximo de epochs
            early_stopping_patience=patience,  # Early stopping
            loss=loss,  # Tipo de loss
            huber_delta=huber_delta,  # Delta Huber
        )

        y_pred_s = model.predict(X_test_s)  # Predicciones escaladas
        y_pred = unscale_y(y_pred_s)  # Predicciones en escala original

        last_torch = model.torch  # Guardamos referencia para serializar

        ts_test = ts[test_idx]  # Timestamps de test
        window_label = f"{ts_test[0].date()} -> {ts_test[-1].date()}"  # Etiqueta de ventana

        # Guardar payload del último fold (modelo + scalers) para inferencia.
        # Elegimos el último fold porque es el más reciente (más relevante operativamente).
        last_model_payload = {
            "ticker": ticker,  # Ticker del activo
            "strategy": "e1_conservative",  # Nombre de estrategia
            "created_at": datetime.utcnow().isoformat(),  # Timestamp UTC de entrenamiento
            "model_class": "GRURegressor",  # Clase del modelo
            "model_kwargs": {
                "input_size": int(X_train.shape[-1]),  # Número de features
                "hidden_sizes": list(gru_units),  # Arquitectura GRU
                "dropout": float(dropout),  # Dropout
                "dense_units": int(dense_units),  # Capa densa final
                "seed": int(seed),  # Semilla
            },
            "lookback_days": int(lookback_days),  # Lookback usado
            "horizon_days": int(horizon_days),  # Horizonte objetivo
            "feature_names": list(feat_names),  # Lista de features
            "scaler_X": {"mean": mean_X.tolist(), "std": std_X.tolist()},  # Stats de X
            "scaler_y": {"mean": float(mean_y), "std": float(std_y)},  # Stats de y
            "train_result": {"epochs_ran": int(res.epochs_ran), "best_val_loss": float(res.best_val_loss)},  # Training info
            "walkforward": {
                "fold": int(fold_idx),  # Fold actual
                "window": window_label,  # Ventana temporal
                "test_size": int(test_size),  # Tamaño test
                "gap_samples": int(gap_samples),  # Gap por embargo
            },
            "state_dict": {k: v.detach().cpu() for k, v in model.model.state_dict().items()},  # Pesos del modelo
        }

        mae = float(np.mean(np.abs(y_test - y_pred)))  # MAE del fold
        rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))  # RMSE del fold
        dir_acc = float(np.mean(np.sign(y_test) == np.sign(y_pred)))  # Accuracy direccional
        ic = compute_information_coefficient(y_test, y_pred)  # IC (Spearman)

        close_prices = ohlcv.loc[ts_test, "close"].to_numpy()  # Precios de cierre test
        bt = backtest_daily_signals(
            timestamps=ts_test,  # Índices temporales
            close_prices=close_prices,  # Precio de cierre
            pred_returns=y_pred,  # Predicción de retornos
            tau_buy=tau_buy,  # Umbral compra
            tau_sell=tau_sell,  # Umbral venta
            round_trip_bps=round_trip_bps,  # Costos
            holding_period_days=holding_period,  # Holding period
            allow_short=allow_short,  # Permitir cortos
            max_position=max_position,  # Máx. exposición
        )
        trading_metrics = summarize_backtest(bt)  # Métricas del backtest

        sharpe = trading_metrics.get("sharpe", float("nan"))  # Sharpe del fold

        ic_str = "nan" if np.isnan(ic) else f"{ic:.3f}"  # IC formateado
        sharpe_str = "nan" if np.isnan(sharpe) else f"{sharpe:.2f}"  # Sharpe formateado
        print(
            f"    Fold {fold_idx}: {window_label} | MAE={mae:.4f} IC={ic_str} Sharpe={sharpe_str}"  # Log por fold
        )

        bt.to_csv(out_dir / f"{ticker}_fold{fold_idx}_backtest.csv")  # Guardar backtest del fold

        fold_summaries.append(
            {
                "fold": fold_idx,  # ID de fold
                "window": window_label,  # Ventana temporal
                "test_start": ts_test[0],  # Inicio de test
                "test_end": ts_test[-1],  # Fin de test
                "n_train": int(len(train_idx)),  # Tamaño train
                "n_val": int(len(val_idx)),  # Tamaño val
                "n_test": int(len(test_idx)),  # Tamaño test
                "epochs_ran": int(res.epochs_ran),  # Epochs efectivas
                "val_loss": float(res.best_val_loss),  # Mejor loss de val
                "ml_mae": mae,  # MAE ML
                "ml_rmse": rmse,  # RMSE ML
                "ml_directional_accuracy": dir_acc,  # Directional accuracy
                "ml_ic": ic,  # IC
                **{f"bt_{k}": float(v) for k, v in trading_metrics.items()},  # Métricas de backtest
            }
        )

        preds_df = pd.DataFrame(
            {
                "fold": fold_idx,  # Fold al que pertenece
                "y_true": y_test,  # Target real
                "y_pred": y_pred,  # Predicción
            },
            index=ts_test,  # Index temporal de test
        )
        preds_df.index.name = "timestamp"  # Etiqueta del índice
        pred_frames.append(preds_df)  # Guardar para concatenar luego

    if not fold_summaries:
        raise ValueError("No se generaron folds válidos para walk-forward")  # Validación final

    fold_df = pd.DataFrame(fold_summaries)  # DataFrame con métricas por fold
    fold_df.to_csv(out_dir / f"{ticker}_walkforward_folds.csv", index=False)  # Guardar CSV

    combined_preds = pd.concat(pred_frames).sort_index()  # Unir predicciones de todos los folds
    combined_preds.to_csv(out_dir / f"{ticker}_walkforward_predictions.csv")  # Guardar predicciones

    y_true_all = combined_preds["y_true"].to_numpy()  # Targets reales combinados
    y_pred_all = combined_preds["y_pred"].to_numpy()  # Predicciones combinadas

    mae_all = float(np.mean(np.abs(y_true_all - y_pred_all)))  # MAE global
    rmse_all = float(np.sqrt(np.mean((y_true_all - y_pred_all) ** 2)))  # RMSE global
    dir_acc_all = float(np.mean(np.sign(y_true_all) == np.sign(y_pred_all)))  # Direccional global
    ic_all = compute_information_coefficient(y_true_all, y_pred_all)  # IC global

    bt_all = backtest_daily_signals(
        timestamps=pd.DatetimeIndex(combined_preds.index),  # Timestamps globales
        close_prices=ohlcv.loc[combined_preds.index, "close"].to_numpy(),  # Precios de cierre
        pred_returns=y_pred_all,  # Predicciones globales
        tau_buy=tau_buy,  # Umbral compra
        tau_sell=tau_sell,  # Umbral venta
        round_trip_bps=round_trip_bps,  # Costos
        holding_period_days=holding_period,  # Holding period
        allow_short=allow_short,  # Permitir cortos
        max_position=max_position,  # Máximo tamaño de posición
    )
    trading_metrics_all = summarize_backtest(bt_all)  # Métricas globales
    bt_all.to_csv(out_dir / f"{ticker}_walkforward_backtest.csv")  # Guardar backtest

    plot_path = out_dir / f"{ticker}_walkforward_metrics.png"  # Path gráfico
    save_walkforward_plot(fold_df, ticker, plot_path)  # Generar gráfico IC/Sharpe

    

    # Guardar modelo entrenado del último fold como artifact por ticker
    if last_model_payload is not None and last_torch is not None:
        model_path = out_dir / f"{ticker}_model.pth"  # Path del modelo
        try:
            last_torch.save(last_model_payload, model_path)  # Serializar payload
        except Exception as exc:
            print(f"⚠️  No se pudo guardar el modelo walk-forward para {ticker}: {exc}")  # Warning

        if require_model_save and not model_path.exists():
            raise RuntimeError(
                f"El pipeline terminó pero NO se guardó el modelo walk-forward en {model_path}. "
                "Si querés permitir continuar sin guardar, setear REQUIRE_MODEL_SAVE=0."
            )  # Error si se exige guardado

    summary = {
        "ticker": ticker,  # Ticker
        "n_samples": int(n_samples),  # Total de muestras
        "n_test": int(len(combined_preds)),  # Total de samples evaluadas
        "folds": int(len(fold_df)),  # Cantidad de folds
        "lookback_days": int(lookback_days),  # Lookback
        "horizon_days": int(horizon_days),  # Horizonte
        "split_method": "walk_forward",  # Método de split
        "walkforward_test_size": int(test_size),  # Tamaño test por fold
        "walkforward_gap": int(gap_samples),  # Gap entre train/test
        "ml_mae": mae_all,  # MAE global
        "ml_rmse": rmse_all,  # RMSE global
        "ml_directional_accuracy": dir_acc_all,  # Accuracy direccional
        "ml_ic": ic_all,  # IC global
        **{f"bt_{k}": float(v) for k, v in trading_metrics_all.items()},  # Métricas backtest
        "folds_file": as_relative(out_dir / f"{ticker}_walkforward_folds.csv"),  # CSV folds
        "predictions_file": as_relative(out_dir / f"{ticker}_walkforward_predictions.csv"),  # CSV preds
        "backtest_file": as_relative(out_dir / f"{ticker}_walkforward_backtest.csv"),  # CSV backtest
        "plot_file": as_relative(plot_path),  # Gráfico
    }  # Fin set REQUIRE_MODEL_SAVE

    return summary  # Retornar resumen final del walk-forward


def run_e1_for_ticker(
    config: dict,  # Configuración global (YAML ya cargado)
    ticker: str,  # Símbolo del activo a entrenar
    raw_dir: Path,  # Ruta de datos raw diarios
    out_dir: Path,  # Directorio base de salida
    benchmark_df: pd.DataFrame | None,  # Benchmark para features relativas
 ) -> dict:  # Retorna resumen
    """Entrena y evalúa la estrategia E1 para un ticker específico."""

    # Copia defensiva para no contaminar el config compartido (Airflow / loops)
    config = copy.deepcopy(config)  # Evita side-effects cuando hay múltiples tickers
    config = _apply_tuned_overrides(config=config, strategy_key="e1_conservative", ticker=ticker)  # Overrides por ticker

    # Flag para exigir guardado del modelo (por defecto sí)
    require_model_save = os.getenv("REQUIRE_MODEL_SAVE", "1").strip().lower() not in {  # Flag guardado
        "0",  # Deshabilitar
        "false",  # Deshabilitar
        "no",  # Deshabilitar
    }  # Fin set REQUIRE_MODEL_SAVE

    # Compatibilidad: si out_dir es base, crear subfolder por ticker
    out_dir = Path(out_dir)  # Normalizar a Path
    if out_dir.name != ticker:  # Asegurar subdir por ticker
        out_dir = out_dir / ticker  # Asegurar carpeta por ticker
    ensure_dir(out_dir)  # Crear carpeta si no existe

    # Parámetros E1 del config
    e1 = config.get("strategies", {}).get("e1_conservative", {})  # Sección E1
    lookback_days = int(e1.get("lookback_days", 180))  # Lookback por default 180
    horizon_days = int(e1.get("horizon_days", 90))  # Horizonte por default 90

    model_cfg = e1.get("model", {})  # Sección de modelo
    gru_units = model_cfg.get("gru_units", [96, 32])  # Arquitectura GRU
    dropout = float(model_cfg.get("dropout", 0.2))  # Dropout
    dense_units = int(model_cfg.get("dense_units", 32))  # Dense units
    lr = float(model_cfg.get("learning_rate", 1e-3))  # Learning rate
    batch_size = int(model_cfg.get("batch_size", 64))  # Batch size
    max_epochs = int(model_cfg.get("max_epochs", 200))  # Max epochs
    patience = int(model_cfg.get("early_stopping_patience", 15))  # Patience
    loss = str(model_cfg.get("loss", "huber"))  # Loss
    huber_delta = float(model_cfg.get("huber_delta", 1.0))  # Huber delta

    seed = int(config.get("project", {}).get("seed", 42))  # Semilla global

    # Cargar datos (priorizar datos limpios si existen)
    # raw_dir es data/raw/daily, entonces data/clean está en raw_dir.parent.parent / "clean"
    clean_dir = raw_dir.parent.parent / "clean"  # Carpeta de datos limpios
    clean_csv_path = clean_dir / f"{ticker}_daily.csv"  # Path del CSV limpio
    
    if clean_csv_path.exists():  # Si hay datos limpios
        csv_path = clean_csv_path  # Usar datos limpios
        print(f"  ✓ Usando datos limpios: {csv_path.name}")  # Log
    else:  # Fallback a datos raw
        csv_path = raw_dir / f"{ticker}_daily.csv"  # Fallback a raw
        if not csv_path.exists():  # Validar existencia
            raise FileNotFoundError(f"Missing daily CSV: {csv_path}")  # Error si no hay data
        print(f"  ⚠️  Usando datos raw (limpieza no ejecutada): {csv_path.name}")  # Warning

    ohlcv = load_ohlcv_csv(csv_path)  # Cargar OHLCV desde CSV

    # Features
    features = compute_e1_features(ohlcv, benchmark_df)  # Features E1
    target = make_target_e1(ohlcv, horizon_days=horizon_days)  # Target futuro

    # Secuencias
    X, y, ts, feat_names = make_sequences(features, target, lookback=lookback_days)  # Construir secuencias

    thresholds = e1.get("thresholds", {})  # Config de umbrales
    tau_buy = float(thresholds.get("tau_buy", 0.04))  # Umbral compra
    tau_sell = float(thresholds.get("tau_sell", -0.02))  # Umbral venta

    costs_cfg = config.get("costs", {})  # Config de costos
    round_trip_bps = float(costs_cfg.get("daily_round_trip_bps", 10))  # Costos diarios

    bt_cfg = e1.get("backtest", {})  # Config de backtest
    holding_period = int(bt_cfg.get("holding_period_days", horizon_days))  # Holding period
    allow_short = bool(bt_cfg.get("allow_short", False))  # Permitir cortos
    max_position = float(bt_cfg.get("max_position", 1.0))  # Máx posición

    splits_cfg = config.get("splits", {})  # Config de splits
    if splits_cfg.get("method") == "walk_forward":  # Si está habilitado walk-forward
        print(  # Log inicio walk-forward
            f"  ▶ Ejecutando walk-forward ({splits_cfg.get('folds', 5)} folds, test={splits_cfg.get('test_size', 'auto')})"  # Log
        )  # Mensaje de inicio
        summary = run_e1_walk_forward(  # Ejecutar walk-forward
            X=X,  # Features
            y=y,  # Targets
            ts=ts,  # Timestamps
            feat_names=feat_names,  # Features
            ticker=ticker,  # Ticker
            out_dir=out_dir,  # Output
            ohlcv=ohlcv,  # OHLCV
            splits_cfg=splits_cfg,  # Config splits
            gru_units=list(gru_units),  # GRU units
            dropout=dropout,  # Dropout
            dense_units=dense_units,  # Dense units
            learning_rate=lr,  # LR
            batch_size=batch_size,  # Batch
            max_epochs=max_epochs,  # Epochs
            patience=patience,  # Patience
            loss=loss,  # Loss
            huber_delta=huber_delta,  # Huber delta
            tau_buy=tau_buy,  # Threshold buy
            tau_sell=tau_sell,  # Threshold sell
            round_trip_bps=round_trip_bps,  # Costs
            holding_period=holding_period,  # Holding
            allow_short=allow_short,  # Shorts
            max_position=max_position,  # Max position
            seed=seed,  # Seed
            lookback_days=lookback_days,  # Lookback
            horizon_days=horizon_days,  # Horizon
        )  # Ejecutar walk-forward
        return summary  # Retornar resumen walk-forward

    # Split temporal: dividir datos en 70% train, 15% val, 15% test
    # Se mantiene el orden temporal para evitar data leakage
    idx_train, idx_val, idx_test = time_split(len(X))  # Índices de split
    X_train, y_train = X[idx_train], y[idx_train]  # Datos train
    X_val, y_val = X[idx_val], y[idx_val]  # Datos val
    X_test, y_test = X[idx_test], y[idx_test]  # Datos test
    ts_test = ts[idx_test]  # Timestamps del test

    # ESTANDARIZAR FEATURES X (Z-score normalization)
    # - Usar solo datos de train para calcular media y std (prevenir data leakage)
    # - Aplicar la misma transformación a val y test
    # - Esto centra y escala los datos a media=0 y std=1
    Xtr2d = X_train.reshape(-1, X_train.shape[-1])  # Flatten para stats
    mean_X = Xtr2d.mean(axis=0)  # Media por feature
    std_X = Xtr2d.std(axis=0) + 1e-12  # Std por feature (+1e-12 evita /0)

    def scale_X(Xa: np.ndarray) -> np.ndarray:  # Escalado Z-score de X
        """Aplica estandarización Z-score a las features."""
        return ((Xa - mean_X) / std_X).astype(np.float32)  # Z-score X

    X_train_s = scale_X(X_train)  # X train escalado
    X_val_s = scale_X(X_val)  # X val escalado
    X_test_s = scale_X(X_test)  # X test escalado

    # ESTANDARIZAR TARGETS y (Z-score normalization)
    # - Normalizamos los targets para estabilizar el entrenamiento
    # - Mejora convergencia durante el backprop
    mean_y = float(y_train.mean())  # Media del target en train
    std_y = float(y_train.std()) + 1e-12  # Std del target en train

    def scale_y(ya: np.ndarray) -> np.ndarray:
        """Aplica estandarización Z-score a los targets."""
        return ((ya - mean_y) / std_y).astype(np.float32)  # Z-score y

    def unscale_y(ya_scaled: np.ndarray) -> np.ndarray:  # Des-escalado de y
        """Deshace la estandarización (vuelve a escala original)."""
        return (ya_scaled * std_y + mean_y).astype(np.float32)  # Des-escalar y

    y_train_s = scale_y(y_train)  # y train escalado
    y_val_s = scale_y(y_val)  # y val escalado
    y_test_s = scale_y(y_test)  # y test escalado

    # Entrenar modelo GRU (Gated Recurrent Unit)
    # - Red recurrente para modelar secuencias temporales
    # - Capaz de capturar dependencias a largo plazo
    model = GRURegressor(
        input_size=X_train_s.shape[-1],  # Número de features (columnas)
        hidden_sizes=gru_units,  # Unidades en capas recurrentes
        dropout=dropout,  # Regularización: desactivar unidades al azar
        dense_units=dense_units,  # Unidades en capa densa final
        seed=seed,  # Para reproducibilidad
    )

    print(f"Entrenando GRU para {ticker}...")  # Log inicio entrenamiento
    res = model.fit(
        X_train_s,  # Features de entrenamiento (normalizados)
        y_train_s,  # Targets de entrenamiento (normalizados)
        X_val_s,  # Features de validación (para early stopping)
        y_val_s,  # Targets de validación
        learning_rate=lr,  # Velocidad de aprendizaje del optimizador
        batch_size=batch_size,  # Muestras por iteración
        max_epochs=max_epochs,  # Épocas máximas de entrenamiento
        early_stopping_patience=patience,  # Parar si no mejora en N épocas
        loss=loss,  # Función de pérdida: "huber" o "mse"
        huber_delta=huber_delta,  # Parámetro delta para loss Huber
    )

    print(f"  Epochs: {res.epochs_ran}, Val Loss: {res.best_val_loss:.6f}")  # Log entrenamiento

    # Guardar modelo entrenado (por ticker)
    model_path = out_dir / f"{ticker}_model.pth"  # Path del modelo
    try:
        torch = model.torch  # Referencia a torch
        payload = {
            "ticker": ticker,  # Ticker
            "strategy": "e1_conservative",  # Estrategia
            "created_at": datetime.utcnow().isoformat(),  # Timestamp
            "model_class": "GRURegressor",  # Clase del modelo
            "model_kwargs": {
                "input_size": int(X_train_s.shape[-1]),  # Input size
                "hidden_sizes": list(gru_units),  # GRU units
                "dropout": float(dropout),  # Dropout
                "dense_units": int(dense_units),  # Dense units
                "seed": int(seed),  # Seed
            },
            "lookback_days": int(lookback_days),  # Lookback
            "horizon_days": int(horizon_days),  # Horizon
            "feature_names": list(feat_names),  # Feature names
            "scaler_X": {"mean": mean_X.tolist(), "std": std_X.tolist()},  # Escalado X
            "scaler_y": {"mean": float(mean_y), "std": float(std_y)},  # Escalado y
            "train_result": {"epochs_ran": int(res.epochs_ran), "best_val_loss": float(res.best_val_loss)},  # Resultados
            "state_dict": {k: v.detach().cpu() for k, v in model.model.state_dict().items()},  # Pesos
        }
        torch.save(payload, model_path)  # Guardar modelo
        print(f"  ✓ Modelo guardado: {model_path}")  # Log guardado
    except Exception as exc:
        print(f"  ⚠️  No se pudo guardar el modelo para {ticker}: {exc}")  # Warning si falla

    if require_model_save and not model_path.exists():
        raise RuntimeError(
            f"El pipeline terminó pero NO se guardó el modelo en {model_path}. "
            "Si querés permitir continuar sin guardar, setear REQUIRE_MODEL_SAVE=0."
        )  # Error si se exige guardado

    # Predicciones en test (normalizadas)
    y_pred_s = model.predict(X_test_s)  # Predicciones escaladas
    
    # Desnormalizar predicciones para métricas y backtesting
    y_pred = unscale_y(y_pred_s)  # Predicciones en escala original

    # Métricas ML
    mae = float(np.mean(np.abs(y_test - y_pred)))  # MAE
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))  # RMSE
    dir_acc = float(np.mean(np.sign(y_test) == np.sign(y_pred)))  # Accuracy direccional
    ic = compute_information_coefficient(y_test, y_pred)  # IC

    ml = {"mae": mae, "rmse": rmse, "directional_accuracy": dir_acc, "ic": ic}  # Métricas ML

    # Backtesting
    # Obtener precios de cierre del período de test
    close_prices_test = ohlcv.loc[ts_test, "close"].to_numpy()  # Close prices
    
    # Ejecutar backtest
    bt = backtest_daily_signals(
        timestamps=ts_test,  # Timestamps
        close_prices=close_prices_test,  # Close prices
        pred_returns=y_pred,  # Predicciones
        tau_buy=tau_buy,  # Umbral compra
        tau_sell=tau_sell,  # Umbral venta
        round_trip_bps=round_trip_bps,  # Costos
        holding_period_days=holding_period,  # Holding
        allow_short=allow_short,  # Shorts
        max_position=max_position,  # Max posición
    )
    
    # Métricas de trading
    trading_metrics = summarize_backtest(bt)  # Métricas de backtest

    

    # Guardar outputs
    ensure_dir(out_dir)  # Asegurar dir

    preds_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred}, index=ts_test)  # DF preds
    preds_df.index.name = "timestamp"  # Nombrar índice
    preds_df.to_csv(out_dir / f"{ticker}_predictions.csv")  # Guardar preds
    
    # Guardar backtest
    bt.to_csv(out_dir / f"{ticker}_backtest.csv")  # Guardar backtest

    # Guardar scaler de features X
    scaler_X_df = pd.DataFrame({"mean": mean_X, "std": std_X}, index=feat_names)  # DF scaler X
    scaler_X_df.to_csv(out_dir / f"{ticker}_scaler.csv")  # Guardar scaler X
    
    # Guardar scaler de target y (para inferencia futura)
    scaler_y_df = pd.DataFrame({
        "mean_y": [mean_y],  # Media y
        "std_y": [std_y]  # Std y
    })
    scaler_y_df.to_csv(out_dir / f"{ticker}_target_scaler.csv", index=False)  # Guardar scaler y

    meta = {
        "ticker": ticker,  # Ticker
        "n_samples": int(len(X)),  # Muestras totales
        "n_test": int(len(X_test)),  # Muestras test
        "lookback_days": lookback_days,  # Lookback
        "horizon_days": horizon_days,  # Horizon
        "split_method": splits_cfg.get("method", "time_split"),  # Método split
        "epochs_ran": res.epochs_ran,  # Epochs ejecutadas
        "val_loss": res.best_val_loss,  # Best val loss
    }

    summary = {
        **meta,  # Metadata
        **{f"ml_{k}": v for k, v in ml.items()},  # Métricas ML
        **{f"bt_{k}": v for k, v in trading_metrics.items()},  # Métricas backtest
    }
    pd.Series(summary).to_csv(out_dir / f"{ticker}_summary.csv")  # Guardar summary

    return summary  # Retornar resumen


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline E1 (GRU conservador)")  # Parser CLI
    parser.add_argument(
        "--config",  # Flag de config
        type=str,  # Tipo str
        default="src/config/base.yaml",  # Config por defecto
        help="Path to config YAML",  # Ayuda CLI
    )
    parser.add_argument(
        "--tickers",  # Flag de tickers
        type=str,  # Tipo str
        default="",  # Por defecto vacío (usa config)
        help="Comma-separated tickers (or uses universe.tickers_by_strategy.e1_conservative)",  # Ayuda CLI
    )
    args = parser.parse_args()  # Parseo de argumentos

    root = project_root()  # Directorio raíz del proyecto
    cfg_path = Path(args.config)  # Path del config
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path  # Resolver a path absoluto

    config = load_yaml(cfg_path)  # Cargar YAML de config

    # Tickers E1
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]  # Tickers por CLI
    if not tickers:
        tickers = list(
            config.get("universe", {})  # Config de universo
            .get("tickers_by_strategy", {})  # Mapa de estrategia a tickers
            .get("e1_conservative", [])  # Tickers E1
        )

    if not tickers:
        raise ValueError("No tickers for E1")  # Error si no hay tickers

    # Benchmark
    benchmark = config.get("universe", {}).get("benchmark", "SPY")  # Benchmark (ej: SPY)
    raw_dir = root / "data" / "raw" / "daily"  # Directorio raw diario

    benchmark_path = raw_dir / f"{benchmark}_daily.csv"  # CSV del benchmark
    if benchmark_path.exists():
        benchmark_df = load_ohlcv_csv(benchmark_path)  # Cargar benchmark
    else:
        print(f"⚠️  Benchmark {benchmark} no encontrado, usando valores vacíos")  # Warning
        benchmark_df = None  # Sin benchmark

    out_base = root / "runs" / "e1_conservative" / datetime.now().strftime("%Y%m%d_%H%M%S")  # Output run
    ensure_dir(out_base)  # Crear folder de salida

    # Guardar config usado
    import shutil  # Import local para copia
    shutil.copy(cfg_path, out_base / "config_used.yaml")  # Copiar YAML usado

    # MLflow (opcional): si está instalado y hay tracking URI, logueamos params/metrics/artifacts.
    mlflow_enabled = False  # Flag MLflow
    mlflow = None  # Referencia a MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()  # URI de tracking
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "E1_Conservative")  # Nombre de experimento
    if tracking_uri:
        try:
            import mlflow as _mlflow  # type: ignore  # Import MLflow si está

            _mlflow.set_tracking_uri(tracking_uri)  # Config tracking
            _mlflow.set_experiment(experiment_name)  # Config experimento
            mlflow = _mlflow  # Guardar referencia
            mlflow_enabled = True  # Habilitar tracking
            print(f"✓ MLflow habilitado: {tracking_uri} (experiment={experiment_name})")  # Log
        except Exception as exc:
            print(f"⚠️  MLflow no disponible, continuando sin tracking: {exc}")  # Warning
            mlflow_enabled = False  # Deshabilitar tracking

    summaries: list[dict] = []  # Resúmenes por ticker
    for ticker in tickers:
        ticker_out = out_base / ticker  # Output por ticker
        try:
            if mlflow_enabled and mlflow is not None:
                timestamp = out_base.name  # Timestamp de run
                with mlflow.start_run(run_name=f"E1_{ticker}_{timestamp}"):
                    mlflow.log_param("strategy", "e1_conservative")  # Param estrategia
                    mlflow.log_param("ticker", ticker)  # Param ticker
                    mlflow.log_param("model_type", "GRU")  # Param tipo modelo
                    mlflow.log_param("timestamp", timestamp)  # Param timestamp

                    e1_cfg = config.get("strategies", {}).get("e1_conservative", {})  # Config E1
                    model_cfg = e1_cfg.get("model", {})  # Config modelo
                    mlflow.log_params(
                        {
                            "lookback_days": int(e1_cfg.get("lookback_days", 180)),  # Lookback
                            "horizon_days": int(e1_cfg.get("horizon_days", 90)),  # Horizon
                            "gru_units": str(model_cfg.get("gru_units", [96, 32])),  # GRU units
                            "dropout": float(model_cfg.get("dropout", 0.2)),  # Dropout
                            "dense_units": int(model_cfg.get("dense_units", 32)),  # Dense units
                            "learning_rate": float(model_cfg.get("learning_rate", 1e-3)),  # LR
                            "batch_size": int(model_cfg.get("batch_size", 64)),  # Batch size
                            "max_epochs": int(model_cfg.get("max_epochs", 200)),  # Max epochs
                            "early_stopping_patience": int(model_cfg.get("early_stopping_patience", 15)),  # Patience
                            "loss": str(model_cfg.get("loss", "huber")),  # Loss
                            "huber_delta": float(model_cfg.get("huber_delta", 1.0)),  # Huber delta
                            "split_method": str(config.get("splits", {}).get("method", "time_split")),  # Split method
                            "seed": int(config.get("project", {}).get("seed", 42)),  # Seed
                        }
                    )

                    summary = run_e1_for_ticker(
                        config,  # Config
                        ticker=ticker,  # Ticker
                        raw_dir=raw_dir,  # Raw dir
                        out_dir=ticker_out,  # Output dir
                        benchmark_df=benchmark_df,  # Benchmark
                    )

                    # Métricas
                    metrics: dict[str, float] = {}  # Dict de métricas
                    for k, v in summary.items():
                        if not (k.startswith("ml_") or k.startswith("bt_")):
                            continue  # Solo métricas ML/BT
                        if isinstance(v, (int, float)):
                            metrics[k] = float(v)  # Convertir a float
                    if metrics:
                        mlflow.log_metrics(metrics)  # Log métricas

                    # Artifacts por ticker
                    config_used = out_base / "config_used.yaml"  # Config usado
                    if config_used.exists():
                        mlflow.log_artifact(str(config_used), artifact_path="config")  # Log config

                    for fname in [
                        f"{ticker}_model.pth",
                        f"{ticker}_predictions.csv",
                        f"{ticker}_summary.csv",
                        f"{ticker}_backtest.csv",
                        f"{ticker}_scaler.csv",
                        f"{ticker}_target_scaler.csv",
                        f"{ticker}_walkforward_folds.csv",
                        f"{ticker}_walkforward_predictions.csv",
                        f"{ticker}_walkforward_backtest.csv",
                        f"{ticker}_walkforward_metrics.png",
                    ]:
                        p = ticker_out / fname  # Path del artefacto
                        if p.exists():
                            # agrupamos por tipo para que sea navegable en la UI
                            if fname.endswith(".pth"):
                                artifact_path = "models"  # Carpeta modelos
                            elif "pred" in fname:
                                artifact_path = "predictions"  # Carpeta preds
                            elif "backtest" in fname:
                                artifact_path = "backtests"  # Carpeta backtests
                            elif "scaler" in fname:
                                artifact_path = "scalers"  # Carpeta scalers
                            elif "walkforward" in fname:
                                artifact_path = "walkforward"  # Carpeta WF
                            else:
                                artifact_path = "artifacts"  # Carpeta default
                            mlflow.log_artifact(str(p), artifact_path=artifact_path)  # Log artifact
            else:
                summary = run_e1_for_ticker(
                    config, ticker=ticker, raw_dir=raw_dir, out_dir=ticker_out, benchmark_df=benchmark_df  # Run sin MLflow
                )
            summaries.append(summary)  # Agregar summary
            print(f"✓ {ticker}: MAE={summary['ml_mae']:.4f} IC={summary['ml_ic']:.3f}\n")  # Log ticker OK
        except Exception as exc:
            print(f"✗ Error en {ticker}: {exc}\n")  # Log error

    pd.DataFrame(summaries).to_csv(out_base / "summary_all.csv", index=False)  # Guardar resumen global

    # Run agregado (opcional) para el summary de todos los tickers.
    if mlflow_enabled and mlflow is not None:
        timestamp = out_base.name  # Timestamp del run
        try:
            with mlflow.start_run(run_name=f"E1_Summary_{timestamp}"):
                summary_path = out_base / "summary_all.csv"  # Path summary
                if summary_path.exists():
                    mlflow.log_artifact(str(summary_path), artifact_path="reports")  # Log summary
                mlflow.log_metric("total_tickers", float(len(tickers)))  # Total tickers
                mlflow.log_metric("successful_tickers", float(len(summaries)))  # Tickers ok

        except Exception as exc:
            print(f"⚠️  No se pudo loguear el summary en MLflow: {exc}")  # Warning MLflow

    print(f"\n✓ Resultados guardados en {out_base}")  # Log final


if __name__ == "__main__":  # Entry-point script
    main()  # Ejecutar main
