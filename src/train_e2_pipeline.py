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
- Lookback: 60 días (vs 180 de E1)
- Horizon: 20 días (vs 90 de E1)
- Early stopping patience: 12 (vs 15 de E1)
- Max epochs: 150 (vs 200 de E1)
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
from functools import lru_cache
import copy

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .features.build_features_e2 import compute_e2_features, make_target_e2
from .features.build_sequences import make_sequences, time_split, temporal_train_val_split
from .models.e2_lstm import LSTMRegressor
from .backtest.daily import backtest_daily_signals, summarize_backtest
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
    tuned_path = (
        os.getenv("E2_TUNED_PARAMS_PATH", "").strip()
        or os.getenv("TUNED_PARAMS_PATH", "").strip()
    )
    if not tuned_path:
        return config

    all_tuned = _load_tuned_params(tuned_path)
    per_ticker = all_tuned.get(ticker)
    if not isinstance(per_ticker, dict):
        return config

    strat = config.setdefault("strategies", {}).setdefault(strategy_key, {})

    # Aplicar thresholds (tau_buy, tau_sell)
    if "tau_buy" in per_ticker or "tau_sell" in per_ticker:
        thresholds = strat.setdefault("thresholds", {})
        if "tau_buy" in per_ticker:
            thresholds["tau_buy"] = per_ticker["tau_buy"]
        if "tau_sell" in per_ticker:
            thresholds["tau_sell"] = per_ticker["tau_sell"]

    # Aplicar parámetros de modelo (lstm_units, dropout, learning_rate, batch_size)
    model = strat.setdefault("model", {})
    
    if "lstm_units_1" in per_ticker and "lstm_units_2" in per_ticker:
        model["lstm_units"] = [per_ticker["lstm_units_1"], per_ticker["lstm_units_2"]]
    
    if "dropout" in per_ticker:
        model["dropout"] = per_ticker["dropout"]
    
    if "learning_rate" in per_ticker:
        model["learning_rate"] = per_ticker["learning_rate"]
    
    if "batch_size" in per_ticker:
        model["batch_size"] = per_ticker["batch_size"]

    # Aplicar filtros (rsi14_min, rsi14_max)
    if "rsi14_min" in per_ticker or "rsi14_max" in per_ticker:
        filters = strat.setdefault("filters", {})
        if "rsi14_min" in per_ticker:
            filters["rsi14_min"] = per_ticker["rsi14_min"]
        if "rsi14_max" in per_ticker:
            filters["rsi14_max"] = per_ticker["rsi14_max"]
    
    # Aplicar parámetros walk-forward (n_folds, internal_val_fraction)
    if "n_folds" in per_ticker or "internal_val_fraction" in per_ticker:
        splits = config.setdefault("splits", {})
        if "n_folds" in per_ticker:
            splits["folds"] = int(per_ticker["n_folds"])
        if "internal_val_fraction" in per_ticker:
            splits["internal_val_fraction"] = float(per_ticker["internal_val_fraction"])

    return config


def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    """Carga CSV OHLCV y lo prepara."""
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing 'timestamp' column in {path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601', utc=True)
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")

    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")

    return df


def compute_information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula el Information Coefficient (correlación) de forma robusta."""
    if len(y_true) <= 1:
        return float("nan")

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def run_e2_walk_forward(
    *,
    X: np.ndarray,
    y: np.ndarray,
    ts: pd.DatetimeIndex,
    feat_names: list[str] | pd.Index,
    ticker: str,
    out_dir: Path,
    ohlcv: pd.DataFrame,
    splits_cfg: dict,
    lstm_units: list[int],
    dropout: float,
    dense_units: int,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    loss: str,
    huber_delta: float,
    tau_buy: float,
    tau_sell: float,
    round_trip_bps: float,
    holding_period: int,
    allow_short: bool,
    max_position: float,
    seed: int,
    lookback_days: int,
    horizon_days: int,
    rsi14_min: float | None = None,
    rsi14_max: float | None = None,
) -> dict:
    """Ejecuta validación walk-forward para la estrategia E2 (LSTM)."""

    require_model_save = os.getenv("REQUIRE_MODEL_SAVE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }

    ensure_dir(out_dir)

    n_samples = len(X)
    if n_samples == 0:
        raise ValueError("No hay muestras disponibles para walk-forward")

    folds = int(splits_cfg.get("folds", 5))
    if folds < 1:
        raise ValueError("'folds' debe ser >= 1 para walk-forward")

    embargo_cfg = splits_cfg.get("embargo_days", {})
    if isinstance(embargo_cfg, dict):
        embargo_days = int(embargo_cfg.get("e2", 0))
    else:
        embargo_days = int(embargo_cfg or 0)

    gap_samples = max(0, int(embargo_days))

    default_test_size = max(1, n_samples // (folds + 1))
    test_size = int(splits_cfg.get("test_size", default_test_size))
    if test_size <= gap_samples:
        test_size = gap_samples + 1

    splitter = TimeSeriesSplit(n_splits=folds, test_size=test_size, gap=gap_samples)

    fold_summaries: list[dict] = []
    pred_frames: list[pd.DataFrame] = []

    last_model_payload: dict | None = None
    last_torch = None

    root = project_root()

    def as_relative(path: Path) -> str:
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    raw_val_fraction = splits_cfg.get(
        "internal_val_fraction", splits_cfg.get("val_fraction", 0.15)
    )
    try:
        val_fraction_cfg = float(raw_val_fraction)
    except (TypeError, ValueError):
        val_fraction_cfg = 0.15

    if not 0 < val_fraction_cfg < 1:
        val_fraction_cfg = 0.15

    for fold_idx, (train_full_idx, test_idx) in enumerate(
        splitter.split(np.arange(n_samples)), start=1
    ):
        if len(test_idx) == 0 or len(train_full_idx) == 0:
            continue

        # Split interno train/val dentro del bloque de entrenamiento
        try:
            train_idx, val_idx = temporal_train_val_split(
                train_full_idx,
                val_fraction=val_fraction_cfg,
            )
        except ValueError:
            split_point = max(1, int(len(train_full_idx) * 0.8))
            train_idx = train_full_idx[:split_point]
            val_idx = train_full_idx[split_point:]

        X_train = X[train_idx]
        X_val = X[val_idx]
        X_test = X[test_idx]

        # Normalizar X por features (eje 0=samples, 1=timesteps, 2=features)
        mean_X = X_train.mean(axis=(0, 1))
        std_X = X_train.std(axis=(0, 1)) + 1e-8

        def scale_X(data: np.ndarray) -> np.ndarray:
            return ((data - mean_X) / std_X).astype(np.float32)

        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]

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

        model = LSTMRegressor(
            input_size=X_train.shape[-1],
            hidden_sizes=list(lstm_units),
            dropout=dropout,
            dense_units=dense_units,
            seed=seed,
        )

        res = model.fit(
            X_train_s,
            y_train_s,
            X_val_s,
            y_val_s,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            early_stopping_patience=patience,
            loss=loss,
            huber_delta=huber_delta,
        )

        y_pred_s = model.predict(X_test_s)
        y_pred = unscale_y(y_pred_s)

        last_torch = model.torch

        ts_test = ts[test_idx]
        window_label = f"{ts_test[0].date()} -> {ts_test[-1].date()}"

        # Guardar payload del último fold
        last_model_payload = {
            "ticker": ticker,
            "strategy": "e2_moderate",
            "created_at": datetime.utcnow().isoformat(),
            "model_class": "LSTMRegressor",
            "model_kwargs": {
                "input_size": int(X_train.shape[-1]),
                "hidden_sizes": list(lstm_units),
                "dropout": float(dropout),
                "dense_units": int(dense_units),
                "seed": int(seed),
            },
            "lookback_days": int(lookback_days),
            "horizon_days": int(horizon_days),
            "feature_names": list(feat_names),
            "scaler_X": {"mean": mean_X.tolist(), "std": std_X.tolist()},
            "scaler_y": {"mean": float(mean_y), "std": float(std_y)},
            "train_result": {"epochs_ran": int(res.epochs_ran), "best_val_loss": float(res.best_val_loss)},
            "walkforward": {
                "fold": int(fold_idx),
                "window": window_label,
                "test_size": int(test_size),
                "gap_samples": int(gap_samples),
            },
            "state_dict": {k: v.detach().cpu() for k, v in model.model.state_dict().items()},
        }

        mae = float(np.mean(np.abs(y_test - y_pred)))
        rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
        dir_acc = float(np.mean(np.sign(y_test) == np.sign(y_pred)))
        ic = compute_information_coefficient(y_test, y_pred)

        close_prices = ohlcv.loc[ts_test, "close"].to_numpy()
        bt = backtest_daily_signals(
            timestamps=ts_test,
            close_prices=close_prices,
            pred_returns=y_pred,
            tau_buy=tau_buy,
            tau_sell=tau_sell,
            round_trip_bps=round_trip_bps,
            holding_period_days=holding_period,
            allow_short=allow_short,
            max_position=max_position,
        )
        trading_metrics = summarize_backtest(bt)

        sharpe = trading_metrics.get("sharpe", float("nan"))

        ic_str = "nan" if np.isnan(ic) else f"{ic:.3f}"
        sharpe_str = "nan" if np.isnan(sharpe) else f"{sharpe:.2f}"
        print(
            f"    Fold {fold_idx}: {window_label} | MAE={mae:.4f} IC={ic_str} Sharpe={sharpe_str}"
        )

        bt.to_csv(out_dir / f"{ticker}_fold{fold_idx}_backtest.csv")

        fold_summaries.append(
            {
                "fold": fold_idx,
                "window": window_label,
                "test_start": ts_test[0],
                "test_end": ts_test[-1],
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_test": int(len(test_idx)),
                "epochs_ran": int(res.epochs_ran),
                "val_loss": float(res.best_val_loss),
                "ml_mae": mae,
                "ml_rmse": rmse,
                "ml_directional_accuracy": dir_acc,
                "ml_ic": ic,
                **{f"bt_{k}": float(v) for k, v in trading_metrics.items()},
            }
        )

        preds_df = pd.DataFrame(
            {
                "fold": fold_idx,
                "y_true": y_test,
                "y_pred": y_pred,
            },
            index=ts_test,
        )
        preds_df.index.name = "timestamp"
        pred_frames.append(preds_df)

    if not fold_summaries:
        raise ValueError("No se generaron folds válidos para walk-forward")

    fold_df = pd.DataFrame(fold_summaries)
    fold_df.to_csv(out_dir / f"{ticker}_walkforward_folds.csv", index=False)

    combined_preds = pd.concat(pred_frames).sort_index()
    combined_preds.to_csv(out_dir / f"{ticker}_walkforward_predictions.csv")

    y_true_all = combined_preds["y_true"].to_numpy()
    y_pred_all = combined_preds["y_pred"].to_numpy()

    mae_all = float(np.mean(np.abs(y_true_all - y_pred_all)))
    rmse_all = float(np.sqrt(np.mean((y_true_all - y_pred_all) ** 2)))
    dir_acc_all = float(np.mean(np.sign(y_true_all) == np.sign(y_pred_all)))
    ic_all = compute_information_coefficient(y_true_all, y_pred_all)

    bt_all = backtest_daily_signals(
        timestamps=pd.DatetimeIndex(combined_preds.index),
        close_prices=ohlcv.loc[combined_preds.index, "close"].to_numpy(),
        pred_returns=y_pred_all,
        tau_buy=tau_buy,
        tau_sell=tau_sell,
        round_trip_bps=round_trip_bps,
        holding_period_days=holding_period,
        allow_short=allow_short,
        max_position=max_position,
    )
    trading_metrics_all = summarize_backtest(bt_all)
    bt_all.to_csv(out_dir / f"{ticker}_walkforward_backtest.csv")

    # Guardar modelo del último fold
    if last_model_payload is not None and last_torch is not None:
        model_path = out_dir / f"{ticker}_model.pth"
        try:
            last_torch.save(last_model_payload, model_path)
        except Exception as exc:
            print(f"⚠️  No se pudo guardar el modelo walk-forward para {ticker}: {exc}")

        if require_model_save and not model_path.exists():
            raise RuntimeError(
                f"El pipeline terminó pero NO se guardó el modelo walk-forward en {model_path}. "
                "Si querés permitir continuar sin guardar, setear REQUIRE_MODEL_SAVE=0."
            )

    summary = {
        "ticker": ticker,
        "n_samples": int(n_samples),
        "n_test": int(len(combined_preds)),
        "folds": int(len(fold_df)),
        "lookback_days": int(lookback_days),
        "horizon_days": int(horizon_days),
        "split_method": "walk_forward",
        "walkforward_test_size": int(test_size),
        "walkforward_gap": int(gap_samples),
        "ml_mae": mae_all,
        "ml_rmse": rmse_all,
        "ml_directional_accuracy": dir_acc_all,
        "ml_ic": ic_all,
        **{f"bt_{k}": float(v) for k, v in trading_metrics_all.items()},
        "folds_file": as_relative(out_dir / f"{ticker}_walkforward_folds.csv"),
        "predictions_file": as_relative(out_dir / f"{ticker}_walkforward_predictions.csv"),
        "backtest_file": as_relative(out_dir / f"{ticker}_walkforward_backtest.csv"),
        # Target columns for reference (E2 defaults)
        "ic_target_min": 0.05,
        "sharpe_target_min": 1.0,
        "mae_target_max": 0.03,
        "rmse_target_max": 0.05,
    }

    return summary


def run_e2_for_ticker(
    config: dict,
    ticker: str,
    raw_dir: Path,
    out_dir: Path,
    benchmark_df: pd.DataFrame | None = None,
) -> dict:
    """Ejecuta pipeline E2 para un ticker."""
    config = copy.deepcopy(config)
    config = _apply_tuned_overrides(config=config, strategy_key="e2_moderate", ticker=ticker)
    ensure_dir(out_dir)

    require_model_save = os.getenv("REQUIRE_MODEL_SAVE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }

    # 1. Cargar datos
    raw_path = raw_dir / f"{ticker}_daily.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"No existe {raw_path}")

    df_raw = load_ohlcv_csv(raw_path)
    print(f"\n{'='*60}")
    print(f"Procesando {ticker} (E2 - Moderada)")
    print(f"{'='*60}")
    print(f"Período: {df_raw.index[0].date()} a {df_raw.index[-1].date()} ({len(df_raw)} días)")

    # 2. Calcular features E2
    feat_df = compute_e2_features(df_raw, benchmark_df=benchmark_df)

    # 3. Crear target
    e2_cfg = config.get("strategies", {}).get("e2_moderate", {})
    horizon_days = e2_cfg.get("horizon_days", 20)
    target_series = make_target_e2(df_raw, horizon_days=horizon_days)

    # Combinar
    df_full = feat_df.join(target_series, how="inner")
    df_full = df_full.dropna()

    if df_full.empty:
        raise ValueError(f"{ticker}: Sin datos después de dropna")

    print(f"Features calculadas: {len(feat_df.columns)}")
    print(f"Datos válidos: {len(df_full)}")

    # 4. Separar X, y
    feat_cols = [c for c in df_full.columns if c != target_series.name]
    lookback_days = e2_cfg.get("lookback_days", 60)
    
    print(f"Lookback: {lookback_days} días, Horizon: {horizon_days} días")

    # Adaptación para usar make_sequences de build_sequences.py
    # La firma retornada es: X, y, ts, feat_cols
    # PERO, make_sequences en train_e2_pipeline invocado recibe 5 args en el código actual: X_raw, y_raw, timestamps, lookback, dropna
    # Y build_sequences.make_sequences espera (features_df, target_series, lookback)
    
    # CORRECCIÓN: Usar build_sequences correctamente
    # Primero necesitamos DataFrame y Series
    # X_raw proviene de df_full[feat_cols]
    # y_raw proviene de df_full[target_series.name]
    
    X_seq, y_seq, ts_seq, _ = make_sequences(
        df_full[feat_cols], df_full[target_series.name], lookback=lookback_days
    )

    print(f"Secuencias: {len(X_seq)} (shape: {X_seq.shape})")

    # 6. Detectar método de split (walk-forward vs time_split)
    splits_cfg = config.get("splits", {})
    split_method = splits_cfg.get("method", "time_split")

    if split_method == "walk_forward":
        print(f"  ▶ Ejecutando walk-forward ({splits_cfg.get('folds', 5)} folds, test={splits_cfg.get('test_size', 'auto')})")
        
        summary = run_e2_walk_forward(
            X=X_seq,
            y=y_seq,
            ts=ts_seq,
            feat_names=feat_cols,
            ticker=ticker,
            out_dir=out_dir,
            ohlcv=df_raw,
            splits_cfg=splits_cfg,
            lstm_units=e2_cfg.get("model", {}).get("lstm_units", [128, 64]),
            dropout=e2_cfg.get("model", {}).get("dropout", 0.2),
            dense_units=e2_cfg.get("model", {}).get("dense_units", 32),
            learning_rate=e2_cfg.get("model", {}).get("learning_rate", 1e-3),
            batch_size=e2_cfg.get("model", {}).get("batch_size", 64),
            max_epochs=e2_cfg.get("model", {}).get("max_epochs", 150),
            patience=e2_cfg.get("model", {}).get("early_stopping_patience", 12),
            loss=e2_cfg.get("model", {}).get("loss", "huber"),
            huber_delta=e2_cfg.get("model", {}).get("huber_delta", 1.0),
            tau_buy=e2_cfg.get("thresholds", {}).get("tau_buy", 0.02),
            tau_sell=e2_cfg.get("thresholds", {}).get("tau_sell", 0.0),
            round_trip_bps=config.get("backtest", {}).get("round_trip_bps", 10),
            holding_period=horizon_days,
            allow_short=e2_cfg.get("backtest", {}).get("allow_short", True),
            max_position=e2_cfg.get("backtest", {}).get("max_position", 1.0),
            seed=config.get("project", {}).get("seed", 42),
            lookback_days=lookback_days,
            horizon_days=horizon_days,
            rsi14_min=e2_cfg.get("filters", {}).get("rsi14_min"),
            rsi14_max=e2_cfg.get("filters", {}).get("rsi14_max"),
        )
        
        return summary
    
    # 6. Split temporal simple (fallback)
    train_ratio = 0.7 
    val_ratio = 0.15
    
    idx_train, idx_val, idx_test = time_split(
        len(X_seq), train_frac=train_ratio, val_frac=val_ratio
    )

    X_train, y_train, ts_train = X_seq[idx_train], y_seq[idx_train], ts_seq[idx_train]
    X_val, y_val, ts_val = X_seq[idx_val], y_seq[idx_val], ts_seq[idx_val]
    X_test, y_test, ts_test = X_seq[idx_test], y_seq[idx_test], ts_seq[idx_test]

    print(f"\nSplit temporal:")
    print(f"  Train: {len(X_train)} ({ts_train[0]} → {ts_train[-1]})")
    print(f"  Val:   {len(X_val)} ({ts_val[0]} → {ts_val[-1]})")
    print(f"  Test:  {len(X_test)} ({ts_test[0]} → {ts_test[-1]})")

    # 7. Estandarizar features (solo sobre train)
    mean_X = X_train.mean(axis=(0, 1))
    std_X = X_train.std(axis=(0, 1)) + 1e-8

    X_train_norm = (X_train - mean_X) / std_X
    X_val_norm = (X_val - mean_X) / std_X
    X_test_norm = (X_test - mean_X) / std_X

    # Estandarizar target
    mean_y = y_train.mean()
    std_y = y_train.std() + 1e-8

    y_train_norm = (y_train - mean_y) / std_y
    y_val_norm = (y_val - mean_y) / std_y

    print(f"\nTarget statistics (train):")
    print(f"  Mean: {mean_y:.4f}, Std: {std_y:.4f}")

    # 8. Entrenar LSTM
    print("\nEntrenando modelo LSTM...")
    model_cfg = e2_cfg.get("model", {}) # Usa config correcta

    model = LSTMRegressor(
        input_size=X_train_norm.shape[2],
        hidden_sizes=model_cfg.get("lstm_units", [128, 64]), # Use lstm_units from yaml
        dropout=model_cfg.get("dropout", 0.2),
        dense_units=model_cfg.get("dense_units", 32),
        seed=config.get("project", {}).get("seed", 42), # Use project seed
    )

    res = model.fit(
        X_train_norm,
        y_train_norm,
        X_val_norm,
        y_val_norm,
        learning_rate=model_cfg.get("learning_rate", 1e-3),
        batch_size=model_cfg.get("batch_size", 64),
        max_epochs=model_cfg.get("max_epochs", 150),
        early_stopping_patience=model_cfg.get("early_stopping_patience", 12),
        loss=model_cfg.get("loss", "huber"),
        huber_delta=model_cfg.get("huber_delta", 1.0),
    )

    print(f"✓ Entrenamiento completado:")
    print(f"  Epochs: {res.epochs_ran}")
    print(f"  Val loss: {res.best_val_loss:.6f}")

    # 8.5 Guardar modelo entrenado (por ticker)
    # Guardamos state_dict + metadata mínima para inferencia/reproducibilidad.
    model_path = out_dir / f"{ticker}_model.pth"
    try:
        torch = model.torch
        payload = {
            "ticker": ticker,
            "strategy": "e2_moderate",
            "created_at": datetime.utcnow().isoformat(),
            "model_class": "LSTMRegressor",
            "model_kwargs": {
                "input_size": int(X_train_norm.shape[2]),
                "hidden_sizes": list(model_cfg.get("lstm_units", [128, 64])),
                "dropout": float(model_cfg.get("dropout", 0.2)),
                "dense_units": int(model_cfg.get("dense_units", 32)),
                "seed": int(config.get("project", {}).get("seed", 42)),
            },
            "lookback_days": int(lookback_days),
            "horizon_days": int(horizon_days),
            "feature_names": list(feat_cols),
            "scaler_X": {"mean": mean_X.tolist(), "std": std_X.tolist()},
            "scaler_y": {"mean": float(mean_y), "std": float(std_y)},
            "train_result": {"epochs_ran": int(res.epochs_ran), "best_val_loss": float(res.best_val_loss)},
            "state_dict": {k: v.detach().cpu() for k, v in model.model.state_dict().items()},
        }
        torch.save(payload, model_path)
        print(f"✓ Modelo guardado: {model_path}")
    except Exception as exc:
        print(f"⚠️  No se pudo guardar el modelo para {ticker}: {exc}")

    if require_model_save and not model_path.exists():
        raise RuntimeError(
            f"El pipeline terminó pero NO se guardó el modelo en {model_path}. "
            "Revisar el bloque de guardado y permisos. "
            "Si querés permitir continuar sin guardar, setear REQUIRE_MODEL_SAVE=0."
        )

    # 9. Predicciones en test (desnormalizar)
    y_pred_norm = model.predict(X_test_norm)
    y_pred = y_pred_norm * std_y + mean_y

    # 10. Métricas ML
    mae = np.abs(y_test - y_pred).mean()
    rmse = np.sqrt(((y_test - y_pred) ** 2).mean())
    ic = compute_information_coefficient(y_test, y_pred)

    # Directional accuracy
    dir_acc = float(np.mean((np.sign(y_test) == np.sign(y_pred))))

    ml_metrics = {
        "mae": mae,
        "rmse": rmse,
        "ic": ic,
        "directional_accuracy": dir_acc,
    }

    print(f"\nMétricas ML (test):")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  IC:   {ic:.3f}")
    print(f"  Directional Accuracy: {dir_acc:.2%}")

    # 11. Backtest simple
    pred_df = pd.DataFrame(
        {
            "timestamp": ts_test,
            "y_true": y_test,
            "y_pred": y_pred,
        }
    )
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
    pred_df = pred_df.set_index("timestamp")

    # Obtener precios de test
    df_test = df_raw.loc[pred_df.index]

    # Señales básicas E2 (Vectorizado simplificado para reporte rápido)
    # Nota: Para backtest riguroso con reglas complejas (TP/SL/TimeStop/Filtros),
    # se debería usar el módulo src.backtest con loop evento a evento.
    # Aquí hacemos una aproximación vectorizada para validar el modelo.
    
    tau_buy = e2_cfg.get("thresholds", {}).get("tau_buy", 0.025)
    tau_sell = e2_cfg.get("thresholds", {}).get("tau_sell", 0.00)

    signals = pd.Series(0, index=pred_df.index)
    signals[pred_df["y_pred"] > tau_buy] = 1   # Long
    signals[pred_df["y_pred"] < tau_sell] = -1  # Short/Exit
    
    # Aplicar filtros simples vectores si existen columnas (RSI, MACD) en df_test
    # Esto es una aproximación de rules_e2.py
    if "rsi_14" in df_test.columns:
         rsi_mask = (df_test["rsi_14"] >= 35) & (df_test["rsi_14"] <= 70)
         # Solo permitimos entrada (1) si RSI ok. Salidas (-1) siempre permitidas.
         signals[(signals == 1) & (~rsi_mask)] = 0
         
    if "macd_hist" in df_test.columns:
        macd_mask = df_test["macd_hist"] > 0
        signals[(signals == 1) & (~macd_mask)] = 0

    # backtest_daily_signals necesita timestamps, close, pred_returns, tau_buy, tau_sell
    # Usar wrapper simplificado
    bt_results = backtest_daily_signals(
        timestamps=pd.DatetimeIndex(pred_df.index),
        close_prices=df_test["close"].to_numpy(),
        pred_returns=pred_df["y_pred"].to_numpy(),
        tau_buy=tau_buy,
        tau_sell=tau_sell,
    )
    trading_metrics = summarize_backtest(bt_results)

    print(f"\nMétricas Trading (test):")
    print(f"  Total Return: {trading_metrics['total_return']:.2%}")
    print(f"  Sharpe:       {trading_metrics['sharpe']:.2f}")
    print(f"  Max Drawdown: {trading_metrics['max_drawdown']:.2%}")

    # 12. Guardar resultados
    pred_df.to_csv(out_dir / f"{ticker}_predictions.csv")

    # Guardar scaler de features
    feat_names = feat_cols
    scaler_X_df = pd.DataFrame({"mean": mean_X, "std": std_X}, index=feat_names)
    scaler_X_df.to_csv(out_dir / f"{ticker}_scaler.csv")

    # Guardar scaler de target
    scaler_y_df = pd.DataFrame({
        "mean_y": [mean_y],
        "std_y": [std_y]
    })
    scaler_y_df.to_csv(out_dir / f"{ticker}_target_scaler.csv", index=False)

    meta = {
        "ticker": ticker,
        "n_samples": int(len(X_seq)),
        "n_test": int(len(X_test)),
        "lookback_days": lookback_days,
        "horizon_days": horizon_days,
        "split_method": splits_cfg.get("method", "time_split"),
        "epochs_ran": res.epochs_ran,
        "val_loss": res.best_val_loss,
    }

    summary = {
        **meta,
        **{f"ml_{k}": v for k, v in ml_metrics.items()},
        **{f"bt_{k}": v for k, v in trading_metrics.items()},
    }
    pd.Series(summary).to_csv(out_dir / f"{ticker}_summary.csv")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline E2 (LSTM moderado)")
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/base.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated tickers (or uses universe.tickers_by_strategy.e2_moderate)",
    )
    args = parser.parse_args()

    root = project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path

    config = load_yaml(cfg_path)

    # Tickers E2
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        tickers = list(
            config.get("universe", {})
            .get("tickers_by_strategy", {})
            .get("e2_moderate", [])
        )

    if not tickers:
        raise ValueError("No tickers for E2")

    # Benchmark
    benchmark = config.get("universe", {}).get("benchmark", "SPY")
    raw_dir = root / "data" / "raw" / "daily"

    benchmark_path = raw_dir / f"{benchmark}_daily.csv"
    if benchmark_path.exists():
        benchmark_df = load_ohlcv_csv(benchmark_path)
    else:
        print(f"⚠️  Benchmark {benchmark} no encontrado, usando valores vacíos")
        benchmark_df = None

    out_base = root / "runs" / "e2_moderate" / datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_dir(out_base)

    # Guardar config usado
    import shutil
    shutil.copy(cfg_path, out_base / "config_used.yaml")

    # MLflow (opcional): si está instalado y hay tracking URI, logueamos params/metrics/artifacts.
    mlflow_enabled = False
    mlflow = None
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "E2_Moderate")
    if tracking_uri:
        try:
            import mlflow as _mlflow  # type: ignore

            _mlflow.set_tracking_uri(tracking_uri)
            _mlflow.set_experiment(experiment_name)
            mlflow = _mlflow
            mlflow_enabled = True
            print(f"✓ MLflow habilitado: {tracking_uri} (experiment={experiment_name})")
        except Exception as exc:
            print(f"⚠️  MLflow no disponible, continuando sin tracking: {exc}")
            mlflow_enabled = False

    summaries: list[dict] = []
    for ticker in tickers:
        ticker_out = out_base / ticker
        try:
            if mlflow_enabled and mlflow is not None:
                timestamp = out_base.name
                with mlflow.start_run(run_name=f"E2_{ticker}_{timestamp}"):
                    mlflow.log_param("strategy", "e2_moderate")
                    mlflow.log_param("ticker", ticker)
                    mlflow.log_param("model_type", "LSTM")
                    mlflow.log_param("timestamp", timestamp)

                    e2_cfg = config.get("strategies", {}).get("e2_moderate", {})
                    model_cfg = e2_cfg.get("model", {})
                    mlflow.log_params(
                        {
                            "lookback_days": int(e2_cfg.get("lookback_days", 60)),
                            "horizon_days": int(e2_cfg.get("horizon_days", 20)),
                            "lstm_units": str(model_cfg.get("lstm_units", [128, 64])),
                            "dropout": float(model_cfg.get("dropout", 0.2)),
                            "dense_units": int(model_cfg.get("dense_units", 32)),
                            "learning_rate": float(model_cfg.get("learning_rate", 1e-3)),
                            "batch_size": int(model_cfg.get("batch_size", 64)),
                            "max_epochs": int(model_cfg.get("max_epochs", 150)),
                            "early_stopping_patience": int(model_cfg.get("early_stopping_patience", 12)),
                            "loss": str(model_cfg.get("loss", "huber")),
                            "huber_delta": float(model_cfg.get("huber_delta", 1.0)),
                            "split_method": str(config.get("splits", {}).get("method", "time_split")),
                            "seed": int(config.get("project", {}).get("seed", 42)),
                        }
                    )

                    summary = run_e2_for_ticker(
                        config, ticker=ticker, raw_dir=raw_dir, out_dir=ticker_out, benchmark_df=benchmark_df
                    )

                    metrics: dict[str, float] = {}
                    for k, v in summary.items():
                        if not (k.startswith("ml_") or k.startswith("bt_")):
                            continue
                        if isinstance(v, (int, float)):
                            metrics[k] = float(v)
                    if metrics:
                        mlflow.log_metrics(metrics)

                    config_used = out_base / "config_used.yaml"
                    if config_used.exists():
                        mlflow.log_artifact(str(config_used), artifact_path="config")

                    for fname in [
                        f"{ticker}_model.pth",
                        f"{ticker}_predictions.csv",
                        f"{ticker}_summary.csv",
                        f"{ticker}_scaler.csv",
                        f"{ticker}_target_scaler.csv",
                    ]:
                        p = ticker_out / fname
                        if p.exists():
                            if fname.endswith(".pth"):
                                artifact_path = "models"
                            elif "pred" in fname:
                                artifact_path = "predictions"
                            elif "scaler" in fname:
                                artifact_path = "scalers"
                            else:
                                artifact_path = "artifacts"
                            mlflow.log_artifact(str(p), artifact_path=artifact_path)
            else:
                summary = run_e2_for_ticker(
                    config, ticker=ticker, raw_dir=raw_dir, out_dir=ticker_out, benchmark_df=benchmark_df
                )
            summaries.append(summary)
            print(f"✓ {ticker}: MAE={summary['ml_mae']:.4f} IC={summary['ml_ic']:.3f}\n")
        except Exception as exc:
            print(f"✗ Error en {ticker}: {exc}\n")

    pd.DataFrame(summaries).to_csv(out_base / "summary_all.csv", index=False)

    # Run agregado (opcional) para el summary de todos los tickers.
    if mlflow_enabled and mlflow is not None:
        timestamp = out_base.name
        try:
            with mlflow.start_run(run_name=f"E2_Summary_{timestamp}"):
                summary_path = out_base / "summary_all.csv"
                if summary_path.exists():
                    mlflow.log_artifact(str(summary_path), artifact_path="reports")
                mlflow.log_metric("total_tickers", float(len(tickers)))
                mlflow.log_metric("successful_tickers", float(len(summaries)))
        except Exception as exc:
            print(f"⚠️  No se pudo loguear el summary en MLflow: {exc}")

    print(f"\n✓ Resultados guardados en {out_base}")


if __name__ == "__main__":
    main()
