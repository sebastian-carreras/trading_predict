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
import socket  # Reachability check MLflow
from datetime import datetime, timezone  # Timestamps
from pathlib import Path  # Paths
from urllib.parse import urlparse  # Parse tracking URI
import time

import numpy as np  # NumPy
import pandas as pd  # Pandas
from sklearn.model_selection import TimeSeriesSplit  # Walk-forward validation

try:
    from dotenv import load_dotenv
    load_dotenv()  # Cargar .env (MLFLOW_TRACKING_URI, MinIO/S3, etc.) en runs por consola
except ImportError:
    pass

from ..backtest.backtest_intraday import (  # Backtest intraday
    backtest_intraday_signals,  # Simula ejecución de órdenes
    compute_max_drawdown,        # Calcula máxima caída del equity
    compute_profit_factor,       # Calcula profit factor (ganancias/pérdidas)
)  # Fin import backtest
from .intraday_data import load_ohlcv_csv  # Data intraday
from .build_features import compute_intraday_features, make_sequences, make_target_return  # Features/target
from .lstm import LSTMRegressor  # Modelo LSTM
from .intraday_metrics import directional_accuracy, information_coefficient, mae, rmse  # Métricas
from ..utils import (
    apply_training_window,
    ensure_dir,
    get_nested,
    load_yaml,
    log_timing_event,
    project_root,
    resolve_lifecycle_paths,
)  # Utils


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


def run_e3_walk_forward(  # Walk-forward completo para E3
    *,  # Solo kwargs
    X: np.ndarray,
    y: np.ndarray,
    ts: pd.DatetimeIndex,
    feat_names: list,
    ticker: str,
    out_dir: Path,
    ohlcv: pd.DataFrame,
    splits_cfg: dict,
    ensemble_members: int,
    hidden_size: int,
    num_layers: int,
    dense_units,
    dropout: float,
    lr: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    loss: str,
    huber_delta: float,
    tau_buy: float,
    tau_sell: float,
    round_trip_bps: float,
    execution_delay_bars: int,
    allow_short: bool,
    max_position: float,
    seed: int,
    horizon_bars: int,
    consensus_tol,
) -> dict:
    """Walk-forward validation para E3 (LSTM ensemble intraday).

    Replica la metodología de E1/E2: folds temporales con embargo mínimo = horizon_bars.
    Por cada fold entrena un ensemble de LSTM y aplica el filtro de consenso configurado.
    Retorna un summary con prefijos ml_/bt_ alineados con E1/E2.
    """
    ensure_dir(out_dir)
    n_samples = len(X)
    if n_samples == 0:
        raise ValueError("No hay muestras disponibles para walk-forward E3")

    folds = int(splits_cfg.get("folds", 3))  # Default 3 (menos datos que diaria)
    if folds < 1:
        raise ValueError("'folds' debe ser >= 1")

    # Embargo en barras (análogo a días en E1/E2): mínimo = horizon_bars para evitar leakage.
    # La clave embargo_days.e3 reutiliza el nombre del campo global; para E3 la unidad es barras.
    embargo_cfg = splits_cfg.get("embargo_days", {})
    if isinstance(embargo_cfg, dict):
        gap_samples = int(embargo_cfg.get("e3", horizon_bars))
    else:
        gap_samples = int(embargo_cfg or horizon_bars)
    gap_samples = max(gap_samples, horizon_bars)  # Garantizar mínimo = horizon

    default_test_size = max(1, n_samples // (folds + 1))
    test_size = int(splits_cfg.get("test_size", default_test_size))
    if test_size <= gap_samples:
        test_size = gap_samples + 1

    splitter = TimeSeriesSplit(n_splits=folds, test_size=test_size, gap=gap_samples)

    raw_val_frac = splits_cfg.get("internal_val_fraction", 0.15)
    try:
        val_fraction = float(raw_val_frac)
    except (TypeError, ValueError):
        val_fraction = 0.15
    if not 0 < val_fraction < 1:
        val_fraction = 0.15

    _bars_per_year = 252 * 78  # ~19,656 barras de 5min por año (factor de anualización)

    fold_summaries: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    total_train_seconds = 0.0
    total_predict_seconds = 0.0

    # Tracking para guardar el último fold como artifact del modelo
    last_fold_models: list = []
    last_fold_mean: np.ndarray | None = None
    last_fold_std: np.ndarray | None = None

    for fold_idx, (train_full_idx, test_idx) in enumerate(
        splitter.split(np.arange(n_samples)), start=1,
    ):
        if len(test_idx) == 0 or len(train_full_idx) == 0:
            continue

        # Split interno train/val manteniendo orden temporal
        sp = max(1, int(len(train_full_idx) * (1.0 - val_fraction)))
        train_idx = train_full_idx[:sp]
        val_idx = train_full_idx[sp:]

        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
        ts_test = ts[test_idx]
        ts_train = ts[train_idx]  # Timestamps de train (para log de ventana completa)

        # Z-score scaling: estadísticas solo del train de este fold (sin leakage)
        Xtr2d = X_train.reshape(-1, X_train.shape[-1])
        fold_mean = Xtr2d.mean(axis=0)
        fold_std = Xtr2d.std(axis=0) + 1e-12

        def _scale_fold(Xa: np.ndarray) -> np.ndarray:  # noqa: E306
            return ((Xa - fold_mean) / fold_std).astype(np.float32)

        X_train_s = _scale_fold(X_train)
        X_val_s = _scale_fold(X_val)
        X_test_s = _scale_fold(X_test)

        # Entrenar ensemble para este fold
        preds_members_fold: list[np.ndarray] = []
        val_losses_fold: list[float] = []
        epochs_ran_fold: list[int] = []  # Épocas corridas por miembro en este fold
        fold_model_objects: list = []  # Referencia a modelos entrenados de este fold

        # Log de fold: train Y test explícitos para evidenciar la ventana CRECIENTE
        # (expanding). Ver solo el test aparenta ventana deslizante. Unidad = barras 5-min.
        fold_pct = 100.0 * fold_idx / folds
        print(
            f"\n  Fold {fold_idx}/{folds} ({fold_pct:5.1f}%): "
            f"train[{ts_train[0].date()} -> {ts_train[-1].date()}] n={len(train_idx)} (expanding) | "
            f"test[{ts_test[0].date()} -> {ts_test[-1].date()}] n={len(test_idx)} | "
            f"embargo={gap_samples} barras"
        )

        for m in range(ensemble_members):
            member_seed = seed + 1000 * m + 100 * fold_idx  # Seed único por miembro y fold
            model_m = LSTMRegressor(
                input_size=X_train_s.shape[-1],
                hidden_size=hidden_size,
                num_layers=num_layers,
                dense_units=dense_units,
                dropout=dropout,
                seed=member_seed,
            )

            train_started_at = datetime.now(timezone.utc).isoformat()
            t0 = time.perf_counter()
            res_m = model_m.fit(
                X_train_s, y_train, X_val_s, y_val,
                learning_rate=lr, batch_size=batch_size, max_epochs=max_epochs,
                early_stopping_patience=patience, loss=loss, huber_delta=huber_delta,
                verbose=False,
            )
            t1 = time.perf_counter()
            train_ended_at = datetime.now(timezone.utc).isoformat()
            total_train_seconds += t1 - t0

            log_timing_event(
                strategy="e3_intraday",
                phase="train",
                duration_seconds=t1 - t0,
                started_at=train_started_at,
                ended_at=train_ended_at,
                ticker=ticker,
                run_dir=out_dir,
                extra={
                    "member": int(m + 1),
                    "split": "walk_forward",
                    "fold": int(fold_idx),
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                },
            )

            val_losses_fold.append(float(res_m.best_val_loss))
            epochs_ran_fold.append(int(res_m.epochs_ran))
            print(
                f"    Miembro {m + 1}/{ensemble_members}: "
                f"epochs={res_m.epochs_ran} val_loss={res_m.best_val_loss:.6f}"
            )

            pred_started_at = datetime.now(timezone.utc).isoformat()
            t0 = time.perf_counter()
            preds_members_fold.append(model_m.predict(X_test_s))
            t1 = time.perf_counter()
            pred_ended_at = datetime.now(timezone.utc).isoformat()
            total_predict_seconds += t1 - t0

            log_timing_event(
                strategy="e3_intraday",
                phase="predict",
                duration_seconds=t1 - t0,
                started_at=pred_started_at,
                ended_at=pred_ended_at,
                ticker=ticker,
                run_dir=out_dir,
                extra={
                    "member": int(m + 1),
                    "split": "walk_forward",
                    "fold": int(fold_idx),
                    "n_test": int(len(test_idx)),
                },
            )
            fold_model_objects.append(model_m)  # Guardar referencia al modelo entrenado

        # Actualizar tracking del último fold (se sobreescribe en cada fold)
        last_fold_models = fold_model_objects
        last_fold_mean = fold_mean
        last_fold_std = fold_std

        # Combinar predicciones del ensemble para este fold
        stacked_fold = np.stack(preds_members_fold, axis=0)  # (n_members, n_test)
        pred_mean_fold = np.mean(stacked_fold, axis=0)
        pred_std_fold = np.std(stacked_fold, axis=0)

        # Filtro de consenso: suprimir señales donde el ensemble no concuerda
        y_pred_fold = pred_mean_fold.copy()
        if consensus_tol is not None and consensus_tol > 0:
            y_pred_fold[pred_std_fold >= consensus_tol] = 0.0

        # Backtest para este fold
        close_series = ohlcv["close"].astype("float64")
        log_close = pd.Series(np.log(close_series.to_numpy()), index=close_series.index)
        bar_ret_fold = log_close.diff().reindex(ts_test).fillna(0.0).to_numpy(dtype=np.float32)

        bt_fold = backtest_intraday_signals(
            timestamps=ts_test,
            bar_returns=bar_ret_fold,
            pred_forward_returns=y_pred_fold.astype(np.float32),
            tau_buy=tau_buy, tau_sell=tau_sell, round_trip_bps=round_trip_bps,
            execution_delay_bars=execution_delay_bars,
            allow_short=allow_short, max_position=max_position,
        )

        # Métricas ML del fold
        ic_fold = (
            float(np.corrcoef(y_test, pred_mean_fold)[0, 1])
            if float(np.std(pred_mean_fold)) > 1e-12 else float("nan")
        )
        mae_fold = float(np.mean(np.abs(y_test - pred_mean_fold)))
        dir_acc_fold = float(np.mean(np.sign(y_test) == np.sign(pred_mean_fold)))

        # Métricas trading del fold
        net_ret_fold = bt_fold["net_ret"].to_numpy(dtype=np.float32)
        sharpe_fold = float(np.mean(net_ret_fold) / (np.std(net_ret_fold) + 1e-12) * np.sqrt(_bars_per_year))
        max_dd_fold = compute_max_drawdown(bt_fold["equity"].to_numpy(dtype=np.float32))
        calmar_fold = float(np.mean(net_ret_fold)) * _bars_per_year / (max_dd_fold + 1e-12)
        pf_fold = compute_profit_factor(net_ret_fold)

        print(f"    → IC={ic_fold:.3f} Sharpe={sharpe_fold:.2f} PF={pf_fold:.2f} DD={max_dd_fold:.3%}")

        bt_fold.to_csv(out_dir / f"{ticker}_fold{fold_idx}_backtest.csv")

        fold_summaries.append({
            "fold": fold_idx,
            "test_start": str(ts_test[0]) if len(ts_test) > 0 else "",
            "test_end": str(ts_test[-1]) if len(ts_test) > 0 else "",
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "val_loss_mean": float(np.mean(val_losses_fold)),
            "epochs_ran_mean": float(np.mean(epochs_ran_fold)),
            "ml_mae": mae_fold,
            "ml_ic": ic_fold,
            "ml_directional_accuracy": dir_acc_fold,
            "bt_sharpe": sharpe_fold,
            "bt_max_drawdown": max_dd_fold,
            "bt_calmar": calmar_fold,
            "bt_profit_factor": pf_fold,
            "bt_num_trades": int((bt_fold["turnover"] > 0).sum()),
        })

        pred_frames.append(
            pd.DataFrame(
                {"fold": fold_idx, "y_true": y_test, "y_pred": pred_mean_fold, "pred_std": pred_std_fold},
                index=ts_test,
            ).rename_axis("timestamp")
        )

    if not fold_summaries:
        raise ValueError("No se generaron folds válidos para walk-forward E3")

    # ---------- Consolidación ----------
    fold_df = pd.DataFrame(fold_summaries)
    fold_df.to_csv(out_dir / f"{ticker}_walkforward_folds.csv", index=False)

    combined = pd.concat(pred_frames).sort_index()
    combined.to_csv(out_dir / f"{ticker}_walkforward_predictions.csv")

    y_true_all = combined["y_true"].to_numpy()
    y_pred_all = combined["y_pred"].to_numpy()
    pred_std_all = combined["pred_std"].to_numpy()

    ic_all = (
        float(np.corrcoef(y_true_all, y_pred_all)[0, 1])
        if float(np.std(y_pred_all)) > 1e-12 else float("nan")
    )
    mae_all = float(np.mean(np.abs(y_true_all - y_pred_all)))
    rmse_all = float(rmse(y_true_all, y_pred_all))
    dir_acc_all = float(np.mean(np.sign(y_true_all) == np.sign(y_pred_all)))

    # Aplicar filtro de consenso en la serie completa para el backtest final
    y_pred_bt = y_pred_all.copy()
    if consensus_tol is not None and consensus_tol > 0:
        y_pred_bt[pred_std_all >= consensus_tol] = 0.0

    close_series = ohlcv["close"].astype("float64")
    log_close = pd.Series(np.log(close_series.to_numpy()), index=close_series.index)
    bar_ret_all = log_close.diff().reindex(combined.index).fillna(0.0).to_numpy(dtype=np.float32)

    bt_all = backtest_intraday_signals(
        timestamps=pd.DatetimeIndex(combined.index),
        bar_returns=bar_ret_all,
        pred_forward_returns=y_pred_bt.astype(np.float32),
        tau_buy=tau_buy, tau_sell=tau_sell, round_trip_bps=round_trip_bps,
        execution_delay_bars=execution_delay_bars,
        allow_short=allow_short, max_position=max_position,
    )
    bt_all.to_csv(out_dir / f"{ticker}_walkforward_backtest.csv")

    net_ret_all = bt_all["net_ret"].to_numpy(dtype=np.float32)
    sharpe_all = float(np.mean(net_ret_all) / (np.std(net_ret_all) + 1e-12) * np.sqrt(_bars_per_year))
    max_dd_all = compute_max_drawdown(bt_all["equity"].to_numpy(dtype=np.float32))
    calmar_all = float(np.mean(net_ret_all)) * _bars_per_year / (max_dd_all + 1e-12)
    pf_all = compute_profit_factor(net_ret_all)

    summary: dict = {
        "ticker": ticker,
        "n_samples": int(n_samples),
        "n_test": int(len(combined)),
        "folds": int(len(fold_df)),
        "lookback_bars": int(X.shape[1]),
        "horizon_bars": int(horizon_bars),
        "tau_buy": float(tau_buy),
        "tau_sell": float(tau_sell),
        "round_trip_bps": float(round_trip_bps),
        "ensemble_members": int(ensemble_members),
        "walkforward_gap_bars": int(gap_samples),
        "consensus_tol": float(consensus_tol) if consensus_tol is not None else None,
        "split_method": "walk_forward",
        "feature_count": int(X.shape[-1]),
        "ml_mae": mae_all,
        "ml_rmse": rmse_all,
        "ml_ic": ic_all,
        "ml_directional_accuracy": dir_acc_all,
        "bt_sharpe": sharpe_all,
        "bt_max_drawdown": max_dd_all,
        "bt_calmar": calmar_all,
        "bt_profit_factor": pf_all,
        "bt_num_trades": int((bt_all["turnover"] > 0).sum()),
        "bt_turnover_mean": float(bt_all["turnover"].mean()),
        "bt_time_in_market": float((bt_all["pos"].abs() > 0).mean()),
        "val_loss": float(fold_df["val_loss_mean"].mean()),
        "epochs_ran_mean": float(fold_df["epochs_ran_mean"].mean()),
        "timing_train_seconds": round(total_train_seconds, 2),
        "timing_predict_seconds": round(total_predict_seconds, 4),
    }

    pd.Series({k: v for k, v in summary.items() if v is not None}).to_csv(
        out_dir / f"{ticker}_summary.csv"
    )

    # Guardar ensemble del último fold como artifact del modelo
    model_path: Path | None = None
    if last_fold_models:
        try:
            _torch = last_fold_models[0].torch
            payload = {
                "ensemble_state_dicts": [m.model.state_dict() for m in last_fold_models],
                "fold_mean": last_fold_mean,
                "fold_std": last_fold_std,
                "input_size": int(X.shape[-1]),
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dense_units": dense_units,
                "dropout": dropout,
                "horizon_bars": horizon_bars,
                "lookback_bars": int(X.shape[1]),
                "feat_names": list(feat_names),
                "ticker": ticker,
            }
            model_path = out_dir / f"{ticker}_model.pth"
            _torch.save(payload, model_path)
            print(f"  ✓ Modelo guardado: {model_path.name} ({len(last_fold_models)} miembros)")
        except Exception as exc:
            print(f"  ⚠️  No se pudo guardar el modelo walk-forward para {ticker}: {exc}")

    if model_path is not None:
        summary["model_file"] = str(model_path.relative_to(out_dir.parents[2]))

    return summary


def _register_e3_candidate(
    *,
    config: dict,
    summary: dict,
    ticker: str,
    out_dir: Path,
    feat_names: list,
    auto_promote: bool,
) -> None:
    """Registra el modelo E3 entrenado en el lifecycle registry (análogo a E1/E2)."""
    root = project_root()
    try:
        from ..lifecycle.registry import ModelRegistry
        from ..lifecycle.guardrails import validate_candidate, log_candidate_metrics

        registry_path, metrics_log_path = resolve_lifecycle_paths(config, root=root)
        log_candidate_metrics(
            metrics=summary, strategy="e3", ticker=ticker,
            run_dir=out_dir, variant="e3_intraday",
            log_path=metrics_log_path,
            feature_names=list(feat_names),
        )
        _guardrail_cfg = config.get("lifecycle", {}).get("guardrails", {})
        _min_sharpe = float(
            _guardrail_cfg.get("min_sharpe_by_strategy", {}).get("e3_intraday", 0.0)
        )
        passed, errors = validate_candidate(
            run_dir=out_dir, ticker=ticker, metrics=summary, min_sharpe=_min_sharpe
        )
        if passed:
            registry = ModelRegistry(registry_path)
            registry.register_candidate(
                strategy="e3", ticker=ticker,
                run_dir=str(out_dir.resolve().relative_to(root)),
                metrics=summary, variant="e3_intraday",
                feature_names=list(feat_names),
            )
            print(f"  ✓ {ticker} registrado como candidato en el registro de ciclo de vida")

            if auto_promote:
                try:
                    from ..lifecycle.promotion import evaluate_and_promote
                    promo_cfg = config.get("lifecycle", {}).get("promotion", {})
                    decision = evaluate_and_promote(registry, "e3", ticker, promo_cfg)
                    if decision.promoted:
                        print(f"  ★ PROMOTED {ticker} to champion ({decision.reason})")
                    else:
                        print(f"  ↳ Kept current champion for {ticker} ({decision.reason})")
                except Exception as promo_exc:
                    print(f"  ⚠️  Auto-promotion failed for {ticker}: {promo_exc}")
        else:
            print(f"  ⚠️  Guardrails failed for {ticker}: {errors}")
    except Exception as exc:
        print(f"  Registracion en el ciclo de vida salteado: {exc}")


def run_for_ticker(config: dict, ticker: str, raw_dir: Path, out_dir: Path, use_latest_data: bool = False, register_lifecycle: bool = True, auto_promote: bool = False) -> dict:  # Pipeline E3
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
    tau_buy = float(thresholds.get("tau_buy", 0.001))      # Umbral para compra
    tau_sell = float(thresholds.get("tau_sell", 0.001))    # Umbral para venta

    # Costos de transacción
    costs = config.get("costs", {})  # Costos
    round_trip_bps = float(costs.get("intraday_round_trip_bps", 20))  # Costo intraday (basis points)

    # Seed global (necesario tanto para walk-forward como para time_split)
    base_seed = int(get_nested(config, ["project", "seed"], 42))  # Seed base

    # Configuración del modelo ensemble
    model_cfg = e3.get("model", {})  # Config modelo
    ensemble_members = int(model_cfg.get("ensemble_members", 3))    # Número de miembros en ensemble
    consensus_tol_raw = model_cfg.get("consensus_tol", None)         # Filtro de consenso del ensemble
    consensus_tol = float(consensus_tol_raw) if consensus_tol_raw is not None else None
    hidden_size = int(model_cfg.get("lstm_hidden_size", 128))       # Unidades LSTM
    num_layers = int(model_cfg.get("lstm_num_layers", 2))           # Capas LSTM
    dense_units = model_cfg.get("dense_units", None)                # Capa densa opcional
    dense_units = int(dense_units) if dense_units is not None else None
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
    ohlcv = apply_training_window(
        ohlcv, config, granularity="intraday", use_latest=use_latest_data, ticker=ticker,
    )

    # Extraer features técnicos intraday
    features = compute_intraday_features(ohlcv)  # Features

    # Crear target: retorno forward (siguiente N barras)
    target = make_target_return(ohlcv, horizon_bars=horizon_bars)  # Target

    # Crear secuencias: [n_samples, lookback_bars, n_features]
    X, y, ts, feat_names = make_sequences(features, target, lookback_bars=lookback_bars)  # Secuencias

    splits_cfg = config.get("splits", {})  # Config splits
    if splits_cfg.get("method") == "walk_forward":  # Walk-forward habilitado
        embargo_val = splits_cfg.get("embargo_days", {}).get("e3", horizon_bars)
        print(f"  ▶ Walk-forward ({splits_cfg.get('folds', 3)} folds, embargo={embargo_val} barras)")
        summary = run_e3_walk_forward(
            X=X, y=y, ts=ts, feat_names=list(feat_names),
            ticker=ticker, out_dir=out_dir, ohlcv=ohlcv,
            splits_cfg=splits_cfg,
            ensemble_members=ensemble_members,
            hidden_size=hidden_size, num_layers=num_layers, dense_units=dense_units,
            dropout=dropout, lr=lr, batch_size=batch_size, max_epochs=max_epochs,
            patience=patience, loss=loss, huber_delta=huber_delta,
            tau_buy=tau_buy, tau_sell=tau_sell, round_trip_bps=round_trip_bps,
            execution_delay_bars=execution_delay_bars, allow_short=allow_short,
            max_position=max_position, seed=base_seed, horizon_bars=horizon_bars,
            consensus_tol=consensus_tol,
        )

        # Registro en lifecycle (walk-forward path)
        if register_lifecycle:
            _register_e3_candidate(
                config=config, summary=summary, ticker=ticker,
                out_dir=out_dir, feat_names=list(feat_names),
                auto_promote=auto_promote,
            )

        return summary

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

    print(f"\nEntrenando ensemble de {ensemble_members} modelos LSTM para {ticker}...")  # Log
    for m in range(ensemble_members):  # Loop miembros
        print(f"\n  Modelo {m+1}/{ensemble_members} (seed={base_seed + 1000 * m}):")  # Log miembro
        seed = base_seed + 1000 * m  # Seed miembro

        # Crear modelo LSTM
        model = LSTMRegressor(  # Instanciar LSTM
            input_size=X_train_s.shape[-1],  # Input size
            hidden_size=hidden_size,  # Hidden
            num_layers=num_layers,  # Capas
            dense_units=dense_units,  # Capa densa opcional
            dropout=dropout,  # Dropout
            seed=seed,  # Seed
        )  # Fin init

        # Entrenar modelo
        train_started_at = datetime.now(timezone.utc).isoformat()
        train_start = time.perf_counter()
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
        train_end = time.perf_counter()
        train_ended_at = datetime.now(timezone.utc).isoformat()
        log_timing_event(
            strategy="e3_intraday",
            phase="train",
            duration_seconds=train_end - train_start,
            started_at=train_started_at,
            ended_at=train_ended_at,
            ticker=ticker,
            run_dir=out_dir,
            extra={
                "member": int(m + 1),
                "split": "time_split",
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
            },
        )

        val_losses.append(res.best_val_loss)  # Guardar val loss
        pred_started_at = datetime.now(timezone.utc).isoformat()
        pred_start = time.perf_counter()
        member_pred = model.predict(X_test_s)  # Guardar preds
        pred_end = time.perf_counter()
        pred_ended_at = datetime.now(timezone.utc).isoformat()
        log_timing_event(
            strategy="e3_intraday",
            phase="predict",
            duration_seconds=pred_end - pred_start,
            started_at=pred_started_at,
            ended_at=pred_ended_at,
            ticker=ticker,
            run_dir=out_dir,
            extra={
                "member": int(m + 1),
                "split": "time_split",
                "n_test": int(len(X_test)),
            },
        )
        preds_members.append(member_pred)  # Guardar preds
        print(f"  Modelo {m+1} final: val_loss={res.best_val_loss:.6f}, epochs={res.epochs_ran}")  # Log

    print(f"\nEnsemble completado. Promediando {len(preds_members)} predicciones...")  # Log

    # COMBINAR PREDICCIONES DEL ENSEMBLE
    stacked_preds = np.stack(preds_members, axis=0)   # (n_members, n_test)
    y_pred = np.mean(stacked_preds, axis=0)            # Promedio
    y_pred_std = np.std(stacked_preds, axis=0)         # Desacuerdo entre miembros

    # FILTRO DE CONSENSO: suprimir señales donde el ensemble no concuerda
    y_pred_filtered = y_pred.copy()
    if consensus_tol is not None and consensus_tol > 0:
        suppressed = int((y_pred_std >= consensus_tol).sum())
        print(f"  Consenso: {suppressed}/{len(y_pred)} señales suprimidas (std >= {consensus_tol:.5f})")
        y_pred_filtered[y_pred_std >= consensus_tol] = 0.0  # Sin señal donde hay desacuerdo

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
        pred_forward_returns=y_pred_filtered.astype(np.float32),  # Pred forward (consenso filtrado)
        tau_buy=tau_buy,  # Tau buy
        tau_sell=tau_sell,  # Tau sell
        round_trip_bps=round_trip_bps,  # Costos
        execution_delay_bars=execution_delay_bars,  # Delay
        allow_short=allow_short,  # Shorts
        max_position=max_position,  # Max posición
    )  # Fin backtest

    # MÉTRICAS DE TRADING
    _net_ret = bt["net_ret"].to_numpy(dtype=np.float32)
    _bars_per_year = 252 * 78  # ~19,656 barras de 5min por año
    _max_dd = compute_max_drawdown(bt["equity"].to_numpy(dtype=np.float32))
    _ann_ret = float(np.mean(_net_ret)) * _bars_per_year
    trading = {  # Métricas trading
        "sharpe": float(np.mean(_net_ret) / (np.std(_net_ret) + 1e-12) * np.sqrt(_bars_per_year)),  # Sharpe anualizado
        "calmar": _ann_ret / (_max_dd + 1e-12),                                                      # Calmar ratio
        "profit_factor": compute_profit_factor(_net_ret),                                            # Profit factor
        "max_drawdown": _max_dd,                                                                     # Max DD
        "num_trades": int((bt["turnover"] > 0).sum()),                                               # Num trades
        "turnover_mean": float(bt["turnover"].mean()),                                               # Turnover
        "time_in_market": float((bt["pos"].abs() > 0).mean()),                                       # Time in market
    }  # Fin métricas trading

    # GUARDAR ARTEFACTOS
    ensure_dir(out_dir)  # Crear dir

    # Guardar backtest completo
    bt.to_csv(out_dir / f"{ticker}_backtest.csv")  # Guardar backtest

    # Guardar predicciones
    preds_df = pd.DataFrame(  # DF preds
        {"y_true": y_test, "y_pred": y_pred, "pred_std": y_pred_std},  # Dict preds (incluye std del ensemble)
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

    summary = {**meta, **{f"ml_{k}": v for k, v in ml.items()}, **{f"bt_{k}": v for k, v in trading.items()}}  # Summary (prefijo bt_ alineado con E1/E2)
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
    parser.add_argument(  # Arg tickers
        "--tickers",  # Flag
        type=str,  # Tipo
        default="",  # Default
        help="Comma-separated tickers override (otherwise uses universe.tickers_by_strategy.e3_intraday)",  # Help
    )  # Fin arg tickers
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="No descargar datos; usar los existentes (por defecto se descargan datos nuevos, ver data.download en base.yaml)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Descargar datos y salir sin entrenar",
    )
    parser.add_argument(
        "--use-latest-data",
        action="store_true",
        help=(
            "Entrenar con el rango extendido hasta hoy "
            "(ignora data.training_window.intraday.end del config; start se preserva)."
        ),
    )
    parser.add_argument(
        "--auto-promote",
        action="store_true",
        help="Auto-promover candidato a champion si supera al actual en composite score",
    )
    args = parser.parse_args()  # Parse args

    if args.skip_download and args.use_latest_data:
        print(
            "⚠️  --skip-download + --use-latest-data: se extiende la ventana hasta hoy "
            "pero no se descargan datos frescos; puede no haber datos recientes."
        )

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
    thresholds = e3.get("thresholds", {})  # Thresholds
    costs = config.get("costs", {})  # Costos

    raw_dir = root / "data" / "raw" / "intraday"  # Dir raw
    out_base = root / "runs" / "e3_intraday" / datetime.now().strftime("%Y%m%d_%H%M%S")  # Dir salida
    ensure_dir(out_base)  # Crear dir

    import shutil
    shutil.copy(cfg_path, out_base / "config_used.yaml")

    # Descarga (config-driven: data.download; opt-out con --skip-download)
    from ..data.ingest import refresh_data_for_training
    refresh_data_for_training(
        config, tickers, granularity="intraday",
        skip_download=args.skip_download, root=root,
        raw_dir=raw_dir,
    )

    if args.download_only:
        print("\n--download-only: datos descargados, sin entrenar.")
        return

    # MLflow setup: intenta servidor remoto; fallback local robusto (igual que E1)
    mlflow_enabled = False  # Flag MLflow
    mlflow = None  # Ref MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()  # Tracking URI
    remote_check_timeout_seconds = float(os.getenv("MLFLOW_REMOTE_CHECK_TIMEOUT_SECONDS", "1.5"))
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "E3_Intraday_Strategy")  # Experimento

    local_sqlite_dir = root / "runs" / "mlflow_local"
    ensure_dir(local_sqlite_dir)
    # DB de fallback SEPARADA de la del server. Si el server MLflow está caído y
    # caemos a SQLite local, NO debe escribir en mlflow.db (la DB compartida que
    # usan server + Airflow): hacerlo contamina la DB con artifact_location del
    # host (file:///Users/...) y rompe los runs en contenedor.
    local_sqlite_db = local_sqlite_dir / "mlflow_fallback.db"
    local_artifacts_dir = local_sqlite_dir / "artifacts"
    ensure_dir(local_artifacts_dir)
    local_sqlite_uri = f"sqlite:///{local_sqlite_db}"

    fallback_local_tracking_uri = os.getenv("MLFLOW_LOCAL_TRACKING_URI", "").strip()
    if fallback_local_tracking_uri:
        local_sqlite_uri = fallback_local_tracking_uri

    local_mlruns = str(root / "mlruns")
    isolated_mlruns = str(root / "runs" / "e3_intraday" / "mlflow_store")
    ensure_dir(Path(isolated_mlruns))

    local_uris = [
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

    def _activate_mlflow(
        _mlflow_module,
        uri: str,
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

    try:  # Import MLflow
        import mlflow as _mlflow  # type: ignore

        if tracking_uri:
            remote_reachable, remote_reachability_err = _is_tracking_uri_reachable(
                tracking_uri,
                remote_check_timeout_seconds,
            )
            if remote_reachable:
                ok_remote, remote_err = _activate_mlflow(_mlflow, tracking_uri)
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
                artifact_dir = local_cfg.get("artifact_dir")
                ok_local, local_err = _activate_mlflow(
                    _mlflow,
                    uri,
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

                    summary = run_for_ticker(config, ticker=ticker, raw_dir=raw_dir, out_dir=ticker_out, use_latest_data=args.use_latest_data, auto_promote=args.auto_promote)  # Ejecutar

                    metrics: dict[str, float] = {}
                    for k, v in summary.items():
                        if not (
                            k.startswith("ml_")
                            or k.startswith("bt_")
                            or k.startswith("timing_")
                        ):
                            continue
                        if isinstance(v, (int, float)):
                            metrics[k] = float(v)
                    if isinstance(summary.get("val_loss"), (int, float)):
                        metrics["val_loss"] = float(summary["val_loss"])
                    if isinstance(summary.get("epochs_ran_mean"), (int, float)):
                        metrics["epochs_ran_mean"] = float(summary["epochs_ran_mean"])
                    if metrics:
                        try:
                            mlflow.log_metrics(metrics)
                        except Exception as met_exc:
                            print(f"  ⚠️  MLflow métricas no guardadas: {met_exc}")

                    # Log artifacts
                    try:
                        for fname in [
                            f"{ticker}_model.pth",
                            f"{ticker}_predictions.csv",
                            f"{ticker}_backtest.csv",
                            f"{ticker}_summary.csv",
                            f"{ticker}_walkforward_folds.csv",
                            f"{ticker}_walkforward_predictions.csv",
                            f"{ticker}_walkforward_backtest.csv",
                        ]:
                            p = ticker_out / fname
                            if p.exists():
                                if fname.endswith(".pth"):
                                    artifact_path = "models"
                                elif "pred" in fname:
                                    artifact_path = "predictions"
                                elif "backtest" in fname:
                                    artifact_path = "backtests"
                                elif "walkforward" in fname or "folds" in fname:
                                    artifact_path = "walkforward"
                                else:
                                    artifact_path = "artifacts"
                                mlflow.log_artifact(str(p), artifact_path=artifact_path)
                    except Exception as art_exc:
                        print(f"  ⚠️  MLflow artifacts no guardados: {art_exc}")
            else:  # Sin MLflow
                summary = run_for_ticker(config, ticker=ticker, raw_dir=raw_dir, out_dir=ticker_out, use_latest_data=args.use_latest_data, auto_promote=args.auto_promote)  # Ejecutar
            summaries.append(summary)  # Append
            print(f"Done {ticker}: PF={summary.get('bt_profit_factor', float('nan')):.3f} DD={summary.get('bt_max_drawdown', float('nan')):.3%} IC={summary.get('ml_ic', float('nan')):.3f}")  # Log
        except Exception as exc:  # Error ticker
            print(f"✗ Error en {ticker}: {exc}")  # Log error

    pd.DataFrame(summaries).to_csv(out_base / "summary_all.csv", index=False)  # Guardar summary
    print(f"Outputs escritos en {out_base}")  # Log salida


if __name__ == "__main__":  # Entry point
    main()  # Ejecutar main
