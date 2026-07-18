"""
Pipeline de entrenamiento LSTM para estrategia E2 (Moderada).

Flujo:
    1. Cargar config base (base.yaml) → extraer params de e2_moderate
    2. Cargar OHLCV (data/clean/ con fallback a raw)
    3. Calcular 16 features (compute_e2_features)
    4. Crear target de retorno a 20 días (make_target_e2)
    5. Construir secuencias (lookback=60)
    6. Walk-forward validation (5 folds, embargo=20 días) o time-split
    7. Por fold: Z-score scaling → entrenar LSTM → predecir → backtest
    8. Guardar modelo .pth (payload del último fold) + CSVs + PNG
    9. Registrar candidato en lifecycle (guardrails + registry)
   10. Tracking en MLflow

Uso:
    python -m src.e2.train_pipeline                              # descarga datos nuevos + entrena
    python -m src.e2.train_pipeline --tickers AAPL --skip-download  # usa datos existentes
    python -m src.e2.train_pipeline --tickers NVDA,GOOGL --auto-promote
"""

from __future__ import annotations

import argparse
import copy
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
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
from .lstm import LSTMRegressor
from ..features.build_sequences_e1e2 import (
    make_sequences,
    temporal_train_val_split,
    time_split,
)
from ..backtest.backtest_daily import backtest_daily_signals, summarize_backtest
from ..utils import (
    apply_training_window,
    ensure_dir,
    load_yaml,
    log_timing_event,
    project_root,
    resolve_lifecycle_paths,
)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    """Carga CSV OHLCV y devuelve DataFrame indexado por timestamp (UTC)."""
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        size = path.stat().st_size
        raise ValueError(
            f"Empty/invalid CSV: {path} (size={size} bytes)"
        ) from exc

    if "timestamp" not in df.columns:
        raise ValueError(f"Missing 'timestamp' column in {path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")

    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")

    return df


def compute_information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """IC = Spearman rank correlation entre retorno real y predicho."""
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


_PLOT_RCPARAMS = {
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 17,
}


def save_walkforward_plot(
    fold_df: pd.DataFrame, ticker: str, out_path: Path,
) -> None:
    """Guarda gráfico de IC y Sharpe por fold."""
    if fold_df.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    plt.rcParams.update(_PLOT_RCPARAMS)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(fold_df["fold"], fold_df["ml_ic"], marker="o")
    axes[0].axhline(0.0, color="black", ls="--", lw=0.8, alpha=0.4)
    axes[0].set_ylabel("IC")
    axes[0].set_title(f"{ticker} — E2 Walk-Forward (LSTM)")

    axes[1].plot(fold_df["fold"], fold_df["bt_sharpe"], marker="o", color="#2ca02c")
    axes[1].axhline(0.0, color="black", ls="--", lw=0.8, alpha=0.4)
    axes[1].set_ylabel("Sharpe")
    axes[1].set_xlabel("Fold")

    if "window" in fold_df.columns:
        axes[1].set_xticks(fold_df["fold"].to_numpy())
        axes[1].set_xticklabels(
            fold_df["window"].to_list(), rotation=35, ha="right",
        )

    for ax in axes:
        ax.grid(alpha=0.3, ls="--", lw=0.8)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def _load_tuned_params(path: str) -> dict:
    if not path:
        return {}
    resolved = Path(path)
    if not resolved.exists():
        return {}
    try:
        data = load_yaml(resolved)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


_SHARED_PARAMS_WARNED: set = set()


def _warn_shared_params(all_tuned: dict, source_path: Path) -> None:
    """Detecta tickers con parámetros idénticos (posibles YAML anchors residuales)."""
    cache_key = str(source_path)
    if cache_key in _SHARED_PARAMS_WARNED:
        return
    _SHARED_PARAMS_WARNED.add(cache_key)

    by_repr: dict[str, list[str]] = {}
    for ticker, params in all_tuned.items():
        if not isinstance(params, dict):
            continue
        key = repr(sorted(params.items()))
        by_repr.setdefault(key, []).append(ticker)

    for group in by_repr.values():
        if len(group) > 1:
            import warnings
            warnings.warn(
                f"Optuna: tickers con params idénticos (posible YAML anchor): "
                f"{', '.join(group)} en {source_path.name}. "
                f"Considerar re-correr --per_ticker para optimización individual.",
                UserWarning,
                stacklevel=3,
            )
            break


def _normalize_e2_tuned_entry(per_ticker: dict) -> dict:
    if not isinstance(per_ticker, dict):
        return {}

    if any(k in per_ticker for k in ("thresholds", "model", "splits")):
        return per_ticker

    thresholds: dict = {}
    model: dict = {}
    splits: dict = {}

    if per_ticker.get("tau_buy") is not None:
        thresholds["tau_buy"] = per_ticker.get("tau_buy")
    if per_ticker.get("tau_sell") is not None:
        thresholds["tau_sell"] = per_ticker.get("tau_sell")

    lstm_units = []
    if per_ticker.get("lstm_units_1") is not None:
        lstm_units.append(per_ticker.get("lstm_units_1"))
    if per_ticker.get("lstm_units_2") is not None:
        lstm_units.append(per_ticker.get("lstm_units_2"))
    if lstm_units:
        model["lstm_units"] = lstm_units

    if per_ticker.get("dropout") is not None:
        model["dropout"] = per_ticker.get("dropout")
    if per_ticker.get("learning_rate") is not None:
        model["learning_rate"] = per_ticker.get("learning_rate")
    if per_ticker.get("weight_decay") is not None:
        model["weight_decay"] = per_ticker.get("weight_decay")
    if per_ticker.get("batch_size") is not None:
        model["batch_size"] = per_ticker.get("batch_size")
    if per_ticker.get("early_stopping_patience") is not None:
        model["early_stopping_patience"] = per_ticker.get("early_stopping_patience")

    if per_ticker.get("n_folds") is not None:
        splits["folds"] = per_ticker.get("n_folds")
    if per_ticker.get("internal_val_fraction") is not None:
        splits["internal_val_fraction"] = per_ticker.get("internal_val_fraction")

    normalized: dict = {}
    if thresholds:
        normalized["thresholds"] = thresholds
    if model:
        normalized["model"] = model
    if splits:
        normalized["splits"] = splits
    return normalized


def _apply_tuned_overrides(*, config: dict, ticker: str) -> tuple[dict, dict]:
    """Aplica overrides de Optuna (por ticker) sobre el config base.

    Resolución de path con prioridad:
      1. E2_TUNED_PARAMS_PATH (env var, override de emergencia)
      2. TUNED_PARAMS_PATH (env var, fallback genérico)
      3. config["optuna"]["e2_moderate"]["tuned_params_path"] (base.yaml, fuente principal)

    Returns:
        (config, hyperparams_info) donde hyperparams_info es un dict con:
          - source: "optuna" | "base_yaml"
          - tuned_params_file: path del archivo de overrides (si aplica)
          - overrides_applied: lista de secciones que se sobreescribieron
    """
    hp_info: dict = {"source": "base_yaml", "tuned_params_file": None, "overrides_applied": []}

    # --- Resolver path: ENV var (deprecado) > config ---
    env_path = (
        os.getenv("E2_TUNED_PARAMS_PATH", "").strip()
        or os.getenv("TUNED_PARAMS_PATH", "").strip()
    )

    if env_path:
        import warnings
        warnings.warn(
            "E2_TUNED_PARAMS_PATH / TUNED_PARAMS_PATH están deprecados. "
            "Usar base.yaml > optuna > e2_moderate > tuned_params_path en su lugar.",
            DeprecationWarning,
            stacklevel=2,
        )
        tuned_path = env_path
    else:
        optuna_cfg = config.get("optuna", {}).get("e2_moderate", {})
        tuned_path = optuna_cfg.get("tuned_params_path", "")
        if not tuned_path:
            return config, hp_info

    resolved = Path(tuned_path)
    if not resolved.is_absolute():
        try:
            resolved = project_root() / resolved
        except Exception:
            pass

    all_tuned = _load_tuned_params(str(resolved))
    if not all_tuned:
        print(f"  ⚠️  E2 tuned params no encontrados o vacíos: {resolved}")
        return config, hp_info

    # Validar params compartidos entre tickers (detecta YAML anchors residuales)
    _warn_shared_params(all_tuned, resolved)

    per_ticker_raw = all_tuned.get(ticker)
    if not isinstance(per_ticker_raw, dict):
        fallback = config.get("optuna", {}).get("e2_moderate", {}).get("fallback_to_base", True)
        if fallback:
            print(f"  ℹ️  Optuna: sin overrides E2 para {ticker} → usando base.yaml")
        return config, hp_info

    per_ticker = _normalize_e2_tuned_entry(per_ticker_raw)
    if not per_ticker:
        return config, hp_info

    # --- Aplica overrides ---
    hp_info["source"] = "optuna"
    hp_info["tuned_params_file"] = str(resolved)

    strat = config.setdefault("strategies", {}).setdefault("e2_moderate", {})
    applied_sections: list[str] = []

    thresholds = per_ticker.get("thresholds")
    if isinstance(thresholds, dict):
        strat.setdefault("thresholds", {}).update(thresholds)
        applied_sections.append("thresholds")

    model = per_ticker.get("model")
    if isinstance(model, dict):
        strat.setdefault("model", {}).update(model)
        applied_sections.append("model")

    splits = per_ticker.get("splits")
    if isinstance(splits, dict):
        config.setdefault("splits", {}).update(splits)
        applied_sections.append("splits")

    hp_info["overrides_applied"] = applied_sections
    if applied_sections:
        print(
            f"  ✓ Optuna overrides E2 para {ticker}: "
            f"{', '.join(applied_sections)} ({resolved.name})"
        )

    return config, hp_info


# ---------------------------------------------------------------------------
# Walk-forward LSTM
# ---------------------------------------------------------------------------

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
    weight_decay: float,
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
) -> dict:
    """Ejecuta walk-forward completo con LSTM y retorna resumen."""

    require_model_save = os.getenv(
        "REQUIRE_MODEL_SAVE", "1",
    ).strip().lower() not in {"0", "false", "no"}

    ensure_dir(out_dir)
    n_samples = len(X)
    if n_samples == 0:
        raise ValueError("No hay muestras disponibles para walk-forward")

    folds = int(splits_cfg.get("folds", 5))
    if folds < 1:
        raise ValueError("'folds' debe ser >= 1")

    embargo_cfg = splits_cfg.get("embargo_days", {})
    if isinstance(embargo_cfg, dict):
        embargo_days = int(embargo_cfg.get("e2", 0))
    else:
        embargo_days = int(embargo_cfg or 0)
    gap_samples = max(0, embargo_days)

    default_test_size = max(1, n_samples // (folds + 1))
    test_size = int(splits_cfg.get("test_size", default_test_size))
    if test_size <= gap_samples:
        test_size = gap_samples + 1

    splitter = TimeSeriesSplit(n_splits=folds, test_size=test_size, gap=gap_samples)

    # Val fraction
    raw_val_fraction = splits_cfg.get(
        "internal_val_fraction", splits_cfg.get("val_fraction", 0.15),
    )
    try:
        val_fraction = float(raw_val_fraction)
    except (TypeError, ValueError):
        val_fraction = 0.15
    if not 0 < val_fraction < 1:
        val_fraction = 0.15

    fold_summaries: list[dict] = []
    pred_frames: list[pd.DataFrame] = []

    last_model_payload: dict | None = None
    last_torch = None
    last_mean_X: np.ndarray | None = None
    last_std_X: np.ndarray | None = None
    last_mean_y: float | None = None
    last_std_y: float | None = None
    total_train_seconds = 0.0
    total_predict_seconds = 0.0
    total_train_samples = 0
    total_val_samples = 0
    val_losses: list[float] = []

    root = project_root()

    def as_relative(path: Path) -> str:
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    # ---------- Main fold loop ----------
    for fold_idx, (train_full_idx, test_idx) in enumerate(
        splitter.split(np.arange(n_samples)), start=1,
    ):
        if len(test_idx) == 0 or len(train_full_idx) == 0:
            continue

        # Train / val split interno
        try:
            train_idx, val_idx = temporal_train_val_split(
                train_full_idx, val_fraction=val_fraction,
            )
        except ValueError:
            sp = max(1, int(len(train_full_idx) * 0.85))
            train_idx = train_full_idx[:sp]
            val_idx = train_full_idx[sp:]

        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

        # Z-score X (per-feature, calculado sobre train aplanado a 2D)
        X_tr_2d = X_train.reshape(-1, X_train.shape[-1])
        mean_X = X_tr_2d.mean(axis=0)
        std_X = X_tr_2d.std(axis=0) + 1e-12

        def scale_X(data: np.ndarray) -> np.ndarray:
            return ((data - mean_X) / std_X).astype(np.float32)

        # Z-score y
        mean_y = float(y_train.mean())
        std_y = float(y_train.std()) + 1e-12

        def scale_y(data: np.ndarray) -> np.ndarray:
            return ((data - mean_y) / std_y).astype(np.float32)

        def unscale_y(data: np.ndarray) -> np.ndarray:
            return (data * std_y + mean_y).astype(np.float32)

        X_train_s = scale_X(X_train)
        X_val_s = scale_X(X_val)
        X_test_s = scale_X(X_test)
        y_train_s = scale_y(y_train)
        y_val_s = scale_y(y_val)

        # Entrenar LSTM
        model = LSTMRegressor(
            input_size=X_train.shape[-1],
            hidden_sizes=list(lstm_units),
            dropout=dropout,
            dense_units=dense_units,
            seed=seed,
        )

        train_started_at = datetime.now(timezone.utc).isoformat()
        train_start = time.perf_counter()
        res = model.fit(
            X_train_s, y_train_s,
            X_val_s, y_val_s,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            early_stopping_patience=patience,
            loss=loss,
            huber_delta=huber_delta,
        )
        train_end = time.perf_counter()
        train_ended_at = datetime.now(timezone.utc).isoformat()
        total_train_seconds += train_end - train_start
        total_train_samples += len(train_idx)
        total_val_samples += len(val_idx)
        val_losses.append(float(res.best_val_loss))

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

        # Predicción
        pred_started_at = datetime.now(timezone.utc).isoformat()
        pred_start = time.perf_counter()
        y_pred_s = model.predict(X_test_s)
        pred_end = time.perf_counter()
        pred_ended_at = datetime.now(timezone.utc).isoformat()
        total_predict_seconds += pred_end - pred_start

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

        y_pred = unscale_y(y_pred_s)

        # Guardar refs del último fold para serialización
        last_torch = model.torch
        last_mean_X = mean_X
        last_std_X = std_X
        last_mean_y = mean_y
        last_std_y = std_y

        ts_test = ts[test_idx]
        ts_train = ts[train_idx]  # Timestamps de train (para log de ventana completa)
        window_label = f"{ts_test[0].date()} -> {ts_test[-1].date()}"  # Etiqueta test (usada por plots/CSV)

        # Model payload (último fold = más reciente)
        last_model_payload = {
            "ticker": ticker,
            "strategy": "e2_moderate",
            "created_at": datetime.now(timezone.utc).isoformat(),
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
            "train_result": {
                "epochs_ran": int(res.epochs_ran),
                "best_val_loss": float(res.best_val_loss),
            },
            "walkforward": {
                "fold": int(fold_idx),
                "window": window_label,
                "test_size": int(test_size),
                "gap_samples": int(gap_samples),
            },
            "state_dict": {
                k: v.detach().cpu()
                for k, v in model.model.state_dict().items()
            },
        }

        # Métricas del fold
        mae = float(np.mean(np.abs(y_test - y_pred)))
        rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
        dir_acc = float(np.mean(np.sign(y_test) == np.sign(y_pred)))
        ic = compute_information_coefficient(y_test, y_pred)

        # Backtest del fold
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
        # Log de fold: train Y test explícitos para evidenciar la ventana CRECIENTE
        # (expanding). Ver solo el test aparenta ventana deslizante.
        fold_pct = 100.0 * fold_idx / folds
        print(
            f"    Fold {fold_idx}/{folds} ({fold_pct:5.1f}%): "
            f"train[{ts_train[0].date()} -> {ts_train[-1].date()}] n={len(train_idx)} (expanding) | "
            f"test[{window_label}] n={len(test_idx)} | "
            f"MAE={mae:.4f} IC={ic_str} Sharpe={sharpe_str}"
        )

        bt.to_csv(out_dir / f"{ticker}_fold{fold_idx}_backtest.csv")

        fold_summaries.append({
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
        })

        preds_df = pd.DataFrame(
            {"fold": fold_idx, "y_true": y_test, "y_pred": y_pred},
            index=ts_test,
        )
        preds_df.index.name = "timestamp"
        pred_frames.append(preds_df)

    # ---------- Consolidación ----------
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

    plot_path = out_dir / f"{ticker}_walkforward_metrics.png"
    save_walkforward_plot(fold_df, ticker, plot_path)

    # Guardar modelo del último fold
    model_path = out_dir / f"{ticker}_model.pth"
    if last_model_payload is not None and last_torch is not None:
        try:
            last_torch.save(last_model_payload, model_path)
        except Exception as exc:
            print(f"⚠️  No se pudo guardar el modelo walk-forward: {exc}")

    if require_model_save and not model_path.exists():
        raise RuntimeError(
            f"No se guardó el modelo walk-forward en {model_path}. "
            "Setear REQUIRE_MODEL_SAVE=0 para permitir continuar sin guardar."
        )

    # Guardar scalers
    scaler_path = None
    if last_mean_X is not None and last_std_X is not None:
        scaler_X_df = pd.DataFrame(
            {"mean": last_mean_X, "std": last_std_X}, index=feat_names,
        )
        scaler_path = out_dir / f"{ticker}_scaler.csv"
        scaler_X_df.to_csv(scaler_path)

    if last_mean_y is not None and last_std_y is not None:
        pd.DataFrame(
            {"mean_y": [last_mean_y], "std_y": [last_std_y]},
        ).to_csv(out_dir / f"{ticker}_target_scaler.csv", index=False)

    summary: dict = {
        "ticker": ticker,
        "n_samples": int(n_samples),
        "n_train": int(total_train_samples),
        "n_val": int(total_val_samples),
        "n_test": int(len(combined_preds)),
        "folds": int(len(fold_df)),
        "lookback_days": int(lookback_days),
        "horizon_days": int(horizon_days),
        "split_method": "walk_forward",
        "walkforward_test_size": int(test_size),
        "walkforward_gap": int(gap_samples),
        "val_loss": float(np.mean(val_losses)) if val_losses else None,
        "ml_mae": mae_all,
        "ml_rmse": rmse_all,
        "ml_directional_accuracy": dir_acc_all,
        "ml_ic": ic_all,
        **{f"bt_{k}": float(v) for k, v in trading_metrics_all.items()},
        "timing_train_seconds": round(total_train_seconds, 2),
        "timing_predict_seconds": round(total_predict_seconds, 4),
        "predictions_file": as_relative(
            out_dir / f"{ticker}_walkforward_predictions.csv",
        ),
        "backtest_file": as_relative(
            out_dir / f"{ticker}_walkforward_backtest.csv",
        ),
        "scaler_file": as_relative(scaler_path) if scaler_path else "",
        "model_file": (
            as_relative(model_path)
            if last_model_payload is not None and last_torch is not None
            else ""
        ),
    }

    return summary


# ---------------------------------------------------------------------------
# Orquestador per-ticker
# ---------------------------------------------------------------------------

def run_e2_for_ticker(
    config: dict,
    ticker: str,
    raw_dir: Path,
    out_dir: Path,
    register_lifecycle: bool = True,
    auto_promote: bool = False,
    use_latest_data: bool = False,
) -> dict:
    """Entrena y evalúa la estrategia E2 (LSTM) para un ticker."""

    config = copy.deepcopy(config)
    config, hp_info = _apply_tuned_overrides(config=config, ticker=ticker)

    require_model_save = os.getenv(
        "REQUIRE_MODEL_SAVE", "1",
    ).strip().lower() not in {"0", "false", "no"}

    out_dir = Path(out_dir)
    if out_dir.name != ticker:
        out_dir = out_dir / ticker
    ensure_dir(out_dir)

    root = project_root()

    def as_relative(path: Path) -> str:
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    # ---------- Config E2 ----------
    e2 = config.get("strategies", {}).get("e2_moderate", {})
    lookback_days = int(e2.get("lookback_days", 60))
    horizon_days = int(e2.get("horizon_days", 20))

    model_cfg = e2.get("model", {})
    lstm_units_raw = model_cfg.get("lstm_units", model_cfg.get("units", [128, 64]))
    if isinstance(lstm_units_raw, (list, tuple)):
        lstm_units = [int(x) for x in lstm_units_raw]
    else:
        lstm_units = [int(lstm_units_raw)]
    dropout = float(model_cfg.get("dropout", 0.2))
    dense_units = int(model_cfg.get("dense_units", 32))
    lr = float(model_cfg.get("learning_rate", 1e-3))
    wd = float(model_cfg.get("weight_decay", 0.0))
    batch_size = int(model_cfg.get("batch_size", 64))
    max_epochs = int(model_cfg.get("max_epochs", 150))
    patience = int(model_cfg.get("early_stopping_patience", 12))
    loss = str(model_cfg.get("loss", "huber"))
    huber_delta = float(model_cfg.get("huber_delta", 1.0))

    seed = int(config.get("project", {}).get("seed", 42))

    # ---------- Cargar datos ----------
    clean_dir = raw_dir.parent.parent / "clean"
    clean_csv_path = clean_dir / f"{ticker}_daily.csv"

    if clean_csv_path.exists():
        csv_path = clean_csv_path
        print(f"  ✓ Usando datos limpios: {csv_path.name}")
    else:
        csv_path = raw_dir / f"{ticker}_daily.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing daily CSV: {csv_path}")
        print(f"  ⚠️  Usando datos raw: {csv_path.name}")

    ohlcv = load_ohlcv_csv(csv_path)
    ohlcv = apply_training_window(
        ohlcv, config, granularity="daily", use_latest=use_latest_data, ticker=ticker,
    )
    # Último día de la ventana de entrenamiento: cutoff para la comparación justa
    # champion-vs-candidate (ventana OOS común). Ver lifecycle.reevaluation.
    train_data_end = str(ohlcv.index.max().date()) if len(ohlcv) else None

    # ---------- Features y target ----------
    active_features = e2["features"]["active"]  # Subset activo para entrenar (ver base.yaml)
    features = compute_e2_features(ohlcv)[active_features]
    target = make_target_e2(ohlcv, horizon_days=horizon_days)

    X, y, ts, feat_names = make_sequences(features, target, lookback=lookback_days)

    print(f"  Datos: X={X.shape}, features={len(feat_names)}")

    # ---------- Thresholds y costos ----------
    thresholds = e2.get("thresholds", {})
    tau_buy = float(thresholds.get("tau_buy", 0.025))
    tau_sell = float(thresholds.get("tau_sell", 0.00))

    costs_cfg = config.get("costs", {})
    round_trip_bps = float(
        costs_cfg.get("daily_round_trip_bps", costs_cfg.get("round_trip_bps_daily", 10)),
    )

    bt_cfg = e2.get("backtest", {})
    holding_period = int(bt_cfg.get("holding_period_days", horizon_days))
    allow_short = bool(bt_cfg.get("allow_short", False))
    max_position = float(bt_cfg.get("max_position", 1.0))

    splits_cfg = config.get("splits", {})

    # ---------- Despachar a walk-forward o time-split ----------
    if splits_cfg.get("method") == "walk_forward":
        print(
            f"  ▶ Walk-forward ({splits_cfg.get('folds', 5)} folds, "
            f"embargo={splits_cfg.get('embargo_days', {}).get('e2', 0)})"
        )
        summary = run_e2_walk_forward(
            X=X, y=y, ts=ts, feat_names=feat_names,
            ticker=ticker, out_dir=out_dir, ohlcv=ohlcv,
            splits_cfg=splits_cfg,
            lstm_units=lstm_units, dropout=dropout,
            dense_units=dense_units, learning_rate=lr,
            weight_decay=wd,
            batch_size=batch_size, max_epochs=max_epochs,
            patience=patience, loss=loss, huber_delta=huber_delta,
            tau_buy=tau_buy, tau_sell=tau_sell,
            round_trip_bps=round_trip_bps,
            holding_period=holding_period,
            allow_short=allow_short, max_position=max_position,
            seed=seed, lookback_days=lookback_days,
            horizon_days=horizon_days,
        )

        # Agregar hiperparámetros al summary
        summary["hp_lstm_units_1"] = lstm_units[0]
        summary["hp_lstm_units_2"] = lstm_units[1] if len(lstm_units) > 1 else None
        summary["hp_dropout"] = dropout
        summary["hp_dense_units"] = dense_units
        summary["hp_learning_rate"] = lr
        summary["hp_weight_decay"] = wd
        summary["hp_batch_size"] = batch_size
        summary["hp_tau_buy"] = tau_buy
        summary["hp_tau_sell"] = tau_sell

        # Lifecycle
        if register_lifecycle:
            try:
                from ..lifecycle.registry import ModelRegistry
                from ..lifecycle.guardrails import validate_candidate, log_candidate_metrics

                registry_path, metrics_log_path = resolve_lifecycle_paths(config, root=root)
                log_candidate_metrics(
                    metrics=summary, strategy="e2", ticker=ticker,
                    run_dir=out_dir, variant="e2_moderate",
                    log_path=metrics_log_path,
                    feature_names=list(feat_names),
                    hyperparams_info=hp_info,
                )
                _guardrail_cfg = config.get("lifecycle", {}).get("guardrails", {})
                _min_sharpe = float(
                    _guardrail_cfg.get("min_sharpe_by_strategy", {}).get("e2_moderate", 0.0)
                )
                passed, errors = validate_candidate(
                    run_dir=out_dir, ticker=ticker, metrics=summary, min_sharpe=_min_sharpe
                )
                if passed:
                    registry = ModelRegistry(registry_path)
                    registry.register_candidate(
                        strategy="e2", ticker=ticker,
                        run_dir=str(out_dir.relative_to(root)),
                        metrics=summary, variant="e2_moderate",
                        feature_names=list(feat_names),
                        hyperparams=hp_info,
                        train_data_end=train_data_end,
                    )
                    print(f"  ✓ {ticker} registrado como candidato en el registro de ciclo de vida")

                    if auto_promote:
                        try:
                            from ..lifecycle.promotion import evaluate_and_promote
                            promo_cfg = config.get("lifecycle", {}).get("promotion", {})
                            decision = evaluate_and_promote(
                                registry, "e2", ticker, promo_cfg,
                            )
                            if decision.promoted:
                                print(f"  ★ PROMOTED {ticker} to champion ({decision.reason})")
                            else:
                                print(f"  ↳ Kept current champion ({decision.reason})")
                        except Exception as promo_exc:
                            print(f"  ⚠️  Auto-promotion failed: {promo_exc}")
                else:
                    print(f"  ⚠️  Guardrails failed for {ticker}: {errors}")
            except Exception as exc:
                print(f"  Registracion en el ciclo de vida salteado: {exc}")

        pd.Series(summary).to_csv(out_dir / f"{ticker}_summary.csv")
        return summary

    # ---------- Time-split branch ----------
    idx_train, idx_val, idx_test = time_split(len(X))
    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]
    ts_test = ts[idx_test]

    # Z-score X
    Xtr2d = X_train.reshape(-1, X_train.shape[-1])
    mean_X = Xtr2d.mean(axis=0)
    std_X = Xtr2d.std(axis=0) + 1e-12

    def scale_X(Xa: np.ndarray) -> np.ndarray:
        return ((Xa - mean_X) / std_X).astype(np.float32)

    X_train_s = scale_X(X_train)
    X_val_s = scale_X(X_val)
    X_test_s = scale_X(X_test)

    # Z-score y
    mean_y = float(y_train.mean())
    std_y = float(y_train.std()) + 1e-12

    def scale_y(ya: np.ndarray) -> np.ndarray:
        return ((ya - mean_y) / std_y).astype(np.float32)

    def unscale_y(ya: np.ndarray) -> np.ndarray:
        return (ya * std_y + mean_y).astype(np.float32)

    y_train_s = scale_y(y_train)
    y_val_s = scale_y(y_val)

    # Train
    model = LSTMRegressor(
        input_size=X_train_s.shape[-1],
        hidden_sizes=lstm_units,
        dropout=dropout,
        dense_units=dense_units,
        seed=seed,
    )

    print(f"  Entrenando LSTM para {ticker}...")
    train_started_at = datetime.now(timezone.utc).isoformat()
    train_start = time.perf_counter()
    res = model.fit(
        X_train_s, y_train_s,
        X_val_s, y_val_s,
        learning_rate=lr,
        weight_decay=wd,
        batch_size=batch_size,
        max_epochs=max_epochs,
        early_stopping_patience=patience,
        loss=loss,
        huber_delta=huber_delta,
    )
    train_end = time.perf_counter()
    train_ended_at = datetime.now(timezone.utc).isoformat()
    train_seconds = train_end - train_start

    log_timing_event(
        strategy="e2_moderate", phase="train",
        duration_seconds=train_seconds,
        started_at=train_started_at, ended_at=train_ended_at,
        ticker=ticker, run_dir=out_dir,
        extra={
            "split": splits_cfg.get("method", "time_split"),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
        },
    )

    print(f"  Epochs: {res.epochs_ran}, Val Loss: {res.best_val_loss:.6f}")

    # Guardar modelo
    model_path = out_dir / f"{ticker}_model.pth"
    try:
        torch = model.torch
        payload = {
            "ticker": ticker,
            "strategy": "e2_moderate",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_class": "LSTMRegressor",
            "model_kwargs": {
                "input_size": int(X_train_s.shape[-1]),
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
            "train_result": {
                "epochs_ran": int(res.epochs_ran),
                "best_val_loss": float(res.best_val_loss),
            },
            "state_dict": {
                k: v.detach().cpu()
                for k, v in model.model.state_dict().items()
            },
        }
        torch.save(payload, model_path)
        print(f"  ✓ Modelo guardado: {model_path.name}")
    except Exception as exc:
        print(f"  ⚠️  No se pudo guardar el modelo: {exc}")

    if require_model_save and not model_path.exists():
        raise RuntimeError(
            f"No se guardó el modelo en {model_path}. "
            "Setear REQUIRE_MODEL_SAVE=0 para continuar sin guardar."
        )

    # Predict
    pred_started_at = datetime.now(timezone.utc).isoformat()
    pred_start = time.perf_counter()
    y_pred_s = model.predict(X_test_s)
    pred_end = time.perf_counter()
    pred_ended_at = datetime.now(timezone.utc).isoformat()
    predict_seconds = pred_end - pred_start

    log_timing_event(
        strategy="e2_moderate", phase="predict",
        duration_seconds=predict_seconds,
        started_at=pred_started_at, ended_at=pred_ended_at,
        ticker=ticker, run_dir=out_dir,
        extra={
            "split": splits_cfg.get("method", "time_split"),
            "n_test": int(len(X_test)),
        },
    )

    y_pred = unscale_y(y_pred_s)

    # Métricas
    mae = float(np.mean(np.abs(y_test - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    dir_acc = float(np.mean(np.sign(y_test) == np.sign(y_pred)))
    ic = compute_information_coefficient(y_test, y_pred)

    # Backtest
    close_prices_test = ohlcv.loc[ts_test, "close"].to_numpy()
    bt = backtest_daily_signals(
        timestamps=ts_test,
        close_prices=close_prices_test,
        pred_returns=y_pred,
        tau_buy=tau_buy, tau_sell=tau_sell,
        round_trip_bps=round_trip_bps,
        holding_period_days=holding_period,
        allow_short=allow_short,
        max_position=max_position,
    )
    trading_metrics = summarize_backtest(bt)

    # Guardar outputs
    preds_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred}, index=ts_test)
    preds_df.index.name = "timestamp"
    preds_df.to_csv(out_dir / f"{ticker}_predictions.csv")
    bt.to_csv(out_dir / f"{ticker}_backtest.csv")

    scaler_X_df = pd.DataFrame({"mean": mean_X, "std": std_X}, index=feat_names)
    scaler_X_df.to_csv(out_dir / f"{ticker}_scaler.csv")
    pd.DataFrame({"mean_y": [mean_y], "std_y": [std_y]}).to_csv(
        out_dir / f"{ticker}_target_scaler.csv", index=False,
    )

    summary = {
        "ticker": ticker,
        "n_samples": int(len(X)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "lookback_days": lookback_days,
        "horizon_days": horizon_days,
        "split_method": splits_cfg.get("method", "time_split"),
        "epochs_ran": res.epochs_ran,
        "val_loss": res.best_val_loss,
        "ml_mae": mae,
        "ml_rmse": rmse,
        "ml_directional_accuracy": dir_acc,
        "ml_ic": ic,
        **{f"bt_{k}": float(v) for k, v in trading_metrics.items()},
        "timing_train_seconds": round(train_seconds, 2),
        "timing_predict_seconds": round(predict_seconds, 4),
        "predictions_file": as_relative(out_dir / f"{ticker}_predictions.csv"),
        "backtest_file": as_relative(out_dir / f"{ticker}_backtest.csv"),
        "scaler_file": as_relative(out_dir / f"{ticker}_scaler.csv"),
        "model_file": as_relative(model_path),
        "hp_lstm_units_1": lstm_units[0],
        "hp_lstm_units_2": lstm_units[1] if len(lstm_units) > 1 else None,
        "hp_dropout": dropout,
        "hp_dense_units": dense_units,
        "hp_learning_rate": lr,
        "hp_weight_decay": wd,
        "hp_batch_size": batch_size,
        "hp_tau_buy": tau_buy,
        "hp_tau_sell": tau_sell,
    }
    pd.Series(summary).to_csv(out_dir / f"{ticker}_summary.csv")

    # Lifecycle
    if register_lifecycle:
        try:
            from ..lifecycle.registry import ModelRegistry
            from ..lifecycle.guardrails import validate_candidate, log_candidate_metrics

            registry_path, metrics_log_path = resolve_lifecycle_paths(config, root=root)
            log_candidate_metrics(
                metrics=summary, strategy="e2", ticker=ticker,
                run_dir=out_dir, variant="e2_moderate",
                log_path=metrics_log_path,
                feature_names=list(feat_names),
                hyperparams_info=hp_info,
            )
            _guardrail_cfg = config.get("lifecycle", {}).get("guardrails", {})
            _min_sharpe = float(
                _guardrail_cfg.get("min_sharpe_by_strategy", {}).get("e2_moderate", 0.0)
            )
            passed, errors = validate_candidate(
                run_dir=out_dir, ticker=ticker, metrics=summary, min_sharpe=_min_sharpe
            )
            if passed:
                registry = ModelRegistry(registry_path)
                registry.register_candidate(
                    strategy="e2", ticker=ticker,
                    run_dir=str(out_dir.relative_to(root)),
                    metrics=summary, variant="e2_moderate",
                    feature_names=list(feat_names),
                    hyperparams=hp_info,
                    train_data_end=train_data_end,
                )
                print(f"  ✓ {ticker} registrado como candidato en el registro de ciclo de vida")

                if auto_promote:
                    try:
                        from ..lifecycle.promotion import evaluate_and_promote
                        promo_cfg = config.get("lifecycle", {}).get("promotion", {})
                        decision = evaluate_and_promote(
                            registry, "e2", ticker, promo_cfg,
                        )
                        if decision.promoted:
                            print(f"  ★ PROMOTED {ticker} to champion ({decision.reason})")
                        else:
                            print(f"  ↳ Kept current champion ({decision.reason})")
                    except Exception as promo_exc:
                        print(f"  ⚠️  Auto-promotion failed: {promo_exc}")
            else:
                print(f"  ⚠️  Guardrails failed for {ticker}: {errors}")
        except Exception as exc:
            print(f"  Registracion en el ciclo de vida salteado: {exc}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline E2 (LSTM moderado)")
    parser.add_argument(
        "--config", type=str, default="src/config/base.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--tickers", type=str, default="",
        help="Comma-separated tickers (default: usa config e2_moderate)",
    )
    parser.add_argument(
        "--auto-promote", action="store_true",
        help="Auto-promover candidato si supera al champion actual",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="No descargar datos; usar los existentes (por defecto se descargan datos nuevos, ver data.download en base.yaml)",
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Descargar datos y salir sin entrenar",
    )
    parser.add_argument(
        "--use-latest-data", action="store_true",
        help=(
            "Entrenar con el rango extendido hasta hoy "
            "(ignora data.training_window.daily.end del config; start se preserva)."
        ),
    )
    args = parser.parse_args()

    if args.skip_download and args.use_latest_data:
        print(
            "⚠️  --skip-download + --use-latest-data: se extiende la ventana hasta hoy "
            "pero no se descargan datos frescos; puede no haber datos recientes."
        )

    root = project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path

    config = load_yaml(cfg_path)

    # Tickers
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        tickers = list(
            config.get("universe", {})
            .get("tickers_by_strategy", {})
            .get("e2_moderate", [])
        )
    if not tickers:
        raise ValueError("No tickers for E2")

    raw_dir = root / "data" / "raw" / "daily"
    clean_dir = root / "data" / "clean"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = root / "runs" / "e2_moderate" / timestamp
    ensure_dir(out_base)

    # Guardar config usado
    import shutil
    shutil.copy(cfg_path, out_base / "config_used.yaml")

    # ------------------------------------------------------------------
    # MLflow setup (remoto → local SQLite fallback)
    # ------------------------------------------------------------------
    mlflow_enabled = False
    mlflow = None
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    remote_check_timeout = float(
        os.getenv("MLFLOW_REMOTE_CHECK_TIMEOUT_SECONDS", "1.5"),
    )
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "E2_Moderate_Strategy")

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

    fallback_uri = os.getenv("MLFLOW_LOCAL_TRACKING_URI", "").strip()
    if fallback_uri:
        local_sqlite_uri = fallback_uri

    def _activate_mlflow(_mod, uri, artifact_dir=None):
        try:
            _mod.set_tracking_uri(uri)
            if artifact_dir is not None:
                exp = _mod.get_experiment_by_name(experiment_name)
                if exp is None:
                    _mod.create_experiment(
                        experiment_name,
                        artifact_location=artifact_dir.resolve().as_uri(),
                    )
                _mod.set_experiment(experiment_name)
            else:
                _mod.set_experiment(experiment_name)
            return True, None
        except Exception as exc:
            return False, str(exc)

    def _is_reachable(uri, timeout):
        parsed = urlparse(uri)
        if parsed.scheme not in {"http", "https"}:
            return True, None
        host = parsed.hostname
        if not host:
            return True, None
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True, None
        except OSError as exc:
            return False, str(exc)

    try:
        import mlflow as _mlflow  # type: ignore

        if tracking_uri:
            ok_reach, _ = _is_reachable(tracking_uri, remote_check_timeout)
            if ok_reach:
                ok, err = _activate_mlflow(_mlflow, tracking_uri)
                if ok:
                    mlflow = _mlflow
                    mlflow_enabled = True
                    print(f"✓ MLflow: {tracking_uri} (exp={experiment_name})")
                else:
                    print(f"⚠️  MLflow remoto no disponible ({err})")
            else:
                print(f"⚠️  MLflow remoto no accesible ({tracking_uri})")

        if not mlflow_enabled:
            ok, err = _activate_mlflow(
                _mlflow, local_sqlite_uri,
                artifact_dir=local_artifacts_dir,
            )
            if ok:
                mlflow = _mlflow
                mlflow_enabled = True
                mode = "local fallback" if tracking_uri else "tracking local"
                print(f"✓ MLflow ({mode}, exp={experiment_name})")
                print(f"  → Store: {local_sqlite_db}")
            else:
                print(f"⚠️  MLflow local falló: {err}")
    except Exception as exc:
        print(f"⚠️  MLflow no disponible: {exc}")

    # ------------------------------------------------------------------
    # Descarga y limpieza (config-driven: data.download; opt-out con --skip-download)
    # ------------------------------------------------------------------
    from ..data.ingest import refresh_data_for_training
    refresh_data_for_training(
        config, tickers, granularity="daily",
        skip_download=args.skip_download, root=root,
        raw_dir=raw_dir, clean_dir=clean_dir,
    )

    if args.download_only:
        print("\n--download-only: datos descargados, sin entrenar.")
        return

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------
    print("Entrenando E2 LSTM...")
    print("=" * 60)

    summaries: list[dict] = []
    for i, ticker in enumerate(tickers, 1):
        ticker_out = out_base / ticker
        print(f"\n[{i}/{len(tickers)}] {ticker}")
        print("-" * 60)
        try:
            if mlflow_enabled and mlflow is not None:
                with mlflow.start_run(run_name=f"E2_{ticker}_{timestamp}"):
                    summary = run_e2_for_ticker(
                        config, ticker=ticker, raw_dir=raw_dir,
                        out_dir=ticker_out,
                        auto_promote=args.auto_promote,
                        use_latest_data=args.use_latest_data,
                    )

                    mlflow.log_param("strategy", "e2_moderate")
                    mlflow.log_param("ticker", ticker)
                    mlflow.log_param("model_type", "LSTM")
                    mlflow.log_param("timestamp", timestamp)
                    mlflow.log_params({
                        "lookback_days": int(summary.get("lookback_days", 60)),
                        "horizon_days": int(summary.get("horizon_days", 20)),
                        "lstm_units": str([
                            int(summary.get("hp_lstm_units_1", 128)),
                            int(summary.get("hp_lstm_units_2", 64) or 64),
                        ]),
                        "dropout": summary.get("hp_dropout", 0.2),
                        "dense_units": int(summary.get("hp_dense_units", 32)),
                        "learning_rate": summary.get("hp_learning_rate", 1e-3),
                        "weight_decay": summary.get("hp_weight_decay", 0.0),
                        "batch_size": int(summary.get("hp_batch_size", 64)),
                        "tau_buy": summary.get("hp_tau_buy", 0.025),
                        "tau_sell": summary.get("hp_tau_sell", 0.00),
                        "split_method": summary.get("split_method", "walk_forward"),
                        "seed": int(config.get("project", {}).get("seed", 42)),
                    })

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
                    if metrics:
                        mlflow.log_metrics(metrics)

                    # Artifacts
                    try:
                        cfg_used = out_base / "config_used.yaml"
                        if cfg_used.exists():
                            mlflow.log_artifact(str(cfg_used), artifact_path="config")
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
                            p = ticker_out / fname
                            if p.exists():
                                if fname.endswith(".pth"):
                                    art = "models"
                                elif "pred" in fname:
                                    art = "predictions"
                                elif "backtest" in fname:
                                    art = "backtests"
                                elif "scaler" in fname:
                                    art = "scalers"
                                elif "walkforward" in fname:
                                    art = "walkforward"
                                else:
                                    art = "artifacts"
                                mlflow.log_artifact(str(p), artifact_path=art)
                    except Exception as art_exc:
                        print(f"  ⚠️  MLflow artifacts: {art_exc}")
            else:
                summary = run_e2_for_ticker(
                    config, ticker=ticker, raw_dir=raw_dir,
                    out_dir=ticker_out,
                    auto_promote=args.auto_promote,
                    use_latest_data=args.use_latest_data,
                )
            summaries.append(summary)
            print(
                f"✓ {ticker}: MAE={summary['ml_mae']:.4f} "
                f"IC={summary['ml_ic']:.3f}\n"
            )
        except Exception as exc:
            print(f"✗ Error en {ticker}: {exc}\n")

    # Summary consolidado
    pd.DataFrame(summaries).to_csv(out_base / "summary_all.csv", index=False)

    # Run agregado en MLflow
    if mlflow_enabled and mlflow is not None:
        try:
            with mlflow.start_run(run_name=f"E2_Summary_{timestamp}"):
                summary_path = out_base / "summary_all.csv"
                if summary_path.exists():
                    mlflow.log_artifact(str(summary_path), artifact_path="reports")
                mlflow.log_metric("total_tickers", float(len(tickers)))
                mlflow.log_metric("successful_tickers", float(len(summaries)))

                summary_df = pd.DataFrame(summaries)
                if not summary_df.empty:
                    numeric_cols = list(
                        summary_df.select_dtypes(include=[np.number]).columns,
                    )
                    agg: dict[str, float] = {}
                    for col in numeric_cols:
                        s = pd.to_numeric(summary_df[col], errors="coerce")
                        s = s.replace([np.inf, -np.inf], np.nan).dropna()
                        if not s.empty:
                            agg[f"summary_{col}_mean"] = float(s.mean())
                    if agg:
                        mlflow.log_metrics(agg)
        except Exception as exc:
            print(f"⚠️  MLflow summary: {exc}")

    print(f"\n✓ Resultados guardados en {out_base}")


if __name__ == "__main__":
    main()
