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
from .features.build_sequences import make_sequences, time_split, temporal_train_val_split
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
    """Aplica overrides por ticker si existe un YAML configurado por env var.

    Formato esperado del YAML (archivo por estrategia):
      <TICKER>:
        thresholds: {tau_buy: ..., tau_sell: ...}
        model: {...}

    Si no hay overrides, devuelve config sin cambios.
    """
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

    # Cargar parámetros de walk-forward validation (si están en el YAML):
    if "n_folds" in per_ticker or "internal_val_fraction" in per_ticker:
        splits = config.setdefault("splits", {})
        if "n_folds" in per_ticker:
            splits["folds"] = int(per_ticker["n_folds"])
        if "internal_val_fraction" in per_ticker:
            splits["internal_val_fraction"] = float(per_ticker["internal_val_fraction"])

    return config


def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    """Carga CSV OHLCV y lo prepara.
    
    - Lee datos OHLCV desde un archivo CSV
    - Convierte la columna 'timestamp' a datetime con timezone UTC
    - Ordena datos cronológicamente y los indexa por timestamp
    - Valida que todas las columnas requeridas (OHLCV) estén presentes
    - Retorna DataFrame indexado por timestamp (timezone-aware)
    """
    # Cargar datos desde CSV
    df = pd.read_csv(path)
    
    # Validar que existe la columna timestamp
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing 'timestamp' column in {path}")

    # Convertir timestamp a datetime con timezone UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601', utc=True)
    
    # Ordenar por timestamp (cronológicamente ascendente)
    df = df.sort_values("timestamp")
    
    # Usar timestamp como índice (más eficiente para acceso temporal)
    df = df.set_index("timestamp")

    # Validar que todas las columnas OHLCV estén presentes
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")

    return df


def compute_information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el Information Coefficient (correlación de Spearman).
    
    El IC mide qué tan bien las predicciones están correlacionadas con valores reales.
    - IC > 0.05 se considera significativo en finanzas
    - IC < 0 indica overfitting o falta de capacidad predictiva
    - IC ≈ 0 indica predicciones aleatorias
    
    Retorna:
        float: IC (rango típico: [-1, 1], aunque puede ser NaN si no hay suficientes datos)
    """
    # Si hay menos de 2 muestras, no se puede calcular correlación
    if len(y_true) <= 1:
        return float("nan")

    # Si alguno de los valores es constante (sin variación), la correlación es indefinida
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")

    try:
        # Usar Spearman (correlación de rangos) que es más robusta a outliers
        from scipy.stats import spearmanr
        ic, _ = spearmanr(y_true, y_pred)
        # Asegurar que devolvemos un número finito
        return float(ic) if np.isfinite(ic) else 0.0
    except Exception:
        # Fallback a correlación de Pearson si scipy no está disponible
        return float(np.corrcoef(y_true, y_pred)[0, 1])


def _resolve_decision_profile(
    *,
    splits_cfg: dict,
    strategy_key: str,
) -> tuple[str, dict, dict, float]:
    """Resuelve perfil/targets/weights/threshold para una estrategia.

    Soporta dos esquemas de config:
    - Nuevo: splits.strategy_profile + splits.decision_profiles
    - Legacy: splits.target_metrics + splits.decision_score
    """

    strategy_profile_map = splits_cfg.get("strategy_profile")
    if not isinstance(strategy_profile_map, dict):
        strategy_profile_map = {}
    profile_name = str(strategy_profile_map.get(strategy_key, "conservative"))

    profiles_cfg = splits_cfg.get("decision_profiles")
    if not isinstance(profiles_cfg, dict):
        profiles_cfg = {}

    profile_cfg = profiles_cfg.get(profile_name)
    if not isinstance(profile_cfg, dict):
        profile_cfg = None

    # Defaults por perfil (target-based; score final se mantiene en escala ~[0,1.5])
    defaults: dict[str, dict] = {
        "conservative": {
            "threshold": 0.65,
            "weights": {
                "directional_accuracy": 0.15,
                "ic": 0.10,
                "sharpe": 0.25,
                "sortino": 0.20,
                "max_drawdown": 0.15,
                "calmar": 0.15,
            },
            "target_metrics": {
                "directional_accuracy_min": 0.58,
                "ic_min": 0.05,
                "sharpe_min": 1.0,
                "sortino_min": 1.0,
                "max_drawdown_max": 0.15,
                "calmar_min": 0.9,
            },
        },
        "moderate": {
            "threshold": 0.60,
            "weights": {
                "directional_accuracy": 0.20,
                "ic": 0.15,
                "sharpe": 0.20,
                "cagr": 0.15,
                "profit_factor": 0.15,
                "hit_rate": 0.15,
            },
            "target_metrics": {
                "directional_accuracy_min": 0.56,
                "ic_min": 0.04,
                "sharpe_min": 0.9,
                "cagr_min": 0.08,
                "profit_factor_min": 1.20,
                "hit_rate_min": 0.52,
            },
        },
        "aggressive": {
            "threshold": 0.55,
            "weights": {
                "mae": 0.10,
                "rmse": 0.10,
                "cagr": 0.25,
                "profit_factor": 0.20,
                "ic": 0.20,
                "sharpe": 0.15,
            },
            "target_metrics": {
                "mae_max": 0.03,
                "rmse_max": 0.05,
                "cagr_min": 0.12,
                "profit_factor_min": 1.25,
                "ic_min": 0.05,
                "sharpe_min": 0.9,
            },
        },
    }

    # Si no existe el perfil configurado, degradamos a conservative.
    if profile_name not in defaults:
        profile_name = "conservative"

    if profile_cfg is None:
        # Legacy fallback (compatibilidad)
        legacy_targets_obj = splits_cfg.get("target_metrics")
        legacy_targets = legacy_targets_obj if isinstance(legacy_targets_obj, dict) else {}
        legacy_score_cfg_obj = splits_cfg.get("decision_score")
        legacy_score_cfg = legacy_score_cfg_obj if isinstance(legacy_score_cfg_obj, dict) else {}

        # Si hay legacy explícito, lo priorizamos (conservative) para no romper setups existentes.
        if legacy_targets or legacy_score_cfg:
            targets = {
                "mae_max": float(legacy_targets.get("mae_max", 0.03)),
                "rmse_max": float(legacy_targets.get("rmse_max", 0.05)),
                "ic_min": float(legacy_targets.get("ic_min", 0.05)),
                "sharpe_min": float(legacy_targets.get("sharpe_min", 1.0)),
                "directional_accuracy_min": float(legacy_targets.get("directional_accuracy_min", 0.58)),
            }
            raw_weights_obj = legacy_score_cfg.get("weights")
            raw_weights = raw_weights_obj if isinstance(raw_weights_obj, dict) else {}
            weights = {
                "mae": float(raw_weights.get("mae", 0.35)),
                "ic": float(raw_weights.get("ic", 0.25)),
                "sharpe": float(raw_weights.get("sharpe", 0.25)),
                "directional_accuracy": float(raw_weights.get("directional_accuracy", 0.15)),
            }
            try:
                threshold = float(legacy_score_cfg.get("threshold", 0.75))
            except (TypeError, ValueError):
                threshold = 0.75

            total = sum(w for w in weights.values() if w > 0)
            if total <= 0:
                total = 1.0
            weights = {k: v / total for k, v in weights.items()}
            return "legacy", targets, weights, threshold

        cfg = defaults[profile_name]
        return profile_name, cfg["target_metrics"], cfg["weights"], float(cfg["threshold"])

    # Nuevo esquema: overlay defaults con config
    base_cfg = defaults[profile_name]
    if profile_cfg is None:
        profile_cfg = {}

    try:
        threshold = float(profile_cfg.get("threshold", base_cfg["threshold"]))
    except (TypeError, ValueError):
        threshold = float(base_cfg["threshold"])
    if not 0 < threshold <= 1.5:
        threshold = float(base_cfg["threshold"])

    raw_weights_obj = profile_cfg.get("weights")
    raw_weights = raw_weights_obj if isinstance(raw_weights_obj, dict) else {}
    weights = {**base_cfg["weights"], **{k: float(v) for k, v in raw_weights.items()}}

    raw_targets_obj = profile_cfg.get("target_metrics")
    raw_targets = raw_targets_obj if isinstance(raw_targets_obj, dict) else {}
    targets = {**base_cfg["target_metrics"], **{k: float(v) for k, v in raw_targets.items()}}

    total = sum(w for w in weights.values() if w > 0)
    if total <= 0:
        weights = dict(base_cfg["weights"])
        total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    return profile_name, targets, weights, float(threshold)


def compute_decision_score(
    *,
    targets: dict,
    weights: dict,
    metrics: dict,
) -> tuple[float, dict]:
    """Calcula un score compuesto para decisión de trading."""

    def _safe_ratio(
        value: float | None,
        target: float,
        *,
        higher_is_better: bool,
        abs_value: bool = False,
    ) -> float:
        if value is None or not np.isfinite(value) or target <= 0:
            return 0.0
        if abs_value:
            value = float(abs(value))
        if higher_is_better:
            ratio = value / target
        else:
            if value <= 0:
                return 1.5
            ratio = target / value
        return float(np.clip(ratio, 0.0, 1.5))

    # Map metric -> (target_key, higher_is_better, abs_value)
    spec: dict[str, tuple[str, bool, bool]] = {
        "mae": ("mae_max", False, False),
        "rmse": ("rmse_max", False, False),
        "directional_accuracy": ("directional_accuracy_min", True, False),
        "ic": ("ic_min", True, False),
        "sharpe": ("sharpe_min", True, False),
        "sortino": ("sortino_min", True, False),
        "calmar": ("calmar_min", True, False),
        "cagr": ("cagr_min", True, False),
        "profit_factor": ("profit_factor_min", True, False),
        "hit_rate": ("hit_rate_min", True, False),
        "max_drawdown": ("max_drawdown_max", False, True),
    }

    raw_components: dict[str, float] = {}
    for name, weight in weights.items():
        if weight <= 0:
            continue
        if name not in spec:
            continue
        target_key, higher_is_better, abs_value = spec[name]
        value = metrics.get(name)
        target = float(targets.get(target_key, 0.0) or 0.0)
        comp = _safe_ratio(
            float(value) if value is not None else None,
            target,
            higher_is_better=higher_is_better,
            abs_value=abs_value,
        )
        # Si no hay dato, omitimos del score (re-normalizamos weights implícitamente)
        if value is None or not np.isfinite(float(value)):
            continue
        raw_components[name] = comp

    if not raw_components:
        return 0.0, {}

    # Re-normalizar pesos solo para componentes disponibles
    active_weights = {k: float(weights.get(k, 0.0)) for k in raw_components.keys()}
    total = sum(w for w in active_weights.values() if w > 0)
    if total <= 0:
        total = 1.0
    active_weights = {k: w / total for k, w in active_weights.items()}

    score = sum(raw_components[k] * active_weights.get(k, 0.0) for k in raw_components)
    return float(score), raw_components


def save_walkforward_plot(fold_df: pd.DataFrame, ticker: str, out_path: Path) -> None:
    """Guarda grafico de IC y Sharpe por fold."""
    if fold_df.empty:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - fallback si matplotlib no está
        print(f"⚠️  No se pudo generar el gráfico walk-forward para {ticker}: {exc}")
        return

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(fold_df["fold"], fold_df["ml_ic"], marker="o")
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    axes[0].set_ylabel("IC")
    axes[0].set_title(f"{ticker} - Walk-Forward Validation")

    axes[1].plot(fold_df["fold"], fold_df["bt_sharpe"], marker="o", color="#2ca02c")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    axes[1].set_ylabel("Sharpe")
    axes[1].set_xlabel("Fold")

    if "window" in fold_df.columns:
        axes[1].set_xticks(fold_df["fold"].to_numpy())
        axes[1].set_xticklabels(fold_df["window"].to_list(), rotation=35, ha="right")

    for ax in axes:
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def run_e1_walk_forward(
    *,
    X: np.ndarray,
    y: np.ndarray,
    ts: pd.DatetimeIndex,
    feat_names: list[str] | pd.Index,
    ticker: str,
    out_dir: Path,
    ohlcv: pd.DataFrame,
    splits_cfg: dict,
    gru_units: list[int],
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
      decision_profile: str,
      decision_targets: dict,
      decision_weights: dict,
      decision_threshold: float,
) -> dict:
    """Ejecuta validación walk-forward para la estrategia E1."""

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
        embargo_days = int(embargo_cfg.get("e1", 0))
    else:
        embargo_days = int(embargo_cfg or 0)

    # Para datos diarios aproximamos gap como cantidad de muestras (1 muestra ≈ 1 día hábil)
    gap_samples = max(0, int(embargo_days))

    # Tamaño de test (en muestras) por fold. Por defecto usamos partición uniforme.
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

    print(f"    Perfil: {decision_profile}")
    print(f"    Score threshold {decision_threshold:.2f} (weights: {decision_weights})")

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

        X_tr_2d = X_train.reshape(-1, X_train.shape[-1])
        mean_X = X_tr_2d.mean(axis=0)
        std_X = X_tr_2d.std(axis=0) + 1e-12

        def scale_X(data: np.ndarray) -> np.ndarray:
            return ((data - mean_X) / std_X).astype(np.float32)

        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]

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

        model = GRURegressor(
            input_size=X_train.shape[-1],
            hidden_sizes=list(gru_units),
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

        # Guardar payload del último fold (modelo + scalers) para inferencia.
        # Elegimos el último fold porque suele ser el más cercano al período final (más relevante operativamente).
        last_model_payload = {
            "ticker": ticker,
            "strategy": "e1_conservative",
            "created_at": datetime.utcnow().isoformat(),
            "model_class": "GRURegressor",
            "model_kwargs": {
                "input_size": int(X_train.shape[-1]),
                "hidden_sizes": list(gru_units),
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

    plot_path = out_dir / f"{ticker}_walkforward_metrics.png"
    save_walkforward_plot(fold_df, ticker, plot_path)

    sharpe_all = float(trading_metrics_all.get("sharpe", float("nan")))
    metrics_for_score = {
        "mae": mae_all,
        "rmse": rmse_all,
        "directional_accuracy": dir_acc_all,
        "ic": ic_all,
        "sharpe": sharpe_all,
        "sortino": float(trading_metrics_all.get("sortino", float("nan"))),
        "calmar": float(trading_metrics_all.get("calmar", float("nan"))),
        "cagr": float(trading_metrics_all.get("cagr", float("nan"))),
        "profit_factor": float(trading_metrics_all.get("profit_factor", float("nan"))),
        "hit_rate": float(
            trading_metrics_all.get(
                "hit_rate",
                trading_metrics_all.get("win_rate", float("nan")),
            )
        ),
        "max_drawdown": float(trading_metrics_all.get("max_drawdown", float("nan"))),
    }
    decision_score, decision_components = compute_decision_score(
        metrics=metrics_for_score,
        targets=decision_targets,
        weights=decision_weights,
    )
    decision_signal = "buy" if decision_score >= decision_threshold else "hold"

    print(
        "    Decision score {:.3f} (threshold {:.2f}) → {}".format(
            decision_score,
            decision_threshold,
            "COMPRAR" if decision_signal == "buy" else "HOLD",
        )
    )

    # Guardar modelo entrenado del último fold como artifact por ticker
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
        "plot_file": as_relative(plot_path),
        "decision_score": decision_score,
        "decision_signal": decision_signal,
        "decision_profile": str(decision_profile),
        "decision_threshold": float(decision_threshold),
        # Target columns for reference
        "ic_target_min": float(decision_targets.get('ic_min', 0.05)),
        "sharpe_target_min": float(decision_targets.get('sharpe_min', 0.9)),
        "mae_target_max": float(decision_targets.get('mae_max', 0.03)),
        "rmse_target_max": float(decision_targets.get('rmse_max', 0.05)),
    }

    return summary


def run_e1_for_ticker(
    config: dict, ticker: str, raw_dir: Path, out_dir: Path, benchmark_df: pd.DataFrame | None
) -> dict:
    """Entrena y evalúa E1 para un ticker."""

    # Copia defensiva para no contaminar el config compartido (Airflow / loops)
    config = copy.deepcopy(config)
    config = _apply_tuned_overrides(config=config, strategy_key="e1_conservative", ticker=ticker)

    require_model_save = os.getenv("REQUIRE_MODEL_SAVE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }

    # Compatibilidad: si out_dir es un directorio base (p.ej. desde Airflow),
    # escribimos en un subdirectorio por ticker. Si ya es el directorio del ticker, lo usamos tal cual.
    out_dir = Path(out_dir)
    if out_dir.name != ticker:
        out_dir = out_dir / ticker
    ensure_dir(out_dir)

    # Parámetros E1 del config
    e1 = config.get("strategies", {}).get("e1_conservative", {})
    lookback_days = int(e1.get("lookback_days", 180))
    horizon_days = int(e1.get("horizon_days", 90))

    model_cfg = e1.get("model", {})
    gru_units = model_cfg.get("gru_units", [96, 32])
    dropout = float(model_cfg.get("dropout", 0.2))
    dense_units = int(model_cfg.get("dense_units", 32))
    lr = float(model_cfg.get("learning_rate", 1e-3))
    batch_size = int(model_cfg.get("batch_size", 64))
    max_epochs = int(model_cfg.get("max_epochs", 200))
    patience = int(model_cfg.get("early_stopping_patience", 15))
    loss = str(model_cfg.get("loss", "huber"))
    huber_delta = float(model_cfg.get("huber_delta", 1.0))

    seed = int(config.get("project", {}).get("seed", 42))

    # Cargar datos (priorizar datos limpios si existen)
    # raw_dir es data/raw/daily, entonces data/clean está en raw_dir.parent.parent / "clean"
    clean_dir = raw_dir.parent.parent / "clean"
    clean_csv_path = clean_dir / f"{ticker}_daily.csv"
    
    if clean_csv_path.exists():
        csv_path = clean_csv_path
        print(f"  ✓ Usando datos limpios: {csv_path.name}")
    else:
        csv_path = raw_dir / f"{ticker}_daily.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing daily CSV: {csv_path}")
        print(f"  ⚠️  Usando datos raw (limpieza no ejecutada): {csv_path.name}")

    ohlcv = load_ohlcv_csv(csv_path)

    # Features
    features = compute_e1_features(ohlcv, benchmark_df)
    target = make_target_e1(ohlcv, horizon_days=horizon_days)

    # Secuencias
    X, y, ts, feat_names = make_sequences(features, target, lookback=lookback_days)

    thresholds = e1.get("thresholds", {})
    tau_buy = float(thresholds.get("tau_buy", 0.04))
    tau_sell = float(thresholds.get("tau_sell", -0.02))

    costs_cfg = config.get("costs", {})
    round_trip_bps = float(costs_cfg.get("daily_round_trip_bps", 10))

    bt_cfg = e1.get("backtest", {})
    holding_period = int(bt_cfg.get("holding_period_days", horizon_days))
    allow_short = bool(bt_cfg.get("allow_short", False))
    max_position = float(bt_cfg.get("max_position", 1.0))

    splits_cfg = config.get("splits", {})
    decision_profile, decision_targets, decision_weights, decision_threshold = _resolve_decision_profile(
        splits_cfg=splits_cfg,
        strategy_key="e1_conservative",
    )
    if splits_cfg.get("method") == "walk_forward":
        print(
            f"  ▶ Ejecutando walk-forward ({splits_cfg.get('folds', 5)} folds, test={splits_cfg.get('test_size', 'auto')})"
        )
        summary = run_e1_walk_forward(
            X=X,
            y=y,
            ts=ts,
            feat_names=feat_names,
            ticker=ticker,
            out_dir=out_dir,
            ohlcv=ohlcv,
            splits_cfg=splits_cfg,
            gru_units=list(gru_units),
            dropout=dropout,
            dense_units=dense_units,
            learning_rate=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            loss=loss,
            huber_delta=huber_delta,
            tau_buy=tau_buy,
            tau_sell=tau_sell,
            round_trip_bps=round_trip_bps,
            holding_period=holding_period,
            allow_short=allow_short,
            max_position=max_position,
            seed=seed,
            lookback_days=lookback_days,
            horizon_days=horizon_days,
                            decision_profile=decision_profile,
            decision_targets=decision_targets,
            decision_weights=decision_weights,
            decision_threshold=decision_threshold,
        )
        return summary

    # Split temporal: dividir datos en 70% train, 15% val, 15% test
    # Se mantiene el orden temporal para evitar data leakage
    idx_train, idx_val, idx_test = time_split(len(X))
    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]
    ts_test = ts[idx_test]

    # ESTANDARIZAR FEATURES X (Z-score normalization)
    # - Usar solo datos de train para calcular media y std (prevenir data leakage)
    # - Aplicar la misma transformación a val y test
    # - Esto centra y escala los datos a media=0 y std=1
    Xtr2d = X_train.reshape(-1, X_train.shape[-1])
    mean_X = Xtr2d.mean(axis=0)
    std_X = Xtr2d.std(axis=0) + 1e-12  # +1e-12 evita división por cero

    def scale_X(Xa: np.ndarray) -> np.ndarray:
        """Aplica estandarización Z-score a las features."""
        return ((Xa - mean_X) / std_X).astype(np.float32)

    X_train_s = scale_X(X_train)
    X_val_s = scale_X(X_val)
    X_test_s = scale_X(X_test)

    # ESTANDARIZAR TARGETS y (Z-score normalization)
    # - Normalizamos los targets para estabilizar el entrenamiento
    # - Mejora convergencia durante el backprop
    mean_y = float(y_train.mean())
    std_y = float(y_train.std()) + 1e-12

    def scale_y(ya: np.ndarray) -> np.ndarray:
        """Aplica estandarización Z-score a los targets."""
        return ((ya - mean_y) / std_y).astype(np.float32)

    def unscale_y(ya_scaled: np.ndarray) -> np.ndarray:
        """Deshace la estandarización (vuelve a escala original)."""
        return (ya_scaled * std_y + mean_y).astype(np.float32)

    y_train_s = scale_y(y_train)
    y_val_s = scale_y(y_val)
    y_test_s = scale_y(y_test)

    # Entrenar modelo GRU (Gated Recurrent Unit)
    # - Red recurrente para modelar secuencias temporales
    # - Capaz de capturar dependencias a largo plazo
    model = GRURegressor(
        input_size=X_train_s.shape[-1],  # Número de features (columnas)
        hidden_sizes=gru_units,          # Unidades en capas recurrentes
        dropout=dropout,                 # Regularización: desactivar unidades al azar
        dense_units=dense_units,         # Unidades en capa densa final
        seed=seed,                       # Para reproducibilidad
    )

    print(f"Entrenando GRU para {ticker}...")
    res = model.fit(
        X_train_s,            # Features de entrenamiento (normalizados)
        y_train_s,            # Targets de entrenamiento (normalizados)
        X_val_s,              # Features de validación (para early stopping)
        y_val_s,              # Targets de validación
        learning_rate=lr,               # Velocidad de aprendizaje del optimizador
        batch_size=batch_size,          # Muestras por iteración
        max_epochs=max_epochs,          # Épocas máximas de entrenamiento
        early_stopping_patience=patience,  # Parar si no mejora en N épocas
        loss=loss,                      # Función de pérdida: "huber" o "mse"
        huber_delta=huber_delta,        # Parámetro delta para loss Huber
    )

    print(f"  Epochs: {res.epochs_ran}, Val Loss: {res.best_val_loss:.6f}")

    # Guardar modelo entrenado (por ticker)
    model_path = out_dir / f"{ticker}_model.pth"
    try:
        torch = model.torch
        payload = {
            "ticker": ticker,
            "strategy": "e1_conservative",
            "created_at": datetime.utcnow().isoformat(),
            "model_class": "GRURegressor",
            "model_kwargs": {
                "input_size": int(X_train_s.shape[-1]),
                "hidden_sizes": list(gru_units),
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
            "state_dict": {k: v.detach().cpu() for k, v in model.model.state_dict().items()},
        }
        torch.save(payload, model_path)
        print(f"  ✓ Modelo guardado: {model_path}")
    except Exception as exc:
        print(f"  ⚠️  No se pudo guardar el modelo para {ticker}: {exc}")

    if require_model_save and not model_path.exists():
        raise RuntimeError(
            f"El pipeline terminó pero NO se guardó el modelo en {model_path}. "
            "Si querés permitir continuar sin guardar, setear REQUIRE_MODEL_SAVE=0."
        )

    # Predicciones en test (normalizadas)
    y_pred_s = model.predict(X_test_s)
    
    # Desnormalizar predicciones para métricas y backtesting
    y_pred = unscale_y(y_pred_s)

    # Métricas ML
    mae = float(np.mean(np.abs(y_test - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    dir_acc = float(np.mean(np.sign(y_test) == np.sign(y_pred)))
    ic = compute_information_coefficient(y_test, y_pred)

    ml = {"mae": mae, "rmse": rmse, "directional_accuracy": dir_acc, "ic": ic}

    # Backtesting
    # Obtener precios de cierre del período de test
    close_prices_test = ohlcv.loc[ts_test, "close"].to_numpy()
    
    # Ejecutar backtest
    bt = backtest_daily_signals(
        timestamps=ts_test,
        close_prices=close_prices_test,
        pred_returns=y_pred,
        tau_buy=tau_buy,
        tau_sell=tau_sell,
        round_trip_bps=round_trip_bps,
        holding_period_days=holding_period,
        allow_short=allow_short,
        max_position=max_position,
    )
    
    # Métricas de trading
    trading_metrics = summarize_backtest(bt)

    sharpe_metric = float(trading_metrics.get("sharpe", float("nan")))
    metrics_for_score = {
        "mae": float(ml["mae"]),
        "rmse": float(ml["rmse"]),
        "directional_accuracy": float(ml["directional_accuracy"]),
        "ic": float(ml["ic"]),
        "sharpe": float(sharpe_metric),
        "sortino": float(trading_metrics.get("sortino", float("nan"))),
        "calmar": float(trading_metrics.get("calmar", float("nan"))),
        "cagr": float(trading_metrics.get("cagr", float("nan"))),
        "profit_factor": float(trading_metrics.get("profit_factor", float("nan"))),
        "hit_rate": float(trading_metrics.get("hit_rate", trading_metrics.get("win_rate", float("nan")))),
        "max_drawdown": float(trading_metrics.get("max_drawdown", float("nan"))),
    }
    decision_score, decision_components = compute_decision_score(
        metrics=metrics_for_score,
        targets=decision_targets,
        weights=decision_weights,
    )
    decision_signal = "buy" if decision_score >= decision_threshold else "hold"

    print(
        "  Decision score {:.3f} (threshold {:.2f}) → {}".format(
            decision_score,
            decision_threshold,
            "COMPRAR" if decision_signal == "buy" else "HOLD",
        )
    )

    # Guardar outputs
    ensure_dir(out_dir)

    preds_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred}, index=ts_test)
    preds_df.index.name = "timestamp"
    preds_df.to_csv(out_dir / f"{ticker}_predictions.csv")
    
    # Guardar backtest
    bt.to_csv(out_dir / f"{ticker}_backtest.csv")

    # Guardar scaler de features X
    scaler_X_df = pd.DataFrame({"mean": mean_X, "std": std_X}, index=feat_names)
    scaler_X_df.to_csv(out_dir / f"{ticker}_scaler.csv")
    
    # Guardar scaler de target y (para inferencia futura)
    scaler_y_df = pd.DataFrame({
        "mean_y": [mean_y],
        "std_y": [std_y]
    })
    scaler_y_df.to_csv(out_dir / f"{ticker}_target_scaler.csv", index=False)

    meta = {
        "ticker": ticker,
        "n_samples": int(len(X)),
        "n_test": int(len(X_test)),
        "lookback_days": lookback_days,
        "horizon_days": horizon_days,
        "split_method": splits_cfg.get("method", "time_split"),
        "epochs_ran": res.epochs_ran,
        "val_loss": res.best_val_loss,
    }

    summary = {
        **meta, 
        **{f"ml_{k}": v for k, v in ml.items()},
        **{f"bt_{k}": v for k, v in trading_metrics.items()},
        "decision_score": decision_score,
        "decision_signal": decision_signal,
        "decision_profile": str(decision_profile),
        "decision_threshold": float(decision_threshold),
    }
    pd.Series(summary).to_csv(out_dir / f"{ticker}_summary.csv")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline E1 (GRU conservador)")
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
        help="Comma-separated tickers (or uses universe.tickers_by_strategy.e1_conservative)",
    )
    args = parser.parse_args()

    root = project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path

    config = load_yaml(cfg_path)

    # Tickers E1
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        tickers = list(
            config.get("universe", {})
            .get("tickers_by_strategy", {})
            .get("e1_conservative", [])
        )

    if not tickers:
        raise ValueError("No tickers for E1")

    # Benchmark
    benchmark = config.get("universe", {}).get("benchmark", "SPY")
    raw_dir = root / "data" / "raw" / "daily"

    benchmark_path = raw_dir / f"{benchmark}_daily.csv"
    if benchmark_path.exists():
        benchmark_df = load_ohlcv_csv(benchmark_path)
    else:
        print(f"⚠️  Benchmark {benchmark} no encontrado, usando valores vacíos")
        benchmark_df = None

    out_base = root / "runs" / "e1_conservative" / datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_dir(out_base)

    # Guardar config usado
    import shutil
    shutil.copy(cfg_path, out_base / "config_used.yaml")

    # MLflow (opcional): si está instalado y hay tracking URI, logueamos params/metrics/artifacts.
    mlflow_enabled = False
    mlflow = None
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "E1_Conservative")
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
                with mlflow.start_run(run_name=f"E1_{ticker}_{timestamp}"):
                    mlflow.log_param("strategy", "e1_conservative")
                    mlflow.log_param("ticker", ticker)
                    mlflow.log_param("model_type", "GRU")
                    mlflow.log_param("timestamp", timestamp)

                    e1_cfg = config.get("strategies", {}).get("e1_conservative", {})
                    model_cfg = e1_cfg.get("model", {})
                    mlflow.log_params(
                        {
                            "lookback_days": int(e1_cfg.get("lookback_days", 180)),
                            "horizon_days": int(e1_cfg.get("horizon_days", 90)),
                            "gru_units": str(model_cfg.get("gru_units", [96, 32])),
                            "dropout": float(model_cfg.get("dropout", 0.2)),
                            "dense_units": int(model_cfg.get("dense_units", 32)),
                            "learning_rate": float(model_cfg.get("learning_rate", 1e-3)),
                            "batch_size": int(model_cfg.get("batch_size", 64)),
                            "max_epochs": int(model_cfg.get("max_epochs", 200)),
                            "early_stopping_patience": int(model_cfg.get("early_stopping_patience", 15)),
                            "loss": str(model_cfg.get("loss", "huber")),
                            "huber_delta": float(model_cfg.get("huber_delta", 1.0)),
                            "split_method": str(config.get("splits", {}).get("method", "time_split")),
                            "seed": int(config.get("project", {}).get("seed", 42)),
                        }
                    )

                    summary = run_e1_for_ticker(
                        config,
                        ticker=ticker,
                        raw_dir=raw_dir,
                        out_dir=ticker_out,
                        benchmark_df=benchmark_df,
                    )

                    # Métricas
                    metrics: dict[str, float] = {}
                    for k, v in summary.items():
                        if not (k.startswith("ml_") or k.startswith("bt_")):
                            continue
                        if isinstance(v, (int, float)):
                            metrics[k] = float(v)
                    if metrics:
                        mlflow.log_metrics(metrics)

                    if isinstance(summary.get("decision_score"), (int, float)):
                        mlflow.log_metric("decision_score", float(summary["decision_score"]))

                    if summary.get("decision_profile"):
                        mlflow.log_param("decision_profile", str(summary["decision_profile"]))

                    # No loguear componentes individuales del decision score en MLflow

                    if summary.get("decision_signal"):
                        mlflow.log_param("decision_signal", str(summary["decision_signal"]))

                    # Artifacts por ticker
                    config_used = out_base / "config_used.yaml"
                    if config_used.exists():
                        mlflow.log_artifact(str(config_used), artifact_path="config")

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
                            # agrupamos por tipo para que sea navegable en la UI
                            if fname.endswith(".pth"):
                                artifact_path = "models"
                            elif "pred" in fname:
                                artifact_path = "predictions"
                            elif "backtest" in fname:
                                artifact_path = "backtests"
                            elif "scaler" in fname:
                                artifact_path = "scalers"
                            elif "walkforward" in fname:
                                artifact_path = "walkforward"
                            else:
                                artifact_path = "artifacts"
                            mlflow.log_artifact(str(p), artifact_path=artifact_path)
            else:
                summary = run_e1_for_ticker(
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
            with mlflow.start_run(run_name=f"E1_Summary_{timestamp}"):
                summary_path = out_base / "summary_all.csv"
                if summary_path.exists():
                    mlflow.log_artifact(str(summary_path), artifact_path="reports")
                mlflow.log_metric("total_tickers", float(len(tickers)))
                mlflow.log_metric("successful_tickers", float(len(summaries)))

                decision_scores: list[float] = []
                for s in summaries:
                    v = s.get("decision_score")
                    if isinstance(v, (int, float)):
                        decision_scores.append(float(v))
                buy_signals = sum(1 for s in summaries if s.get("decision_signal") == "buy")

                mlflow.log_metric("decision_buy_signals", float(buy_signals))
                if decision_scores:
                    mlflow.log_metric(
                        "decision_score_mean", float(sum(decision_scores) / len(decision_scores))
                    )
                    mlflow.log_metric("decision_score_min", float(min(decision_scores)))
                    mlflow.log_metric("decision_score_max", float(max(decision_scores)))
        except Exception as exc:
            print(f"⚠️  No se pudo loguear el summary en MLflow: {exc}")

    print(f"\n✓ Resultados guardados en {out_base}")


if __name__ == "__main__":
    main()
