"""Fair re-evaluation of a frozen model on a recent out-of-sample window.

Runs an already-trained model (typically the current champion) *forward* over a
set of dates using today's data, WITHOUT retraining, and recomputes backtest
metrics with the very same engine used during training. This is what lets
promotion compare champion vs. candidate on a **common out-of-sample window**
(dates neither model trained on) instead of pitting the champion's stale stored
metrics against the candidate's fresh ones.

Why this is correct
--------------------
* ``models/<run>/<TICKER>_model.pth`` stores the *last walk-forward fold's*
  model + scalers — i.e. exactly the frozen model that is served in production.
  Re-running it over new dates is honest inference, not retraining.
* The E1/E2 features are causal (built from past prices only), so the lookback
  window ending at a date ``d`` is identical regardless of how much *future*
  data ``ohlcv`` contains. That is what makes it valid to reproduce a model's
  past predictions — and what the inference-correctness test asserts.

The heavy training modules (MLflow, optimisers, …) are intentionally NOT
imported here; only the lightweight feature/sequence/backtest helpers are.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.backtest_daily import backtest_daily_signals, summarize_backtest
from ..features.build_sequences_e1e2 import make_sequences
from ..metrics.skill import (
    directional_accuracy_edge,
    naive_up_rate,
    pesaran_timmermann,
    sharpe_excess,
)
from .loader import ModelLoader
from .registry import ModelRegistry


# ---------------------------------------------------------------
# Strategy dispatch (feature/target builders + model classes)
# ---------------------------------------------------------------
def _strategy_kit(strategy: str):
    """Return ``(compute_features, make_target)`` for a strategy prefix.

    Model reconstruction no longer lives here: the model class is resolved from
    the global model registry inside ``_rebuild_model`` (keyed by the payload's
    ``model_class`` string). Adding a new model therefore requires **no edit**
    to this function — only registering the class via ``@register_model``.
    """
    prefix = strategy.split("_", 1)[0].lower()
    if prefix == "e1":
        from ..e1.build_features import compute_e1_features, make_target_e1
        return compute_e1_features, make_target_e1
    if prefix == "e2":
        from ..e2.build_features import compute_e2_features, make_target_e2
        return compute_e2_features, make_target_e2
    raise ValueError(f"Re-evaluation not supported for strategy={strategy!r}")


def _exog_for(strategy: str, ticker: str | None, feat_names: list[str], root: Path | None = None):
    """Carga el frame exógeno si el modelo (por sus feature_names) usa features exógenas.

    Devuelve None para modelos base, E1, o ticker desconocido → la reevaluación recompone
    features solo del OHLCV, como siempre. NO construye el cache por red
    (build_if_missing=False): si falta, propaga y el llamador degrada a 'stored'/None.
    """
    if not ticker or not strategy.lower().startswith(("e1", "e2")):
        return None
    from ..data.macro import EXOG_FEATURE_COLS, load_exog_for

    if not any(f in set(EXOG_FEATURE_COLS) for f in feat_names):
        return None
    return load_exog_for(ticker, root=root, build_if_missing=False)


def _rebuild_model(payload: dict[str, Any]):
    """Reconstruct a frozen predictor from a saved ``model.pth`` payload.

    Resolves the class from the global model registry (single source of truth),
    then delegates to the class's ``from_payload`` when present. Falls back to
    the historical state_dict path for older payloads or classes without the
    torch mixin.
    """
    from ..models.registry import get_model_class

    cls = get_model_class(payload["model_class"])  # raises ValueError if unknown
    if hasattr(cls, "from_payload"):
        return cls.from_payload(payload)
    model = cls(**payload["model_kwargs"])
    model.model.load_state_dict(payload["state_dict"])
    model.model.eval()
    return model


def _information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman IC (falls back to Pearson); NaN when undefined.

    Mirrors ``src.e1.train_pipeline.compute_information_coefficient`` so the
    recomputed metric is comparable to the stored one.
    """
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        ic, _ = spearmanr(y_true, y_pred)
        return float(ic)
    except Exception:
        try:
            return float(np.corrcoef(y_true, y_pred)[0, 1])
        except Exception:
            return float("nan")


# ---------------------------------------------------------------
# Frozen-model inference
# ---------------------------------------------------------------
def predict_series(
    strategy: str,
    payload: dict[str, Any],
    ohlcv: pd.DataFrame,
    *,
    ticker: str | None = None,
    root: Path | None = None,
) -> pd.Series:
    """Predicted forward returns for every reconstructable window-end date.

    Uses the frozen model + scalers stored in ``payload`` and features built
    from ``ohlcv``. Returns a Series (unscaled, real return space) indexed by
    the window-end timestamp. Only dates that have BOTH a full feature window
    and a realised target survive ``make_sequences``' dropna — which is exactly
    the set usable for a fair backtest comparison.
    """
    compute_features, make_target = _strategy_kit(strategy)
    model = _rebuild_model(payload)

    lookback = int(payload["lookback_days"])
    horizon = int(payload["horizon_days"])
    feat_names = list(payload["feature_names"])

    exog = _exog_for(strategy, ticker, feat_names, root)
    features = compute_features(ohlcv, exog=exog) if exog is not None else compute_features(ohlcv)
    # Select + order features exactly as during training.
    features = features[feat_names]
    target = make_target(ohlcv, horizon_days=horizon)

    X, _y, ts, _ = make_sequences(features, target, lookback=lookback)

    mean_X = np.asarray(payload["scaler_X"]["mean"], dtype=np.float32)
    std_X = np.asarray(payload["scaler_X"]["std"], dtype=np.float32)
    X_s = ((X - mean_X) / std_X).astype(np.float32)

    y_pred_s = model.predict(X_s)
    mean_y = float(payload["scaler_y"]["mean"])
    std_y = float(payload["scaler_y"]["std"])
    y_pred = np.asarray(y_pred_s, dtype=np.float64) * std_y + mean_y

    return pd.Series(y_pred, index=ts, name="y_pred")


def champion_predictions(
    strategy: str,
    ticker: str,
    registry: ModelRegistry,
    ohlcv: pd.DataFrame,
    *,
    root: Path | None = None,
) -> pd.Series | None:
    """Frozen-champion predictions over all reconstructable dates, or None."""
    loader = ModelLoader(registry, root)
    payload = loader.load_champion(strategy, ticker)
    if payload is None:
        return None
    return predict_series(strategy, payload, ohlcv, ticker=ticker, root=root)


# ---------------------------------------------------------------
# Live "today" forecast (leaderboard / serving)
# ---------------------------------------------------------------
def predict_latest(
    strategy: str,
    payload: dict[str, Any],
    ohlcv: pd.DataFrame,
    *,
    ticker: str | None = None,
    root: Path | None = None,
) -> tuple[float, pd.Timestamp] | None:
    """The model's forecast for the MOST RECENT window (no realised target).

    Unlike ``predict_series`` (which aligns to a target and therefore lags by
    the horizon), this builds the last ``lookback`` rows of features up to the
    latest available date and returns ``(pred_return, as_of)`` — the frozen
    model's predicted forward return as of today. Returns None when there is
    not enough history or the champion's feature set no longer exists (drift).
    """
    compute_features, _make_target = _strategy_kit(strategy)
    lookback = int(payload["lookback_days"])
    feat_names = list(payload["feature_names"])

    try:
        exog = _exog_for(strategy, ticker, feat_names, root)
    except Exception:
        return None  # cache exógeno ausente/ilegible → tratar como no-servable (drift)
    features = compute_features(ohlcv, exog=exog) if exog is not None else compute_features(ohlcv)
    try:
        features = features[feat_names].dropna()
    except KeyError:
        return None  # champion trained with features the current code no longer builds
    if len(features) < lookback:
        return None

    window = features.to_numpy(dtype=np.float32)[-lookback:][None, ...]  # (1, lookback, n_feat)
    mean_X = np.asarray(payload["scaler_X"]["mean"], dtype=np.float32)
    std_X = np.asarray(payload["scaler_X"]["std"], dtype=np.float32)
    X_s = ((window - mean_X) / std_X).astype(np.float32)

    model = _rebuild_model(payload)
    y_pred_s = model.predict(X_s)
    mean_y = float(payload["scaler_y"]["mean"])
    std_y = float(payload["scaler_y"]["std"])
    pred = float(np.asarray(y_pred_s, dtype=np.float64).ravel()[0]) * std_y + mean_y
    return pred, features.index[-1]


def champion_latest_forecast(
    strategy: str,
    ticker: str,
    registry: ModelRegistry,
    ohlcv: pd.DataFrame,
    *,
    root: Path | None = None,
) -> tuple[float, pd.Timestamp] | None:
    """Live forecast from the current champion, or None if it can't be run."""
    loader = ModelLoader(registry, root)
    try:
        payload = loader.load_champion(strategy, ticker)
    except ModuleNotFoundError:
        # torch no instalado (ej. FastAPI): el llamador cae al y_pred guardado.
        return None
    if payload is None:
        return None
    try:
        return predict_latest(strategy, payload, ohlcv, ticker=ticker, root=root)
    except Exception:
        return None


def is_champion_servable(
    strategy: str,
    ticker: str,
    registry: ModelRegistry,
    ohlcv: pd.DataFrame | None,
    *,
    root: Path | None = None,
) -> bool:
    """True iff the champion can produce a live forecast on ``ohlcv`` today.

    Used to force-promote a fresh candidate when the incumbent champion is
    unservable (feature drift or missing model file).
    """
    if ohlcv is None or len(ohlcv) == 0:
        return False
    return champion_latest_forecast(strategy, ticker, registry, ohlcv, root=root) is not None


def load_clean_ohlcv(ticker: str, *, root: Path | None = None) -> pd.DataFrame | None:
    """Load ``data/clean/<ticker>_daily.csv`` (ts-indexed), or None if missing.

    Reuses the pipeline's ``load_ohlcv_csv`` parser so the leaderboard and the
    DAG share one loader.
    """
    if root is None:
        from ..utils import project_root
        root = project_root()
    path = Path(root) / "data" / "clean" / f"{ticker}_daily.csv"
    if not path.exists():
        return None
    from ..e1.train_pipeline import load_ohlcv_csv
    return load_ohlcv_csv(path)


# ---------------------------------------------------------------
# Walk-forward predictions (candidate side — already OOS on disk)
# ---------------------------------------------------------------
def load_walkforward_predictions(
    run_dir: str | Path,
    ticker: str,
    *,
    root: Path | None = None,
) -> pd.DataFrame | None:
    """Load ``<run_dir>/<ticker>_walkforward_predictions.csv`` (ts-indexed)."""
    rd = Path(run_dir)
    if not rd.is_absolute() and root is not None:
        rd = Path(root) / rd
    path = rd / f"{ticker}_walkforward_predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "timestamp"
    return df


# ---------------------------------------------------------------
# Backtest metrics on a given window
# ---------------------------------------------------------------
def recompute_metrics_on_window(
    ohlcv: pd.DataFrame,
    dates: pd.DatetimeIndex,
    pred_returns: np.ndarray,
    y_true: np.ndarray,
    backtest_params: dict[str, Any],
) -> dict[str, float]:
    """Backtest ``pred_returns`` over ``dates`` and return registry-style metrics.

    Keys mirror what training writes to the registry (``bt_*`` from
    ``summarize_backtest`` plus ``ml_ic`` / ``ml_directional_accuracy`` /
    ``ml_mae`` / ``ml_rmse``), so ``promotion.compute_score`` consumes them
    unchanged.
    """
    dates = pd.DatetimeIndex(dates)
    close = ohlcv.loc[dates, "close"].to_numpy()
    bt = backtest_daily_signals(
        timestamps=dates,
        close_prices=close,
        pred_returns=np.asarray(pred_returns, dtype=np.float64),
        tau_buy=float(backtest_params.get("tau_buy", 0.02)),
        tau_sell=float(backtest_params.get("tau_sell", 0.00)),
        round_trip_bps=float(backtest_params.get("round_trip_bps", 10.0)),
        holding_period_days=int(backtest_params.get("holding_period_days", 90)),
        allow_short=bool(backtest_params.get("allow_short", False)),
        max_position=float(backtest_params.get("max_position", 1.0)),
    )
    trading = summarize_backtest(bt)
    metrics: dict[str, float] = {f"bt_{k}": float(v) for k, v in trading.items()}

    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(pred_returns, dtype=np.float64)
    metrics["ml_directional_accuracy"] = float(np.mean(np.sign(yt) == np.sign(yp)))
    metrics["ml_ic"] = _information_coefficient(yt, yp)
    metrics["ml_mae"] = float(np.mean(np.abs(yt - yp)))
    metrics["ml_rmse"] = float(np.sqrt(np.mean((yt - yp) ** 2)))

    # Métricas neutrales a la deriva, mismas claves que escribe el entrenamiento, para
    # que la comparación justa champion-vs-candidato pueda usarlas. Ver src/metrics/skill.py.
    metrics["ml_naive_up_rate"] = naive_up_rate(yt)
    metrics["ml_dir_acc_edge"] = directional_accuracy_edge(yt, yp)
    pt_stat, pt_pvalue = pesaran_timmermann(yt, yp)
    metrics["ml_pt_stat"] = pt_stat
    metrics["ml_pt_pvalue"] = pt_pvalue
    metrics["bt_sharpe_excess"] = sharpe_excess(metrics.get("bt_sharpe", float("nan")), close)
    return metrics


def backtest_params_from_config(config: dict[str, Any], variant: str) -> dict[str, Any]:
    """Assemble backtest params for a strategy variant from ``base.yaml``.

    Reuses the same thresholds/costs/holding that ``train_pipeline`` uses, so
    the re-backtest is apples-to-apples with the stored metrics.
    """
    strat = config.get("strategies", {}).get(variant, {})
    thresholds = strat.get("thresholds", {})
    bt_cfg = strat.get("backtest", {})
    horizon = int(strat.get("horizon_days", 90))
    costs = config.get("costs", {})
    return {
        "tau_buy": float(thresholds.get("tau_buy", 0.02)),
        "tau_sell": float(thresholds.get("tau_sell", 0.00)),
        "round_trip_bps": float(costs.get("daily_round_trip_bps", 10.0)),
        "holding_period_days": int(bt_cfg.get("holding_period_days", horizon)),
        "allow_short": bool(bt_cfg.get("allow_short", False)),
        "max_position": float(bt_cfg.get("max_position", 1.0)),
    }
