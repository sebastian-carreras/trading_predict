"""
MLflow dashboard helpers — log standardized tags and metrics for dashboard views.

Every pipeline should call ``log_dashboard_tags`` at the start of an MLflow run
and ``log_dashboard_timing`` after training/prediction to ensure the dashboard
can filter, group and alert correctly.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

_THRESHOLDS_CACHE: dict[str, Any] | None = None


def _load_thresholds() -> dict[str, Any]:
    """Load dashboard_thresholds.yaml (cached)."""
    global _THRESHOLDS_CACHE
    if _THRESHOLDS_CACHE is not None:
        return _THRESHOLDS_CACHE

    from ..utils import load_yaml, project_root

    path = project_root() / "src" / "config" / "dashboard_thresholds.yaml"
    if not path.exists():
        _THRESHOLDS_CACHE = {}
        return _THRESHOLDS_CACHE
    _THRESHOLDS_CACHE = load_yaml(path)
    return _THRESHOLDS_CACHE


def get_strategy_thresholds(strategy_key: str) -> dict[str, Any]:
    """Return the thresholds block for *strategy_key* or ``{}``."""
    cfg = _load_thresholds()
    return cfg.get("strategies", {}).get(strategy_key, {})


# ── Tags ────────────────────────────────────────────────────

def log_dashboard_tags(
    *,
    strategy: str,
    ticker: str | None = None,
    run_type: str = "train",
    extra_tags: dict[str, str] | None = None,
) -> None:
    """Log standard tags so the MLflow UI can filter by strategy/ticker/type.

    Parameters
    ----------
    strategy : str
        Strategy key (e.g. ``"e1_simple"``).
    ticker : str | None
        Ticker symbol or pair identifier.
    run_type : str
        One of ``"train"``, ``"predict"``, ``"evaluate"``, ``"aggregate"``.
    extra_tags : dict | None
        Additional tags to log.
    """
    try:
        import mlflow  # type: ignore

        if mlflow.active_run() is None:
            return

        mlflow.set_tag("dashboard.strategy", strategy)
        mlflow.set_tag("dashboard.run_type", run_type)
        if ticker:
            mlflow.set_tag("dashboard.ticker", ticker)

        # Friendly display name from config
        thr = get_strategy_thresholds(strategy)
        if thr.get("display_name"):
            mlflow.set_tag("dashboard.strategy_display", thr["display_name"])

        if extra_tags:
            for k, v in extra_tags.items():
                mlflow.set_tag(f"dashboard.{k}", str(v))
    except Exception:
        pass


# ── Timing ──────────────────────────────────────────────────

def log_dashboard_timing(
    *,
    train_seconds: float | None = None,
    predict_seconds: float | None = None,
) -> None:
    """Log timing metrics under a ``dashboard.*`` namespace.

    These are the canonical metric names used by the dashboard checker.
    """
    try:
        import mlflow  # type: ignore

        if mlflow.active_run() is None:
            return

        if train_seconds is not None:
            mlflow.log_metric("dashboard.train_seconds", float(train_seconds))
        if predict_seconds is not None:
            mlflow.log_metric("dashboard.predict_seconds", float(predict_seconds))
    except Exception:
        pass


# ── Alert evaluation ────────────────────────────────────────

_COLOR_GREEN = "🟢"
_COLOR_YELLOW = "🟡"
_COLOR_RED = "🔴"


def evaluate_alert(
    metric_name: str,
    value: float,
    strategy_key: str,
) -> str:
    """Return an alert color (emoji) for a metric value.

    Returns one of ``"🟢"``, ``"🟡"``, ``"🔴"`` or ``"⚪"`` (unknown).
    """
    if not math.isfinite(value):
        return "⚪"

    thr = get_strategy_thresholds(strategy_key)
    metric_cfg = thr.get("metrics", {}).get(metric_name)
    if metric_cfg is None:
        return "⚪"

    direction = metric_cfg.get("direction", "higher_is_better")

    if direction == "lower_is_better":
        green_max = metric_cfg.get("green_max", float("inf"))
        red_min = metric_cfg.get("red_min", float("inf"))
        if value <= green_max:
            return _COLOR_GREEN
        if value > red_min:
            return _COLOR_RED
        return _COLOR_YELLOW
    else:  # higher_is_better
        green_min = metric_cfg.get("green_min", float("-inf"))
        red_max = metric_cfg.get("red_max", float("-inf"))
        if value >= green_min:
            return _COLOR_GREEN
        if value < red_max:
            return _COLOR_RED
        return _COLOR_YELLOW


def evaluate_timing_alert(
    phase: str,
    seconds: float,
    strategy_key: str,
) -> str:
    """Return alert color for a timing value (train or predict)."""
    if not math.isfinite(seconds):
        return "⚪"

    thr = get_strategy_thresholds(strategy_key)
    timing_cfg = thr.get("timing", {}).get(f"{phase}_seconds")
    if timing_cfg is None:
        return "⚪"

    critical = timing_cfg.get("critical_above", float("inf"))
    warn = timing_cfg.get("warn_above", float("inf"))

    if seconds > critical:
        return _COLOR_RED
    if seconds > warn:
        return _COLOR_YELLOW
    return _COLOR_GREEN


# ── Log alerts as tags ──────────────────────────────────────

def log_metric_alerts(
    strategy_key: str,
    metrics: dict[str, float],
) -> None:
    """Evaluate each metric against thresholds and store alert color as MLflow tag.

    The dashboard UI can then filter runs by ``dashboard.alert.*`` tags.
    """
    try:
        import mlflow  # type: ignore

        if mlflow.active_run() is None:
            return

        worst = _COLOR_GREEN
        for name, value in metrics.items():
            color = evaluate_alert(name, value, strategy_key)
            mlflow.set_tag(f"dashboard.alert.{name}", color)
            if color == _COLOR_RED:
                worst = _COLOR_RED
            elif color == _COLOR_YELLOW and worst != _COLOR_RED:
                worst = _COLOR_YELLOW

        mlflow.set_tag("dashboard.alert.overall", worst)
    except Exception:
        pass
