"""
Dashboard integration helpers for all pipelines.

Provides a single ``with_dashboard_logging`` context manager that pipelines
can wrap around their MLflow runs to automatically log dashboard tags,
timing, and alert colours.

Usage inside any pipeline:

    from src.dashboard.integration import with_dashboard_logging

    with mlflow.start_run(...):
        with with_dashboard_logging("e1_simple", ticker="AAPL"):
            # ... train, predict, log metrics ...
            pass
"""

from __future__ import annotations

import contextlib
import time
from typing import Any, Generator

from . import (
    log_dashboard_tags,
    log_dashboard_timing,
    log_metric_alerts,
)


@contextlib.contextmanager
def with_dashboard_logging(
    strategy: str,
    *,
    ticker: str | None = None,
    run_type: str = "train",
    extra_tags: dict[str, str] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Context manager that wraps an MLflow run body with dashboard logging.

    Yields a mutable dict where the caller should deposit:
      - ``"metrics"``: ``dict[str, float]``  — ML + trading metrics
      - ``"train_seconds"``: ``float``        — training duration
      - ``"predict_seconds"``: ``float``      — prediction duration

    On exit, the context manager will log dashboard tags, timing, and alerts.

    Example::

        with mlflow.start_run(run_name=...):
            with with_dashboard_logging("e2_simple", ticker="AAPL") as ctx:
                # ... training code ...
                ctx["train_seconds"] = train_duration
                ctx["predict_seconds"] = pred_duration
                ctx["metrics"] = {"ml_mae": 0.02, "bt_sharpe": 1.1, ...}
    """
    ctx: dict[str, Any] = {}
    wall_start = time.perf_counter()

    # Log tags at the start
    log_dashboard_tags(
        strategy=strategy,
        ticker=ticker,
        run_type=run_type,
        extra_tags=extra_tags,
    )

    try:
        yield ctx
    finally:
        wall_end = time.perf_counter()

        # Fall back to wall-clock time if caller didn't set explicit timing
        train_t = ctx.get("train_seconds")
        pred_t = ctx.get("predict_seconds")
        if train_t is None and pred_t is None:
            # Use total elapsed as train time estimate
            train_t = wall_end - wall_start

        log_dashboard_timing(
            train_seconds=train_t,
            predict_seconds=pred_t,
        )

        metrics = ctx.get("metrics")
        if isinstance(metrics, dict):
            log_metric_alerts(strategy, metrics)
