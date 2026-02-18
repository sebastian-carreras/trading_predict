"""Phase-1 guardrails: only reject technically broken models.

All metrics are logged to a JSONL file for future empirical threshold
calibration (Phase 2).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def validate_candidate(
    run_dir: str | Path,
    ticker: str,
    *,
    baseline_metrics: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Run permissive guardrails on a trained model.

    Only rejects models that are technically broken:
    1. Model file missing or unloadable
    2. NaN / Inf in predictions
    3. NaN / Inf in backtest
    4. Sharpe < 0 (loses money)
    5. Worse than baseline on IC (if baseline provided)

    Returns:
        (passed, errors): True if model passes, list of error messages.
    """
    run_dir = Path(run_dir)
    errors: list[str] = []

    # 1. Model file exists
    model_files = list(run_dir.glob(f"{ticker}*_model.pth")) + list(
        run_dir.glob(f"{ticker}*model.pth")
    )
    if not model_files:
        errors.append(f"No model file found in {run_dir}")

    # 2. Check predictions for NaN/Inf
    pred_files = list(run_dir.glob(f"{ticker}*predictions*.csv"))
    for pred_file in pred_files:
        try:
            df = pd.read_csv(pred_file)
            for col in df.select_dtypes(include=[np.number]).columns:
                vals = df[col].to_numpy(dtype=float)
                if not np.isfinite(vals).all():
                    errors.append(f"NaN/Inf in predictions column '{col}': {pred_file.name}")
        except Exception as exc:
            errors.append(f"Cannot read predictions {pred_file.name}: {exc}")

    # 3. Check backtest for NaN/Inf
    bt_files = list(run_dir.glob(f"{ticker}*backtest*.csv"))
    for bt_file in bt_files:
        try:
            bt = pd.read_csv(bt_file)
            check_cols = [c for c in ("net_ret", "equity", "costs", "turnover") if c in bt.columns]
            for col in check_cols:
                vals = bt[col].to_numpy(dtype=float)
                if not np.isfinite(vals).all():
                    errors.append(f"NaN/Inf in backtest column '{col}': {bt_file.name}")
        except Exception as exc:
            errors.append(f"Cannot read backtest {bt_file.name}: {exc}")

    # 4. Check summary metrics
    summary_files = list(run_dir.glob("*summary*.csv")) + list(
        run_dir.parent.glob("summary_all.csv")
    )
    sharpe = _extract_metric(run_dir, ticker, "bt_sharpe", "sharpe")
    if sharpe is not None and sharpe < 0:
        errors.append(f"Sharpe ratio {sharpe:.4f} < 0 (model loses money)")

    # 5. Compare against baseline
    if baseline_metrics is not None:
        baseline_ic = baseline_metrics.get("ml_ic")
        candidate_ic = _extract_metric(run_dir, ticker, "ml_ic")
        if baseline_ic is not None and candidate_ic is not None:
            if candidate_ic < baseline_ic:
                errors.append(
                    f"IC {candidate_ic:.4f} worse than baseline {baseline_ic:.4f}"
                )

    return len(errors) == 0, errors


def log_candidate_metrics(
    metrics: dict[str, Any],
    strategy: str,
    ticker: str,
    run_dir: str | Path,
    variant: str,
    *,
    log_path: str | Path | None = None,
) -> Path:
    """Append all metrics to JSONL for future threshold calibration.

    Returns the path to the log file.
    """
    if log_path is None:
        log_path = Path(run_dir).parents[2] / "models" / "metrics_log.jsonl"

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "ticker": ticker,
        "variant": variant,
        "run_dir": str(run_dir),
    }

    # Include all numeric metrics
    for key, value in metrics.items():
        try:
            fval = float(value)
            if fval == fval:  # not NaN
                entry[key] = round(fval, 6)
        except (TypeError, ValueError):
            pass

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")

    return log_path


def _extract_metric(
    run_dir: Path,
    ticker: str,
    *metric_names: str,
) -> float | None:
    """Try to extract a metric from summary CSV in run_dir or parent."""
    for summary_path in [
        run_dir / f"{ticker}_summary.csv",
        run_dir.parent / "summary_all.csv",
    ]:
        if not summary_path.exists():
            continue
        try:
            df = pd.read_csv(summary_path)
            row = df[df["ticker"] == ticker]
            if row.empty:
                continue
            for name in metric_names:
                if name in row.columns:
                    val = row[name].iloc[0]
                    fval = float(val)
                    if fval == fval:  # not NaN
                        return fval
        except Exception:
            continue
    return None
