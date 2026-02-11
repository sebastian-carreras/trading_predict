"""
Dashboard health checker — CLI tool that queries MLflow and the timing log
to produce a per-strategy summary with alert colors.

Usage (local):
    python -m src.dashboard.checker [--strategy e1_simple] [--last N]

Usage (Docker):
    docker compose exec mlflow python -m src.dashboard.checker

The output is a colour-coded table printed to stdout **and** saved as
``reports/dashboard/dashboard_report.csv``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils import ensure_dir, load_yaml, project_root
from . import (
    evaluate_alert,
    evaluate_timing_alert,
    get_strategy_thresholds,
)


# ── Timing log reader ──────────────────────────────────────

def read_timing_log(log_path: Path | None = None) -> pd.DataFrame:
    """Read the JSONL timing log into a DataFrame."""
    if log_path is None:
        log_path = project_root() / "reports" / "timing" / "timing_log.jsonl"
    if not log_path.exists():
        return pd.DataFrame()

    records: list[dict] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "ended_at" in df.columns:
        df["ended_at"] = pd.to_datetime(df["ended_at"], utc=True, errors="coerce")
    return df


def timing_summary_for_strategy(
    df_timing: pd.DataFrame,
    strategy: str,
) -> dict[str, Any]:
    """Compute avg and last timing for train/predict phases."""
    result: dict[str, Any] = {}
    if df_timing.empty:
        return result

    subset = df_timing[df_timing["strategy"] == strategy].copy()
    if subset.empty:
        return result

    for phase in ("train", "predict"):
        phase_df = subset[subset["phase"] == phase]
        if phase_df.empty:
            continue

        durations = phase_df["duration_seconds"].astype(float)
        result[f"{phase}_avg_seconds"] = float(durations.mean())
        result[f"{phase}_p50_seconds"] = float(durations.median())
        result[f"{phase}_p95_seconds"] = float(durations.quantile(0.95)) if len(durations) > 1 else float(durations.iloc[0])

        # Last run
        if "ended_at" in phase_df.columns and phase_df["ended_at"].notna().any():
            last_row = phase_df.sort_values("ended_at").iloc[-1]
        else:
            last_row = phase_df.iloc[-1]
        result[f"{phase}_last_seconds"] = float(last_row["duration_seconds"])
        if "ended_at" in last_row and pd.notna(last_row["ended_at"]):
            result[f"{phase}_last_at"] = str(last_row["ended_at"])

    return result


# ── MLflow reader ───────────────────────────────────────────

def _mlflow_runs_for_experiment(
    experiment_name: str,
    max_results: int = 200,
) -> pd.DataFrame:
    """Query MLflow for runs in an experiment."""
    try:
        import mlflow  # type: ignore

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return pd.DataFrame()

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=max_results,
        )
        if not runs:
            return pd.DataFrame()

        rows = []
        for run in runs:
            row: dict[str, Any] = {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name or "",
                "status": run.info.status,
                "start_time": datetime.fromtimestamp(
                    run.info.start_time / 1000, tz=timezone.utc
                )
                if run.info.start_time
                else None,
            }
            row.update(run.data.params)
            row.update(run.data.metrics)
            for k, v in run.data.tags.items():
                row[f"tag.{k}"] = v
            rows.append(row)

        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def mlflow_metrics_summary(
    experiment_name: str,
    metric_keys: list[str],
    max_results: int = 200,
) -> dict[str, Any]:
    """Compute avg and last-run values for requested metric keys."""
    df = _mlflow_runs_for_experiment(experiment_name, max_results)
    if df.empty:
        return {}

    # Only FINISHED runs
    if "status" in df.columns:
        df = df[df["status"] == "FINISHED"]
    if df.empty:
        return {}

    # Drop _init_test / empty runs (no metrics logged)
    if "run_name" in df.columns:
        df = df[~df["run_name"].str.contains("_init_test", na=False)]
    if df.empty:
        return {}

    result: dict[str, Any] = {}

    # Last run (most recent with actual metrics)
    if "start_time" in df.columns:
        df_sorted = df.sort_values("start_time", ascending=False)
    else:
        df_sorted = df

    # Find first run that has at least one of the requested metric keys
    last = df_sorted.iloc[0]
    for _, candidate in df_sorted.iterrows():
        has_any = any(
            key in candidate and pd.notna(candidate.get(key))
            for key in metric_keys
        )
        if has_any:
            last = candidate
            break

    result["last_run_name"] = last.get("run_name", "")
    result["last_run_time"] = str(last.get("start_time", ""))
    result["total_runs"] = len(df)

    for key in metric_keys:
        if key in df.columns:
            vals = df[key].dropna().astype(float)
            if not vals.empty:
                result[f"{key}_avg"] = float(vals.mean())
                result[f"{key}_last"] = float(last.get(key, float("nan")))
            else:
                result[f"{key}_avg"] = float("nan")
                result[f"{key}_last"] = float("nan")
        else:
            result[f"{key}_avg"] = float("nan")
            result[f"{key}_last"] = float("nan")

    return result


# ── Report builder ──────────────────────────────────────────

def build_dashboard_report(
    strategies: list[str] | None = None,
) -> pd.DataFrame:
    """Build the full dashboard report as a DataFrame.

    Each row is a (strategy, metric) pair with avg/last values and alert color.
    """
    cfg_path = project_root() / "src" / "config" / "dashboard_thresholds.yaml"
    if not cfg_path.exists():
        print("⚠️  dashboard_thresholds.yaml not found")
        return pd.DataFrame()

    cfg = load_yaml(cfg_path)
    all_strategies = cfg.get("strategies", {})

    if strategies:
        all_strategies = {k: v for k, v in all_strategies.items() if k in strategies}

    # Read timing log once
    df_timing = read_timing_log()

    rows: list[dict[str, Any]] = []

    for strat_key, strat_cfg in all_strategies.items():
        display = strat_cfg.get("display_name", strat_key)
        experiment = strat_cfg.get("experiment_name", strat_key)

        # --- Timing ---
        timing = timing_summary_for_strategy(df_timing, strat_key)

        for phase in ("train", "predict"):
            avg_key = f"{phase}_avg_seconds"
            last_key = f"{phase}_last_seconds"
            p50_key = f"{phase}_p50_seconds"
            p95_key = f"{phase}_p95_seconds"
            avg_val = timing.get(avg_key, float("nan"))
            last_val = timing.get(last_key, float("nan"))
            p50_val = timing.get(p50_key, float("nan"))
            p95_val = timing.get(p95_key, float("nan"))

            alert = evaluate_timing_alert(phase, last_val, strat_key)

            rows.append({
                "strategy": display,
                "strategy_key": strat_key,
                "category": "⏱ Timing",
                "metric": f"{phase}_seconds",
                "avg": _fmt(avg_val, 1),
                "p50": _fmt(p50_val, 1),
                "p95": _fmt(p95_val, 1),
                "last": _fmt(last_val, 1),
                "alert": alert,
            })

        # --- ML / Trading metrics ---
        metric_keys = list(strat_cfg.get("metrics", {}).keys())
        mlflow_data = mlflow_metrics_summary(experiment, metric_keys)

        for metric_name, metric_cfg in strat_cfg.get("metrics", {}).items():
            avg_val = mlflow_data.get(f"{metric_name}_avg", float("nan"))
            last_val = mlflow_data.get(f"{metric_name}_last", float("nan"))
            alert = evaluate_alert(metric_name, last_val, strat_key)

            cat = "📈 Trading" if any(
                p in metric_name for p in ("bt_", "tr_", "sharpe", "cagr", "drawdown", "profit", "return", "win_rate")
            ) else "🤖 ML"

            rows.append({
                "strategy": display,
                "strategy_key": strat_key,
                "category": cat,
                "metric": metric_name,
                "avg": _fmt(avg_val, 4),
                "p50": "",
                "p95": "",
                "last": _fmt(last_val, 4),
                "alert": alert,
            })

        # Add MLflow run count
        total = mlflow_data.get("total_runs", 0)
        last_time = mlflow_data.get("last_run_time", "")
        rows.append({
            "strategy": display,
            "strategy_key": strat_key,
            "category": "ℹ️  Info",
            "metric": "total_mlflow_runs",
            "avg": str(total),
            "p50": "",
            "p95": "",
            "last": str(last_time)[:19] if last_time else "",
            "alert": "⚪",
        })

    return pd.DataFrame(rows)


def _fmt(val: float, decimals: int = 4) -> str:
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return "—"
    return f"{val:.{decimals}f}"


# ── Pretty print ────────────────────────────────────────────

def print_report(df: pd.DataFrame) -> None:
    """Print a formatted dashboard report to stdout."""
    if df.empty:
        print("No data available for dashboard report.")
        return

    strategies = df["strategy"].unique()

    for strat in strategies:
        sdf = df[df["strategy"] == strat]
        strat_key = sdf["strategy_key"].iloc[0]

        print(f"\n{'='*70}")
        print(f"  {strat}")
        thr = get_strategy_thresholds(strat_key)
        if thr.get("description"):
            print(f"  {thr['description']}")
        print(f"{'='*70}")

        for cat in sdf["category"].unique():
            cat_df = sdf[sdf["category"] == cat]
            print(f"\n  {cat}")
            print(f"  {'─'*64}")
            print(f"  {'Metric':<30} {'Avg':>8} {'Last':>8} {'Alert':>6}")
            print(f"  {'─'*64}")
            for _, row in cat_df.iterrows():
                metric = row["metric"]
                avg = row["avg"]
                last = row["last"]
                alert = row["alert"]
                print(f"  {metric:<30} {avg:>8} {last:>8}  {alert}")

    print()


# ── Save report ─────────────────────────────────────────────

def save_report(df: pd.DataFrame) -> Path:
    """Save report to CSV."""
    out_dir = project_root() / "reports" / "dashboard"
    ensure_dir(out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"dashboard_report_{ts}.csv"
    df.to_csv(path, index=False)

    # Also save as latest
    latest = out_dir / "dashboard_report_latest.csv"
    df.to_csv(latest, index=False)

    return path


# ── CLI ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dashboard health checker — per-strategy metrics + alerts"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Filter by strategy key (e.g. e1_simple). Omit for all.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save report CSV (default: True).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout output.",
    )
    args = parser.parse_args()

    strategies = [args.strategy] if args.strategy else None
    df = build_dashboard_report(strategies)

    if not args.quiet:
        print_report(df)

    if args.save and not df.empty:
        path = save_report(df)
        print(f"✓ Report saved: {path.relative_to(project_root())}")


if __name__ == "__main__":
    main()
