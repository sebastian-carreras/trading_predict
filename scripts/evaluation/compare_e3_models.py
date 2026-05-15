"""Compare E3 champion (LSTM Ensemble intraday) vs E3 baseline (Ridge).

By default reads champions from models/registry.json. E3 baseline is not in the
registry, so it falls back to the latest run under runs/e3_baseline.

Usage:
    python -m scripts.evaluation.compare_e3_models
    python -m scripts.evaluation.compare_e3_models --model-run runs/e3_intraday/20260421_115847
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluation._compare_common import (
    compare_runs,
    load_latest_run,
    load_summary_for_comparison,
    print_comparison,
    save_comparison_figure,
    save_comparison_markdown,
    validate_improvement_signs,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "conclusiones" / "comparaciones"
STRATEGY = "e3"


def _resolve_ticker_dir(row: pd.Series, fallback_run_dir: Path | None) -> Path | None:
    """Return the per-ticker subfolder where raw run files live.

    Champions loaded from the registry carry `_run_dir` (already per-ticker).
    Otherwise compose <fallback_run_dir>/<ticker>.
    """
    run_dir_str = row.get("_run_dir") if "_run_dir" in row.index else None
    if run_dir_str:
        return Path(run_dir_str)
    if fallback_run_dir is None:
        return None
    return fallback_run_dir / row["ticker"]


def _patch_missing_metrics(
    df: pd.DataFrame,
    fallback_run_dir: Path | None,
    is_baseline: bool,
) -> pd.DataFrame:
    """Compute ml_rmse, bt_cagr, bt_hit_rate from per-ticker files when absent.

    If a row has `_run_dir` (registry source), uses that ticker dir directly.
    Otherwise falls back to <fallback_run_dir>/<ticker>.
    """
    rows = []
    for _, row in df.iterrows():
        ticker = row["ticker"]
        row = row.copy()

        ticker_dir = _resolve_ticker_dir(row, fallback_run_dir)
        if ticker_dir is None:
            rows.append(row)
            continue

        if is_baseline:
            pred_file = ticker_dir / f"{ticker}_baseline_predictions.csv"
            bt_file = ticker_dir / f"{ticker}_baseline_backtest.csv"
        else:
            pred_file = ticker_dir / f"{ticker}_walkforward_predictions.csv"
            bt_file = ticker_dir / f"{ticker}_walkforward_backtest.csv"

        # RMSE
        if "ml_rmse" not in row.index or pd.isna(row["ml_rmse"]):
            if pred_file.exists():
                pred = pd.read_csv(pred_file)
                row["ml_rmse"] = np.sqrt(np.mean((pred["y_true"] - pred["y_pred"]) ** 2))

        # CAGR and hit_rate both need the backtest file — read once
        if bt_file.exists() and (
            "bt_cagr" not in row.index or pd.isna(row.get("bt_cagr", np.nan))
            or "bt_hit_rate" not in row.index or pd.isna(row.get("bt_hit_rate", np.nan))
        ):
            if is_baseline:
                bt = pd.read_csv(bt_file, index_col=0, parse_dates=True).sort_index()
                t0, t1 = bt.index[0], bt.index[-1]
            else:
                bt = pd.read_csv(bt_file, parse_dates=["timestamp"]).sort_values("timestamp")
                t0, t1 = bt["timestamp"].iloc[0], bt["timestamp"].iloc[-1]

            if "bt_cagr" not in row.index or pd.isna(row.get("bt_cagr", np.nan)):
                years = (t1 - t0).total_seconds() / (365.25 * 24 * 3600)
                final_eq = bt["equity"].iloc[-1]
                row["bt_cagr"] = (final_eq ** (1 / years)) - 1 if years > 0 else np.nan

            if "bt_hit_rate" not in row.index or pd.isna(row.get("bt_hit_rate", np.nan)):
                active = bt[bt["pos"] != 0]
                row["bt_hit_rate"] = (active["net_ret"] > 0).mean() if len(active) > 0 else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare E3 Intraday LSTM Ensemble vs Baseline")
    parser.add_argument("--baseline-run", default=None)
    parser.add_argument("--model-run", default=None)
    args = parser.parse_args()

    explicit_baseline = Path(args.baseline_run) if args.baseline_run else None
    explicit_model = Path(args.model_run) if args.model_run else None

    df_baseline, baseline_src = load_summary_for_comparison(
        STRATEGY, "baseline",
        explicit_run=explicit_baseline,
        runs_subdir=ROOT / "runs" / "e3_baseline",
        project_root=ROOT,
    )
    df_model, model_src = load_summary_for_comparison(
        STRATEGY, "champion",
        explicit_run=explicit_model,
        runs_subdir=ROOT / "runs" / "e3_intraday",
        project_root=ROOT,
    )

    print(f"[INFO] Baseline source: {baseline_src}")
    print(f"[INFO] Model source:    {model_src}")

    if df_baseline is None:
        print("ERROR: could not load baseline data")
        sys.exit(1)
    if df_model is None:
        print("ERROR: could not load model data")
        sys.exit(1)

    # For E3 baseline (which has no registry slot), fall back to the latest baseline
    # run dir so the patch function can find raw files (predictions/backtest CSVs).
    baseline_fallback_run = explicit_baseline or load_latest_run(ROOT / "runs" / "e3_baseline")

    df_baseline = _patch_missing_metrics(df_baseline, baseline_fallback_run, is_baseline=True)
    df_model = _patch_missing_metrics(df_model, explicit_model, is_baseline=False)

    print(f"Baseline tickers: {len(df_baseline)}, Model tickers: {len(df_model)}")

    df_comp = compare_runs(df_baseline, df_model, model_label="LSTM Ensemble E3")
    validate_improvement_signs(df_comp)
    print_comparison(df_comp, "LSTM Ensemble E3")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_comparison_figure(
        df_comp,
        model_label="LSTM Ensemble E3",
        strategy_label="E3 Intraday (LSTM Ensemble, 30-min bars)",
        out_path=OUT_DIR / "fig_e3_vs_baseline.png",
        trading_split_per_group=True,
    )

    # Markdown report
    save_comparison_markdown(
        df_comp,
        model_label="LSTM Ensemble E3",
        strategy_label="E3 Intraday (LSTM Ensemble, 30-min bars)",
        out_path=OUT_DIR / "fig_e3_vs_baseline.md",
    )

    out_csv = OUT_DIR / "e3_comparison.csv"
    df_comp.to_csv(out_csv, index=False)
    print(f"[OK] {out_csv}")


if __name__ == "__main__":
    main()
