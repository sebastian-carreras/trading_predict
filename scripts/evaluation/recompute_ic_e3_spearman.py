"""Recompute IC (Spearman) for E3 LSTM ensemble and Ridge baseline.

Reads existing predictions.csv files from the latest LSTM and baseline runs,
recomputes Spearman IC per ticker (instead of the Pearson recorded in
summary.csv), and writes a consolidated CSV plus prints the means.

Usage:
    python scripts/evaluation/recompute_ic_e3_spearman.py \
        --lstm-run runs/e3_intraday/20260503_155038 \
        --baseline-run runs/e3_baseline/20260503_211623
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


TICKERS = ("SPY", "AAPL", "NVDA", "QQQ")


def spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    ic, _ = spearmanr(y_true, y_pred)
    return float(ic) if np.isfinite(ic) else float("nan")


def lstm_predictions_path(run_dir: Path, ticker: str) -> Path:
    return run_dir / ticker / f"{ticker}_walkforward_predictions.csv"


def baseline_predictions_path(run_dir: Path, ticker: str) -> Path:
    return run_dir / ticker / f"{ticker}_baseline_predictions.csv"


def compute_for_run(run_dir: Path, label: str, path_fn) -> dict[str, float]:
    out: dict[str, float] = {}
    for ticker in TICKERS:
        path = path_fn(run_dir, ticker)
        df = pd.read_csv(path)
        ic = spearman_ic(df["y_true"].to_numpy(), df["y_pred"].to_numpy())
        print(f"  [{label}] {ticker:5s}  n={len(df):6d}  IC_spearman={ic:+.4f}")
        out[ticker] = ic
    mean_ic = float(np.nanmean(list(out.values())))
    print(f"  [{label}] mean IC_spearman = {mean_ic:+.4f}")
    out["__mean__"] = mean_ic
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lstm-run", required=True, type=Path)
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/e3_recompute_ic_spearman.csv"),
        help="Path to write consolidated CSV (relative to repo root)",
    )
    args = parser.parse_args()

    print(f"LSTM run:     {args.lstm_run}")
    print(f"Baseline run: {args.baseline_run}")
    print()

    print("LSTM ensemble (champion run):")
    lstm = compute_for_run(args.lstm_run, "LSTM", lstm_predictions_path)
    print()
    print("Ridge baseline:")
    base = compute_for_run(args.baseline_run, "Ridge", baseline_predictions_path)
    print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ticker", "ic_spearman_lstm", "ic_spearman_baseline"])
        for ticker in TICKERS:
            writer.writerow([ticker, lstm[ticker], base[ticker]])
        writer.writerow(["__mean__", lstm["__mean__"], base["__mean__"]])
    print(f"Written: {args.out}")


if __name__ == "__main__":
    main()
