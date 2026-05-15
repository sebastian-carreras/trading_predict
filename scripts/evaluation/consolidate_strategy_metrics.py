"""Consolidate champion/candidate/baseline metrics from registry.json into chapter_4 tables.

Outputs:
    reports/tables/chapter_4/consolidated_metrics.csv   — one row per (strategy, ticker, slot)
    reports/tables/chapter_4/consolidated_aggregates.csv — mean/median/std/p25/p75/p95 per (strategy, slot)

Usage:
    python -m scripts.evaluation.consolidate_strategy_metrics
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "models" / "registry.json"
OUT_DIR = ROOT / "reports" / "conclusiones" / "tablas"

METRIC_COLS = [
    "ml_mae",
    "ml_rmse",
    "ml_ic",
    "ml_directional_accuracy",
    "bt_sharpe",
    "bt_cagr",
    "bt_max_drawdown",
    "bt_calmar",
    "bt_profit_factor",
    "bt_hit_rate",
    "bt_num_trades",
]

STRATEGY_DISPLAY = {
    "e1": "E1 Conservative (GRU 90d)",
    "e2": "E2 Moderate (LSTM 20d)",
    "e3": "E3 Intraday (LSTM Ensemble)",
    "e3_intraday": "E3 Intraday (LSTM Ensemble)",
}

SLOTS = ["baseline", "champion", "candidate"]


def load_registry() -> dict:
    with REGISTRY_PATH.open() as f:
        return json.load(f)


def extract_rows(registry: dict) -> list[dict]:
    rows = []
    strategies = registry.get("strategies", {})
    for strategy_key, strategy_data in strategies.items():
        display = STRATEGY_DISPLAY.get(strategy_key, strategy_key)
        tickers_data = strategy_data.get("tickers", {})
        for ticker, ticker_entry in tickers_data.items():
            for slot in SLOTS:
                entry = ticker_entry.get(slot)
                if entry is None:
                    continue
                metrics = entry.get("metrics", {})
                row = {
                    "strategy": strategy_key,
                    "strategy_display": display,
                    "ticker": ticker,
                    "slot": slot,
                    "variant": entry.get("variant", ""),
                    "run_dir": entry.get("run_dir", ""),
                    "registered_at": entry.get("registered_at", ""),
                }
                for col in METRIC_COLS:
                    val = metrics.get(col)
                    row[col] = float(val) if val is not None else np.nan
                rows.append(row)
    return rows


def compute_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    agg_rows = []
    metric_cols_present = [c for c in METRIC_COLS if c in df.columns]
    for (strategy, slot), group in df.groupby(["strategy", "slot"]):
        row: dict = {
            "strategy": strategy,
            "strategy_display": STRATEGY_DISPLAY.get(strategy, strategy),
            "slot": slot,
            "n_tickers": len(group),
        }
        for col in metric_cols_present:
            vals = group[col].dropna()
            if len(vals) == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_median"] = np.nan
                row[f"{col}_std"] = np.nan
                row[f"{col}_p25"] = np.nan
                row[f"{col}_p75"] = np.nan
                row[f"{col}_p95"] = np.nan
            else:
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_median"] = float(vals.median())
                row[f"{col}_std"] = float(vals.std())
                row[f"{col}_p25"] = float(vals.quantile(0.25))
                row[f"{col}_p75"] = float(vals.quantile(0.75))
                row[f"{col}_p95"] = float(vals.quantile(0.95))
        agg_rows.append(row)
    return pd.DataFrame(agg_rows)


def flag_issues(df: pd.DataFrame) -> list[str]:
    """Return list of (strategy, ticker) pairs with NaN/Inf in key champion metrics."""
    issues = []
    champions = df[df["slot"] == "champion"]
    for _, row in champions.iterrows():
        for col in ["bt_sharpe", "ml_ic", "bt_cagr"]:
            val = row.get(col)
            if val is not None and not np.isfinite(float(val)):
                issues.append(f"  {row['strategy']}:{row['ticker']} has {col}={val}")
    return issues


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    registry = load_registry()
    rows = extract_rows(registry)

    if not rows:
        print("ERROR: No rows extracted from registry. Check registry.json.")
        return

    df = pd.DataFrame(rows)

    # Sort for readability
    df = df.sort_values(["strategy", "slot", "ticker"]).reset_index(drop=True)

    out_metrics = OUT_DIR / "consolidated_metrics.csv"
    df.to_csv(out_metrics, index=False)
    print(f"[OK] {out_metrics}  ({len(df)} rows)")

    # Aggregates
    df_agg = compute_aggregates(df)
    df_agg = df_agg.sort_values(["strategy", "slot"]).reset_index(drop=True)
    out_agg = OUT_DIR / "consolidated_aggregates.csv"
    df_agg.to_csv(out_agg, index=False)
    print(f"[OK] {out_agg}  ({len(df_agg)} rows)")

    # Print summary table
    print("\n--- Champion summary ---")
    champions = df[df["slot"] == "champion"][
        ["strategy", "ticker", "bt_sharpe", "bt_cagr", "bt_max_drawdown", "ml_ic", "ml_directional_accuracy"]
    ]
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.width", 120)
    print(champions.to_string(index=False))

    # Flag issues
    issues = flag_issues(df)
    if issues:
        print("\n[WARN] Champions with NaN/Inf metrics (consider re-training):")
        for issue in issues:
            print(issue)
    else:
        print("\n[OK] No NaN/Inf detected in champion metrics.")


if __name__ == "__main__":
    main()
