"""One-shot batch generation of history_report CSVs for all (strategy, ticker) pairs.

Reads MLflow runs via the tracking server (MLFLOW_TRACKING_URI, same env var
and mlflow: http://mlflow:5000 endpoint every training pipeline already uses),
so it works regardless of which backend store the mlflow server is configured
with (Postgres in docker-compose, SQLite locally, etc.). The generated CSVs
are read by scripts/evaluation/stability_analysis.py to produce the stability
figures.

Usage:
    python3 scripts/evaluation/_batch_history_reports.py
"""
from __future__ import annotations

import math
import os
import re
import sys
import time
from pathlib import Path

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "dashboard"

STRATEGY_EXPERIMENTS = {
    "e1_conservative": "E1_Conservative_Strategy",
    "e2_moderate": "E2_Moderate_Strategy",
    "e3_intraday": "E3_Intraday_Strategy",
}

UNIVERSE = {
    "e1_conservative": [
        "YPFD.BA", "GGAL.BA", "PAMP.BA", "BYMA.BA", "CEPU.BA",
        "AAPL", "MSFT", "JNJ", "PG", "V",
    ],
    "e2_moderate": [
        "NVDA", "GOOGL", "AMZN", "META", "NFLX",
        "BBAR.BA", "BMA.BA", "EDN.BA", "TGSU2.BA", "LOMA.BA",
    ],
    "e3_intraday": ["SPY", "AAPL", "NVDA", "QQQ"],
}

METRIC_COLS = [
    "ml_mae", "ml_rmse", "ml_ic", "ml_directional_accuracy",
    "bt_sharpe", "bt_sortino", "bt_cagr", "bt_max_drawdown",
    "bt_calmar", "bt_profit_factor",
]

# Run names look like "E1_AAPL_20260501_180312". The TICKER segment may contain
# dots (e.g. BYMA.BA), so the run name uses the literal ticker between the
# strategy prefix and the YYYYMMDD_HHMMSS suffix.
RUN_NAME_RE = re.compile(r"^E[123]_(?P<ticker>[A-Za-z0-9.]+?)_\d{8}_\d{6}$")


def fetch_runs(client: MlflowClient, experiment_name: str) -> list[tuple[str, str, int]]:
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return []
    runs: list[tuple[str, str, int]] = []
    page_token = None
    while True:
        page = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=["attributes.start_time DESC"],
            page_token=page_token,
        )
        for run in page:
            metrics = {
                k: v for k, v in run.data.metrics.items()
                if k in METRIC_COLS and not math.isnan(v)
            }
            runs.append((run.info.run_id, run.info.run_name or "", run.info.start_time, metrics))
        page_token = page.token
        if not page_token:
            break
    return runs


def ms_to_date(ms: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def write_history_csv(out_path: Path, rows: list[dict]) -> None:
    header = ["date", "run_name"] + METRIC_COLS
    with out_path.open("w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            vals = [row["date"], row["run_name"]]
            for col in METRIC_COLS:
                v = row.get(col)
                vals.append("" if v is None else f"{v:.10g}")
            f.write(",".join(vals) + "\n")


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        print("[ERROR] MLFLOW_TRACKING_URI is not set", file=sys.stderr)
        sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    client = MlflowClient(tracking_uri=tracking_uri)
    total_saved = 0
    total_skipped = 0
    t0 = time.time()

    for strategy, experiment_name in STRATEGY_EXPERIMENTS.items():
        all_runs = fetch_runs(client, experiment_name)
        # Group runs by ticker (extracted from run name)
        by_ticker: dict[str, list[tuple[str, str, int, dict]]] = {}
        for run_id, name, start_time, metrics in all_runs:
            m = RUN_NAME_RE.match(name or "")
            if not m:
                continue
            ticker = m.group("ticker")
            if ticker.lower() == "summary":
                continue
            by_ticker.setdefault(ticker.upper(), []).append((run_id, name, start_time, metrics))

        for ticker in UNIVERSE[strategy]:
            runs = by_ticker.get(ticker.upper(), [])
            if not runs:
                print(f"  [SKIP] {strategy} {ticker}: no runs found")
                total_skipped += 1
                continue
            rows = []
            for run_id, name, start_time, metrics in runs[:50]:  # last 50 per ticker
                row = {
                    "date": ms_to_date(start_time),
                    "run_name": name,
                }
                row.update(metrics)
                rows.append(row)
            safe_ticker = ticker.replace(".", "_")
            out_path = OUT_DIR / f"history_report_{strategy}_{safe_ticker}_latest.csv"
            write_history_csv(out_path, rows)
            print(f"  [OK]   {strategy} {ticker:10s} -> {out_path.name} ({len(rows)} rows)")
            total_saved += 1

    print(f"\nDone in {time.time()-t0:.1f}s. Saved {total_saved}, skipped {total_skipped}.")


if __name__ == "__main__":
    main()
