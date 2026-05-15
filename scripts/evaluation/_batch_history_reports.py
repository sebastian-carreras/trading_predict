"""One-shot batch generation of history_report CSVs for all (strategy, ticker) pairs.

Reads MLflow runs directly from the local SQLite tracking store, bypassing the
mlflow library to avoid environment setup issues. The generated CSVs are read by
scripts/evaluation/stability_analysis.py to produce the stability figures.

Usage:
    python3 scripts/evaluation/_batch_history_reports.py
"""
from __future__ import annotations

import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "runs" / "mlflow_local" / "mlflow.db"
OUT_DIR = ROOT / "reports" / "dashboard"

STRATEGY_EXPERIMENTS = {
    "e1_conservative": "E1_Conservative_Strategy",
    "e2_moderate": "E2_Moderate",
    "e3_intraday": "E3_Intraday",
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


def fetch_runs(conn: sqlite3.Connection, experiment_name: str) -> list[tuple[str, str, int]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT r.run_uuid, r.name, r.start_time
        FROM runs r
        JOIN experiments e ON e.experiment_id = r.experiment_id
        WHERE e.name = ?
          AND r.status = 'FINISHED'
          AND r.lifecycle_stage = 'active'
        ORDER BY r.start_time DESC
        """,
        (experiment_name,),
    )
    return cur.fetchall()


def fetch_metrics(conn: sqlite3.Connection, run_uuids: list[str]) -> dict[str, dict[str, float]]:
    if not run_uuids:
        return {}
    placeholders = ",".join("?" * len(run_uuids))
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT run_uuid, key, value, is_nan
        FROM latest_metrics
        WHERE run_uuid IN ({placeholders}) AND key IN ({",".join("?" * len(METRIC_COLS))})
        """,
        (*run_uuids, *METRIC_COLS),
    )
    result: dict[str, dict[str, float]] = {u: {} for u in run_uuids}
    for run_uuid, key, value, is_nan in cur.fetchall():
        if is_nan:
            continue
        result.setdefault(run_uuid, {})[key] = float(value)
    return result


def ms_to_date(ms: int) -> str:
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
    if not DB.exists():
        print(f"[ERROR] SQLite store not found at {DB}", file=sys.stderr)
        sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB))
    total_saved = 0
    total_skipped = 0
    t0 = time.time()

    for strategy, experiment_name in STRATEGY_EXPERIMENTS.items():
        all_runs = fetch_runs(conn, experiment_name)
        # Group runs by ticker (extracted from run name)
        by_ticker: dict[str, list[tuple[str, str, int]]] = {}
        for run_uuid, name, start_time in all_runs:
            m = RUN_NAME_RE.match(name or "")
            if not m:
                continue
            ticker = m.group("ticker")
            if ticker.lower() == "summary":
                continue
            by_ticker.setdefault(ticker.upper(), []).append((run_uuid, name, start_time))

        # Preload metrics for ALL runs of this strategy in one go
        all_uuids = [r[0] for r in all_runs]
        metrics_by_run = fetch_metrics(conn, all_uuids)

        for ticker in UNIVERSE[strategy]:
            runs = by_ticker.get(ticker.upper(), [])
            if not runs:
                print(f"  [SKIP] {strategy} {ticker}: no runs found")
                total_skipped += 1
                continue
            rows = []
            for run_uuid, name, start_time in runs[:50]:  # last 50 per ticker
                row = {
                    "date": ms_to_date(start_time),
                    "run_name": name,
                }
                row.update(metrics_by_run.get(run_uuid, {}))
                rows.append(row)
            safe_ticker = ticker.replace(".", "_")
            out_path = OUT_DIR / f"history_report_{strategy}_{safe_ticker}_latest.csv"
            write_history_csv(out_path, rows)
            print(f"  [OK]   {strategy} {ticker:10s} -> {out_path.name} ({len(rows)} rows)")
            total_saved += 1

    conn.close()
    print(f"\nDone in {time.time()-t0:.1f}s. Saved {total_saved}, skipped {total_skipped}.")


if __name__ == "__main__":
    main()
