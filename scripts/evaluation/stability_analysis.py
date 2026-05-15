"""Stability analysis: champion metric evolution over time, and promotions timeline.

Reads:
    reports/dashboard/history_report_*.csv
    models/promotion_log.jsonl

Outputs:
    reports/conclusiones/figuras/fig_champion_stability_sharpe.png
    reports/conclusiones/figuras/fig_champion_stability_ic.png
    reports/conclusiones/tablas/promotions_timeline.csv

Usage:
    python -m scripts.evaluation.stability_analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = ROOT / "reports" / "dashboard"
PROMOTION_LOG = ROOT / "models" / "promotion_log.jsonl"
_BASE = ROOT / "reports" / "conclusiones"
OUT_FIG_DIR = _BASE / "figuras"
OUT_TABLE_DIR = _BASE / "tablas"

STYLE = {
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 17,
}

STRATEGY_CONFIG = {
    "e1_conservative": {"label": "E1 Conservative", "color": "#2196F3"},
    # E2: ancla la leyenda con bbox para correrla a la izquierda y evitar la
    # superposicion con los puntos del spike de fin de marzo.
    "e2_moderate": {
        "label": "E2 Moderate", "color": "#4CAF50",
        "legend_loc": "center left",
        "legend_bbox": (0.22, 0.55),
    },
    "e3_intraday": {"label": "E3 Intraday", "color": "#FF9800"},
}


def load_history_reports() -> dict[str, pd.DataFrame]:
    """Load all history_report_*.csv files, grouped by strategy-like key."""
    files = sorted(DASHBOARD_DIR.glob("history_report_*.csv"))
    dfs: dict[str, list[pd.DataFrame]] = {}
    for f in files:
        # filename: history_report_{strategy}_{ticker}_latest.csv
        stem = f.stem  # history_report_e1_conservative_AAPL_latest
        parts = stem.split("_")
        # Try to identify strategy: e1_conservative, e2_moderate, e3_intraday
        strategy_key = None
        for key in STRATEGY_CONFIG:
            if key.replace("_", "_") in stem:
                strategy_key = key
                break
        if strategy_key is None:
            continue
        try:
            df = pd.read_csv(f, parse_dates=["date"])
            df["strategy"] = strategy_key
            df["source_file"] = f.name
            dfs.setdefault(strategy_key, []).append(df)
        except Exception as exc:
            print(f"  [WARN] Cannot load {f.name}: {exc}")
    return {k: pd.concat(v, ignore_index=True) for k, v in dfs.items()}


def load_promotion_log() -> pd.DataFrame:
    rows = []
    if not PROMOTION_LOG.exists():
        return pd.DataFrame()
    with PROMOTION_LOG.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                rows.append(entry)
            except json.JSONDecodeError:
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df


def build_promotions_timeline(df_promo: pd.DataFrame) -> pd.DataFrame:
    """Filter actual promotions (promoted=True) and shape as timeline."""
    if df_promo.empty:
        return pd.DataFrame()
    promoted = df_promo[df_promo.get("promoted", pd.Series(dtype=bool)) == True].copy()
    cols = ["timestamp", "strategy", "ticker", "candidate_score", "champion_score",
            "improvement_pct", "reason"]
    cols_present = [c for c in cols if c in promoted.columns]
    promoted = promoted[cols_present].sort_values("timestamp").reset_index(drop=True)
    return promoted


def _plot_metric_figure(history_by_strategy: dict[str, pd.DataFrame],
                        out_dir: Path,
                        metric_col: str,
                        metric_label: str,
                        marker: str,
                        ref_lines: list[tuple[float, str, str]],
                        out_filename: str,
                        suptitle: str) -> None:
    """Generate a vertically stacked figure (one panel per strategy) for a single metric.

    Args:
        history_by_strategy: history dataframes keyed by strategy.
        out_dir: directory to save the PNG.
        metric_col: column to plot (e.g., "bt_sharpe", "ml_ic").
        metric_label: y-axis label.
        marker: matplotlib marker style.
        ref_lines: list of (y_value, color, linestyle) reference lines.
        out_filename: output PNG filename.
        suptitle: figure-level title.
    """
    plt.rcParams.update(STYLE)
    strategies = [k for k in STRATEGY_CONFIG if k in history_by_strategy]
    if not strategies:
        print(f"[WARN] No history data found for {metric_col} plot.")
        return

    fig, axes = plt.subplots(len(strategies), 1, figsize=(12, 5.99 * len(strategies)),
                             squeeze=False)
    fig.suptitle(suptitle, fontsize=24, fontweight="bold", y=1.005)

    for row_idx, strat in enumerate(strategies):
        df = history_by_strategy[strat].sort_values("date")
        cfg = STRATEGY_CONFIG[strat]
        label = cfg["label"]
        ax = axes[row_idx, 0]

        for ticker_file, grp in df.groupby("source_file"):
            ticker_name = (ticker_file
                           .split(f"{strat}_")[-1]
                           .replace("_latest", "")
                           .replace(".csv", "")
                           .replace("_BA", ".BA"))
            grp = grp.dropna(subset=[metric_col]).sort_values("date")
            if len(grp) < 1:
                continue
            ax.plot(grp["date"], grp[metric_col], marker=marker, markersize=5,
                    linewidth=1.8, label=ticker_name, alpha=0.85)

        for y_val, color, linestyle in ref_lines:
            ax.axhline(y_val, color=color, linestyle=linestyle, linewidth=1, alpha=0.5)

        ax.set_title(f"{label} — {metric_label}", fontsize=22, fontweight="bold")
        ax.set_ylabel(metric_label, fontsize=21)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.tick_params(axis="both", labelsize=19)
        ax.tick_params(axis="x", rotation=30)
        legend_kwargs = {"fontsize": 18, "ncol": 2, "loc": cfg.get("legend_loc", "best")}
        if "legend_bbox" in cfg:
            legend_kwargs["bbox_to_anchor"] = cfg["legend_bbox"]
        ax.legend(**legend_kwargs)

    fig.tight_layout()
    out = out_dir / out_filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def fig_champion_stability(history_by_strategy: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Generate two separate figures: one for Sharpe ratio and one for IC (Spearman)."""
    _plot_metric_figure(
        history_by_strategy=history_by_strategy,
        out_dir=out_dir,
        metric_col="bt_sharpe",
        metric_label="Sharpe Ratio",
        marker="o",
        ref_lines=[(0, "red", "--"), (1, "green", ":")],
        out_filename="fig_champion_stability_sharpe.png",
        suptitle="Evolución temporal del Sharpe ratio del champion por estrategia",
    )
    _plot_metric_figure(
        history_by_strategy=history_by_strategy,
        out_dir=out_dir,
        metric_col="ml_ic",
        metric_label="IC (Spearman)",
        marker="s",
        ref_lines=[(0, "gray", "--"), (0.05, "green", ":")],
        out_filename="fig_champion_stability_ic.png",
        suptitle="Evolución temporal del IC (Spearman) del champion por estrategia",
    )


def main() -> None:
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # Load history reports
    history_by_strategy = load_history_reports()
    print(f"Loaded history data for strategies: {list(history_by_strategy.keys())}")
    for k, df in history_by_strategy.items():
        print(f"  {k}: {len(df)} rows from {df['source_file'].nunique()} files")

    # Generate stability plot
    fig_champion_stability(history_by_strategy, OUT_FIG_DIR)

    # Load and save promotions timeline
    df_promo = load_promotion_log()
    if not df_promo.empty:
        df_timeline = build_promotions_timeline(df_promo)
        out_csv = OUT_TABLE_DIR / "promotions_timeline.csv"
        df_timeline.to_csv(out_csv, index=False)
        print(f"[OK] {out_csv}  ({len(df_timeline)} actual promotions out of {len(df_promo)} log entries)")
    else:
        print("[WARN] promotion_log.jsonl empty or not found")


if __name__ == "__main__":
    main()
