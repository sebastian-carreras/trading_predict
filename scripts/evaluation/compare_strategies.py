"""Generate cross-strategy comparison figures for Chapter 4.

Reads consolidated_metrics.csv (produced by consolidate_strategy_metrics.py)
and produces 4 figures in reports/figures/chapter_4/:

    fig_sharpe_boxplot_by_strategy.png   — boxplot Sharpe by strategy (champions)
    fig_risk_return_scatter.png           — scatter CAGR vs |MaxDD| by ticker
    fig_directional_accuracy_heatmap.png  — heatmap ticker x strategy (dir. acc.)
    fig_ic_vs_sharpe.png                  — scatter IC vs Sharpe (ML → trading)

Usage:
    python -m scripts.evaluation.compare_strategies
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_CSV = ROOT / "reports" / "conclusiones" / "tablas" / "consolidated_metrics.csv"
OUT_DIR = ROOT / "reports" / "conclusiones" / "figuras"

STRATEGY_LABELS = {
    "e1": "E1 Conservative\n(GRU 90d)",
    "e2": "E2 Moderate\n(LSTM 20d)",
    "e3": "E3 Intraday\n(LSTM Ens.)",
    "e3_intraday": "E3 Intraday\n(LSTM Ens.)",
}

COLORS = {
    "e1": "#2196F3",
    "e2": "#4CAF50",
    "e3": "#FF9800",
    "e3_intraday": "#FF9800",
}

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


def load_champions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["slot"] == "champion"].copy()
    # Merge e3 and e3_intraday as one strategy for display
    df["strategy_plot"] = df["strategy"].replace({"e3": "e3_intraday"})
    return df


def fig_sharpe_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update(STYLE)
    strategies = ["e1", "e2", "e3_intraday"]
    data_groups = []
    labels = []
    colors = []
    for s in strategies:
        group = df[df["strategy_plot"] == s]["bt_sharpe"].dropna()
        if len(group) > 0:
            data_groups.append(group.values)
            labels.append(STRATEGY_LABELS.get(s, s))
            colors.append(COLORS.get(s, "gray"))

    fig, ax = plt.subplots(figsize=(8, 4.2768))
    bp = ax.boxplot(data_groups, patch_artist=True, widths=0.4,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.6, label="Sharpe = 0")
    ax.axhline(1, color="green", linestyle=":", linewidth=1, alpha=0.6, label="Sharpe = 1.0")
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Sharpe Ratio", fontsize=14)
    ax.set_title("Sharpe Ratio por Estrategia — Modelos Champion", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)

    # Scatter individual points
    for i, (group, color) in enumerate(zip(data_groups, colors), start=1):
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, size=len(group))
        ax.scatter(np.full(len(group), i) + jitter, group,
                   color=color, alpha=0.6, s=40, zorder=5)

    fig.tight_layout()
    out = out_dir / "fig_sharpe_boxplot_by_strategy.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def fig_risk_return_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(9, 6))

    strategies = ["e1", "e2", "e3_intraday"]
    for s in strategies:
        group = df[df["strategy_plot"] == s].copy()
        group = group.dropna(subset=["bt_cagr", "bt_max_drawdown", "bt_sharpe"])
        if group.empty:
            continue
        x = group["bt_cagr"] * 100  # % CAGR
        y = group["bt_max_drawdown"].abs() * 100  # % MaxDD
        size = np.clip(group["bt_sharpe"] * 40, 10, 300)
        color = COLORS.get(s, "gray")
        ax.scatter(x, y, s=size, color=color, alpha=0.7,
                   label=STRATEGY_LABELS.get(s, s), edgecolors="white", linewidth=0.5)
        for _, row in group.iterrows():
            ax.annotate(row["ticker"], (row["bt_cagr"] * 100, row["bt_max_drawdown"] * 100),
                        fontsize=10, alpha=0.7, ha="left", va="bottom")

    ax.set_xlabel("CAGR (%)", fontsize=14)
    ax.set_ylabel("|Max Drawdown| (%)", fontsize=14)
    ax.set_title("Riesgo vs Retorno — Champions por Ticker y Estrategia\n(tamaño ∝ Sharpe Ratio)",
                 fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

    fig.tight_layout()
    out = out_dir / "fig_risk_return_scatter.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def fig_directional_accuracy_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update(STYLE)

    short_codes = {"e1": "E1", "e2": "E2", "e3": "E3", "e3_intraday": "E3"}

    # Build pivot: ticker x strategy (etiquetas cortas para evitar superposicion).
    df_plot = df.dropna(subset=["ml_directional_accuracy"]).copy()
    df_plot["strategy_short"] = df_plot["strategy_plot"].map(
        lambda s: short_codes.get(s, s.upper())
    )
    pivot = df_plot.pivot_table(
        index="ticker", columns="strategy_short",
        values="ml_directional_accuracy", aggfunc="mean"
    )
    pivot = pivot.reindex(columns=["E1", "E2", "E3"]).sort_index()

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.7 + 1.5)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0.45, vmax=0.75)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=18, fontweight="bold")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=17)
    ax.set_title("Directional Accuracy (%) — Champions por Ticker y Estrategia",
                 fontsize=18, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, label="Directional Accuracy")
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label("Directional Accuracy", fontsize=14)
    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=18,
                        fontweight="bold",
                        color="black" if 0.5 < val < 0.7 else "white")

    fig.tight_layout()
    out = out_dir / "fig_directional_accuracy_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def fig_ic_vs_sharpe(df: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(8, 6))

    strategies = ["e1", "e2", "e3_intraday"]
    for s in strategies:
        group = df[df["strategy_plot"] == s].dropna(subset=["ml_ic", "bt_sharpe"])
        if group.empty:
            continue
        ax.scatter(group["ml_ic"], group["bt_sharpe"],
                   color=COLORS.get(s, "gray"), alpha=0.75, s=60,
                   label=STRATEGY_LABELS.get(s, s).replace("\n", " "),
                   edgecolors="white", linewidth=0.5)
        for _, row in group.iterrows():
            ax.annotate(row["ticker"], (row["ml_ic"], row["bt_sharpe"]),
                        fontsize=10, alpha=0.7)

    ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("IC (Information Coefficient — Spearman)", fontsize=14)
    ax.set_ylabel("Sharpe Ratio", fontsize=14)
    ax.set_title("Señal ML (IC) vs Performance Trading (Sharpe) — Champions",
                 fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)

    fig.tight_layout()
    out = out_dir / "fig_ic_vs_sharpe.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def main() -> None:
    if not IN_CSV.exists():
        print(f"ERROR: {IN_CSV} not found. Run consolidate_strategy_metrics.py first.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_champions(IN_CSV)
    print(f"Loaded {len(df)} champion rows from {IN_CSV}")

    fig_sharpe_boxplot(df, OUT_DIR)
    fig_risk_return_scatter(df, OUT_DIR)
    fig_directional_accuracy_heatmap(df, OUT_DIR)
    fig_ic_vs_sharpe(df, OUT_DIR)

    print("\n[OK] All figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
