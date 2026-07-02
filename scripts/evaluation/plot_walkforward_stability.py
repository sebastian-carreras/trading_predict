"""Figura de tesis: estabilidad del walk-forward (IC y Sharpe por fold).

Compara un ticker ESTABLE (receta confiable) contra uno INESTABLE (receta
no confiable) usando las métricas por fold del walk-forward. Ilustra que el
WF no mejora el modelo: audita su consistencia entre regímenes de mercado.

Salida:
    reports/conclusiones/figuras/fig_conclusions_walkforward_stability.png

Uso:
    python -m scripts.evaluation.plot_walkforward_stability
    python -m scripts.evaluation.plot_walkforward_stability --run runs/e1_conservative/20260615_050125 --stable V --unstable YPFD.BA
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "conclusiones" / "figuras"

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
    "legend.fontsize": 11,
    "figure.titlesize": 17,
}

COLOR_STABLE = "#2196F3"    # azul — receta confiable
COLOR_UNSTABLE = "#E53935"  # rojo — receta inestable


def _load_folds(run_dir: Path, ticker: str) -> pd.DataFrame:
    path = run_dir / ticker / f"{ticker}_walkforward_folds.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")
    return pd.read_csv(path)


def _plot_metric(ax, df_s, df_u, col, stable, unstable, title, ylabel, zero_line):
    folds_s = df_s["fold"].to_numpy()
    folds_u = df_u["fold"].to_numpy()

    if zero_line:
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)

    for df, folds, color, name in (
        (df_s, folds_s, COLOR_STABLE, stable),
        (df_u, folds_u, COLOR_UNSTABLE, unstable),
    ):
        vals = df[col].to_numpy()
        mu, sigma = float(np.nanmean(vals)), float(np.nanstd(vals))
        ax.plot(folds, vals, marker="o", markersize=8, linewidth=2.2,
                color=color, label=f"{name}  (μ={mu:+.2f}, σ={sigma:.2f})")
        # banda media ± desvío
        ax.axhline(mu, color=color, linewidth=0.9, alpha=0.35)
        ax.fill_between(folds, mu - sigma, mu + sigma, color=color, alpha=0.08)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Fold (ventana temporal creciente →)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(set(folds_s).union(folds_u)))
    ax.legend(loc="best")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/e1_conservative/20260615_050125")
    ap.add_argument("--stable", default="V")
    ap.add_argument("--unstable", default="YPFD.BA")
    args = ap.parse_args()

    run_dir = ROOT / args.run
    df_s = _load_folds(run_dir, args.stable)
    df_u = _load_folds(run_dir, args.unstable)

    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(
        "Walk-Forward: estabilidad de la receta entre folds (E1 Conservative)",
        fontweight="bold",
    )

    _plot_metric(
        axes[0], df_s, df_u, "ml_ic", args.stable, args.unstable,
        "Information Coefficient (IC) por fold", "IC (Spearman)", zero_line=True,
    )
    _plot_metric(
        axes[1], df_s, df_u, "bt_sharpe", args.stable, args.unstable,
        "Sharpe del backtest por fold", "Sharpe ratio", zero_line=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "fig_conclusions_walkforward_stability.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
