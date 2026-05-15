"""Generate conclusion-level figures for the thesis final chapter.

Reads champion predictions and backtest CSVs from the model registry and
produces 8 figures in reports/conclusiones/figuras/:

    fig_conclusions_roc.png              — ROC curves (1x3 per strategy)
    fig_conclusions_confusion.png        — Confusion matrices (1x3)
    fig_conclusions_monthly_heatmap_e1.png — Monthly returns heatmap E1
    fig_conclusions_monthly_heatmap_e2.png — Monthly returns heatmap E2
    fig_conclusions_monthly_heatmap_e3.png — Monthly returns heatmap E3
    fig_conclusions_underwater.png       — Drawdown underwater chart (1x3)
    fig_conclusions_rolling_ic.png       — Rolling Spearman IC (1x3)
    fig_conclusions_quantile_returns.png — Prediction quintile analysis (1x3)

Usage:
    python -m scripts.evaluation.plot_conclusions
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, auc, confusion_matrix


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "models" / "registry.json"
RUNS_DIR = ROOT / "runs"
OUT_DIR = ROOT / "reports" / "conclusiones" / "figuras"

STRATEGY_CONFIG = {
    "e1": {
        "label": "E1 Conservative\n(GRU 90d)",
        "variant": "e1_conservative",
        "color": "#2196F3",
        "rolling_window": 30,
    },
    "e2": {
        "label": "E2 Moderate\n(LSTM 20d)",
        "variant": "e2_moderate",
        "color": "#4CAF50",
        "rolling_window": 60,
    },
    "e3": {
        "label": "E3 Intraday\n(LSTM Ens.)",
        "variant": "e3_intraday",
        "color": "#FF9800",
        "rolling_window": 100,
    },
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _to_utc(series: pd.Series) -> pd.Series:
    """Convert a timestamp column to tz-aware UTC, handling mixed formats."""
    if pd.api.types.is_datetime64_any_dtype(series):
        if series.dt.tz is None:
            return series.dt.tz_localize("UTC")
        return series.dt.tz_convert("UTC")
    try:
        return pd.to_datetime(series, utc=True, format="ISO8601")
    except (ValueError, TypeError):
        return pd.to_datetime(series, utc=True)


def _load_ticker_run(run_dir_rel: str, ticker: str) -> dict | None:
    run_dir = ROOT / run_dir_rel
    pred_path = run_dir / f"{ticker}_walkforward_predictions.csv"
    bt_path = run_dir / f"{ticker}_walkforward_backtest.csv"
    if not pred_path.exists() or not bt_path.exists():
        return None
    preds = pd.read_csv(pred_path)
    preds["timestamp"] = _to_utc(preds["timestamp"])
    bt = pd.read_csv(bt_path)
    bt["timestamp"] = _to_utc(bt["timestamp"])
    first_eq = bt["equity"].iloc[0] if len(bt) > 0 else 1.0
    bt["equity_norm"] = bt["equity"] / (first_eq if first_eq != 0 else 1.0)
    return {"ticker": ticker, "preds": preds, "bt": bt}


def load_strategy_data(strategy: str) -> list[dict]:
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    results = []
    strat_data = registry.get("strategies", {}).get(strategy, {})
    for ticker, slots in strat_data.get("tickers", {}).items():
        champ = slots.get("champion") or {}
        run_dir_rel = champ.get("run_dir")
        if run_dir_rel:
            item = _load_ticker_run(run_dir_rel, ticker)
            if item:
                results.append(item)

    # Fallback for strategies with no registered champions (E3)
    if not results:
        variant = STRATEGY_CONFIG[strategy]["variant"]
        variant_dir = RUNS_DIR / variant
        if variant_dir.exists():
            runs = sorted([d for d in variant_dir.iterdir()
                           if d.is_dir() and d.name[0].isdigit()])
            if runs:
                for ticker_dir in runs[-1].iterdir():
                    if not ticker_dir.is_dir():
                        continue
                    ticker = ticker_dir.name
                    rel = str((runs[-1] / ticker).relative_to(ROOT))
                    item = _load_ticker_run(rel, ticker)
                    if item:
                        results.append(item)
    return results


# ---------------------------------------------------------------------------
# Shared save helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Figure 1: ROC curves
# ---------------------------------------------------------------------------

def fig_roc_curves(all_data: dict[str, list[dict]], out_dir: Path) -> None:
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Curvas ROC — Clasificacion Direccional del Retorno",
        fontsize=16, fontweight="bold",
    )

    for ax, (strategy, items) in zip(axes, all_data.items()):
        cfg = STRATEGY_CONFIG[strategy]
        color = cfg["color"]
        aucs: list[float] = []

        for item in items:
            preds = item["preds"].dropna(subset=["y_true", "y_pred"])
            if len(preds) < 20:
                continue
            y_true_bin = (preds["y_true"] > 0).astype(int)
            fpr, tpr, _ = roc_curve(y_true_bin, preds["y_pred"])
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            ax.plot(fpr, tpr, color=color, alpha=0.35, linewidth=1.2,
                    label=f"{item['ticker']} ({roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Azar")
        mean_auc = float(np.mean(aucs)) if aucs else 0.0
        ax.set_title(
            f"{cfg['label'].replace(chr(10), ' ')}\nAUC medio = {mean_auc:.3f}",
            fontsize=13,
        )
        ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
        ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
        ax.legend(fontsize=10, loc="lower right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

    fig.tight_layout()
    _save(fig, out_dir / "fig_conclusions_roc.png")


# ---------------------------------------------------------------------------
# Figure 2: Confusion matrix
# ---------------------------------------------------------------------------

def fig_confusion_matrix(all_data: dict[str, list[dict]], out_dir: Path) -> None:
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        "Matriz de Confusion — Prediccion Direccional (pooled por estrategia)",
        fontsize=16, fontweight="bold",
    )

    for ax, (strategy, items) in zip(axes, all_data.items()):
        cfg = STRATEGY_CONFIG[strategy]
        all_true: list[int] = []
        all_pred: list[int] = []
        for item in items:
            preds = item["preds"].dropna(subset=["y_true", "y_pred"])
            all_true.extend((preds["y_true"] > 0).astype(int).tolist())
            all_pred.extend((preds["y_pred"] > 0).astype(int).tolist())

        if not all_true:
            ax.set_visible(False)
            continue

        cm = confusion_matrix(all_true, all_pred, normalize="true")
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        labels = ["Baja", "Sube"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([f"Pred: {la}" for la in labels], fontsize=12)
        ax.set_yticklabels([f"Real: {la}" for la in labels], fontsize=12)
        ax.set_title(cfg["label"].replace("\n", " "), fontsize=13, fontweight="bold")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        fontsize=16, fontweight="bold",
                        color="white" if cm[i, j] > 0.6 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    _save(fig, out_dir / "fig_conclusions_confusion.png")


# ---------------------------------------------------------------------------
# Figure 3: Monthly returns heatmap (one per strategy)
# ---------------------------------------------------------------------------

def fig_monthly_heatmap(
    items: list[dict], cfg: dict, out_dir: Path, strategy_key: str,
) -> None:
    plt.rcParams.update(STYLE)

    monthly_series: list[pd.Series] = []
    for item in items:
        bt = item["bt"].copy()
        monthly = (
            (1 + bt.set_index("timestamp")["net_ret"])
            .resample("ME")
            .prod()
            .sub(1)
            .rename(item["ticker"])
        )
        monthly_series.append(monthly)

    if not monthly_series:
        return

    avg_monthly = pd.concat(monthly_series, axis=1).mean(axis=1)
    df_pivot = pd.DataFrame({
        "year": avg_monthly.index.year,
        "month": avg_monthly.index.month,
        "ret": avg_monthly.values,
    }).pivot(index="year", columns="month", values="ret")

    month_names = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                   "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    df_pivot.columns = [month_names[m - 1] for m in df_pivot.columns]

    valid_vals = df_pivot.values[~np.isnan(df_pivot.values)]
    vmax = float(max(abs(valid_vals).max(), 0.01)) if len(valid_vals) > 0 else 0.05

    fig, ax = plt.subplots(figsize=(12, max(3, len(df_pivot) * 0.65 + 1.5)))
    im = ax.imshow(df_pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(df_pivot.columns)))
    ax.set_xticklabels(df_pivot.columns, fontsize=12)
    ax.set_yticks(range(len(df_pivot.index)))
    ax.set_yticklabels(df_pivot.index, fontsize=12)
    ax.set_title(
        f"Retornos Mensuales — {cfg['label'].replace(chr(10), ' ')}"
        f" (promedio {len(items)} tickers)",
        fontsize=15, fontweight="bold",
    )
    plt.colorbar(
        im, ax=ax, label="Retorno mensual",
        format=mticker.PercentFormatter(xmax=1, decimals=1),
    )
    for i in range(len(df_pivot.index)):
        for j in range(len(df_pivot.columns)):
            val = df_pivot.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val * 100:.1f}%", ha="center", va="center",
                        fontsize=10,
                        color="black" if abs(val) < vmax * 0.6 else "white")

    fig.tight_layout()
    _save(fig, out_dir / f"fig_conclusions_monthly_heatmap_{strategy_key}.png")


# ---------------------------------------------------------------------------
# Figure 4: Underwater (drawdown) chart
# ---------------------------------------------------------------------------

def fig_underwater(all_data: dict[str, list[dict]], out_dir: Path) -> None:
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Grafico Underwater (Drawdown) — Modelos Champion",
        fontsize=16, fontweight="bold",
    )

    for ax, (strategy, items) in zip(axes, all_data.items()):
        cfg = STRATEGY_CONFIG[strategy]
        color = cfg["color"]
        dd_series: list[pd.Series] = []

        for item in items:
            bt = item["bt"].copy()
            eq = bt.set_index("timestamp")["equity_norm"]
            dd = eq / eq.cummax() - 1
            ax.fill_between(dd.index, dd.values, 0, alpha=0.12, color=color)
            ax.plot(dd.index, dd.values, color=color, alpha=0.3, linewidth=0.8)
            dd_series.append(dd)

        if dd_series:
            combined = pd.concat(dd_series, axis=1).sort_index().ffill()
            avg_dd = combined.mean(axis=1)
            ax.plot(avg_dd.index, avg_dd.values, color="darkred",
                    linewidth=2, label="Media", zorder=5)

        ax.axhline(0, color="black", linewidth=1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_title(cfg["label"].replace("\n", " "), fontsize=13, fontweight="bold")
        ax.set_ylabel("Drawdown (%)")
        ax.legend(fontsize=11)
        ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    _save(fig, out_dir / "fig_conclusions_underwater.png")


# ---------------------------------------------------------------------------
# Figure 5: Rolling IC
# ---------------------------------------------------------------------------

def _rolling_spearman(y_pred: np.ndarray, y_true: np.ndarray, window: int) -> np.ndarray:
    n = len(y_pred)
    ic = np.full(n, np.nan)
    for i in range(window - 1, n):
        ic_val, _ = stats.spearmanr(
            y_pred[i - window + 1: i + 1],
            y_true[i - window + 1: i + 1],
        )
        ic[i] = 0.0 if np.isnan(ic_val) else ic_val
    return ic


def fig_rolling_ic(all_data: dict[str, list[dict]], out_dir: Path) -> None:
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "IC Rolling (Spearman) — Estabilidad Temporal del Signal Predictivo",
        fontsize=16, fontweight="bold",
    )

    for ax, (strategy, items) in zip(axes, all_data.items()):
        cfg = STRATEGY_CONFIG[strategy]
        color = cfg["color"]
        window = cfg["rolling_window"]
        ic_series_list: list[pd.Series] = []

        for item in items:
            preds = (
                item["preds"]
                .dropna(subset=["y_true", "y_pred"])
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            if len(preds) < window:
                continue
            ic_vals = _rolling_spearman(
                preds["y_pred"].values, preds["y_true"].values, window
            )
            ts = pd.to_datetime(preds["timestamp"])
            ic_s = pd.Series(ic_vals, index=ts).dropna()
            ax.plot(ic_s.index, ic_s.values, color=color, alpha=0.3, linewidth=0.9)
            ic_series_list.append(ic_s)

        if ic_series_list:
            combined = pd.concat(ic_series_list, axis=1).sort_index().fillna(0)
            avg_ic = combined.mean(axis=1)
            std_ic = combined.std(axis=1)
            ax.plot(avg_ic.index, avg_ic.values, color=color, linewidth=2.5,
                    label=f"Media (ventana={window})", zorder=5)
            ax.fill_between(avg_ic.index, avg_ic - std_ic, avg_ic + std_ic,
                            color=color, alpha=0.12)

        ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="IC=0")
        ax.axhline(0.05, color="green", linestyle=":", linewidth=1, alpha=0.6, label="IC=0.05")
        ax.set_title(cfg["label"].replace("\n", " "), fontsize=13, fontweight="bold")
        ax.set_ylabel(f"IC Spearman (ventana {window})")
        ax.legend(fontsize=11)
        ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    _save(fig, out_dir / "fig_conclusions_rolling_ic.png")


# ---------------------------------------------------------------------------
# Figure 6: Quantile return analysis
# ---------------------------------------------------------------------------

def fig_quantile_returns(all_data: dict[str, list[dict]], out_dir: Path) -> None:
    plt.rcParams.update(STYLE)
    n_quantiles = 5
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Retorno Medio por Cuantil de Prediccion — Validacion del Skill de Ranking",
        fontsize=16, fontweight="bold",
    )

    for ax, (strategy, items) in zip(axes, all_data.items()):
        cfg = STRATEGY_CONFIG[strategy]
        color = cfg["color"]

        frames = [
            item["preds"].dropna(subset=["y_true", "y_pred"])[["y_pred", "y_true"]]
            for item in items
        ]
        if not frames:
            ax.set_visible(False)
            continue
        pooled = pd.concat(frames, ignore_index=True)

        pooled["quantile"] = pd.qcut(
            pooled["y_pred"], n_quantiles, labels=False, duplicates="drop"
        )
        grp = pooled.groupby("quantile")["y_true"]
        q_mean = grp.mean()
        q_std = grp.std()
        q_count = grp.count()
        global_mean = pooled["y_true"].mean()

        x = list(range(len(q_mean)))
        bar_colors = [color if v >= 0 else "#EF5350" for v in q_mean.values]
        ax.bar(x, q_mean.values * 100, color=bar_colors, alpha=0.85,
               width=0.6, edgecolor="white")
        stderr = (q_std / np.sqrt(q_count)).values * 100
        ax.errorbar(x, q_mean.values * 100, yerr=stderr,
                    fmt="none", color="black", capsize=4, linewidth=1.5)
        ax.axhline(global_mean * 100, color="gray", linestyle="--", linewidth=1.5,
                   alpha=0.7, label=f"Media global ({global_mean * 100:.3f}%)")

        ax.set_xticks(x)
        ax.set_xticklabels([f"Q{i + 1}" for i in x], fontsize=13)
        ax.set_xlabel("Cuantil de prediccion (Q1=menor, Q5=mayor)")
        ax.set_ylabel("Retorno real medio (%)")
        ax.set_title(cfg["label"].replace("\n", " "), fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)

    fig.tight_layout()
    _save(fig, out_dir / "fig_conclusions_quantile_returns.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando datos de champions por estrategia...")
    all_data: dict[str, list[dict]] = {}
    for strategy in STRATEGY_CONFIG:
        items = load_strategy_data(strategy)
        all_data[strategy] = items
        tickers = [i["ticker"] for i in items]
        print(f"  {strategy}: {len(items)} tickers — {tickers}")

    print("\nGenerando figuras de conclusiones...")
    fig_roc_curves(all_data, OUT_DIR)
    fig_confusion_matrix(all_data, OUT_DIR)
    fig_underwater(all_data, OUT_DIR)
    fig_rolling_ic(all_data, OUT_DIR)
    fig_quantile_returns(all_data, OUT_DIR)
    for strategy_key, items in all_data.items():
        if items:
            fig_monthly_heatmap(items, STRATEGY_CONFIG[strategy_key], OUT_DIR, strategy_key)

    print(f"\n[OK] Figuras escritas en: {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
