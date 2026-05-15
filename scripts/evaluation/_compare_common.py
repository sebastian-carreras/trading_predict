"""Shared helpers for strategy comparison scripts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS_CONFIG = [
    # (display_name, column_name, direction)
    ("MAE", "ml_mae", "lower_better"),
    ("RMSE", "ml_rmse", "lower_better"),
    ("Dir. Accuracy", "ml_directional_accuracy", "higher_better"),
    ("IC (Spearman)", "ml_ic", "higher_better"),
    ("Sharpe Ratio", "bt_sharpe", "higher_better"),
    ("CAGR", "bt_cagr", "higher_better"),
    ("Max Drawdown", "bt_max_drawdown", "lower_better"),
    ("Calmar Ratio", "bt_calmar", "higher_better"),
    ("Profit Factor", "bt_profit_factor", "higher_better"),
    ("Hit Rate", "bt_hit_rate", "higher_better"),
]

ML_METRICS = ["MAE", "RMSE", "Dir. Accuracy", "IC (Spearman)"]

# Subgrupos por escala. Cada tupla: (titulo, slug_archivo, [metricas]).
ML_METRIC_GROUPS = [
    ("Errores (escala retorno)", "errores", ["MAE", "RMSE"]),
    ("Calidad de señal",        "calidad", ["Dir. Accuracy", "IC (Spearman)"]),
]
TRADING_METRIC_GROUPS = [
    ("Riesgo/retorno ajustado", "riesgo",     ["Sharpe Ratio", "Calmar Ratio"]),
    ("Retorno y drawdown",      "retorno",    ["CAGR", "Max Drawdown"]),
    ("Calidad de operaciones",  "operaciones",["Profit Factor", "Hit Rate"]),
]

FS_SUPTITLE = 17
FS_SUBTITLE = 15
FS_TICK = 15
FS_LEGEND = 14
FS_BAR_LABEL = 13


def load_latest_run(runs_dir: Path) -> Path | None:
    if not runs_dir.exists():
        return None
    candidates = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
    )
    return candidates[-1] if candidates else None


def load_summary_all(run_dir: Path) -> pd.DataFrame | None:
    for candidate in [
        run_dir / "summary_all.csv",
        run_dir / "baseline_summary_all.csv",
    ]:
        if candidate.exists():
            return pd.read_csv(candidate)
    return None


def load_champions_summary(
    strategy: str,
    slot: str,
    project_root: Path,
) -> pd.DataFrame | None:
    """Build a virtual summary by reading inline `metrics` from each ticker's slot.

    Each ticker may point to a different `run_dir` (champions are picked per-ticker
    based on composite score, so they often come from different historical runs).
    The returned DataFrame has one row per ticker with the same column names that
    `summary_all.csv` would have for those metrics, plus `_run_dir` for downstream
    helpers that need to read raw files.

    Args:
        strategy: registry top-level key, e.g. "e1", "e2", "e3".
        slot: "champion", "baseline", or "candidate".
        project_root: project root path (registry is at <root>/models/registry.json).

    Returns:
        DataFrame with one row per ticker, or None if the strategy is missing.
    """
    reg_path = project_root / "models" / "registry.json"
    if not reg_path.exists():
        print(f"[WARN] Registry not found at {reg_path}")
        return None
    reg = json.loads(reg_path.read_text())

    strat_data = reg.get("strategies", {}).get(strategy)
    if not strat_data:
        print(f"[WARN] Strategy '{strategy}' not in registry")
        return None

    rows = []
    missing = []
    for ticker, slots in strat_data.get("tickers", {}).items():
        slot_info = slots.get(slot)
        if not slot_info or not isinstance(slot_info, dict):
            missing.append(ticker)
            continue
        metrics = slot_info.get("metrics", {})
        if not metrics:
            missing.append(ticker)
            continue
        row = {"ticker": ticker, **metrics, "_run_dir": slot_info.get("run_dir", "")}
        rows.append(row)

    if missing:
        print(f"[WARN] {len(missing)} tickers in {strategy} have no '{slot}' slot or metrics: {missing}")

    if not rows:
        return None
    return pd.DataFrame(rows)


def load_summary_for_comparison(
    strategy: str,
    slot: str,
    explicit_run: Path | None,
    runs_subdir: Path,
    project_root: Path,
) -> tuple[pd.DataFrame | None, str]:
    """Load comparison data with a 3-level fallback chain.

    Order:
        1. If `explicit_run` is provided, load `summary_all.csv` from that single run.
        2. Else, try the registry slot for this strategy (champions per ticker).
        3. Else, fall back to the latest run under `runs_subdir`.

    Returns:
        (DataFrame or None, human-readable source description)
    """
    if explicit_run is not None:
        df = load_summary_all(explicit_run)
        return df, f"explicit run {explicit_run}"

    df = load_champions_summary(strategy, slot, project_root)
    if df is not None:
        return df, f"registry slot '{slot}' for {strategy}"

    latest = load_latest_run(runs_subdir)
    if latest is not None:
        df = load_summary_all(latest)
        return df, f"latest run under {runs_subdir} ({latest.name})"

    return None, "no source available"


def compare_runs(
    df_baseline: pd.DataFrame,
    df_model: pd.DataFrame,
    model_label: str = "Model",
) -> pd.DataFrame:
    rows = []
    for display, col, direction in METRICS_CONFIG:
        baseline_val = df_baseline[col].mean() if col in df_baseline.columns else np.nan
        model_val = df_model[col].mean() if col in df_model.columns else np.nan
        diff = model_val - baseline_val

        if direction == "higher_better":
            model_better = diff > 0
            improvement_pct = (diff / abs(baseline_val) * 100) if baseline_val != 0 else 0
        else:
            model_better = diff < 0
            improvement_pct = (-diff / abs(baseline_val) * 100) if baseline_val != 0 else 0

        rows.append({
            "Metrica": display,
            "Baseline": baseline_val,
            model_label: model_val,
            "Diferencia": diff,
            "Mejora_%": improvement_pct,
            "Modelo_mejor": model_better,
        })
    return pd.DataFrame(rows)


def _draw_grouped_bars(
    ax,
    subset: pd.DataFrame,
    model_label: str,
) -> None:
    metrics = subset["Metrica"].tolist()
    base_vals = subset["Baseline"].tolist()
    model_vals = subset[model_label].tolist()
    x = np.arange(len(metrics))
    width = 0.35

    bars_b = ax.bar(x - width / 2, base_vals, width, label="Baseline",
                    color="#90CAF9", edgecolor="white")
    bars_m = ax.bar(x + width / 2, model_vals, width, label=model_label,
                    color="#1565C0", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20, ha="right", fontsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Etiquetas numéricas en cada barra (formato compacto).
    def _fmt(v: float) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if abs(v) >= 100:
            return f"{v:.1f}"
        if abs(v) >= 1:
            return f"{v:.2f}"
        return f"{v:.3f}"

    for bars, vals in ((bars_b, base_vals), (bars_m, model_vals)):
        labels = [_fmt(v) for v in vals]
        ax.bar_label(bars, labels=labels, fontsize=FS_BAR_LABEL, padding=2)


def _save_single_group_figure(
    df_comp: pd.DataFrame,
    model_label: str,
    strategy_label: str,
    section_title: str,
    group_title: str,
    metric_names: list[str],
    dest: Path,
) -> None:
    subset = df_comp[df_comp["Metrica"].isin(metric_names)].copy()
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(
        f"{strategy_label} — Baseline vs {model_label}\n{section_title}: {group_title}",
        fontsize=FS_SUPTITLE,
        fontweight="bold",
    )

    _draw_grouped_bars(ax, subset, model_label)
    ax.legend(fontsize=FS_LEGEND, frameon=True)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(dest, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[OK] {dest}")


def _save_grouped_figure(
    df_comp: pd.DataFrame,
    model_label: str,
    strategy_label: str,
    title: str,
    metric_groups: list[tuple[str, str, list[str]]],
    dest: Path,
) -> None:
    visible_groups = []
    for group_title, _slug, metric_names in metric_groups:
        subset = df_comp[df_comp["Metrica"].isin(metric_names)].copy()
        if not subset.empty:
            visible_groups.append((group_title, subset))

    if not visible_groups:
        return

    n = len(visible_groups)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.5), squeeze=False)
    axes = axes[0]

    fig.suptitle(
        f"{strategy_label} — Baseline vs {model_label}\n{title}",
        fontsize=FS_SUPTITLE,
        fontweight="bold",
    )

    for ax, (group_title, subset) in zip(axes, visible_groups):
        _draw_grouped_bars(ax, subset, model_label)
        ax.set_title(group_title, fontsize=FS_SUBTITLE, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.965),
        fontsize=FS_LEGEND,
        frameon=True,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(dest, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[OK] {dest}")


def save_comparison_figure(
    df_comp: pd.DataFrame,
    model_label: str,
    strategy_label: str,
    out_path: Path,
    trading_split_per_group: bool = False,
) -> None:
    """Generate ML and trading comparison figures.

    Args:
        trading_split_per_group: if True, produces one trading figure per group
            (recommended when scales differ across groups, used by E1/E2/E3).
    """
    # ML: una figura por subgrupo (más espacio por barra para incluir en informes).
    for group_title, slug, metric_names in ML_METRIC_GROUPS:
        dest = out_path.with_name(f"{out_path.stem}_ml_{slug}{out_path.suffix}")
        _save_single_group_figure(
            df_comp=df_comp,
            model_label=model_label,
            strategy_label=strategy_label,
            section_title="Metricas ML (Offline)",
            group_title=group_title,
            metric_names=metric_names,
            dest=dest,
        )

    if trading_split_per_group:
        # Una figura independiente por grupo de trading (útil cuando las escalas
        # son demasiado distintas para compartir subplots).
        for group_title, slug, metric_names in TRADING_METRIC_GROUPS:
            dest = out_path.with_name(f"{out_path.stem}_trading_{slug}{out_path.suffix}")
            _save_single_group_figure(
                df_comp=df_comp,
                model_label=model_label,
                strategy_label=strategy_label,
                section_title="Metricas Trading (Backtest)",
                group_title=group_title,
                metric_names=metric_names,
                dest=dest,
            )
    else:
        # Trading: figura combinada con subplots por escala (se mantiene como estaba).
        dest = out_path.with_name(out_path.stem + "_trading" + out_path.suffix)
        _save_grouped_figure(
            df_comp=df_comp,
            model_label=model_label,
            strategy_label=strategy_label,
            title="Metricas Trading (Backtest)",
            metric_groups=TRADING_METRIC_GROUPS,
            dest=dest,
        )


def save_comparison_markdown(
    df_comp: pd.DataFrame,
    model_label: str,
    strategy_label: str,
    out_path: Path,
) -> None:
    ml = df_comp[df_comp["Metrica"].isin(ML_METRICS)].copy()
    trading = df_comp[~df_comp["Metrica"].isin(ML_METRICS)].copy()

    win_col = "Modelo_mejor"
    ml["Mejor"] = ml[win_col].map({True: "✓", False: "✗"})
    trading["Mejor"] = trading[win_col].map({True: "✓", False: "✗"})

    display_cols = ["Metrica", "Baseline", model_label, "Diferencia", "Mejora_%", "Mejor"]

    n_wins = int(df_comp[win_col].sum())
    total = len(df_comp)

    with open(out_path, "w") as f:
        f.write(f"# Comparación {strategy_label}\n\n")
        f.write("## Métricas ML\n\n")
        f.write(ml[display_cols].to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## Métricas Trading\n\n")
        f.write(trading[display_cols].to_markdown(index=False, floatfmt=".4f"))
        f.write(f"\n\n## Resumen\n\n")
        f.write(f"- **Total métricas**: {total}\n")
        f.write(f"- **{model_label} superior**: {n_wins} ({n_wins/total*100:.1f}%)\n")
        f.write(f"- **Baseline superior**: {total - n_wins} ({(total - n_wins)/total*100:.1f}%)\n")

    print(f"[OK] {out_path}")


def validate_improvement_signs(
    df: pd.DataFrame,
    mejora_col: str = "Mejora_%",
    mejor_col: str = "Modelo_mejor",
    metrica_col: str = "Metrica",
) -> None:
    """Warn when the sign of Mejora_% is inconsistent with Modelo_mejor.

    A positive improvement must mean the model is better, and vice-versa.
    This catches formula bugs where division by a negative baseline inverts the sign.
    """
    issues = []
    for _, row in df.iterrows():
        pct = row[mejora_col]
        better = row[mejor_col]
        if isinstance(better, str):
            better = better.strip() == "✓"
        if pd.isna(pct):
            continue
        if better and pct < 0:
            issues.append(f"  {row[metrica_col]}: modelo MEJOR pero Mejora = {pct:+.2f}% (negativo — bug de signo)")
        elif not better and pct > 0:
            issues.append(f"  {row[metrica_col]}: modelo PEOR pero Mejora = {pct:+.2f}% (positivo — bug de signo)")
    if issues:
        print("[WARN] Inconsistencias de signo detectadas:")
        for msg in issues:
            print(msg)
    else:
        print("[OK] Signos de Mejora_% consistentes en todas las métricas")


def print_comparison(df_comp: pd.DataFrame, model_label: str) -> None:
    ml = df_comp[df_comp["Metrica"].isin(ML_METRICS)]
    trading = df_comp[~df_comp["Metrica"].isin(ML_METRICS)]

    print("\n--- ML Metrics ---")
    for _, row in ml.iterrows():
        mark = ">" if row["Modelo_mejor"] else "<"
        print(f"  {mark} {row['Metrica']:<20} Baseline={row['Baseline']:.4f}  "
              f"{model_label}={row[model_label]:.4f}  Mejora={row['Mejora_%']:+.1f}%")

    print("\n--- Trading Metrics ---")
    for _, row in trading.iterrows():
        mark = ">" if row["Modelo_mejor"] else "<"
        print(f"  {mark} {row['Metrica']:<20} Baseline={row['Baseline']:.4f}  "
              f"{model_label}={row[model_label]:.4f}  Mejora={row['Mejora_%']:+.1f}%")

    n_wins = df_comp["Modelo_mejor"].sum()
    print(f"\n{model_label} wins {n_wins}/{len(df_comp)} metrics.")
