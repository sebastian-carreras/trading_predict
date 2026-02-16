"""Comparación de resultados entre las 3 versiones de E1.

Versiones:
- e1_baseline
- e1_simple
- e1_conservative

Métricas comparadas:
- tiempo de entrenamiento (promedio y total)
- mae, rmse, ic, directional_accuracy
- bt_sharpe, bt_cagr, bt_max_drawdown, bt_num_trades

Uso:
    python scripts/evaluation/compare_e1_versions.py
    python scripts/evaluation/compare_e1_versions.py --tickers AAPL,MSFT
    python scripts/evaluation/compare_e1_versions.py --no-common-tickers
    python scripts/evaluation/compare_e1_versions.py --per-ticker
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml


REQUIRED_COLUMNS = [
    "ticker",
    "ml_mae",
    "ml_rmse",
    "ml_ic",
    "ml_directional_accuracy",
    "bt_sharpe",
    "bt_cagr",
    "bt_max_drawdown",
    "bt_num_trades",
    "timing_train_seconds",
]

PER_TICKER_METRIC_COLUMNS = [
    "timing_train_seconds",
    "ml_mae",
    "ml_rmse",
    "ml_ic",
    "ml_directional_accuracy",
    "bt_sharpe",
    "bt_cagr",
    "bt_max_drawdown",
    "bt_num_trades",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def latest_run(runs_dir: Path) -> Path | None:
    if not runs_dir.exists():
        return None
    candidates = []
    for path in runs_dir.iterdir():
        if not path.is_dir():
            continue
        has_summary = (path / "summary_all.csv").exists() or (path / "baseline_summary_all.csv").exists()
        if has_summary:
            candidates.append(path)
    candidates = sorted(candidates)
    return candidates[-1] if candidates else None


def load_summary(run_dir: Path) -> pd.DataFrame:
    candidates = [
        run_dir / "summary_all.csv",
        run_dir / "baseline_summary_all.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"No se encontró summary_all.csv en {run_dir}")


def resolve_thresholds(df: pd.DataFrame, run_dir: Path, strategy_key: str) -> tuple[float, float]:
    buy_candidates = ["tau_buy", "bt_tau_buy", "threshold_tau_buy", "train_tau_buy"]
    sell_candidates = ["tau_sell", "bt_tau_sell", "threshold_tau_sell", "train_tau_sell"]

    def _from_df(candidates: list[str]) -> float | None:
        for col in candidates:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                if not series.empty:
                    return float(series.iloc[0])
        return None

    tau_buy_df = _from_df(buy_candidates)
    tau_sell_df = _from_df(sell_candidates)
    if tau_buy_df is not None and tau_sell_df is not None:
        return tau_buy_df, tau_sell_df

    cfg_path = run_dir / "config_used.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        thresholds = (
            cfg.get("strategies", {})
            .get(strategy_key, {})
            .get("thresholds", {})
        )
        if isinstance(thresholds, dict):
            return float(thresholds.get("tau_buy", np.nan)), float(thresholds.get("tau_sell", np.nan))

    return float("nan"), float("nan")


def ensure_required_columns(df: pd.DataFrame, label: str) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{label}: faltan columnas requeridas: {', '.join(missing)}")


def filter_tickers(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if not tickers:
        return df
    selected = {ticker.strip() for ticker in tickers if ticker.strip()}
    if not selected:
        return df
    return df[df["ticker"].astype(str).isin(selected)].copy()


def compute_row(version: str, df: pd.DataFrame, tau_buy: float, tau_sell: float) -> dict[str, float | str | int]:
    return {
        "version": version,
        "n_tickers": int(df["ticker"].nunique()),
        "train_time_seconds_avg": float(df["timing_train_seconds"].mean()),
        "train_time_seconds_total": float(df["timing_train_seconds"].sum()),
        "mae": float(df["ml_mae"].mean()),
        "rmse": float(df["ml_rmse"].mean()),
        "ic": float(df["ml_ic"].mean()),
        "directional_accuracy": float(df["ml_directional_accuracy"].mean()),
        "bt_sharpe": float(df["bt_sharpe"].mean()),
        "bt_cagr": float(df["bt_cagr"].mean()),
        "bt_max_drawdown": float(df["bt_max_drawdown"].mean()),
        "tau_buy": float(tau_buy),
        "tau_sell": float(tau_sell),
        "bt_num_trades": float(df["bt_num_trades"].mean()),
    }


def format_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        "train_time_seconds_avg",
        "train_time_seconds_total",
        "mae",
        "rmse",
        "ic",
        "directional_accuracy",
        "bt_sharpe",
        "bt_cagr",
        "bt_max_drawdown",
        "tau_buy",
        "tau_sell",
        "bt_num_trades",
    ]
    for col in numeric_cols:
        out[col] = out[col].astype(float).round(6)
    return out


def build_per_ticker_comparison(
    baseline_df: pd.DataFrame,
    simple_df: pd.DataFrame,
    conservative_df: pd.DataFrame,
    *,
    tau_by_version: dict[str, tuple[float, float]],
    allow_distinct_universes: bool,
) -> pd.DataFrame:
    metric_rename = {
        "timing_train_seconds": "train_time_seconds",
        "ml_mae": "mae",
        "ml_rmse": "rmse",
        "ml_ic": "ic",
        "ml_directional_accuracy": "directional_accuracy",
        "bt_sharpe": "bt_sharpe",
        "bt_cagr": "bt_cagr",
        "bt_max_drawdown": "bt_max_drawdown",
        "bt_num_trades": "bt_num_trades",
    }

    def _normalize(df: pd.DataFrame, version: str) -> pd.DataFrame:
        out = df[["ticker", *PER_TICKER_METRIC_COLUMNS]].copy()
        out["version"] = version
        tau_buy, tau_sell = tau_by_version.get(version, (float("nan"), float("nan")))
        out["tau_buy"] = float(tau_buy)
        out["tau_sell"] = float(tau_sell)
        out = out.rename(columns=metric_rename)
        return out[
            [
                "ticker",
                "version",
                "tau_buy",
                "tau_sell",
                "train_time_seconds",
                "mae",
                "rmse",
                "ic",
                "directional_accuracy",
                "bt_sharpe",
                "bt_cagr",
                "bt_max_drawdown",
                "bt_num_trades",
            ]
        ]

    base = _normalize(baseline_df, "e1_baseline")
    simp = _normalize(simple_df, "e1_simple")
    cons = _normalize(conservative_df, "e1_conservative")

    combined = pd.concat([base, simp, cons], ignore_index=True)

    if not allow_distinct_universes:
        common = set(base["ticker"]) & set(simp["ticker"]) & set(cons["ticker"])
        combined = combined[combined["ticker"].isin(common)].copy()

    version_order = {"e1_baseline": 0, "e1_simple": 1, "e1_conservative": 2}
    combined["_version_order"] = combined["version"].map(version_order).fillna(99)
    combined = combined.sort_values(["ticker", "_version_order"]).drop(columns=["_version_order"]).reset_index(drop=True)

    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara resultados de E1 baseline/simple/conservative")
    parser.add_argument("--baseline-run", type=str, default=None, help="Path run e1_baseline")
    parser.add_argument("--simple-run", type=str, default=None, help="Path run e1_simple")
    parser.add_argument("--conservative-run", type=str, default=None, help="Path run e1_conservative")
    parser.add_argument("--tickers", type=str, default="", help="Tickers separados por coma")
    parser.add_argument(
        "--no-common-tickers",
        action="store_true",
        help="No forzar intersección de tickers entre versiones",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/tables/e1_versions_comparison",
        help="Base de salida (se crea carpeta y archivos con timestamp)",
    )
    parser.add_argument(
        "--per-ticker",
        action="store_true",
        help="Genera comparación por ticker entre versiones (ticker vs ticker)",
    )
    args = parser.parse_args()

    root = project_root()

    baseline_run = Path(args.baseline_run) if args.baseline_run else latest_run(root / "runs" / "e1_baseline")
    simple_run = Path(args.simple_run) if args.simple_run else latest_run(root / "runs" / "e1_simple")
    conservative_run = (
        Path(args.conservative_run) if args.conservative_run else latest_run(root / "runs" / "e1_conservative")
    )

    if baseline_run is None or simple_run is None or conservative_run is None:
        print("❌ No se pudieron detectar runs para las 3 versiones.")
        print("   Revisa runs/e1_baseline, runs/e1_simple y runs/e1_conservative")
        sys.exit(1)

    baseline_df = load_summary(baseline_run)
    simple_df = load_summary(simple_run)
    conservative_df = load_summary(conservative_run)

    ensure_required_columns(baseline_df, "e1_baseline")
    ensure_required_columns(simple_df, "e1_simple")
    ensure_required_columns(conservative_df, "e1_conservative")

    ticker_filter = [t.strip() for t in args.tickers.split(",") if t.strip()]
    baseline_df = filter_tickers(baseline_df, ticker_filter)
    simple_df = filter_tickers(simple_df, ticker_filter)
    conservative_df = filter_tickers(conservative_df, ticker_filter)

    if baseline_df.empty or simple_df.empty or conservative_df.empty:
        print("❌ Después del filtro, una o más versiones quedaron sin tickers.")
        sys.exit(1)

    if not args.no_common_tickers:
        common = (
            set(baseline_df["ticker"].astype(str))
            & set(simple_df["ticker"].astype(str))
            & set(conservative_df["ticker"].astype(str))
        )
        if not common:
            print("❌ No hay tickers en común entre las 3 versiones.")
            print("   Usa --no-common-tickers para comparar universos distintos.")
            sys.exit(1)

        baseline_df = baseline_df[baseline_df["ticker"].astype(str).isin(common)].copy()
        simple_df = simple_df[simple_df["ticker"].astype(str).isin(common)].copy()
        conservative_df = conservative_df[conservative_df["ticker"].astype(str).isin(common)].copy()

    rows = [
        compute_row("e1_baseline", baseline_df, *resolve_thresholds(baseline_df, baseline_run, "e1_conservative")),
        compute_row("e1_simple", simple_df, *resolve_thresholds(simple_df, simple_run, "e1_simple")),
        compute_row(
            "e1_conservative",
            conservative_df,
            *resolve_thresholds(conservative_df, conservative_run, "e1_conservative"),
        ),
    ]
    comparison_df = format_numeric(pd.DataFrame(rows))

    tau_by_version = {
        "e1_baseline": resolve_thresholds(baseline_df, baseline_run, "e1_conservative"),
        "e1_simple": resolve_thresholds(simple_df, simple_run, "e1_simple"),
        "e1_conservative": resolve_thresholds(conservative_df, conservative_run, "e1_conservative"),
    }

    print("\n" + "=" * 100)
    print("COMPARACIÓN E1 (3 VERSIONES)")
    print("=" * 100)
    print(f"baseline     : {baseline_run}")
    print(f"simple       : {simple_run}")
    print(f"conservative : {conservative_run}")
    print("\nMétricas: tau_buy, tau_sell, train_time_seconds_avg, train_time_seconds_total,")
    print("          mae, rmse, ic, directional_accuracy, bt_sharpe, bt_cagr,")
    print("          bt_max_drawdown, bt_num_trades\n")
    print(comparison_df.to_string(index=False))

    output_base = Path(args.output)
    if not output_base.is_absolute():
        output_base = root / output_base

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base.parent / output_base.name
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{output_base.name}_{timestamp}"
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    comparison_df.to_csv(csv_path, index=False)

    md_df = comparison_df.copy()
    md_df = md_df.replace({np.nan: ""})
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# Comparación E1 (Baseline vs Simple vs Conservative)\n\n")
        handle.write(md_df.to_markdown(index=False))
        handle.write("\n")

    if args.per_ticker:
        per_ticker_df = build_per_ticker_comparison(
            baseline_df,
            simple_df,
            conservative_df,
            tau_by_version=tau_by_version,
            allow_distinct_universes=args.no_common_tickers,
        )

        per_ticker_csv = output_dir / f"{stem}_per_ticker.csv"
        per_ticker_md = output_dir / f"{stem}_per_ticker.md"

        per_ticker_df.to_csv(per_ticker_csv, index=False)
        with open(per_ticker_md, "w", encoding="utf-8") as handle:
            handle.write("# Comparación E1 por Ticker (Baseline vs Simple vs Conservative)\n\n")
            handle.write(per_ticker_df.to_markdown(index=False))
            handle.write("\n")

        print(f"✓ CSV por ticker: {per_ticker_csv}")
        print(f"✓ MD  por ticker: {per_ticker_md}")

    print(f"✓ Carpeta de salida: {output_dir}")
    print(f"\n✓ CSV: {csv_path}")
    print(f"✓ MD : {md_path}")


if __name__ == "__main__":
    main()
