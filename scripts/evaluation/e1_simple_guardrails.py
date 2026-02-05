#!/usr/bin/env python
"""
Guardrails rápidos para E1 Simple.

Valida métricas mínimas y ausencia de NaNs en backtest de un run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs" / "e1_simple"


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def find_latest_run_dir(runs_dir: Path) -> Path | None:
    if not runs_dir.exists():
        return None
    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def resolve_summary_path(*, run_dir: Path | None, summary_file: Path | None) -> Path:
    if summary_file is not None:
        return summary_file
    if run_dir is None:
        raise ValueError("run_dir es requerido si no se pasa summary_file")
    return run_dir / "summary_all.csv"


def resolve_backtest_path(
    *,
    row: pd.Series,
    run_dir: Path | None,
    project_root: Path,
) -> Path | None:
    backtest_file = row.get("backtest_file")
    if isinstance(backtest_file, str) and backtest_file:
        backtest_path = Path(backtest_file)
        if not backtest_path.is_absolute():
            backtest_path = project_root / backtest_path
        return backtest_path

    ticker = row.get("ticker")
    if run_dir is not None and ticker:
        return run_dir / str(ticker) / f"{ticker}_backtest.csv"
    return None


def _get_metric(row: pd.Series, *keys: str) -> float | None:
    for key in keys:
        if key in row:
            value = row.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def validate_backtest(backtest_path: Path) -> list[str]:
    if not backtest_path.exists():
        return [f"Backtest no existe: {backtest_path}"]

    try:
        bt = pd.read_csv(backtest_path)
    except Exception as exc:
        return [f"Backtest ilegible {backtest_path}: {exc}"]

    required_cols = ["net_ret", "equity", "costs", "turnover"]
    missing = [c for c in required_cols if c not in bt.columns]
    if missing:
        return [f"Backtest sin columnas {missing}: {backtest_path}"]

    for col in required_cols:
        series = bt[col].to_numpy(dtype=float, copy=False)
        if not np.isfinite(series).all():
            return [f"Backtest con NaN/Inf en {col}: {backtest_path}"]

    return []


def validate_summary(
    *,
    summary_path: Path,
    project_root: Path,
    run_dir: Path | None,
    sharpe_min: float | None,
    max_drawdown_max: float | None,
    cagr_min: float | None,
) -> list[str]:
    if not summary_path.exists():
        return [f"No existe summary_all.csv: {summary_path}"]

    try:
        summary = pd.read_csv(summary_path)
    except Exception as exc:
        return [f"Summary ilegible {summary_path}: {exc}"]

    errors: list[str] = []
    if summary.empty:
        return ["Summary vacío"]

    for _, row in summary.iterrows():
        ticker = row.get("ticker", "<unknown>")
        sharpe = _get_metric(row, "bt_sharpe", "sharpe")
        max_dd = _get_metric(row, "bt_max_drawdown", "max_drawdown")
        cagr = _get_metric(row, "bt_cagr", "cagr")

        if sharpe_min is not None and sharpe is not None and sharpe < sharpe_min:
            errors.append(f"{ticker}: Sharpe {sharpe:.3f} < min {sharpe_min:.3f}")
        if max_drawdown_max is not None and max_dd is not None and max_dd > max_drawdown_max:
            errors.append(
                f"{ticker}: Max DD {max_dd:.3f} > max {max_drawdown_max:.3f}"
            )
        if cagr_min is not None and cagr is not None and cagr < cagr_min:
            errors.append(f"{ticker}: CAGR {cagr:.3f} < min {cagr_min:.3f}")

        backtest_path = resolve_backtest_path(row=row, run_dir=run_dir, project_root=project_root)
        if backtest_path is None:
            errors.append(f"{ticker}: No se pudo resolver backtest_file")
        else:
            errors.extend(validate_backtest(backtest_path))

    return errors


def build_thresholds_from_config(config: dict) -> dict:
    decision_cfg = config.get("decision", {})
    targets = decision_cfg.get("targets", {})
    return {
        "sharpe_min": float(targets.get("sharpe_min", 1.0)),
        "max_drawdown_max": 0.20,
        "cagr_min": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Guardrails rápidos para E1 Simple")
    parser.add_argument("--config", default="src/config/base.yaml", help="Ruta al config YAML")
    parser.add_argument("--run-dir", help="Ruta del run (ej: runs/e1_simple/20260124_175222)")
    parser.add_argument("--summary-file", help="Ruta a summary_all.csv")
    parser.add_argument("--sharpe-min", type=float, help="Sharpe mínimo")
    parser.add_argument("--max-drawdown-max", type=float, help="Max drawdown máximo")
    parser.add_argument("--cagr-min", type=float, help="CAGR mínimo")

    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)
    defaults = build_thresholds_from_config(config)

    sharpe_min = args.sharpe_min if args.sharpe_min is not None else defaults["sharpe_min"]
    max_dd_max = (
        args.max_drawdown_max
        if args.max_drawdown_max is not None
        else defaults["max_drawdown_max"]
    )
    cagr_min = args.cagr_min if args.cagr_min is not None else defaults["cagr_min"]

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run_dir(DEFAULT_RUNS_DIR)
    summary_file = Path(args.summary_file) if args.summary_file else None

    summary_path = resolve_summary_path(run_dir=run_dir, summary_file=summary_file)

    errors = validate_summary(
        summary_path=summary_path,
        project_root=PROJECT_ROOT,
        run_dir=run_dir,
        sharpe_min=sharpe_min,
        max_drawdown_max=max_dd_max,
        cagr_min=cagr_min,
    )

    if errors:
        print("❌ Guardrails fallaron:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("✅ Guardrails OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
