"""Compare E2 champion (LSTM 20d) vs E2 baseline (Ridge).

By default reads champions/baselines from models/registry.json (best per ticker,
possibly from different historical runs). Pass --baseline-run / --model-run to
override with a specific run directory.

Usage:
    python -m scripts.evaluation.compare_e2_models
    python -m scripts.evaluation.compare_e2_models --model-run runs/e2_moderate/20260501_141523
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.evaluation._compare_common import (
    compare_runs,
    load_summary_for_comparison,
    print_comparison,
    save_comparison_figure,
    save_comparison_markdown,
    validate_improvement_signs,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "conclusiones" / "comparaciones"
STRATEGY = "e2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare E2 Moderate LSTM vs Baseline")
    parser.add_argument("--baseline-run", default=None)
    parser.add_argument("--model-run", default=None)
    args = parser.parse_args()

    df_baseline, baseline_src = load_summary_for_comparison(
        STRATEGY, "baseline",
        explicit_run=Path(args.baseline_run) if args.baseline_run else None,
        runs_subdir=ROOT / "runs" / "e2_baseline",
        project_root=ROOT,
    )
    df_model, model_src = load_summary_for_comparison(
        STRATEGY, "champion",
        explicit_run=Path(args.model_run) if args.model_run else None,
        runs_subdir=ROOT / "runs" / "e2_moderate",
        project_root=ROOT,
    )

    print(f"[INFO] Baseline source: {baseline_src}")
    print(f"[INFO] Model source:    {model_src}")

    if df_baseline is None:
        print("ERROR: could not load baseline data")
        sys.exit(1)
    if df_model is None:
        print("ERROR: could not load model data")
        sys.exit(1)

    print(f"Baseline tickers: {len(df_baseline)}, Model tickers: {len(df_model)}")

    df_comp = compare_runs(df_baseline, df_model, model_label="LSTM E2")
    validate_improvement_signs(df_comp)
    print_comparison(df_comp, "LSTM E2")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_comparison_figure(
        df_comp,
        model_label="LSTM E2",
        strategy_label="E2 Moderate (LSTM 20-day)",
        out_path=OUT_DIR / "fig_e2_vs_baseline.png",
        trading_split_per_group=True,
    )

    # Markdown report
    save_comparison_markdown(
        df_comp,
        model_label="LSTM E2",
        strategy_label="E2 Moderate (LSTM 20-day)",
        out_path=OUT_DIR / "fig_e2_vs_baseline.md",
    )

    # Also save CSV
    out_csv = OUT_DIR / "e2_comparison.csv"
    df_comp.to_csv(out_csv, index=False)
    print(f"[OK] {out_csv}")


if __name__ == "__main__":
    main()
