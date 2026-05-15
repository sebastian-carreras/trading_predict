"""Replay the E3 lifecycle decision using Spearman-IC instead of Pearson-IC.

Reads the current registry, recomputes Spearman IC from the predictions of
both champion and candidate (per ticker), substitutes ``metrics.ml_ic`` in
copies of those metric dicts, and reruns ``compare_candidate_vs_champion``
with the E3 promotion config.

This is a SIMULATION: it does not modify the registry.

Usage:
    python scripts/evaluation/replay_lifecycle_e3.py
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.lifecycle.promotion import compare_candidate_vs_champion  # noqa: E402

TICKERS = ("SPY", "AAPL", "NVDA", "QQQ")

E3_CONFIG = {
    "require_positive_sharpe": False,
    "min_improvement": 0.05,
    "scoring_weights": {
        "bt_sharpe": 0.30,
        "ml_ic": 0.20,
        "ml_directional_accuracy": 0.25,
        "bt_calmar": 0.25,
    },
}


def spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return 0.0
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    ic, _ = spearmanr(y_true, y_pred)
    return float(ic) if np.isfinite(ic) else 0.0


def predictions_path_for(run_dir: Path, ticker: str) -> Path:
    """Resolve the walkforward predictions file (LSTM ensemble runs)."""
    candidate = run_dir / ticker / f"{ticker}_walkforward_predictions.csv"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No predictions found in {run_dir} for {ticker}")


def recompute_ic_from_run(run_dir: Path, ticker: str) -> float:
    df = pd.read_csv(predictions_path_for(run_dir, ticker))
    return spearman_ic(df["y_true"].to_numpy(), df["y_pred"].to_numpy())


def main() -> None:
    registry_path = REPO_ROOT / "models" / "registry.json"
    with registry_path.open() as fh:
        registry = json.load(fh)

    e3_tickers = registry["strategies"]["e3"]["tickers"]

    print(f"Replay using E3 config: weights={E3_CONFIG['scoring_weights']}, "
          f"min_improvement={E3_CONFIG['min_improvement']}, "
          f"require_positive_sharpe={E3_CONFIG['require_positive_sharpe']}\n")

    print(f"{'Ticker':6s} {'IC_p (champ)':>12s} {'IC_s (champ)':>12s} "
          f"{'IC_p (cand)':>12s} {'IC_s (cand)':>12s} "
          f"{'Score (P)':>10s} {'Score (S)':>10s} "
          f"{'Improv P':>10s} {'Improv S':>10s} "
          f"{'Decision (P)':>14s} {'Decision (S)':>14s}")
    print("-" * 156)

    summary_rows = []
    for ticker in TICKERS:
        entry = e3_tickers.get(ticker)
        if not entry:
            print(f"{ticker}: no entry in registry; skipping")
            continue
        champion = entry.get("champion")
        candidate = entry.get("candidate")
        if not champion or not candidate:
            print(f"{ticker}: missing champion or candidate; skipping")
            continue

        champ_run = REPO_ROOT / champion["run_dir"].lstrip("./")
        cand_run = REPO_ROOT / candidate["run_dir"].lstrip("./")
        # `run_dir` already includes the ticker, so step up one level
        if champ_run.name == ticker:
            champ_run = champ_run.parent
        if cand_run.name == ticker:
            cand_run = cand_run.parent

        ic_p_champ = float(champion["metrics"].get("ml_ic"))
        ic_p_cand = float(candidate["metrics"].get("ml_ic"))

        ic_s_champ = recompute_ic_from_run(champ_run, ticker)
        ic_s_cand = recompute_ic_from_run(cand_run, ticker)

        # Pearson decision (as currently in registry)
        decision_p = compare_candidate_vs_champion(
            candidate["metrics"], champion["metrics"], E3_CONFIG
        )

        # Spearman decision: substitute ml_ic on copies
        cand_metrics_s = deepcopy(candidate["metrics"])
        champ_metrics_s = deepcopy(champion["metrics"])
        cand_metrics_s["ml_ic"] = ic_s_cand
        champ_metrics_s["ml_ic"] = ic_s_champ
        decision_s = compare_candidate_vs_champion(
            cand_metrics_s, champ_metrics_s, E3_CONFIG
        )

        flag = "FLIP" if decision_p.should_promote != decision_s.should_promote else "same"
        print(f"{ticker:6s} "
              f"{ic_p_champ:>+12.4f} {ic_s_champ:>+12.4f} "
              f"{ic_p_cand:>+12.4f} {ic_s_cand:>+12.4f} "
              f"{decision_p.champion_score:>+10.4f} {decision_s.champion_score:>+10.4f} "
              f"{decision_p.improvement_pct:>+10.4f} {decision_s.improvement_pct:>+10.4f} "
              f"{('PROMOTE' if decision_p.should_promote else 'KEEP'):>14s} "
              f"{('PROMOTE' if decision_s.should_promote else 'KEEP'):>14s}  [{flag}]")

        summary_rows.append({
            "ticker": ticker,
            "ic_pearson_champ": ic_p_champ,
            "ic_spearman_champ": ic_s_champ,
            "ic_pearson_cand": ic_p_cand,
            "ic_spearman_cand": ic_s_cand,
            "score_pearson_champ": decision_p.champion_score,
            "score_pearson_cand": decision_p.candidate_score,
            "score_spearman_champ": decision_s.champion_score,
            "score_spearman_cand": decision_s.candidate_score,
            "improvement_pearson": decision_p.improvement_pct,
            "improvement_spearman": decision_s.improvement_pct,
            "decision_pearson": "PROMOTE" if decision_p.should_promote else "KEEP",
            "decision_spearman": "PROMOTE" if decision_s.should_promote else "KEEP",
            "flipped": decision_p.should_promote != decision_s.should_promote,
        })

    print("-" * 156)
    flipped = [r for r in summary_rows if r["flipped"]]
    if flipped:
        print(f"\n{len(flipped)} ticker(s) flipped decision under Spearman:")
        for r in flipped:
            print(f"  {r['ticker']}: {r['decision_pearson']} -> {r['decision_spearman']}")
    else:
        print("\nNo ticker flipped its promotion decision under Spearman IC.")

    out_path = REPO_ROOT / "runs" / "e3_replay_lifecycle_spearman.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_path, index=False)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
