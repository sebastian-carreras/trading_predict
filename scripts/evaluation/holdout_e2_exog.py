"""
Validación HELD-OUT temporal de las features exógenas (Fase 1) para E2.

Motivación: la ablación seleccionó los 13 tickers "ganadores" sobre la MISMA ventana
walk-forward (2016-2026) donde se midió el score → riesgo de sobreajustar la decisión.
Esta validación separa selección de evaluación en el TIEMPO:

  1. Entrena base y exog con datos SOLO hasta ``--train-end`` (default 2023-12-31).
  2. Toma el modelo congelado y lo corre FORWARD sobre la ventana no vista
     ``[--eval-start, --eval-end]`` (default 2024-01-01 .. 2026-03-03) usando
     src.lifecycle.reevaluation (inferencia honesta, sin reentrenar).
  3. Recomputa el score compuesto en esa ventana held-out y compara base vs exog.

Si los ganadores de la ablación siguen ganando en 2024-2026 (datos que el modelo nunca
vio), la ventaja es robusta al tiempo, no un artefacto de la ventana.

Corre con register_lifecycle=False → NO toca models/registry.json.

Uso:
    PYTHONPATH=. python -m scripts.evaluation.holdout_e2_exog                    # 13 winners + controles
    PYTHONPATH=. python -m scripts.evaluation.holdout_e2_exog --tickers NVDA
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from src.e2.build_features import make_target_e2
from src.e2.train_pipeline import run_e2_for_ticker
from src.lifecycle import reevaluation
from src.lifecycle.promotion import compute_score
from src.utils import ensure_dir, get_nested, load_yaml, project_root

# Ganadores robustos de la ablación JUSTA v3 (params exog-tuneados, 30 tickers) + perdedores
# robustos como control negativo (no deberían mejorar out-of-sample). Ver docs/ABLATION_EXOG.md.
_WINNERS = ["COST", "GE", "NFLX", "NVDA", "GOOGL", "LOMA.BA", "AVGO", "CRM", "ORCL", "AMZN", "MA", "ISRG"]
_CONTROLS = ["GS", "WMT", "VRTX"]
_METRIC_KEYS = ("bt_sharpe", "ml_ic", "ml_directional_accuracy", "bt_calmar")
_FIELDS = ["ticker", "group", "mode", "n_seeds", "n_eval", "score_mean", "score_std"] + [f"{k}_mean" for k in _METRIC_KEYS]

# Params Optuna tuneados para el set exógeno (21 features) — el modo exog los usa si existen
# (comparación justa base-vs-exog). Ver el plan Paso 3/4.
_EXOG_TUNED_PATH = "reports/hyperparameter_optimization/exog/e2_tuned_params_by_ticker.yaml"


def _eval_one_seed(cfg: dict, ticker: str, mode: str, seed: int, train_end: str,
                   eval_start: str, eval_end: str, root: Path, weights: dict):
    """Entrena un seed (≤train_end) y evalúa el modelo congelado en [eval_start, eval_end]."""
    c = copy.deepcopy(cfg)
    c.setdefault("data", {}).setdefault("training_window", {}).setdefault("daily", {})["end"] = train_end
    c.setdefault("data", {}).setdefault("exog", {})["enabled"] = (mode == "exog")
    c.setdefault("project", {})["seed"] = int(seed)
    if mode == "exog" and (root / _EXOG_TUNED_PATH).exists():
        c.setdefault("optuna", {}).setdefault("e2_moderate", {})["tuned_params_path"] = _EXOG_TUNED_PATH

    out_dir = root / "runs" / "holdout" / mode
    run_e2_for_ticker(c, ticker, root / "data" / "raw" / "daily", out_dir,
                      register_lifecycle=False, auto_promote=False)

    # Modelo congelado (última fold del walk-forward ≤ train_end).
    import torch
    payload = torch.load(out_dir / ticker / f"{ticker}_model.pth", map_location="cpu", weights_only=False)

    # OHLCV completo (incluye 2024-2026) para inferencia forward honesta.
    ohlcv = reevaluation.load_clean_ohlcv(ticker, root=root)
    preds = reevaluation.predict_series("e2_moderate", payload, ohlcv, ticker=ticker, root=root)
    y_true = make_target_e2(ohlcv, horizon_days=int(payload["horizon_days"])).reindex(preds.index)

    lo = pd.Timestamp(eval_start, tz="UTC")
    hi = pd.Timestamp(eval_end, tz="UTC")
    mask = (preds.index >= lo) & (preds.index <= hi) & y_true.notna().to_numpy()
    window = preds.index[mask]
    if len(window) < 20:
        raise ValueError(f"ventana held-out insuficiente: n={len(window)}")

    bt_params = reevaluation.backtest_params_from_config(cfg, "e2_moderate")
    metrics = reevaluation.recompute_metrics_on_window(
        ohlcv, window, preds.loc[window].to_numpy(), y_true.loc[window].to_numpy(), bt_params,
    )
    score, _ = compute_score(metrics, weights)
    return score, metrics, len(window)


def _train_and_eval(cfg: dict, ticker: str, mode: str, seeds: list[int], train_end: str,
                    eval_start: str, eval_end: str, root: Path, weights: dict):
    """Multi-seed: entrena/evalúa el modelo congelado por cada seed y promedia el score held-out."""
    scores: list[float] = []
    mets: dict[str, list[float]] = {k: [] for k in _METRIC_KEYS}
    n_eval = 0
    for seed in seeds:
        s, m, n = _eval_one_seed(cfg, ticker, mode, seed, train_end, eval_start, eval_end, root, weights)
        scores.append(s)
        n_eval = n
        for k in _METRIC_KEYS:
            v = m.get(k)
            if isinstance(v, (int, float)) and v == v:  # descarta NaN
                mets[k].append(float(v))
    smean = float(np.mean(scores))
    row = [ticker, "", mode, len(scores), n_eval, round(smean, 6), round(float(np.std(scores)), 6)] + [
        round(float(np.mean(mets[k])), 6) if mets[k] else "" for k in _METRIC_KEYS
    ]
    return row, smean


def run(tickers: list[str], groups: dict[str, str], seeds: list[int], train_end: str,
        eval_start: str, eval_end: str, out_path: Path) -> None:
    root = project_root()
    cfg = load_yaml(root / "src" / "config" / "base.yaml")
    promo = get_nested(cfg, ["lifecycle", "promotion"], default={}) or {}
    weights = get_nested(promo, ["strategy_overrides", "e2_moderate", "scoring_weights"], default=None) \
        or promo.get("scoring_weights", {})

    ensure_dir(out_path.parent)
    rows: list[list] = []
    scores: dict[str, dict[str, float]] = {}

    print(f"Held-out E2 | train≤{train_end} · eval [{eval_start}..{eval_end}] | "
          f"{len(tickers)} tickers × {len(seeds)} seeds={seeds} | pesos={weights}\n", flush=True)
    for i, tk in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {tk} ({groups.get(tk, '?')})", flush=True)
        for mode in ("base", "exog"):
            try:
                row, score = _train_and_eval(cfg, tk, mode, seeds, train_end, eval_start, eval_end, root, weights)
                scores.setdefault(tk, {})[mode] = score
            except Exception as exc:
                row = [tk, groups.get(tk, ""), mode, 0, 0, "ERR", str(exc)[:40]] + [""] * len(_METRIC_KEYS)
                score = None
            row[1] = groups.get(tk, "")
            rows.append(row)
            print("   " + " | ".join(str(c) for c in row), flush=True)
            with open(out_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(_FIELDS)
                w.writerows(rows)
        d = scores.get(tk, {})
        if "base" in d and "exog" in d:
            delta = d["exog"] - d["base"]
            print(f"   >>> {tk}: base={d['base']:.4f} exog={d['exog']:.4f} Δ={delta:+.4f} "
                  f"{'MEJORA ✅' if delta > 0 else 'peor ❌'}\n", flush=True)

    _summary(scores, groups, out_path)


def _summary(scores: dict, groups: dict, out_path: Path) -> None:
    pairs = {t: d for t, d in scores.items() if "base" in d and "exog" in d}
    print("=" * 60)
    for grp in ("winner", "control"):
        ts = [t for t in pairs if groups.get(t) == grp]
        if not ts:
            continue
        dd = np.array([pairs[t]["exog"] - pairs[t]["base"] for t in ts])
        print(f"{grp.upper()} (n={len(ts)}): Δ mediana={np.median(dd):+.4f} media={dd.mean():+.4f} | "
              f"siguen mejorando held-out {(dd > 0).sum()}/{len(ts)}")
    print(f"\nCSV: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validación held-out temporal exógenas E2")
    ap.add_argument("--tickers", type=str, default="", help="Coma-separados (default: 13 winners + 2 controles)")
    ap.add_argument("--train-end", type=str, default="2023-12-31")
    ap.add_argument("--eval-start", type=str, default="2024-01-01")
    ap.add_argument("--eval-end", type=str, default="2026-03-03")
    ap.add_argument("--out", type=str, default="reports/ablation/e2_exog_holdout.csv")
    ap.add_argument("--seeds", type=str, default="42", help="Coma-separados (default: 42 = single seed)")
    args = ap.parse_args()

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        groups = {t: ("winner" if t in _WINNERS else "control" if t in _CONTROLS else "other") for t in tickers}
    else:
        tickers = _WINNERS + _CONTROLS
        groups = {**{t: "winner" for t in _WINNERS}, **{t: "control" for t in _CONTROLS}}

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    root = project_root()
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    run(tickers, groups, seeds, args.train_end, args.eval_start, args.eval_end, out)


if __name__ == "__main__":
    main()
