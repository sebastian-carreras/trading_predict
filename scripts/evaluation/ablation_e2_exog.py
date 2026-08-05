"""
Ablación E2: features base (12) vs base + núcleo exógeno (Bloques A/B/C/D).

Entrena cada ticker en ambos modos —sin y con exógenas— y compara el score compuesto
(los mismos pesos que la promoción: 0.35·Sharpe + 0.25·IC + 0.20·DirAcc + 0.20·Calmar).

Robustez multi-seed: con --seeds se promedia el score sobre varios seeds por (ticker, modo),
para separar la SEÑAL de las exógenas del RUIDO de entrenamiento del LSTM (init aleatorio).
El Δ per-ticker de un solo seed es ruidoso; el promedio sobre seeds es lo que hay que mirar.

Corre con register_lifecycle=False → NO ensucia models/registry.json. Modelos efímeros en
runs/ablation/{base,exog}/<TICKER>/.

Uso:
    PYTHONPATH=. python -m scripts.evaluation.ablation_e2_exog --seeds 42,7,123      # 30 tickers × 3 seeds
    PYTHONPATH=. python -m scripts.evaluation.ablation_e2_exog --tickers NVDA,GS     # subconjunto
    PYTHONPATH=. python -m scripts.evaluation.ablation_e2_exog                        # single seed (42)
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import numpy as np

from src.lifecycle.promotion import compute_score
from src.e2.train_pipeline import run_e2_for_ticker
from src.utils import ensure_dir, get_nested, load_yaml, project_root

_METRIC_KEYS = ("bt_sharpe", "ml_ic", "ml_directional_accuracy", "bt_calmar")
_FIELDS = ["ticker", "mode", "n_ok", "score_mean", "score_std"] + [f"{k}_mean" for k in _METRIC_KEYS]

# Params Optuna tuneados PARA el set exógeno (21 features). Si existe, el modo exog los usa
# → comparación justa (base con params base, exog con params exog). Ver el plan Paso 3.
_EXOG_TUNED_PATH = "reports/hyperparameter_optimization/exog/e2_tuned_params_by_ticker.yaml"


def _run_mode(cfg: dict, ticker: str, mode: str, seeds: list[int], raw_dir: Path, root: Path, weights: dict):
    """Entrena (ticker, modo) sobre todos los seeds. Devuelve (fila_csv, score_mean|None)."""
    scores: list[float] = []
    metrics: dict[str, list[float]] = {k: [] for k in _METRIC_KEYS}
    for seed in seeds:
        c = copy.deepcopy(cfg)
        c.setdefault("data", {}).setdefault("exog", {})["enabled"] = (mode == "exog")
        c.setdefault("project", {})["seed"] = int(seed)
        # Comparación justa: el modo exog usa los params tuneados para 21 features (si existen).
        if mode == "exog" and (root / _EXOG_TUNED_PATH).exists():
            c.setdefault("optuna", {}).setdefault("e2_moderate", {})["tuned_params_path"] = _EXOG_TUNED_PATH
        out_dir = root / "runs" / "ablation" / mode
        try:
            summary = run_e2_for_ticker(c, ticker, raw_dir, out_dir, register_lifecycle=False, auto_promote=False)
            s, _ = compute_score(summary, weights)
            scores.append(s)
            for k in _METRIC_KEYS:
                v = summary.get(k)
                if isinstance(v, (int, float)):
                    metrics[k].append(float(v))
        except Exception as exc:
            print(f"      seed={seed} ERR: {str(exc)[:80]}", flush=True)

    if not scores:
        return [ticker, mode, 0, "ERR", "", "", "", "", ""], None
    smean = float(np.mean(scores))
    row = [ticker, mode, len(scores), round(smean, 6), round(float(np.std(scores)), 6)] + \
        [round(float(np.mean(metrics[k])), 6) if metrics[k] else "" for k in _METRIC_KEYS]
    return row, smean


def run(tickers: list[str], seeds: list[int], out_path: Path) -> None:
    root = project_root()
    cfg = load_yaml(root / "src" / "config" / "base.yaml")
    promo = get_nested(cfg, ["lifecycle", "promotion"], default={}) or {}
    weights = get_nested(promo, ["strategy_overrides", "e2_moderate", "scoring_weights"], default=None) \
        or promo.get("scoring_weights", {})
    raw_dir = root / "data" / "raw" / "daily"

    ensure_dir(out_path.parent)
    rows: list[list] = []
    scores: dict[str, dict[str, float]] = {}

    print(f"Ablación E2 base vs exog | {len(tickers)} tickers × {len(seeds)} seeds={seeds} | pesos={weights}\n", flush=True)
    for i, tk in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {tk}", flush=True)
        for mode in ("base", "exog"):
            row, smean = _run_mode(cfg, tk, mode, seeds, raw_dir, root, weights)
            rows.append(row)
            if smean is not None:
                scores.setdefault(tk, {})[mode] = smean
            print("   " + " | ".join(str(c) for c in row), flush=True)
            with open(out_path, "w", newline="") as f:  # escritura incremental
                w = csv.writer(f)
                w.writerow(_FIELDS)
                w.writerows(rows)
        d = scores.get(tk, {})
        if "base" in d and "exog" in d:
            delta = d["exog"] - d["base"]
            print(f"   >>> {tk}: base={d['base']:.4f} exog={d['exog']:.4f} Δ={delta:+.4f} "
                  f"{'MEJORA ✅' if delta > 0 else 'peor ❌'}\n", flush=True)

    _summary(scores, out_path)


def _summary(scores: dict[str, dict[str, float]], out_path: Path) -> None:
    pairs = {tk: d for tk, d in scores.items() if "base" in d and "exog" in d}
    if not pairs:
        print("Sin pares base/exog válidos para agregar.")
        return
    base = np.array([d["base"] for d in pairs.values()])
    exog = np.array([d["exog"] for d in pairs.values()])
    ba = [t for t in pairs if t.endswith(".BA")]
    us = [t for t in pairs if not t.endswith(".BA")]

    def grp(name: str, ts: list[str]) -> None:
        if not ts:
            return
        dd = np.array([pairs[t]["exog"] - pairs[t]["base"] for t in ts])
        print(f"  {name} (n={len(ts)}): Δ mediana={np.median(dd):+.4f} media={dd.mean():+.4f} | mejoran {(dd > 0).sum()}/{len(ts)}")

    print("=" * 60)
    print(f"AGREGADO ({len(pairs)} tickers):")
    print(f"  score mediana:  base={np.median(base):.4f}  exog={np.median(exog):.4f}  Δ={np.median(exog) - np.median(base):+.4f}")
    print(f"  score media:    base={base.mean():.4f}  exog={exog.mean():.4f}  Δ={exog.mean() - base.mean():+.4f}")
    print(f"  tickers que MEJORAN con exógenas: {int((exog > base).sum())}/{len(pairs)}")
    grp("US ", us)
    grp(".BA", ba)
    print(f"\nCSV: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ablación E2 base vs exógenas (A/B/C/D), multi-seed")
    ap.add_argument("--tickers", type=str, default="", help="Coma-separados (default: universo E2)")
    ap.add_argument("--seeds", type=str, default="42", help="Coma-separados (default: 42 = single seed)")
    ap.add_argument("--out", type=str, default="reports/ablation/e2_exog_ablation.csv")
    args = ap.parse_args()

    root = project_root()
    cfg = load_yaml(root / "src" / "config" / "base.yaml")
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] \
        or list(get_nested(cfg, ["universe", "tickers_by_strategy", "e2_moderate"], default=[]))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    run(tickers, seeds, out)


if __name__ == "__main__":
    main()
