"""Piso de ruido de la métrica primaria de promoción (``bt_sharpe_excess``).

Para qué
--------
El gate de mejora de la v2 (``lifecycle.promotion.v2.min_improvement_abs``) tiene que ser
mayor que la variación que produce **el mismo modelo entrenado con otra semilla**. Sin ese
número el umbral es arbitrario: se promueven candidatos cuya "mejora" es ruido de
inicialización de la red.

Precedente: la ablación de exógenas midió un std entre semillas de ~0.09 para el score viejo,
**mayor que el efecto que intentaba medir** (~0.03-0.05), lo que invalidó sus conclusiones
per-ticker. Ver ``reports/ablation/ablation_report.md``.

Qué hace
--------
Entrena cada ticker con N semillas manteniendo TODO lo demás fijo, y reporta la dispersión
de ``bt_sharpe_excess`` (y del score viejo, para comparar). Corre con
``register_lifecycle=False``: no toca ``models/registry.json``. Modelos efímeros en
``runs/noise_floor/``.

Uso:
    PYTHONPATH=. python -m scripts.evaluation.noise_floor                       # E2, 6 tickers, 5 semillas
    PYTHONPATH=. python -m scripts.evaluation.noise_floor --strategy e1
    PYTHONPATH=. python -m scripts.evaluation.noise_floor --tickers NVDA,GGAL.BA --seeds 42,7,123
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import numpy as np

from src.lifecycle.promotion import compute_score, get_strategy_promotion_config
from src.utils import ensure_dir, get_nested, load_yaml, project_root

# Subconjunto por defecto: mezcla US y .BA, que tienen regímenes de deriva muy distintos
# (deriva nominal anualizada mediana 0.15 vs 0.49) y no tienen por qué compartir piso de ruido.
_DEFAULT_TICKERS = ["NVDA", "AAPL", "GS", "GGAL.BA", "TGSU2.BA", "LOMA.BA"]
_DEFAULT_SEEDS = [42, 7, 123, 2024, 31]
_FIELDS = [
    "strategy", "ticker", "n_seeds",
    "sharpe_excess_mean", "sharpe_excess_std", "sharpe_excess_min", "sharpe_excess_max",
    "score_v1_mean", "score_v1_std",
    "bt_sharpe_mean", "ml_ic_mean", "ml_dir_acc_edge_mean",
]


def _runner(strategy: str):
    """Importa el pipeline perezosamente (arrastra torch)."""
    if strategy == "e1":
        from src.e1.train_pipeline import run_e1_for_ticker
        return run_e1_for_ticker
    if strategy == "e2":
        from src.e2.train_pipeline import run_e2_for_ticker
        return run_e2_for_ticker
    raise SystemExit(f"Estrategia no soportada: {strategy} (usar e1 o e2)")


def _run_ticker(
    cfg: dict, strategy: str, ticker: str, seeds: list[int],
    raw_dir: Path, root: Path, weights: dict,
) -> tuple[list, dict[str, list[float]]]:
    run_for_ticker = _runner(strategy)
    collected: dict[str, list[float]] = {
        "bt_sharpe_excess": [], "score_v1": [], "bt_sharpe": [],
        "ml_ic": [], "ml_dir_acc_edge": [],
    }

    for seed in seeds:
        c = copy.deepcopy(cfg)
        c.setdefault("project", {})["seed"] = int(seed)
        out_dir = root / "runs" / "noise_floor" / strategy / f"seed_{seed}"
        try:
            summary = run_for_ticker(
                c, ticker, raw_dir, out_dir,
                register_lifecycle=False, auto_promote=False,
            )
        except Exception as exc:
            print(f"      seed={seed} ERR: {str(exc)[:90]}", flush=True)
            continue

        score_v1, _ = compute_score(summary, weights)
        collected["score_v1"].append(score_v1)
        for key in ("bt_sharpe_excess", "bt_sharpe", "ml_ic", "ml_dir_acc_edge"):
            val = summary.get(key)
            if isinstance(val, (int, float)) and np.isfinite(val):
                collected[key].append(float(val))
        print(f"      seed={seed}: sharpe_excess={summary.get('bt_sharpe_excess')} "
              f"score_v1={score_v1:.4f}", flush=True)

    ex = collected["bt_sharpe_excess"]
    if not ex:
        return [strategy, ticker, 0] + ["ERR"] * (len(_FIELDS) - 3), collected

    def _m(key: str) -> float:
        vals = collected[key]
        return round(float(np.mean(vals)), 6) if vals else float("nan")

    row = [
        strategy, ticker, len(ex),
        round(float(np.mean(ex)), 6), round(float(np.std(ex, ddof=1)) if len(ex) > 1 else 0.0, 6),
        round(float(np.min(ex)), 6), round(float(np.max(ex)), 6),
        _m("score_v1"),
        round(float(np.std(collected["score_v1"], ddof=1)) if len(collected["score_v1"]) > 1 else 0.0, 6),
        _m("bt_sharpe"), _m("ml_ic"), _m("ml_dir_acc_edge"),
    ]
    return row, collected


def run(strategy: str, tickers: list[str], seeds: list[int], out_path: Path) -> None:
    root = project_root()
    cfg = load_yaml(root / "src" / "config" / "base.yaml")
    promo = get_nested(cfg, ["lifecycle", "promotion"], default={}) or {}
    weights = get_strategy_promotion_config(promo, strategy).get("scoring_weights", {})
    raw_dir = root / "data" / "raw" / "daily"

    ensure_dir(out_path.parent)
    rows: list[list] = []

    print(f"Piso de ruido {strategy.upper()} | {len(tickers)} tickers × {len(seeds)} semillas "
          f"{seeds}\nMétrica primaria: bt_sharpe_excess\n", flush=True)

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}", flush=True)
        row, _ = _run_ticker(cfg, strategy, ticker, seeds, raw_dir, root, weights)
        rows.append(row)
        std = row[4] if isinstance(row[4], float) else None
        if std is not None:
            print(f"   → std entre semillas = {std:.4f}\n", flush=True)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(_FIELDS)
            w.writerows(rows)

    _summary(rows, out_path)


def _summary(rows: list[list], out_path: Path) -> None:
    stds = [r[4] for r in rows if isinstance(r[4], float)]
    print("=" * 72)
    if not stds:
        print("Sin resultados válidos.")
        return

    floor = float(np.median(stds))
    print(f"PISO DE RUIDO (mediana del std entre semillas de bt_sharpe_excess): {floor:.4f}")
    print(f"  rango por ticker: {min(stds):.4f} .. {max(stds):.4f}")
    print()
    print("Umbral sugerido para lifecycle.promotion.v2.min_improvement_abs:")
    print(f"  conservador (2× piso): {2 * floor:.3f}")
    print(f"  moderado    (1× piso): {floor:.3f}")
    print()
    print("Una mejora menor a ese valor es indistinguible de reentrenar con otra semilla.")
    print(f"\nCSV: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Piso de ruido de bt_sharpe_excess entre semillas")
    ap.add_argument("--strategy", type=str, default="e2", choices=["e1", "e2"])
    ap.add_argument("--tickers", type=str, default="", help="coma-separados (default: 6 mixtos US/.BA)")
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in _DEFAULT_SEEDS))
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] or _DEFAULT_TICKERS
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    root = project_root()
    out = Path(args.out or f"reports/promotion/noise_floor_{args.strategy}.csv")
    if not out.is_absolute():
        out = root / out
    run(args.strategy, tickers, seeds, out)


if __name__ == "__main__":
    main()
