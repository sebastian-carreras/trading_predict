"""Rollback de promociones forzadas: restaura champions retirados mejores.

Deshace la regresión del 2026-07-15, donde la regla ``force = not
is_champion_servable`` de los DAGs promovió modelos PEORES sobre los champions
buenos (que quedaron en ``retired[]``). Para cada ``(strategy, ticker)`` compara
el **score compuesto** (el mismo de la promoción) del champion actual vs.
``retired[0]``; si el retirado es mejor, lo restaura como champion.

**Dry-run por defecto** — no escribe nada hasta pasar ``--execute``.

Uso:
    python -m scripts.evaluation.rollback_forced_promotions              # dry-run
    python -m scripts.evaluation.rollback_forced_promotions --strategies e1,e2
    python -m scripts.evaluation.rollback_forced_promotions --execute    # aplica
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.lifecycle.promotion import compute_score, get_strategy_promotion_config
from src.lifecycle.registry import ModelRegistry
from src.utils import load_yaml, project_root

ROOT = project_root()
REGISTRY_PATH = ROOT / "models" / "registry.json"
CONFIG_PATH = ROOT / "src" / "config" / "base.yaml"

EXCLUDE_TICKERS = {"AAA"}


def _score(metrics: dict[str, Any], weights: dict[str, float]) -> float:
    s, _ = compute_score(metrics or {}, weights)
    return float(s)


def _artifacts_ok(run_dir: str, ticker: str) -> bool:
    rd = Path(run_dir)
    if not rd.is_absolute():
        rd = ROOT / rd
    if not rd.exists():
        return False
    return bool(list(rd.glob(f"{ticker}*_model.pth"))) and bool(
        list(rd.glob(f"{ticker}*_walkforward_predictions.csv"))
    )


def find_rollbacks(registry: ModelRegistry, config: dict, strategies: list[str]) -> list[dict[str, Any]]:
    """(strategy, ticker) donde retired[0] tiene mejor score que el champion actual."""
    promo_cfg = config.get("lifecycle", {}).get("promotion", {})
    plans: list[dict[str, Any]] = []
    for strat in strategies:
        weights = get_strategy_promotion_config(promo_cfg, strat).get("scoring_weights", {})
        tickers = registry.data.get("strategies", {}).get(strat, {}).get("tickers", {})
        for ticker, entry in tickers.items():
            if ticker in EXCLUDE_TICKERS:
                continue
            champ = entry.get("champion")
            retired = entry.get("retired", [])
            if not champ or not retired:
                continue
            r0 = retired[0]
            champ_score = _score(champ.get("metrics", {}), weights)
            ret_score = _score(r0.get("metrics", {}), weights)
            if ret_score > champ_score:
                plans.append({
                    "strategy": strat,
                    "ticker": ticker,
                    "champ_score": champ_score,
                    "ret_score": ret_score,
                    "champ_sharpe": (champ.get("metrics", {}) or {}).get("bt_sharpe"),
                    "ret_sharpe": (r0.get("metrics", {}) or {}).get("bt_sharpe"),
                    "artifacts_ok": _artifacts_ok(r0.get("run_dir", ""), ticker),
                })
    return plans


def main() -> int:
    parser = argparse.ArgumentParser(description="Rollback de promociones forzadas (dry-run por defecto).")
    parser.add_argument("--strategies", default="e1,e2,e3", help="Estrategias, separadas por coma.")
    parser.add_argument("--execute", action="store_true", help="Aplicar los cambios (por defecto: dry-run).")
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    config = load_yaml(CONFIG_PATH)
    registry = ModelRegistry(REGISTRY_PATH)

    plans = find_rollbacks(registry, config, strategies)

    mode = "EJECUCIÓN" if args.execute else "DRY-RUN (no escribe nada)"
    print("=" * 84)
    print(f"  ROLLBACK de promociones forzadas — {mode}")
    print("=" * 84)
    if not plans:
        print("No hay champions retirados mejores que el actual. Nada que revertir.")
        return 0

    print(f"{'STRAT':<5} {'TICKER':<10} {'champ→ret Sharpe':>20} {'champ→ret score':>22} {'artefactos':>11}")
    print("-" * 84)
    for p in plans:
        cs = p["champ_sharpe"]; rs = p["ret_sharpe"]
        sharpe = f"{cs:.2f} → {rs:.2f}" if cs is not None and rs is not None else "—"
        score = f"{p['champ_score']:.4f} → {p['ret_score']:.4f}"
        art = "ok" if p["artifacts_ok"] else "FALTAN ⚠"
        print(f"{p['strategy']:<5} {p['ticker']:<10} {sharpe:>20} {score:>22} {art:>11}")
    print("-" * 84)
    n_missing = sum(1 for p in plans if not p["artifacts_ok"])
    print(f"Total a revertir: {len(plans)}"
          + (f"  ({n_missing} con artefactos faltantes ⚠)" if n_missing else ""))

    if not args.execute:
        print("\nDry-run: no se escribió nada. Repetí con --execute para aplicar.")
        return 0

    restored = 0
    for p in plans:
        r = registry.restore_champion_from_retired(
            p["strategy"], p["ticker"], reason="rollback_20260715_forced_regression",
        )
        if r is not None:
            restored += 1
    print(f"\n✅ Restaurados {restored}/{len(plans)} champions. Registry actualizado (escritura atómica).")
    print("   Regenerá la demo si corresponde:  python -m demo.bundle_assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
