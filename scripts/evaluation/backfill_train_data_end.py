"""Backfill de ``train_data_end`` en champions viejos (pre-tracking).

Champions promovidos antes de que ``train_data_end`` se empezara a trackear
(ver ``src/lifecycle/registry.py::register_candidate``) no lo tienen guardado.
Sin ese campo, ``promotion.compare_on_common_window`` no puede construir una
ventana out-of-sample común y cae a la comparación legacy stored-vs-stored
(métricas de épocas distintas, no comparables entre sí).

Este script backfillea el valor correcto usando la MISMA definición que ya
usa el leaderboard (``leaderboard.champion_data_cutoff``): la fecha embebida
en ``run_dir`` (proxy verificado — coincide con ``train_data_end`` en los
champions que trackean ambos campos). **No** usa la última predicción del
walk-forward: esa es la fecha de la última señal, que cae ``horizonte`` días
ANTES del corte real, no el corte en sí.

Efecto al aplicar: en el PRÓXIMO retrain, los champions backfilleados pasan
de la comparación "stored" a la ruta "fair_window" en la promoción — más
rigurosa, misma metodología que ya usa `trading_predict` para los champions
nuevos. No cambia nada retroactivamente ni al correr este script.

**Dry-run por defecto** — no escribe nada hasta pasar ``--apply``. Al aplicar,
hace un backup de ``models/registry.json`` primero (el registry está
gitignored, sin red de git).

Lee:
    models/registry.json   (champions por estrategia)

Escribe (solo con --apply):
    models/registry.json                       (train_data_end backfilleado)
    models/registry.backup.<timestamp>.json    (backup previo a escribir)

Uso:
    python -m scripts.evaluation.backfill_train_data_end              # dry-run
    python -m scripts.evaluation.backfill_train_data_end --strategies e1,e2
    python -m scripts.evaluation.backfill_train_data_end --apply      # aplica
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.evaluation.leaderboard import champion_data_cutoff
from src.lifecycle.reevaluation import load_walkforward_predictions
from src.lifecycle.registry import ModelRegistry
from src.utils import project_root

ROOT = project_root()
REGISTRY_PATH = ROOT / "models" / "registry.json"

DEFAULT_STRATEGIES = ["e1", "e2", "e3"]


def _wf_last(run_dir: str, ticker: str) -> str | None:
    """Fecha de la última señal del walk-forward — solo para eyeballear el dry-run."""
    wf = load_walkforward_predictions(run_dir, ticker, root=ROOT)
    if wf is None or wf.empty:
        return None
    import pandas as pd
    return str(pd.Timestamp(wf.index.max()).date())


def find_backfills(registry: ModelRegistry, strategies: list[str]) -> list[dict[str, Any]]:
    """Champions sin train_data_end, con el valor propuesto a backfillear."""
    plans: list[dict[str, Any]] = []
    for strat in strategies:
        tickers = registry.data.get("strategies", {}).get(strat, {}).get("tickers", {})
        for ticker, entry in tickers.items():
            champ = entry.get("champion")
            if not champ or champ.get("train_data_end"):
                continue  # ya tiene el campo, o no hay champion
            proposed = champion_data_cutoff(champ)
            if proposed is None:
                continue  # sin run_dir parseable: no hay de dónde sacar el valor
            plans.append({
                "strategy": strat,
                "ticker": ticker,
                "run_dir": champ.get("run_dir", ""),
                "proposed": proposed,
                "wf_last": _wf_last(champ.get("run_dir", ""), ticker),
            })
    return plans


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill de train_data_end en champions viejos (dry-run por defecto)."
    )
    parser.add_argument(
        "--strategies",
        default=",".join(DEFAULT_STRATEGIES),
        help="Estrategias a revisar, separadas por coma (default: e1,e2,e3).",
    )
    parser.add_argument("--apply", action="store_true", help="Aplicar los cambios (por defecto: dry-run).")
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    registry = ModelRegistry(REGISTRY_PATH)

    plans = find_backfills(registry, strategies)

    mode = "APLICANDO" if args.apply else "DRY-RUN (no escribe nada)"
    print("=" * 92)
    print(f"  BACKFILL train_data_end — {mode}")
    print("=" * 92)
    if not plans:
        print("Todos los champions ya tienen train_data_end. Nada que backfillear.")
        return 0

    print(f"{'STRAT':<5} {'TICKER':<10} {'proposed (train_data_end)':>26} {'wf_last (señal, ref.)':>22}")
    print("-" * 92)
    for p in plans:
        wf = p["wf_last"] or "—"
        print(f"{p['strategy']:<5} {p['ticker']:<10} {p['proposed']:>26} {wf:>22}")
    print("-" * 92)
    print(f"Total a backfillear: {len(plans)}")

    if not args.apply:
        print("\nDry-run: no se escribió nada. Repetí con --apply para aplicar.")
        return 0

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = REGISTRY_PATH.parent / f"registry.backup.{ts}.json"
    shutil.copy2(REGISTRY_PATH, backup_path)
    print(f"\nBackup creado: {backup_path.relative_to(ROOT)}")

    updated = 0
    for p in plans:
        if registry.set_train_data_end(p["strategy"], p["ticker"], p["proposed"]):
            updated += 1
    print(f"✅ Backfilleados {updated}/{len(plans)} champions. Registry actualizado (escritura atómica).")
    print("   En el próximo retrain, estos champions pasan a la comparación fair_window en la promoción.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
