#!/usr/bin/env python3
"""Refresca los datos de mercado de TODOS los tickers que son champion, sin entrenar.

Pertenece al **track de señales**, no al de lifecycle. El leaderboard corre el
champion congelado sobre los datos más recientes; para que esa señal sea de hoy,
los datos tienen que ser de hoy. Este script se ocupa solo de eso.

Por qué el registry y no base.yaml
----------------------------------
El universo sale de ``registry.list_all(stage="champion")``: un champion recién
promovido entra solo, y un ticker que dejó de tener champion deja de descargarse.
Los DAGs de entrenamiento eligen a quién entrenarle un rival — una decisión de
costo distinta, que no debería gobernar qué tickers tienen datos frescos.

Delega en ``src.data.ingest.refresh_data_for_training``, que ya encadena refresco
del cache exógeno + descarga incremental + limpieza.

Uso:
    python -m scripts.data.refresh_champion_data
    python -m scripts.data.refresh_champion_data --strategies e1,e2
    python -m scripts.data.refresh_champion_data --dry-run   # solo lista tickers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.ingest import refresh_data_for_training  # noqa: E402
from src.lifecycle.registry import ModelRegistry  # noqa: E402
from src.utils import load_yaml  # noqa: E402

DEFAULT_STRATEGIES = ["e1", "e2"]


def champion_tickers(registry: ModelRegistry, strategies: list[str]) -> list[str]:
    """Tickers con champion en alguna de las estrategias pedidas, sin duplicados.

    El orden es estable (estrategia, luego orden del registry) para que la salida
    del script sea diffeable entre corridas.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for strategy in strategies:
        for champ in registry.list_all(strategy=strategy, stage="champion"):
            ticker = champ.get("ticker")
            if ticker and ticker not in seen:
                seen.add(ticker)
                ordered.append(ticker)
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresca OHLCV + exógenas de todos los champions (sin entrenar)."
    )
    parser.add_argument(
        "--strategies",
        default=",".join(DEFAULT_STRATEGIES),
        help="Estrategias cuyos champions refrescar, separadas por coma (default: e1,e2).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo lista los tickers que se refrescarían; no descarga nada.",
    )
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    config = load_yaml(ROOT / "src" / "config" / "base.yaml")
    registry = ModelRegistry(ROOT / "models" / "registry.json")

    tickers = champion_tickers(registry, strategies)
    if not tickers:
        print(f"No hay champions para {strategies}: nada que refrescar.")
        return 0

    print(f"Champions a refrescar ({len(tickers)}) en {strategies}:")
    print(f"  {', '.join(tickers)}")

    if args.dry_run:
        print("\n--dry-run: no se descargó nada.")
        return 0

    refresh_data_for_training(config, tickers, granularity="daily", root=ROOT)
    print(f"\n✓ Refresco completo para {len(tickers)} champions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
