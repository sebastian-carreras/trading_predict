"""Runner unificado para entrenar todas las variantes de E1.

Ejecuta, en este orden:
1) e1_baseline      -> src.train_e1_baseline
2) e1_simple        -> src.train_e1_simple_pipeline
3) e1_conservador   -> src.train_e1_pipeline

Uso:
    python -m src.train_e1_all
    python -m src.train_e1_all --tickers AAPL,MSFT
    python -m src.train_e1_all --config src/config/base.yaml --tickers YPFD.BA,GGAL.BA
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from .utils import load_yaml, project_root
except ImportError:  # pragma: no cover
    from src.utils import load_yaml, project_root


def _parse_tickers(raw: str) -> list[str]:
    return [ticker.strip() for ticker in raw.split(",") if ticker.strip()]


def _resolve_default_tickers(config: dict) -> list[str]:
    by_strategy = config.get("universe", {}).get("tickers_by_strategy", {})
    conservative = by_strategy.get("e1_conservative", [])
    simple = by_strategy.get("e1_simple", [])

    merged = []
    for ticker in list(conservative) + list(simple):
        if ticker and ticker not in merged:
            merged.append(ticker)

    if merged:
        return merged

    universe = config.get("universe", {}).get("tickers", [])
    return [ticker for ticker in universe if ticker]


def _run_command(command: list[str]) -> int:
    print(f"\n$ {' '.join(command)}")
    result = subprocess.run(command, check=False)
    return int(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena E1 en baseline, simple y conservador"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Tickers separados por coma (ej: AAPL,MSFT). Si se omite, usa todos los tickers E1 del config.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/base.yaml",
        help="Ruta al archivo de configuración YAML",
    )
    args = parser.parse_args()

    root = project_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"No existe config: {config_path}")

    config = load_yaml(config_path)

    tickers = _parse_tickers(args.tickers) if args.tickers else _resolve_default_tickers(config)
    if not tickers:
        raise ValueError("No hay tickers para E1 en argumentos ni en config")

    tickers_csv = ",".join(tickers)

    print("=" * 72)
    print("RUNNER E1 - baseline + simple + conservador")
    print("=" * 72)
    print(f"Config : {config_path}")
    print(f"Tickers ({len(tickers)}): {tickers_csv}")

    steps = [
        ("e1_baseline", "src.train_e1_baseline"),
        ("e1_simple", "src.train_e1_simple_pipeline"),
        ("e1_conservador", "src.train_e1_pipeline"),
    ]

    exit_codes: dict[str, int] = {}
    for name, module in steps:
        print("\n" + "-" * 72)
        print(f"Ejecutando {name}...")
        command = [
            sys.executable,
            "-m",
            module,
            "--config",
            str(config_path),
            "--tickers",
            tickers_csv,
        ]
        exit_codes[name] = _run_command(command)

    print("\n" + "=" * 72)
    print("RESUMEN")
    print("=" * 72)
    has_errors = False
    for name, code in exit_codes.items():
        status = "OK" if code == 0 else f"ERROR ({code})"
        if code != 0:
            has_errors = True
        print(f"- {name}: {status}")

    if has_errors:
        raise SystemExit(1)

    print("\n" + "=" * 72)
    print("COMPARACIÓN AUTOMÁTICA")
    print("=" * 72)
    compare_command = [
        sys.executable,
        str(root / "scripts" / "evaluation" / "compare_e1_versions.py"),
        "--tickers",
        tickers_csv,
        "--per-ticker",
    ]
    compare_exit_code = _run_command(compare_command)
    if compare_exit_code != 0:
        print("⚠️  No se pudo ejecutar la comparación automática de versiones E1.")
        print("   Puedes correrla manualmente con:")
        print("   python scripts/evaluation/compare_e1_versions.py")


if __name__ == "__main__":
    main()
