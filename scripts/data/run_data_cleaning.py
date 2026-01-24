#!/usr/bin/env python3
"""
Script para ejecutar limpieza de datos y ver reporte de calidad.

Uso:
    python scripts/run_data_cleaning.py              # Forward fill (recomendado)
    python scripts/run_data_cleaning.py --interpolate  # Interpolación lineal
    python scripts/run_data_cleaning.py --drop         # Eliminar filas con nulos
"""

import sys
from pathlib import Path

# Agregar src al path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from src.data.clean_daily import process_daily_data_with_cleaning

def main():
    raw_dir = root / "data" / "raw" / "daily"
    clean_dir = root / "data" / "clean"

    # Determinar estrategia
    if "--interpolate" in sys.argv:
        strategy = "interpolate"
    elif "--drop" in sys.argv:
        strategy = "drop"
    else:
        strategy = "forward_fill"

    print(f"\n{'='*80}")
    print(f"LIMPIEZA DE DATOS - Trading Predict")
    print(f"{'='*80}")
    print(f"Directorio raw:   {raw_dir}")
    print(f"Directorio clean: {clean_dir}")
    print(f"Estrategia:       {strategy.upper()}")
    print(f"{'='*80}\n")

    if not raw_dir.exists():
        print(f"❌ Error: Directorio raw no existe: {raw_dir}")
        print(f"\nEjecuta primero: python -m src.data.download_daily")
        return 1

    reports = process_daily_data_with_cleaning(
        raw_dir=raw_dir,
        clean_dir=clean_dir,
        strategy=strategy,
        min_days=252,
        remove_zero_volume=True,  # Eliminar días con volumen=0
        verbose=True,
    )

    # Guardar reporte JSON
    import json
    report_path = clean_dir / "data_quality_report.json"
    with open(report_path, "w") as f:
        json.dump(reports, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print(f"📄 Reporte de calidad guardado en: {report_path}")
    print(f"{'='*80}\n")

    # Mostrar tickers problemáticos
    tickers_with_many_nulls = [
        (ticker, sum(r["null_counts"].values()))
        for ticker, r in reports.items()
        if r.get("null_counts")
    ]
    
    if tickers_with_many_nulls:
        tickers_with_many_nulls.sort(key=lambda x: x[1], reverse=True)
        print("\n⚠️  TICKERS CON MÁS VALORES NULOS:")
        for ticker, total_nulls in tickers_with_many_nulls[:10]:
            report = reports[ticker]
            print(f"\n  {ticker}: {total_nulls} nulos totales")
            for feat, count in report["null_counts"].items():
                pct = report["null_percentages"][feat]
                print(f"    - {feat}: {count} ({pct}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
