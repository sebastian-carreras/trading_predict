#!/usr/bin/env python3
"""
Script para ejecutar limpieza de datos y ver reporte de calidad.

Este script es un wrapper CLI que encapsula la lógica de limpieza de datos
ubicada en src/data/clean_daily.py. Proporciona una interfaz simple para
usuarios finales y genera reportes de calidad en JSON.

Estrategias de limpieza disponibles:
  - forward_fill (DEFAULT): Propaga último valor válido hacia adelante.
    Recomendado para series financieras con gaps pequeños.
  
  - interpolate: Interpola linealmente entre valores válidos.
    Útil para llenar gaps medianos de forma suave.
  
  - drop: Elimina filas completas que contienen nulos.
    Más conservador pero puede perder muchos datos.

Uso:
    python scripts/run_data_cleaning.py              # Forward fill (recomendado)
    python scripts/run_data_cleaning.py --interpolate  # Interpolación lineal
    python scripts/run_data_cleaning.py --drop         # Eliminar filas con nulos

Salida:
    - CSVs limpios: data/clean/{ticker}_daily.csv
    - Reporte JSON: data/clean/data_quality_report.json
    - Consola: diagnóstico y warnings detallados
"""

import sys
from pathlib import Path

# Agregar src al path para importar módulos del proyecto
# (permite que el script encuentre src.data.clean_daily)
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

# Importar la función principal de limpieza desde el módulo reutilizable
from src.data.clean_daily import process_daily_data_with_cleaning

def main():
    """
    Función principal que ejecuta el pipeline de limpieza de datos.
    
    Pasos:
    1. Determina la estrategia de limpieza desde argumentos CLI
    2. Valida que exista el directorio raw con datos descargados
    3. Procesa todos los CSVs del directorio raw
    4. Aplica la estrategia seleccionada para cada ticker
    5. Guarda CSVs limpios en directorio clean/
    6. Genera reporte JSON con estadísticas de calidad
    7. Imprime warnings sobre tickers problemáticos
    
    Retorna:
        0 si exitoso, 1 si error (ej: directorio raw no existe)
    """
    # Directorios de entrada y salida
    raw_dir = root / "data" / "raw" / "daily"      # Datos descargados sin procesar
    clean_dir = root / "data" / "clean"            # Datos limpios y validados

    # Determinar estrategia desde argumentos de línea de comandos
    # Por defecto: forward_fill (recomendado para datos financieros)
    if "--interpolate" in sys.argv:
        strategy = "interpolate"
    elif "--drop" in sys.argv:
        strategy = "drop"
    else:
        strategy = "forward_fill"

    # Imprimir encabezado con configuración
    print(f"\n{'='*80}")
    print(f"LIMPIEZA DE DATOS - Trading Predict")
    print(f"{'='*80}")
    print(f"Directorio raw:   {raw_dir}")
    print(f"Directorio clean: {clean_dir}")
    print(f"Estrategia:       {strategy.upper()}")
    print(f"{'='*80}\n")

    # Validar que exista el directorio raw con datos descargados
    if not raw_dir.exists():
        print(f"❌ Error: Directorio raw no existe: {raw_dir}")
        print(f"\nEjecuta primero: python -m src.data.download_daily")
        return 1

    # Ejecutar limpieza para todos los CSVs en directorio raw
    # Retorna diccionario con reportes de calidad por ticker
    reports = process_daily_data_with_cleaning(
        raw_dir=raw_dir,
        clean_dir=clean_dir,
        strategy=strategy,
        min_days=252,                  # Mínimo 1 año de datos (días hábiles)
        remove_zero_volume=True,       # Eliminar días con volumen=0 (sin trading real)
        verbose=True,                  # Mostrar detalles de cada ticker
    )

    # Guardar reporte de calidad en JSON para auditoría y análisis posterior
    import json
    report_path = clean_dir / "data_quality_report.json"
    with open(report_path, "w") as f:
        json.dump(reports, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print(f"📄 Reporte de calidad guardado en: {report_path}")
    print(f"{'='*80}\n")

    # Analizar y mostrar tickers con más problemas (muchos nulos)
    # Útil para identificar fuentes de datos problemáticas
    tickers_with_many_nulls = [
        (ticker, sum(r["null_counts"].values()))
        for ticker, r in reports.items()
        if r.get("null_counts")
    ]
    
    # Si hay tickers con nulos, mostrar los 10 peores
    if tickers_with_many_nulls:
        tickers_with_many_nulls.sort(key=lambda x: x[1], reverse=True)
        print("\n⚠️  TICKERS CON MÁS VALORES NULOS:")
        for ticker, total_nulls in tickers_with_many_nulls[:10]:
            report = reports[ticker]
            print(f"\n  {ticker}: {total_nulls} nulos totales")
            # Desglose por feature (columna)
            for feat, count in report["null_counts"].items():
                pct = report["null_percentages"][feat]
                print(f"    - {feat}: {count} ({pct}%)")

    return 0


if __name__ == "__main__":
    """
    Entry point del script. Se ejecuta cuando se llama directamente:
        python scripts/run_data_cleaning.py
    
    Llama a main() y pasa su código de retorno al sistema operativo.
    """
    sys.exit(main())
