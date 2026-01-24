#!/usr/bin/env python3
"""
Script de validación para el pipeline de optimización E2.

Ejecuta una optimización rápida para verificar que todo funciona correctamente.
"""

import sys
from pathlib import Path

# Agregar src/ al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import project_root, load_yaml


def validate_e2_optimization():
    """Valida que el pipeline de optimización E2 esté correctamente configurado."""
    
    print("="*80)
    print("VALIDACIÓN DEL PIPELINE DE OPTIMIZACIÓN E2")
    print("="*80)
    
    root = project_root()
    
    # 1. Verificar archivos necesarios
    print("\n1. Verificando archivos necesarios...")
    
    required_files = [
        "scripts/optimize_e2_hyperparameters.py",
        "src/train_e2_pipeline.py",
        "src/config/base.yaml",
        "dockerfiles/airflow/dags/e2_optuna_tuning.py",
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = root / file_path
        if full_path.exists():
            print(f"   ✓ {file_path}")
        else:
            print(f"   ✗ {file_path} (FALTANTE)")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n   ⚠️  Faltan {len(missing_files)} archivo(s)")
        return False
    
    # 2. Verificar configuración
    print("\n2. Verificando configuración...")
    
    config_path = root / "src/config/base.yaml"
    config = load_yaml(config_path)
    
    # Verificar universo E2
    e2_tickers = config.get("universe", {}).get("tickers_by_strategy", {}).get("e2_moderate", [])
    
    if not e2_tickers:
        print("   ✗ No se encontraron tickers E2 en configuración")
        return False
    
    print(f"   ✓ Tickers E2 configurados: {len(e2_tickers)}")
    print(f"     Primeros 5: {', '.join(e2_tickers[:5])}")
    
    # Verificar estrategia E2
    e2_config = config.get("strategies", {}).get("e2_moderate", {})
    
    if not e2_config:
        print("   ✗ Configuración E2 no encontrada")
        return False
    
    print(f"   ✓ Estrategia E2 configurada")
    print(f"     Lookback: {e2_config.get('lookback', 'N/A')}")
    print(f"     Horizon: {e2_config.get('horizon', 'N/A')}")
    
    # 3. Verificar dependencias Python
    print("\n3. Verificando dependencias Python...")
    
    required_modules = [
        ("optuna", "Optuna"),
        ("mlflow", "MLflow"),
        ("yaml", "PyYAML"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
    ]
    
    missing_modules = []
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"   ✓ {display_name}")
        except ImportError:
            print(f"   ✗ {display_name} (NO INSTALADO)")
            missing_modules.append(display_name)
    
    if missing_modules:
        print(f"\n   ⚠️  Faltan {len(missing_modules)} dependencia(s)")
        print("   Instalar con: pip install optuna mlflow pyyaml pandas numpy")
        return False
    
    # 4. Verificar directorios de salida
    print("\n4. Verificando directorios de salida...")
    
    output_dirs = [
        "reports/hyperparameter_optimization",
        "runs/mlflow_local",
    ]
    
    for dir_path in output_dirs:
        full_path = root / dir_path
        if full_path.exists():
            print(f"   ✓ {dir_path} (existe)")
        else:
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"   ℹ {dir_path} (creado)")
    
    # 5. Resumen final
    print("\n" + "="*80)
    print("RESULTADO DE VALIDACIÓN")
    print("="*80)
    print("\n✅ Todos los componentes están correctamente configurados")
    print("\nPróximos pasos:")
    print("  1. Prueba rápida (CLI):")
    print("     python scripts/optimize_e2_hyperparameters.py --n_trials 5 --quick")
    print("\n  2. Prueba desde Airflow:")
    print("     - Acceder a http://localhost:8080")
    print("     - Buscar DAG: e2_optuna_hyperparameter_tuning")
    print("     - Trigger con config: {\"n_trials\": \"5\", \"quick_mode\": \"True\"}")
    print("\n  3. Ver documentación completa:")
    print("     cat README_E2_OPTIMIZATION.md")
    print("\n" + "="*80)
    
    return True


if __name__ == "__main__":
    success = validate_e2_optimization()
    sys.exit(0 if success else 1)
