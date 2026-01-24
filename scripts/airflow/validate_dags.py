#!/usr/bin/env python3
"""
Script de verificación de DAGs de Airflow.

Valida sintaxis y estructura de los DAGs sin necesidad de levantar Airflow.

Uso:
    python scripts/validate_dags.py
    python scripts/validate_dags.py --dag e1_baseline_linear_regression
"""

import argparse
import sys
from pathlib import Path


def validate_dag(dag_path: Path) -> tuple[bool, str]:
    """
    Valida un DAG de Airflow.
    
    Returns:
        (is_valid, message)
    """
    if not dag_path.exists():
        return False, f"Archivo no existe: {dag_path}"
    
    try:
        # Leer el archivo
        with open(dag_path, 'r') as f:
            code = f.read()
        
        # Validar sintaxis Python
        compile(code, str(dag_path), 'exec')
        
        # Validar que tenga imports de Airflow
        required_imports = ['from airflow', 'import dag', 'DAG']
        has_airflow = any(imp.lower() in code.lower() for imp in required_imports)
        
        if not has_airflow:
            return False, "No se encontraron imports de Airflow"
        
        # Validar que defina un DAG
        if 'dag = DAG' not in code and 'with DAG' not in code:
            return False, "No se encontró definición de DAG"
        
        return True, "✓ DAG válido"
        
    except SyntaxError as e:
        return False, f"Error de sintaxis: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Validar DAGs de Airflow")
    parser.add_argument(
        '--dag',
        type=str,
        default=None,
        help='Nombre del DAG específico a validar (ej: e1_baseline_linear_regression)'
    )
    
    args = parser.parse_args()
    
    # Detectar project root
    script_path = Path(__file__).resolve()
    if script_path.parent.name == 'scripts':
        root = script_path.parent.parent
    else:
        root = Path.cwd()
    
    dags_dir = root / 'dockerfiles' / 'airflow' / 'dags'
    
    if not dags_dir.exists():
        print(f"❌ Directorio de DAGs no existe: {dags_dir}")
        sys.exit(1)
    
    # Encontrar DAGs a validar
    if args.dag:
        # DAG específico
        dag_file = dags_dir / f"{args.dag}.py"
        if not dag_file.exists():
            print(f"❌ DAG no encontrado: {dag_file}")
            sys.exit(1)
        dag_files = [dag_file]
    else:
        # Todos los DAGs
        dag_files = list(dags_dir.glob('*.py'))
        if '__pycache__' in str(dags_dir):
            dag_files = [f for f in dag_files if '__pycache__' not in str(f)]
    
    if not dag_files:
        print("⚠️  No se encontraron DAGs para validar")
        sys.exit(0)
    
    print(f"\n{'='*70}")
    print(f"Validando {len(dag_files)} DAG(s)")
    print(f"{'='*70}\n")
    
    results = []
    for dag_path in sorted(dag_files):
        is_valid, message = validate_dag(dag_path)
        results.append((dag_path.stem, is_valid, message))
        
        status = "✓" if is_valid else "✗"
        print(f"{status} {dag_path.stem:40} {message}")
    
    print(f"\n{'='*70}")
    
    # Resumen
    valid_count = sum(1 for _, is_valid, _ in results if is_valid)
    total_count = len(results)
    
    print(f"\nResumen: {valid_count}/{total_count} DAGs válidos")
    
    if valid_count < total_count:
        print("\n⚠️  Algunos DAGs tienen errores. Revisar antes de deployar.")
        sys.exit(1)
    else:
        print("\n✅ Todos los DAGs son válidos")
        sys.exit(0)


if __name__ == '__main__':
    main()
