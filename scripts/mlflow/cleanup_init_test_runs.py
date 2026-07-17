"""
Limpia runs '_init_test' obsoletos de MLflow.

Uso:
    python scripts/mlflow/cleanup_init_test_runs.py
    python scripts/mlflow/cleanup_init_test_runs.py --tracking-uri http://localhost:5050
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Agregar src al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import project_root


def cleanup_init_test_runs(tracking_uri: str | None = None) -> None:
    """
    Elimina runs con nombre '_init_test' del backend de MLflow.
    
    Args:
        tracking_uri: URI del tracking store. Si None, usa MLFLOW_TRACKING_URI
                     o default local sqlite.
    """
    try:
        import mlflow
    except ImportError:
        print("❌ MLflow no está instalado")
        return

    load_dotenv()  # Cargar .env (MLFLOW_TRACKING_URI, etc.)

    root = project_root()

    # Determinar tracking URI
    if tracking_uri is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    
    if not tracking_uri:
        # Usar SQLite local por defecto
        local_db = root / "runs" / "mlflow_local" / "mlflow.db"
        tracking_uri = f"sqlite:///{local_db}"
    
    print(f"🔍 Conectando a MLflow: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Buscar runs en todos los experimentos
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    
    total_deleted = 0
    
    for exp in experiments:
        exp_id = exp.experiment_id
        exp_name = exp.name
        
        # Buscar runs con nombre _init_test
        runs = client.search_runs(
            experiment_ids=[exp_id],
            filter_string="",
            max_results=10000,
        )
        
        init_test_runs = [r for r in runs if r.info.run_name == "_init_test"]
        
        if init_test_runs:
            print(f"\n📂 Experimento: {exp_name} (ID: {exp_id})")
            print(f"   Encontrados {len(init_test_runs)} runs '_init_test'")
            
            for run in init_test_runs:
                try:
                    client.delete_run(run.info.run_id)
                    total_deleted += 1
                    print(f"   ✓ Eliminado run {run.info.run_id[:8]}...")
                except Exception as e:
                    print(f"   ✗ Error eliminando {run.info.run_id[:8]}: {e}")
    
    if total_deleted == 0:
        print("\n✓ No se encontraron runs '_init_test' para eliminar")
    else:
        print(f"\n✓ Total eliminados: {total_deleted} runs '_init_test'")


def main():
    parser = argparse.ArgumentParser(
        description="Limpia runs '_init_test' obsoletos de MLflow"
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="URI del tracking store (default: MLFLOW_TRACKING_URI o local SQLite)",
    )
    
    args = parser.parse_args()
    cleanup_init_test_runs(tracking_uri=args.tracking_uri)


if __name__ == "__main__":
    main()
