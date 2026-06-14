#!/usr/bin/env python3
"""Re-apunta el artifact_location de los experimentos MLflow a S3/MinIO.

Contexto: experimentos creados desde la consola del host (o por el fallback local
con el server caído) quedaron con artifact_location apuntando a rutas del host
(file:///Users/...). Esas rutas no existen dentro de los contenedores, así que
Airflow falla con PermissionError al hacer log_artifact. Al pasar el server a
--default-artifact-root s3://<bucket>/, los experimentos NUEVOS quedan en S3, pero
los VIEJOS conservan su ruta vieja (el artifact_location es inmutable vía la API
de MLflow). Este script normaliza la columna directamente en el backend SQLite.

Uso (con el server mlflow DETENIDO, para evitar lock de SQLite):

    # Dry-run: muestra qué cambiaría, no toca nada
    python scripts/mlflow_repoint_to_s3.py

    # Aplica los cambios
    python scripts/mlflow_repoint_to_s3.py --apply

Opciones:
    --db PATH        Ruta al SQLite de MLflow (default: runs/mlflow_local/mlflow.db)
    --bucket NAME    Bucket S3 destino (default: $MLFLOW_BUCKET_NAME o 'mlflow')
    --apply          Confirma y escribe los cambios (sin esto: dry-run)
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-apuntar artifact_location de MLflow a S3")
    parser.add_argument("--db", default="runs/mlflow_local/mlflow.db", help="Ruta al SQLite de MLflow")
    parser.add_argument("--bucket", default=os.getenv("MLFLOW_BUCKET_NAME", "mlflow"), help="Bucket S3 destino")
    parser.add_argument("--apply", action="store_true", help="Aplicar cambios (sin esto: dry-run)")
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"❌ No existe el SQLite: {db_path}")
        return 1

    s3_prefix = f"s3://{args.bucket}/"

    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute("SELECT experiment_id, name, artifact_location FROM experiments ORDER BY experiment_id")
        rows = cur.fetchall()

        to_fix = [(eid, name, loc) for (eid, name, loc) in rows if not (loc or "").startswith("s3://")]

        print(f"DB: {db_path}")
        print(f"Destino: {s3_prefix}<experiment_id>\n")
        print(f"Experimentos totales: {len(rows)} | a re-apuntar (no-s3): {len(to_fix)}\n")
        for eid, name, loc in rows:
            mark = "→ FIX" if not (loc or "").startswith("s3://") else "   ok"
            new = f"  ==>  {s3_prefix}{eid}" if mark == "→ FIX" else ""
            print(f"  [{mark}] id={eid:>3} {name[:34]:34s} {loc}{new}")

        if not to_fix:
            print("\n✓ Nada que re-apuntar; todos los experimentos ya usan s3://")
            return 0

        if not args.apply:
            print(f"\n[DRY-RUN] Se re-apuntarían {len(to_fix)} experimentos. Repetí con --apply para escribir.")
            return 0

        for eid, _name, _loc in to_fix:
            cur.execute(
                "UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?",
                (f"{s3_prefix}{eid}", eid),
            )
        con.commit()
        print(f"\n✓ Re-apuntados {len(to_fix)} experimentos a {s3_prefix}<id>")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
