#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${MLFLOW_VERSION:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    MLFLOW_VERSION="$(python -c "import mlflow; print(mlflow.__version__)" 2>/dev/null || true)"
  fi
fi

if [[ -z "${MLFLOW_VERSION:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    MLFLOW_VERSION="$(python3 -c "import mlflow; print(mlflow.__version__)" 2>/dev/null || true)"
  fi
fi

if [[ -z "${MLFLOW_VERSION:-}" ]]; then
  echo "No se pudo detectar mlflow local. Definí MLFLOW_VERSION manualmente."
  echo "Ejemplo: MLFLOW_VERSION=3.8.1 bash scripts/mlflow/up_transparent_mlflow.sh"
  exit 1
fi

export MLFLOW_VERSION
export MLFLOW_BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI:-sqlite:////mlflow_data/mlflow_local/mlflow.db}"
export MLFLOW_DEFAULT_ARTIFACT_ROOT="${MLFLOW_DEFAULT_ARTIFACT_ROOT:-file:///mlflow_data/mlflow_local/artifacts}"

echo "[MLflow Transparent] Root: ${ROOT_DIR}"
echo "[MLflow Transparent] MLFLOW_VERSION=${MLFLOW_VERSION}"
echo "[MLflow Transparent] BACKEND=${MLFLOW_BACKEND_STORE_URI}"
echo "[MLflow Transparent] ARTIFACT_ROOT=${MLFLOW_DEFAULT_ARTIFACT_ROOT}"

docker compose build mlflow
docker compose up -d postgres mlflow

echo
echo "MLflow levantado en http://localhost:5050"
echo "Runs offline + online comparten el mismo store."
