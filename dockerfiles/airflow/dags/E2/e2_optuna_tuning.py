"""
DAG para ejecutar la optimización de hiperparámetros E2 con Optuna desde la UI de Airflow.

Permite ajustar tickers, número de trials, overrides del espacio de búsqueda y parámetros
clave de MLflow/Optuna sin modificar código. Específico para estrategia E2 (LSTM).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from airflow import DAG
from airflow.operators.python import PythonOperator

DAG_ID = "e2_optuna_hyperparameter_tuning"
SCRIPT_REL_PATH = Path("scripts/optimization/optimize_e2_hyperparameters.py")


def resolve_project_root() -> Path:
    """Busca la raíz del proyecto donde está el script de optimización."""
    candidates = []

    env_root = os.environ.get("TRADING_PREDICT_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    dag_dir = Path(__file__).resolve().parent
    candidates.append(dag_dir)
    candidates.append(dag_dir.parent)
    candidates.append(dag_dir / "trading_predict")
    candidates.append(dag_dir.parent / "trading_predict")

    candidates.append(Path("/opt/airflow"))
    candidates.append(Path("/opt/airflow/trading_predict"))

    # Evitar duplicados preservando orden
    seen: set[Path] = set()
    unique_candidates = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(resolved)

    for candidate in unique_candidates:
        script_path = candidate / SCRIPT_REL_PATH
        if script_path.exists():
            return candidate

    searched = ", ".join(str(c) for c in unique_candidates)
    raise FileNotFoundError(
        f"No se encontró {SCRIPT_REL_PATH} en ninguno de los candidatos: {searched}. "
        "Configura la variable de entorno TRADING_PREDICT_ROOT o monta el repo en el scheduler."
    )


DEFAULT_PARAMS: Dict[str, Optional[str]] = {
    # Configuración general
    "config_path": "src/config/base.yaml",
    "n_trials": "20",
    "study_name": "e2_hyperparameter_optimization_timesplit",
    "per_ticker": "False",

    # Tickers (universo E2 moderado)
    "tickers": "NVDA, AMD, MSFT, AAPL, TSLA, GOOGL, META, NFLX, AMZN, BABA",
    "single_ticker": "",
    "quick_mode": "False",

    # Timeouts y outputs
    "timeout_seconds": "",
    "mlflow_uri": "local",
    "output_dir": "reports/hyperparameter_optimization",

    # --- Thresholds de Trading ---
    "tau_buy_min": "0.015",
    "tau_buy_max": "0.05",
    "tau_sell_min": "-0.01",
    "tau_sell_max": "0.01",

    # --- Arquitectura LSTM ---
    "lstm_units_1_min": "64",
    "lstm_units_1_max": "256",
    "lstm_units_2_min": "32",
    "lstm_units_2_max": "128",

    # --- Regularización ---
    "dropout_min": "0.1",
    "dropout_max": "0.4",

    # --- Entrenamiento ---
    "learning_rate_min": "5e-5",
    "learning_rate_max": "5e-3",
    "batch_sizes": "32,64,128",

    # Argumentos CLI extra
    "extra_cli_args": "",
}


def _as_bool(value: Optional[str]) -> bool:
    """Convierte string a booleano."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _is_not_blank(value: Optional[str]) -> bool:
    """Verifica si un valor no está vacío."""
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    return bool(str(value).strip())


def run_optuna_tuning(**context) -> str:
    """
    Ejecuta el script de optimización de hiperparámetros E2.

    Lee parámetros desde Airflow params y construye el comando CLI.
    """
    params = context["params"]
    log = context["ti"].log

    project_root = resolve_project_root()
    script_path = project_root / SCRIPT_REL_PATH
    if not script_path.exists():
        raise FileNotFoundError(
            f"No se encontró el script en {script_path}. "
            "Ajusta TRADING_PREDICT_ROOT o monta el repo completo."
        )

    log.info("Usando PROJECT_ROOT=%s", project_root)
    log.info("Script Optuna E2=%s", script_path)

    # Comando base
    base_cmd: List[str] = [sys.executable, str(script_path)]
    base_cmd.extend(["--config", str(params.get("config_path") or DEFAULT_PARAMS["config_path"])])
    base_cmd.extend(["--n_trials", str(params.get("n_trials") or DEFAULT_PARAMS["n_trials"])])
    base_cmd.extend(["--study_name", str(params.get("study_name") or DEFAULT_PARAMS["study_name"])])
    base_cmd.extend(["--output_dir", str(params.get("output_dir") or DEFAULT_PARAMS["output_dir"])])
    base_cmd.extend(["--mlflow_uri", str(params.get("mlflow_uri") or DEFAULT_PARAMS["mlflow_uri"])])

    # Modo per_ticker
    per_ticker = _as_bool(params.get("per_ticker"))
    if per_ticker:
        base_cmd.append("--per_ticker")

    # Timeout
    timeout_value = params.get("timeout_seconds")
    if _is_not_blank(timeout_value):
        base_cmd.extend(["--timeout", str(timeout_value)])

    # Batch sizes
    batch_sizes = params.get("batch_sizes")
    if _is_not_blank(batch_sizes):
        base_cmd.extend(["--batch_sizes", str(batch_sizes)])

    # Selección de tickers
    tickers_list = str(params.get("tickers") or "").strip()
    single_ticker = str(params.get("single_ticker") or "").strip()
    quick_mode = _as_bool(params.get("quick_mode"))

    # En modo per_ticker, el script espera una lista de tickers
    if per_ticker and (not tickers_list) and single_ticker:
        tickers_list = single_ticker
        single_ticker = ""

    if tickers_list:
        base_cmd.extend(["--tickers", tickers_list])
        if quick_mode:
            log.warning("Ignorando quick_mode porque se proporcionaron tickers personalizados")
    elif single_ticker:
        if per_ticker:
            base_cmd.extend(["--tickers", single_ticker])
        else:
            base_cmd.extend(["--ticker", single_ticker])
    elif quick_mode:
        base_cmd.append("--quick")

    # Overrides del espacio de búsqueda
    override_args = (
        # Trading thresholds
        "tau_buy_min",
        "tau_buy_max",
        "tau_sell_min",
        "tau_sell_max",
        # Arquitectura LSTM
        "lstm_units_1_min",
        "lstm_units_1_max",
        "lstm_units_2_min",
        "lstm_units_2_max",
        # Regularización
        "dropout_min",
        "dropout_max",
        # Entrenamiento
        "learning_rate_min",
        "learning_rate_max",
    )

    for override in override_args:
        value = params.get(override)
        if _is_not_blank(value):
            base_cmd.extend([f"--{override}", str(value)])

    # Argumentos CLI extra
    extra_cli = str(params.get("extra_cli_args") or "").strip()
    if extra_cli:
        base_cmd.extend(shlex.split(extra_cli))

    # Configurar entorno
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{project_root}:{current_pythonpath}" if current_pythonpath else str(project_root)
    )

    log.info("Ejecutando comando Optuna E2: %s", shlex.join(base_cmd))

    # Ejecutar script
    try:
        result = subprocess.run(
            base_cmd,
            cwd=str(project_root),
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        log.info("Salida del script:\n%s", result.stdout)
        if result.stderr:
            log.warning("Stderr del script:\n%s", result.stderr)
    except subprocess.CalledProcessError as e:
        log.error("El script falló con código de salida %d", e.returncode)
        if e.stdout:
            log.error("STDOUT:\n%s", e.stdout)
        if e.stderr:
            log.error("STDERR:\n%s", e.stderr)
        raise

    # Guardar comando ejecutado en XCom
    rendered_command = shlex.join(base_cmd)
    context["ti"].xcom_push(key="cli_command", value=rendered_command)
    return rendered_command


def build_dag() -> DAG:
    """Construye el DAG de Airflow."""
    dag = DAG(
        dag_id=DAG_ID,
        description="Ejecuta Optuna para E2 (LSTM) con parámetros configurables",
        default_args={"owner": "ml_team"},
        schedule_interval=None,  # Manual trigger
        start_date=datetime(2024, 1, 1),
        catchup=False,
        params=DEFAULT_PARAMS,
        tags=["optuna", "mlflow", "e2", "lstm", "hyperparameter_tuning"],
    )

    PythonOperator(
        task_id="run_optuna_tuning",
        python_callable=run_optuna_tuning,
        dag=dag,
    )

    return dag


dag = build_dag()
