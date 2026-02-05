"""
DAG para ejecutar la optimización de hiperparámetros E1 con Optuna desde la UI de Airflow.

Permite ajustar tickers, número de trials, overrides del espacio de búsqueda y parámetros
clave de MLflow/Optuna sin modificar código.
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

DAG_ID = "e1_optuna_hyperparameter_tuning"
SCRIPT_REL_PATH = Path("scripts/optimize_e1_hyperparameters.py")


def resolve_project_root() -> Path:
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
    "config_path": "src/config/base.yaml",
    "n_trials": "20",
    "study_name": "e1_hyperparameter_optimization",
    "per_ticker": "False",
    "tickers": "YPFD.BA, GGAL.BA, PAMP.BA, BYMA.BA, CEPU.BA, AAPL, MSFT, JNJ, PG, V",
    "single_ticker": "",
    "quick_mode": "False",
    "timeout_seconds": "",
    "mlflow_uri": "local",
    "output_dir": "reports/hyperparameter_optimization",
    "tau_buy_min": "0.02",
    "tau_buy_max": "0.10",
    "tau_sell_min": "-0.02",
    "tau_sell_max": "0.02",
    "dropout_min": "0.2",
    "dropout_max": "0.5",
    "learning_rate_min": "1e-4",
    "learning_rate_max": "1e-2",
    "gru_units_1_min": "32",
    "gru_units_1_max": "128",
    "gru_units_2_min": "16",
    "gru_units_2_max": "64",
    "batch_sizes": "32,64,128",
    "extra_cli_args": "",
}


def _as_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _is_not_blank(value: Optional[str]) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    return bool(str(value).strip())


def run_optuna_tuning(**context) -> str:
    params = context["params"]
    log = context["ti"].log

    project_root = resolve_project_root()
    script_path = project_root / SCRIPT_REL_PATH
    if not script_path.exists():
        raise FileNotFoundError(
            f"No se encontró el script en {script_path}. Ajusta TRADING_PREDICT_ROOT o monta el repo completo."
        )

    log.info("Usando PROJECT_ROOT=%s", project_root)
    log.info("Script Optuna=%s", script_path)

    base_cmd: List[str] = [sys.executable, str(script_path)]
    base_cmd.extend(["--config", str(params.get("config_path") or DEFAULT_PARAMS["config_path"])])
    base_cmd.extend(["--n_trials", str(params.get("n_trials") or DEFAULT_PARAMS["n_trials"])])
    base_cmd.extend(["--study_name", str(params.get("study_name") or DEFAULT_PARAMS["study_name"])])
    base_cmd.extend(["--output_dir", str(params.get("output_dir") or DEFAULT_PARAMS["output_dir"])])
    base_cmd.extend(["--mlflow_uri", str(params.get("mlflow_uri") or DEFAULT_PARAMS["mlflow_uri"])])

    per_ticker = _as_bool(params.get("per_ticker"))
    if per_ticker:
        base_cmd.append("--per_ticker")

    timeout_value = params.get("timeout_seconds")
    if _is_not_blank(timeout_value):
        base_cmd.extend(["--timeout", str(timeout_value)])

    batch_sizes = params.get("batch_sizes")
    if _is_not_blank(batch_sizes):
        base_cmd.extend(["--batch_sizes", str(batch_sizes)])

    tickers_list = str(params.get("tickers") or "").strip()
    single_ticker = str(params.get("single_ticker") or "").strip()
    quick_mode = _as_bool(params.get("quick_mode"))

    # En modo per_ticker, el script espera una lista de tickers; para evitar confusiones
    # desde la UI, si sólo se setea single_ticker lo tratamos como lista.
    if per_ticker and (not tickers_list) and single_ticker:
        tickers_list = single_ticker
        single_ticker = ""

    if tickers_list:
        base_cmd.extend(["--tickers", tickers_list])
        if quick_mode:
            log.warning("Ignorando quick_mode porque se proporcionaron tickers personalizados")
    elif single_ticker:
        # En modo per_ticker preferimos siempre pasar lista (aunque sea de 1)
        if per_ticker:
            base_cmd.extend(["--tickers", single_ticker])
        else:
            base_cmd.extend(["--ticker", single_ticker])
    elif quick_mode:
        base_cmd.append("--quick")

    override_args = (
        "tau_buy_min",
        "tau_buy_max",
        "tau_sell_min",
        "tau_sell_max",
        "dropout_min",
        "dropout_max",
        "learning_rate_min",
        "learning_rate_max",
        "gru_units_1_min",
        "gru_units_1_max",
        "gru_units_2_min",
        "gru_units_2_max",
    )

    for override in override_args:
        value = params.get(override)
        if _is_not_blank(value):
            base_cmd.extend([f"--{override}", str(value)])

    extra_cli = str(params.get("extra_cli_args") or "").strip()
    if extra_cli:
        base_cmd.extend(shlex.split(extra_cli))

    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{project_root}:{current_pythonpath}" if current_pythonpath else str(project_root)
    )

    log.info("Ejecutando comando Optuna: %s", shlex.join(base_cmd))
    
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

    rendered_command = shlex.join(base_cmd)
    context["ti"].xcom_push(key="cli_command", value=rendered_command)
    return rendered_command


def build_dag() -> DAG:
    dag = DAG(
        dag_id=DAG_ID,
        description="Ejecuta Optuna para E1 con parámetros configurables",
        default_args={"owner": "ml_team"},
        schedule_interval=None,
        start_date=datetime(2024, 1, 1),
        catchup=False,
        params=DEFAULT_PARAMS,
        tags=["optuna", "mlflow", "e1"],
    )

    PythonOperator(
        task_id="run_optuna_tuning",
        python_callable=run_optuna_tuning,
        dag=dag,
    )

    return dag


dag = build_dag()
