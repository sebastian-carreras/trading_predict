"""
DAG de reporting DIARIO — salud del CICLO DE VIDA. NO re-entrena modelos.

Responde: **¿los champions están mejorando o empeorando con los días?** Es la
contraparte del DAG `daily_signals`, que responde "¿qué invierto hoy?".

    daily_report   → calidad de los modelos     (data-aware: tras el retrain)
    daily_signals  → decisión de inversión      (por cron: corre igual si falla)

El leaderboard vivía acá y se movió a `daily_signals`: es una salida del track de
señales, necesita datos frescos y debe producirse aunque el retrain se caiga.
Dejarlo en los dos DAGs generaba dos snapshots por día en leaderboard_history.jsonl.

Flujo:
    refresh_dashboard  (checker: summary + ticker views → CSVs)
        → refresh_history_reports  (regenera los history_report_*.csv desde MLflow)
        → stability                (evolución del champion: Sharpe/IC + promociones)

Todas las tareas reutilizan módulos ya existentes. No descarga datos ni entrena.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.datasets import Dataset
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow"

# Mismos URIs que producen los pipelines E1 y E2 al terminar. Con esto el DAG es
# "data-aware": no corre por reloj, sino cuando AMBOS datasets se actualizan
# (E1 y E2 terminaron su retrain del día). Así el reporte siempre ve champions frescos.
DATASET_E1 = Dataset("trading://registry/e1_conservative")
DATASET_E2 = Dataset("trading://registry/e2_moderate")

default_args = {
    "owner": "trading_predict",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 6),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

dag = DAG(
    "daily_report",
    default_args=default_args,
    description="Reporte diario: estabilidad y salud de los champions (sin re-entrenar)",
    schedule=[DATASET_E1, DATASET_E2],  # Corre cuando E1 y E2 terminaron su retrain (no por reloj)
    catchup=False,
    tags=["trading", "reporting", "lifecycle", "daily"],
)

# 1) Refrescar los CSV del dashboard (summary global + per-ticker de E1 y E2).
#    Estos alimentan tanto la inspección manual como stability_analysis.
refresh_dashboard = BashOperator(
    task_id="refresh_dashboard",
    bash_command=(
        f"cd {PROJECT_DIR} && "
        "python -m src.dashboard.checker --view summary --quiet && "
        "python -m src.dashboard.checker --view ticker --strategy e1_conservative --quiet && "
        "python -m src.dashboard.checker --view ticker --strategy e2_moderate --quiet"
    ),
    dag=dag,
)

# 2) Regenerar los history_report_*.csv de TODOS los tickers desde la MLflow DB.
#    stability_analysis solo grafica los CSV que existan; sin este paso plotearía
#    un subconjunto viejo/parcial.
refresh_history = BashOperator(
    task_id="refresh_history_reports",
    bash_command=(
        f"cd {PROJECT_DIR} && "
        "python3 scripts/evaluation/_batch_history_reports.py"
    ),
    dag=dag,
)

# 3) Estabilidad: evolución del champion (Sharpe/IC en el tiempo) + timeline de
#    promociones. Responde "¿los champions mejoran con los días?" a largo plazo.
stability = BashOperator(
    task_id="stability_analysis",
    bash_command=(
        f"cd {PROJECT_DIR} && "
        "python -m scripts.evaluation.stability_analysis"
    ),
    dag=dag,
)

refresh_dashboard >> refresh_history >> stability
