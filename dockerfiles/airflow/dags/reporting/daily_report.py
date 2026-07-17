"""
DAG de reporting DIARIO — NO re-entrena modelos.

Bajo la cadencia "split" (E1/E2 se re-entrenan semanalmente), este DAG corre
todas las mañanas para responder dos preguntas sin tocar los modelos:

  1. ¿Cuáles son los mejores tickers para invertir hoy?  → leaderboard
  2. ¿Los champions están mejorando o empeorando con los días?
        → leaderboard (Δ score vs. ayer) + stability_analysis (tendencia larga)

Flujo:
    refresh_dashboard  (checker: summary + ticker views → CSVs)
        → leaderboard  (ranking de champions por score compuesto + Δ vs. ayer)
        → stability    (evolución del champion: Sharpe/IC + timeline de promociones)

Todas las tareas reutilizan módulos ya existentes del repo; la única pieza
nueva es scripts.evaluation.leaderboard. No descarga datos ni entrena.
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
    description="Reporte diario: leaderboard de tickers + estabilidad de champions (sin re-entrenar)",
    schedule=[DATASET_E1, DATASET_E2],  # Corre cuando E1 y E2 terminaron su retrain (no por reloj)
    catchup=False,
    tags=["trading", "reporting", "leaderboard", "daily"],
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

# 2) Leaderboard: "mejores tickers para invertir hoy" + Δ score/rank vs. ayer.
#    Escribe reports/dashboard/leaderboard_<fecha>.csv, _latest.csv y history.jsonl.
leaderboard = BashOperator(
    task_id="leaderboard",
    bash_command=(
        f"cd {PROJECT_DIR} && "
        "python -m scripts.evaluation.leaderboard --strategies e1,e2"
    ),
    dag=dag,
)

# 3) Regenerar los history_report_*.csv de TODOS los tickers desde la MLflow DB.
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

# 4) Estabilidad: evolución del champion (Sharpe/IC en el tiempo) + timeline de
#    promociones. Responde "¿los champions mejoran con los días?" a largo plazo.
stability = BashOperator(
    task_id="stability_analysis",
    bash_command=(
        f"cd {PROJECT_DIR} && "
        "python -m scripts.evaluation.stability_analysis"
    ),
    dag=dag,
)

refresh_dashboard >> leaderboard >> refresh_history >> stability
