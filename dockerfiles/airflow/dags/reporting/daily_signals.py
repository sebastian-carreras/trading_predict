"""
DAG de SEÑALES DIARIAS — qué invertir hoy. NO entrena modelos.

Responde una sola pregunta: **¿qué predice hoy el champion que ya tenemos?**
Toma el champion congelado del registry y lo corre sobre los datos más recientes
(inferencia, ~0.13 s por ticker). No entrena, no promueve, no toca el lifecycle.

Por qué es un DAG aparte y por cron
-----------------------------------
El track de señales no debe depender de que el entrenamiento salga bien. Un
retrain fallido no puede dejarte sin saber qué dicen tus champions hoy.

Por eso NO usa Dataset ni ExternalTaskSensor:
  - Si E2 falla, ningún Dataset se emite y un DAG data-aware nunca arrancaría —
    que es justo el caso que hay que cubrir.
  - E1 se dispara por Dataset, así que su logical date no alinea con un cron y un
    ExternalTaskSensor quedaría colgado esperando una fecha que no existe.

El cron a las 11:00 UTC corre 3h después del ancla de E2 (08:00 UTC): da tiempo a
que la cadena de entrenamiento termine y evita solaparse con sus escrituras en
data/raw y data/clean. Si el entrenamiento se cayó, este DAG corre igual.

Flujo:
    refresh_champion_data  (OHLCV + exógenas de los 40 champions, sin entrenar)
        → live_signals     (leaderboard con inferencia en vivo + AS_OF por fila)

Sobre el registry: solo lo LEE, así que no compite con las escrituras del
lifecycle aunque llegara a solaparse.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow"

default_args = {
    "owner": "trading_predict",
    "depends_on_past": False,
    "start_date": datetime(2026, 7, 30),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "daily_signals",
    default_args=default_args,
    description="Señales diarias: inferencia en vivo del champion sobre datos frescos (sin entrenar)",
    # Por reloj, no por Dataset: tiene que correr aunque el retrain haya fallado.
    schedule_interval="0 11 * * *",
    catchup=False,
    tags=["trading", "signals", "leaderboard", "daily"],
)

# 1) Datos frescos para TODOS los champions del registry — no solo los que el
#    lifecycle entrena. Incluye el refresco del cache exógeno (macro/cross-asset),
#    que ningún otro DAG hacía.
refresh_champion_data = BashOperator(
    task_id="refresh_champion_data",
    bash_command=(
        f"cd {PROJECT_DIR} && "
        "python -m scripts.data.refresh_champion_data --strategies e1,e2"
    ),
    dag=dag,
)

# 2) Leaderboard: PRED%/SIG salen de correr el champion congelado sobre los datos
#    recién refrescados. AS_OF deja explícito el día de datos de cada predicción;
#    las que superan reporting.max_signal_age_days se muestran pero sin señal.
live_signals = BashOperator(
    task_id="live_signals",
    bash_command=(
        f"cd {PROJECT_DIR} && "
        "python -m scripts.evaluation.leaderboard --strategies e1,e2"
    ),
    dag=dag,
)

refresh_champion_data >> live_signals
