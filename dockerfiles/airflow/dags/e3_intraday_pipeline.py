"""
DAG de Airflow para Estrategia E3 - Intradía (LSTM Ensemble)

Flujo:
1. Descargar datos 5-min (últimos 60 días)
2. Entrenar ensemble LSTM con MLflow tracking
3. Ejecutar backtest con costos 20 bps
4. Actualizar modelos en FastAPI
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import mlflow
import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

default_args = {
    'owner': 'trading_predict',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 6),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=3),
}

dag = DAG(
    'e3_intraday_pipeline',
    default_args=default_args,
    description='Pipeline E3: Trading intradía (30 min, LSTM Ensemble)',
    schedule_interval='0 1 * * *',  # Diario 1 AM (re-entrenar con datos frescos)
    catchup=False,
    tags=['trading', 'e3', 'intraday', 'lstm', 'ensemble'],
)


def download_intraday_data(**context):
    """Task 1: Descargar datos 5-min."""
    import sys
    sys.path.insert(0, '/opt/airflow')
    
    from src.data.intraday_yfinance import download_ohlcv_5m
    from src.utils import load_yaml
    from pathlib import Path
    
    root = Path("/opt/airflow")
    config = load_yaml(root / "src/config/base.yaml")
    
    # Tickers E3
    tickers = list(
        config.get("universe", {})
        .get("tickers_by_strategy", {})
        .get("e3_intraday", [])
    )
    
    out_dir = root / "data/raw/intraday"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    e3_cfg = config.get("strategies", {}).get("e3_intraday", {})
    period = e3_cfg.get("data", {}).get("period", "60d")
    
    # Download all tickers at once
    try:
        written_paths = download_ohlcv_5m(
            tickers=tickers,
            out_dir=out_dir,
            period=period,
            interval="5m",
        )
        downloaded = [p.stem.replace("_5m", "") for p in written_paths]
        print(f"✓ Downloaded {len(downloaded)} tickers: {downloaded}")
    except Exception as e:
        print(f"⚠️ Error descargando tickers: {e}")
        downloaded = []
    
    context['task_instance'].xcom_push(key='tickers_downloaded', value=downloaded)
    return f"Descargados {len(downloaded)} tickers 5-min"


def train_e3_with_mlflow(**context):
    """Task 2: Entrenar ensemble E3 con MLflow."""
    import sys
    sys.path.insert(0, '/opt/airflow')
    
    from src.train_e3_pipeline import run_for_ticker
    from src.utils import load_yaml
    from pathlib import Path
    
    mlflow.set_experiment("E3_Intraday_Strategy")
    
    root = Path("/opt/airflow")
    config = load_yaml(root / "src/config/base.yaml")
    
    tickers = context['task_instance'].xcom_pull(task_ids='download_intraday', key='tickers_downloaded')
    if not tickers:
        return "No tickers to train"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = root / "runs/e3_intraday" / timestamp
    out_base.mkdir(parents=True, exist_ok=True)
    
    # Set MLflow environment variables for the training pipeline
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "E3_Intraday_Strategy"
    
    results = []
    for ticker in tickers:
        ticker_out = out_base / ticker
        print(f"\n{'='*60}")
        print(f"Training E3 ensemble for {ticker}")
        print(f"{'='*60}")
        
        try:
            summary = run_for_ticker(
                config=config,
                ticker=ticker,
                raw_dir=root / "data/raw/intraday",
                out_dir=ticker_out,
            )
            
            print(f"\n✓ {ticker} complete:")
            print(f"  MAE: {summary.get('ml_mae', 0):.6f}")
            print(f"  RMSE: {summary.get('ml_rmse', 0):.6f}")
            print(f"  IC: {summary.get('ml_ic', 0):.4f}")
            print(f"  Dir Acc: {summary.get('ml_directional_accuracy', 0):.3f}")
            print(f"  Profit Factor: {summary.get('tr_profit_factor', 0):.3f}")
            print(f"  Max DD: {summary.get('tr_max_drawdown', 0):.3%}")
            
            results.append({
                "ticker": ticker,
                "status": "success",
                "profit_factor": summary.get("tr_profit_factor"),
                "ic": summary.get("ml_ic"),
            })
            
        except Exception as e:
            print(f"✗ Error en {ticker}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"ticker": ticker, "status": "failed", "error": str(e)})
    
    # Guardar y loggear resumen agregado
    summary_df = pd.DataFrame(results)
    summary_path = out_base / "summary_all.csv"
    summary_df.to_csv(summary_path, index=False)
    
    # Targets desde config
    decision_cfg = config.get("decision", {})
    default_targets = {
        'ic_min': 0.05,
    }
    targets = {**default_targets, **decision_cfg.get("targets", {})}
    
    ic_values = [r.get("ic") for r in results if r.get("ic") is not None]
    
    with mlflow.start_run(run_name=f"E3_Intraday_Summary_{timestamp}"):
        mlflow.log_artifact(str(summary_path))
        mlflow.log_metric("total_tickers", len(results))
        mlflow.log_metric("successful_tickers", sum(1 for r in results if r["status"] == "success"))
        if ic_values:
            mlflow.log_metric("ic_mean", float(sum(ic_values) / len(ic_values)))
            mlflow.log_metric("ic_median", float(sorted(ic_values)[len(ic_values)//2]))
            mlflow.log_metric("ic_above_threshold", sum(1 for ic in ic_values if ic > targets['ic_min']))
            mlflow.log_param("ic_target_min", targets['ic_min'])
    
    context['task_instance'].xcom_push(key='run_dir', value=str(out_base))
    successful = sum(1 for r in results if r["status"] == "success")
    return f"Entrenados {successful}/{len(results)} modelos E3"


# Definir tareas
task_download = PythonOperator(
    task_id='download_intraday',
    python_callable=download_intraday_data,
    dag=dag,
)

task_train = PythonOperator(
    task_id='train_e3_ensemble',
    python_callable=train_e3_with_mlflow,
    dag=dag,
)

# Flujo
task_download >> task_train
