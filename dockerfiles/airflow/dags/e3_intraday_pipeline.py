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
    
    from src.data.intraday_yfinance import download_yfinance_intraday
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
    
    downloaded = []
    for ticker in tickers:
        try:
            df = download_yfinance_intraday(
                ticker=ticker,
                period=period,
                interval="5m",
            )
            out_path = out_dir / f"{ticker}_5min.csv"
            df.to_csv(out_path, index=True)
            downloaded.append(ticker)
        except Exception as e:
            print(f"⚠️ Error descargando {ticker}: {e}")
    
    context['task_instance'].xcom_push(key='tickers_downloaded', value=downloaded)
    return f"Descargados {len(downloaded)} tickers 5-min"


def train_e3_with_mlflow(**context):
    """Task 2: Entrenar ensemble E3 con MLflow."""
    import sys
    sys.path.insert(0, '/opt/airflow')
    
    from src.e3_intraday_pipeline import run_intraday_for_ticker
    from src.utils import load_yaml
    from pathlib import Path
    
    mlflow.set_experiment("E3_Intraday_Strategy")
    
    root = Path("/opt/airflow")
    config = load_yaml(root / "src/config/base.yaml")
    
    tickers = context['task_instance'].xcom_pull(task_ids='download_intraday', key='tickers_downloaded')
    if not tickers:
        return "No tickers to train"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "runs/e3_intraday" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for ticker in tickers:
        with mlflow.start_run(run_name=f"E3_{ticker}_{timestamp}"):
            # Extraer configuración del modelo
            e3_config = config["strategies"]["e3_intraday"]
            model_config = e3_config.get("model", {})
            bt_config = e3_config.get("backtest", {})
            thresholds = e3_config.get("thresholds", {})
            
            # Log parámetros de estrategia
            mlflow.log_params({
                "strategy": "e3_intraday",
                "ticker": ticker,
                "frequency": "5min",
                "horizon_bars": e3_config.get("horizon_bars", 6),
                "lookback_bars": e3_config.get("lookback_bars", 96),
                "model_type": "LSTM_Ensemble",
            })
            
            # Log hiperparámetros del modelo
            mlflow.log_params({
                "ensemble_members": model_config.get("ensemble_members", 3),
                "lstm_hidden_size": model_config.get("lstm_hidden_size", 64),
                "lstm_num_layers": model_config.get("lstm_num_layers", 2),
                "dropout": model_config.get("dropout", 0.2),
                "learning_rate": model_config.get("learning_rate", 1e-3),
                "batch_size": model_config.get("batch_size", 256),
                "max_epochs": model_config.get("max_epochs", 30),
                "early_stopping_patience": model_config.get("early_stopping_patience", 5),
                "loss": model_config.get("loss", "huber"),
                "huber_delta": model_config.get("huber_delta", 1.0),
            })
            
            # Log parámetros de backtest
            mlflow.log_params({
                "tau_buy": thresholds.get("tau_buy", 0.001),
                "tau_sell": thresholds.get("tau_sell", 0.001),
                "round_trip_bps": config.get("costs", {}).get("intraday_round_trip_bps", 20),
                "execution_delay_bars": bt_config.get("execution_delay_bars", 1),
                "allow_short": bt_config.get("allow_short", True),
                "max_position": bt_config.get("max_position", 1.0),
            })
            
            try:
                result = run_intraday_for_ticker(
                    config=config,
                    ticker=ticker,
                    root=root,
                    out_dir=out_dir,
                )
                
                # Log métricas ML
                mlflow.log_metrics({
                    "mae": result.get("ml_mae", 0),
                    "rmse": result.get("ml_rmse", 0),
                    "directional_accuracy": result.get("ml_directional_accuracy", 0),
                })
                
                # Log métricas de backtest
                if "backtest" in result:
                    bt = result["backtest"]
                    mlflow.log_metrics({
                        "profit_factor": bt.get("profit_factor", 0),
                        "sharpe_intraday": bt.get("sharpe", 0),
                        "max_drawdown": bt.get("max_drawdown", 0),
                        "total_trades": bt.get("total_trades", 0),
                    })
                
                results.append({
                    "ticker": ticker,
                    "status": "success",
                    "profit_factor": result.get("backtest", {}).get("profit_factor"),
                })
                
            except Exception as e:
                mlflow.log_param("error", str(e))
                results.append({"ticker": ticker, "status": "failed", "error": str(e)})
    
    context['task_instance'].xcom_push(key='run_dir', value=str(out_dir))
    return f"Entrenados {len(results)} modelos E3"


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
