"""
DAG de Airflow para Estrategia E2 - Moderada (LSTM)

Flujo:
1. Descargar datos diarios → data/raw/daily/
2. Calcular features E2 (momentum-focused: RSI, MACD, Stochastic, OBV)
3. Entrenar modelo LSTM con MLflow tracking
4. Guardar predicciones y métricas
5. Notificar a FastAPI (modelo disponible)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import mlflow
import os
import json

# Configuración MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

default_args = {
    'owner': 'trading_predict',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 8),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'e2_moderate_pipeline',
    default_args=default_args,
    description='Pipeline E2: Predicción de retornos a 20 días (LSTM)',
    schedule_interval='0 3 * * 1',  # Lunes 3 AM (después de E1)
    catchup=False,
    tags=['trading', 'e2', 'moderate', 'lstm'],
    params={
        'tickers': 'BBAR.BA, BMA.BA, EDN.BA, TGSUD.BA, LOMA.BA, NVDA, GOOGL, AMZN, META, NFLX',
        'use_tuned_params': 'False',
        'tuned_params_path': 'reports/hyperparameter_optimization/e2_tuned_params_by_ticker.yaml',
    },
)


def _as_bool(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def download_daily_data(**context):
    """Task 1: Descargar datos diarios con control skip_existing."""
    import sys
    sys.path.insert(0, '/opt/airflow')
    
    from src.data.download_daily import download_daily_ohlcv
    from src.utils import load_yaml
    from pathlib import Path
    
    root = Path("/opt/airflow")
    config = load_yaml(root / "src/config/base.yaml")
    
    # Obtener tickers del parámetro del DAG o del config
    tickers_param = context['params'].get('tickers', '').strip()
    if tickers_param:
        tickers = [t.strip() for t in tickers_param.split(',') if t.strip()]
        # Agregar benchmark si no está en la lista
        benchmark = config.get("universe", {}).get("benchmark", "SPY")
        if benchmark not in tickers:
            tickers.append(benchmark)
    else:
        # Usar tickers E2 del config
        tickers = list(
            config.get("universe", {})
            .get("tickers_by_strategy", {})
            .get("e2_moderate", [])
        )
    
    out_dir = root / "data/raw/daily"
    
    written = download_daily_ohlcv(
        tickers, 
        out_dir=out_dir, 
        period="10y",
        skip_existing=True,
        min_days_fresh=1,
    )
    
    context['task_instance'].xcom_push(key='files_downloaded', value=len(written))
    context['task_instance'].xcom_push(key='tickers_to_train', value=tickers)
    return f"Descargados {len(written)} archivos nuevos para {len(tickers)} tickers"


def clean_daily_data(**context):
    """Task 1.5: Limpiar datos descargados."""
    import sys
    sys.path.insert(0, '/opt/airflow')
    
    from src.data.clean_daily import process_daily_data_with_cleaning
    from pathlib import Path
    
    root = Path("/opt/airflow")
    raw_dir = root / "data/raw/daily"
    clean_dir = root / "data/clean"
    
    print("Ejecutando limpieza y validación de datos...")
    reports = process_daily_data_with_cleaning(
        raw_dir=raw_dir,
        clean_dir=clean_dir,
        strategy="forward_fill",
        min_days=252,
        remove_zero_volume=True,
        verbose=True,
    )
    
    total_cleaned = sum(1 for r in reports.values() if r.get("status") == "ok")
    return f"Limpiados {total_cleaned} archivos"


def train_e2_with_mlflow(**context):
    """Task 2: Entrenar modelos E2 (LSTM) con MLflow tracking."""
    import sys
    sys.path.insert(0, '/opt/airflow')
    
    from src.train_e2_pipeline import run_e2_for_ticker
    from src.utils import load_yaml, ensure_dir
    from pathlib import Path
    import pandas as pd
    
    mlflow.set_experiment("E2_Moderate_Strategy")
    
    root = Path("/opt/airflow")
    config = load_yaml(root / "src/config/base.yaml")

    # Elegir baseline vs tuned params (Optuna) desde la UI del DAG.
    use_tuned_params = _as_bool(context['params'].get('use_tuned_params'))
    tuned_params_path = str(context['params'].get('tuned_params_path') or '').strip()

    tuned_map = None
    if use_tuned_params:
        if not tuned_params_path:
            tuned_params_path = 'reports/hyperparameter_optimization/e2_tuned_params_by_ticker.yaml'

        resolved_tuned_path = root / tuned_params_path
        os.environ['E2_TUNED_PARAMS_PATH'] = str(resolved_tuned_path)

        if resolved_tuned_path.exists():
            try:
                tuned_map = load_yaml(resolved_tuned_path)
            except Exception as e:
                print(f"⚠️ No se pudo cargar tuned params YAML: {resolved_tuned_path}: {e}")
                tuned_map = None
        else:
            print(f"⚠️ use_tuned_params=True pero no existe: {resolved_tuned_path}")
    else:
        os.environ.pop('E2_TUNED_PARAMS_PATH', None)
        os.environ.pop('TUNED_PARAMS_PATH', None)
    
    # Tickers
    tickers = context['task_instance'].xcom_pull(task_ids='download_daily_data', key='tickers_to_train')
    if not tickers:
        tickers = list(
            config.get("universe", {})
            .get("tickers_by_strategy", {})
            .get("e2_moderate", [])
        )
    
    # Benchmark
    benchmark = config.get("universe", {}).get("benchmark", "SPY")
    raw_dir = root / "data/raw/daily"
    
    benchmark_path = raw_dir / f"{benchmark}_daily.csv"
    if benchmark_path.exists():
        from src.train_e2_pipeline import load_ohlcv_csv
        benchmark_df = load_ohlcv_csv(benchmark_path)
    else:
        benchmark_df = None
    
    # Directorio de salida con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "runs" / "e2_moderate" / timestamp
    ensure_dir(out_dir)
    
    # Entrenar cada ticker en un run de MLflow
    results = []
    for ticker in tickers:
        ticker_out = out_dir / ticker
        
        with mlflow.start_run(run_name=f"E2_{ticker}_{timestamp}"):
            mlflow.log_param("use_tuned_params", use_tuned_params)
            mlflow.log_param("tuned_params_path", tuned_params_path if use_tuned_params else "")
            if use_tuned_params and isinstance(tuned_map, dict):
                overrides_for_ticker = tuned_map.get(ticker) or {}
                mlflow.log_param(
                    "tuned_overrides",
                    json.dumps(overrides_for_ticker, sort_keys=True, ensure_ascii=False),
                )

            mlflow.log_param("strategy", "e2_moderate")
            mlflow.log_param("ticker", ticker)
            mlflow.log_param("model_type", "LSTM")
            mlflow.log_param("lookback_days", config.get("strategies", {}).get("e2_moderate", {}).get("lookback_days", 60))
            mlflow.log_param("horizon_days", config.get("strategies", {}).get("e2_moderate", {}).get("horizon_days", 20))
            mlflow.log_param("timestamp", timestamp)
            
            # Loggear hiperparámetros del modelo
            model_cfg = config.get("strategies", {}).get("e2_moderate", {}).get("model", {})
            mlflow.log_params({
                "lstm_units": str(model_cfg.get("lstm_units", [128, 64])),
                "dropout": model_cfg.get("dropout", 0.2),
                "dense_units": model_cfg.get("dense_units", 32),
                "learning_rate": model_cfg.get("learning_rate", 0.001),
                "batch_size": model_cfg.get("batch_size", 64),
                "max_epochs": model_cfg.get("max_epochs", 150),
                "early_stopping_patience": model_cfg.get("early_stopping_patience", 12),
            })
            
            try:
                result = run_e2_for_ticker(
                    config=config,
                    ticker=ticker,
                    raw_dir=raw_dir,
                    out_dir=ticker_out,
                    benchmark_df=benchmark_df,
                )
                
                # Log métricas ML a MLflow
                mlflow.log_metrics({
                    "mae": result.get("ml_mae", 0),
                    "rmse": result.get("ml_rmse", 0),
                    "ic": result.get("ml_ic", 0),
                    "directional_accuracy": result.get("ml_directional_accuracy", 0),
                })
                
                # Log métricas de backtesting a MLflow
                mlflow.log_metrics({
                    "bt_sharpe": result.get("bt_sharpe", 0),
                    "bt_cagr": result.get("bt_cagr", 0),
                    "bt_max_drawdown": result.get("bt_max_drawdown", 0),
                    "bt_calmar": result.get("bt_calmar", 0),
                    "bt_profit_factor": result.get("bt_profit_factor", 0),
                    "bt_win_rate": result.get("bt_win_rate", 0),
                    "bt_num_trades": result.get("bt_num_trades", 0),
                })
                
                # Log predicciones como artifact
                pred_path = ticker_out / f"{ticker}_predictions.csv"
                if pred_path.exists():
                    mlflow.log_artifact(str(pred_path), artifact_path="predictions")

                # Log modelo LSTM como artifact
                model_path = ticker_out / f"{ticker}_model.pth"
                if model_path.exists():
                    mlflow.log_artifact(str(model_path), artifact_path="models")
                
                results.append({
                    "ticker": ticker,
                    "status": "success",
                    "mae": result.get("ml_mae"),
                    "ic": result.get("ml_ic"),
                    "sharpe": result.get("bt_sharpe"),
                    "max_dd": result.get("bt_max_drawdown"),
                })
                
            except Exception as e:
                mlflow.log_param("error", str(e))
                results.append({
                    "ticker": ticker,
                    "status": "failed",
                    "error": str(e),
                })
    
    # Guardar resumen
    summary_df = pd.DataFrame(results)
    summary_path = out_dir / "summary_all.csv"
    summary_df.to_csv(summary_path, index=False)
    
    # Calcular métricas agregadas
    successful_results = [r for r in results if r["status"] == "success" and r.get("ic") is not None]
    ic_values = [r["ic"] for r in successful_results]
    
    # Log resumen como artifact en MLflow
    with mlflow.start_run(run_name=f"E2_Summary_{timestamp}"):
        mlflow.log_artifact(str(summary_path))
        mlflow.log_metric("total_tickers", len(tickers))
        mlflow.log_metric("successful_tickers", len(successful_results))
        
        # Métricas agregadas de IC
        if ic_values:
            mlflow.log_metric("ic_mean", float(sum(ic_values) / len(ic_values)))
            mlflow.log_metric("ic_median", float(sorted(ic_values)[len(ic_values) // 2]))
            mlflow.log_metric("ic_min", float(min(ic_values)))
            mlflow.log_metric("ic_max", float(max(ic_values)))
            mlflow.log_metric("ic_positive_count", sum(1 for ic in ic_values if ic > 0))
            mlflow.log_metric("ic_above_threshold", sum(1 for ic in ic_values if ic > 0.05))
    
    context['task_instance'].xcom_push(key='run_dir', value=str(out_dir))
    return f"Entrenados {len(results)} modelos E2"


def notify_api_model_ready(**context):
    """Task 3: Notificar a FastAPI que nuevos modelos están listos."""
    import requests
    
    run_dir = context['task_instance'].xcom_pull(task_ids='train_e2_models', key='run_dir')
    
    try:
        response = requests.post(
            "http://fastapi:8800/models/register",
            json={
                "strategy": "e2_moderate",
                "run_dir": run_dir,
                "timestamp": datetime.now().isoformat(),
            },
            timeout=10,
        )
        response.raise_for_status()
        return f"API notificada: {response.status_code}"
    except Exception as e:
        print(f"⚠️ Error notificando API: {e}")
        return "API notification skipped"


# Definir tareas
task_download = PythonOperator(
    task_id='download_daily_data',
    python_callable=download_daily_data,
    dag=dag,
)

task_clean = PythonOperator(
    task_id='clean_daily_data',
    python_callable=clean_daily_data,
    dag=dag,
)

task_train = PythonOperator(
    task_id='train_e2_models',
    python_callable=train_e2_with_mlflow,
    dag=dag,
)

task_notify = PythonOperator(
    task_id='notify_api',
    python_callable=notify_api_model_ready,
    dag=dag,
)

# Flujo del DAG: Download → Clean → Train → Notify
task_download >> task_clean >> task_train >> task_notify
