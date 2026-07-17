"""
DAG de Airflow para Estrategia E2 Simple - Simplificada (LSTM sin walk-forward)

Flujo:
1. Descargar datos diarios → data/raw/daily/
2. Limpiar y validar datos
3. Entrenar modelo LSTM simple con MLflow tracking
4. Guardar predicciones y métricas
5. Notificar a FastAPI (modelo disponible)

Diferencias vs E2 Moderate:
- Sin walk-forward: usa un único split temporal (train/val/test)
- Decision score simplificado: 5 componentes (IC, directional accuracy, sharpe, MAE, RMSE)
- Misma arquitectura LSTM: 2 capas (128 → 64)
- Menos epochs: 100 vs 150 de E2 Moderate
- Más rápido de entrenar (~40% menos tiempo), ideal para experimentos y prototipado
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
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
    'e2_simple_pipeline',
    default_args=default_args,
    description='Pipeline E2 Simple: Predicción de retornos a 20 días (LSTM simplificado)',
    schedule_interval=None,  # Solo manual: la automatización diaria corre únicamente e2_moderate
    catchup=False,
    tags=['trading', 'e2', 'simple', 'lstm', 'simplified'],
    params={
        'tickers': 'BBAR.BA, BMA.BA, EDN.BA, TGSU2.BA, LOMA.BA, NVDA, GOOGL, AMZN, META, NFLX',
        'skip_download': 'False',  # Permite reutilizar datos descargados por E2 Moderate
        'skip_cleaning': 'False',   # Permite reutilizar datos limpios
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
    
    skip_download = _as_bool(context['params'].get('skip_download'))
    
    if skip_download:
        context['task_instance'].xcom_push(key='files_downloaded', value=0)
        context['task_instance'].xcom_push(key='tickers_to_train', value=[])
        return "Descarga omitida (skip_download=True)"
    
    root = Path("/opt/airflow")
    config = load_yaml(root / "src/config/base.yaml")
    
    # Obtener tickers del parámetro del DAG o del config
    tickers_param = context['params'].get('tickers', '').strip()
    if tickers_param:
        tickers = [t.strip() for t in tickers_param.split(',') if t.strip()]
    else:
        # Usar todos los tickers del config para E2 Simple
        tickers = list(
            config.get("universe", {})
            .get("tickers_by_strategy", {})
            .get("e2_simple", [])
        )
        if not tickers:
            # Fallback a E2 moderate
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
    """Task 1.5: Limpiar datos descargados (detectar y corregir nulos/duplicados)."""
    import sys
    sys.path.insert(0, '/opt/airflow')
    
    from src.data.clean_daily import process_daily_data_with_cleaning
    from pathlib import Path
    
    skip_cleaning = _as_bool(context['params'].get('skip_cleaning'))
    
    if skip_cleaning:
        context['task_instance'].xcom_push(key='cleaned_tickers', value=0)
        context['task_instance'].xcom_push(key='rejected_tickers', value=0)
        return "Limpieza omitida (skip_cleaning=True)"
    
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
    
    # Resumen para XCom
    cleaned_count = sum(1 for r in reports.values() if r.get("status") == "cleaned")
    rejected_count = sum(1 for r in reports.values() if r.get("status") == "rejected")
    
    context['task_instance'].xcom_push(key='cleaned_tickers', value=cleaned_count)
    context['task_instance'].xcom_push(key='rejected_tickers', value=rejected_count)
    
    return f"Limpiados {cleaned_count} tickers, rechazados {rejected_count}"


def train_e2_simple_with_mlflow(**context):
    """Task 2: Entrenar modelos E2 Simple con tracking MLflow."""
    import sys
    sys.path.insert(0, '/opt/airflow')
    
    from src.e2.train_simple_pipeline import run_e2_simple_for_ticker
    from src.utils import load_yaml
    from pathlib import Path
    import pandas as pd
    import numpy as np
    
    def sanitize_metric(v):
        """Sanitiza valor para MLflow: NaN -> 0, Inf -> max_float."""
        if not isinstance(v, (int, float)):
            return v
        if np.isnan(v):
            return 0.0
        if np.isinf(v):
            return 1e8 if v > 0 else -1e8
        return v
    
    mlflow.set_experiment("E2_Simple_Strategy")
    
    root = Path("/opt/airflow")
    config = load_yaml(root / "src/config/base.yaml")
    
    # Obtener tickers del XCom (pasados desde download_daily_data)
    tickers_from_download = context['task_instance'].xcom_pull(
        task_ids='download_daily_data', 
        key='tickers_to_train'
    )
    
    if tickers_from_download:
        # Filtrar solo los tickers de E2 Simple que fueron descargados
        e2_simple_tickers_config = set(
            config.get("universe", {})
            .get("tickers_by_strategy", {})
            .get("e2_simple", [])
        )
        if not e2_simple_tickers_config:
            # Fallback a E2 moderate
            e2_simple_tickers_config = set(
                config.get("universe", {})
                .get("tickers_by_strategy", {})
                .get("e2_moderate", [])
            )
        tickers = [t for t in tickers_from_download if t in e2_simple_tickers_config]
    else:
        # Usar todos los tickers E2 Simple del config
        tickers = list(
            config.get("universe", {})
            .get("tickers_by_strategy", {})
            .get("e2_simple", [])
        )
        if not tickers:
            # Fallback a E2 moderate
            tickers = list(
                config.get("universe", {})
                .get("tickers_by_strategy", {})
                .get("e2_moderate", [])
            )
    
    if not tickers:
        return "No tickers to train for E2 Simple strategy"
    
    # Directorio de salida con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "runs" / "e2_simple" / timestamp
    from src.utils import ensure_dir
    ensure_dir(out_dir)
    
    # Entrenar cada ticker en un run de MLflow
    results = []
    for ticker in tickers:
        ticker_out = out_dir / ticker
        
        with mlflow.start_run(run_name=f"E2_Simple_{ticker}_{timestamp}"):
            mlflow.log_param("strategy", "e2_simple")
            mlflow.log_param("ticker", ticker)
            mlflow.log_param("model_type", "LSTM")
            mlflow.log_param("lookback_days", config.get("strategies", {}).get("e2_simple", {}).get("lookback_days", 60))
            mlflow.log_param("horizon_days", config.get("strategies", {}).get("e2_simple", {}).get("horizon_days", 20))
            mlflow.log_param("timestamp", timestamp)
            
            # Loggear hiperparámetros del modelo
            model_cfg = config.get("strategies", {}).get("e2_simple", {}).get("model", {})
            mlflow.log_params({
                "lstm_units": str(model_cfg.get("lstm_units", [128, 64])),
                "dropout": model_cfg.get("dropout", 0.2),
                "dense_units": model_cfg.get("dense_units", 16),
                "learning_rate": model_cfg.get("learning_rate", 0.001),
                "batch_size": model_cfg.get("batch_size", 64),
                "max_epochs": model_cfg.get("max_epochs", 100),
                "early_stopping_patience": model_cfg.get("early_stopping_patience", 10),
            })
            
            # Loggear filtros
            filters_cfg = config.get("strategies", {}).get("e2_simple", {}).get("filters", {})
            mlflow.log_params({
                "rsi14_min": filters_cfg.get("rsi14_min", 35),
                "rsi14_max": filters_cfg.get("rsi14_max", 70),
                "macd_confirmation": filters_cfg.get("macd_confirmation", True),
                "volume_zscore_window": filters_cfg.get("volume_zscore_window", 20),
                "volume_zscore_min": filters_cfg.get("volume_zscore_min", 0),
            })
            
            # Loggear thresholds
            thresholds_cfg = config.get("strategies", {}).get("e2_simple", {}).get("thresholds", {})
            mlflow.log_params({
                "tau_buy": thresholds_cfg.get("tau_buy", 0.025),
                "tau_sell": thresholds_cfg.get("tau_sell", 0.00),
            })
            
            try:
                result = run_e2_simple_for_ticker(
                    config=config,
                    ticker=ticker,
                    raw_dir=raw_dir,
                    out_dir=ticker_out,
                )
                
                # Log métricas ML a MLflow (sanitizadas)
                mlflow.log_metrics({
                    "mae": sanitize_metric(result.get("ml_mae", 0)),
                    "rmse": sanitize_metric(result.get("ml_rmse", 0)),
                    "ic": sanitize_metric(result.get("ml_ic", 0)),
                    "directional_accuracy": sanitize_metric(result.get("ml_directional_accuracy", 0)),
                })
                
                # Log métricas de backtesting a MLflow (sanitizadas)
                mlflow.log_metrics({
                    "bt_sharpe": sanitize_metric(result.get("bt_sharpe", 0)),
                    "bt_cagr": sanitize_metric(result.get("bt_cagr", 0)),
                    "bt_max_drawdown": sanitize_metric(result.get("bt_max_drawdown", 0)),
                    "bt_calmar": sanitize_metric(result.get("bt_calmar", 0)),
                    "bt_profit_factor": sanitize_metric(result.get("bt_profit_factor", 0)),
                    "bt_win_rate": sanitize_metric(result.get("bt_win_rate", 0)),
                    "bt_num_trades": sanitize_metric(result.get("bt_num_trades", 0)),
                })
                
                # Log predicciones como artifact
                pred_path = ticker_out / f"{ticker}_predictions.csv"
                if pred_path.exists():
                    mlflow.log_artifact(str(pred_path), artifact_path="predictions")

                # Log modelo LSTM como artifact
                model_path = ticker_out / f"{ticker}_model.pth"
                if model_path.exists():
                    mlflow.log_artifact(str(model_path), artifact_path="models")
                
                # Log scaler como artifact
                scaler_path = ticker_out / f"{ticker}_scaler.csv"
                if scaler_path.exists():
                    mlflow.log_artifact(str(scaler_path), artifact_path="scalers")
                
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
    sharpe_values = [r["sharpe"] for r in successful_results if r.get("sharpe") is not None]
    
    
    # Targets/thresholds de las métricas (desde config)
    e2_simple_cfg = config.get("strategies", {}).get("e2_simple", {})
    decision_cfg = config.get("decision", {})
    
    # Targets por defecto para E2 Simple (mismo que en train_e2_simple_pipeline.py)
    default_targets = {
        'ic_min': 0.05,
        'directional_accuracy_min': 0.55,
        'sharpe_min': 1.0,
        'mae_max': 0.03,
        'rmse_max': 0.05,
    }
    targets = {**default_targets, **decision_cfg.get("targets", {})}
    
    # Log resumen como artifact en MLflow
    with mlflow.start_run(run_name=f"E2_Simple_Summary_{timestamp}"):
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
            mlflow.log_metric("ic_above_threshold", sum(1 for ic in ic_values if ic > targets['ic_min']))
            # Loggear target
            mlflow.log_param("ic_target_min", targets['ic_min'])
        
        # Métricas agregadas de Sharpe
        if sharpe_values:
            mlflow.log_metric("sharpe_mean", float(sum(sharpe_values) / len(sharpe_values)))
            mlflow.log_metric("sharpe_median", float(sorted(sharpe_values)[len(sharpe_values) // 2]))
            mlflow.log_metric("sharpe_min", float(min(sharpe_values)))
            mlflow.log_metric("sharpe_max", float(max(sharpe_values)))
            mlflow.log_metric("sharpe_above_threshold", sum(1 for s in sharpe_values if s > targets['sharpe_min']))
            # Loggear target
            mlflow.log_param("sharpe_target_min", targets['sharpe_min'])
        
    
    context['task_instance'].xcom_push(key='run_dir', value=str(out_dir))
    return f"Entrenados {len(results)} modelos E2 Simple"


def notify_api_model_ready(**context):
    """Task 3: Notificar a FastAPI que nuevos modelos están listos."""
    import requests
    
    run_dir = context['task_instance'].xcom_pull(task_ids='train_e2_simple_models', key='run_dir')
    
    try:
        response = requests.post(
            "http://fastapi:8800/models/register",
            json={
                "strategy": "e2_simple",
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
    task_id='train_e2_simple_models',
    python_callable=train_e2_simple_with_mlflow,
    dag=dag,
)

task_notify = PythonOperator(
    task_id='notify_api',
    python_callable=notify_api_model_ready,
    dag=dag,
)

# Flujo del DAG: Download → Clean → Train → Notify
task_download >> task_clean >> task_train >> task_notify
