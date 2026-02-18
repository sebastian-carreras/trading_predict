"""
DAG de Airflow para Estrategia E1 Simple - Simplificada (GRU sin walk-forward)

Flujo:
1. Descargar datos diarios → data/raw/daily/
2. Limpiar y validar datos
3. Entrenar modelo GRU simple con MLflow tracking
4. Guardar predicciones y métricas
5. Notificar a FastAPI (modelo disponible)

Diferencias vs E1 Conservative:
- Sin walk-forward: usa un único split temporal (train/val/test)
- Decision score simplificado: 5 componentes (IC, directional accuracy, sharpe, MAE, RMSE)
- Arquitectura GRU más simple: 1 capa vs 2-3 capas en E1 Conservative
- Más rápido de entrenar, ideal para experimentos y prototipado
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
    'start_date': datetime(2026, 1, 6),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'e1_simple_pipeline',
    default_args=default_args,
    description='Pipeline E1 Simple: GRU simplificado sin walk-forward',
    schedule_interval='0 3 * * 1',  # Lunes 3 AM (después del E1 Conservative)
    catchup=False,
    tags=['trading', 'e1', 'simple', 'gru', 'simplified'],
    params={
        'tickers': 'YPFD.BA, GGAL.BA, PAMP.BA, BYMA.BA, CEPU.BA, AAPL, MSFT, JNJ, PG, V',
        'skip_download': 'False',  # Permite reutilizar datos descargados por E1 Conservative
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
        # Usar todos los tickers del config para E1 Simple
        tickers = list(
            config.get("universe", {})
            .get("tickers_by_strategy", {})
            .get("e1_simple", [])
        )
        if not tickers:
            # Fallback a E1 conservative
            tickers = list(
                config.get("universe", {})
                .get("tickers_by_strategy", {})
                .get("e1_conservative", [])
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


def train_e1_simple_with_mlflow(**context):
    """Task 2: Entrenar modelos E1 Simple con tracking MLflow."""
    import sys
    sys.path.insert(0, '/opt/airflow')
    
    from src.e1.train_simple_pipeline import run_e1_simple_for_ticker
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
    
    mlflow.set_experiment("E1_Simple_Strategy")
    
    root = Path("/opt/airflow")
    config = load_yaml(root / "src/config/base.yaml")
    
    # Obtener tickers del XCom (pasados desde download_daily_data)
    tickers_from_download = context['task_instance'].xcom_pull(
        task_ids='download_daily_data', 
        key='tickers_to_train'
    )
    
    if tickers_from_download:
        # Filtrar solo los tickers de E1 Simple que fueron descargados
        e1_simple_tickers_config = set(
            config.get("universe", {})
            .get("tickers_by_strategy", {})
            .get("e1_simple", [])
        )
        if not e1_simple_tickers_config:
            # Fallback a E1 conservative
            e1_simple_tickers_config = set(
                config.get("universe", {})
                .get("tickers_by_strategy", {})
                .get("e1_conservative", [])
            )
        tickers = [t for t in tickers_from_download if t in e1_simple_tickers_config]
    else:
        # Usar todos los tickers E1 Simple del config
        tickers = list(
            config.get("universe", {})
            .get("tickers_by_strategy", {})
            .get("e1_simple", [])
        )
        if not tickers:
            # Fallback a E1 conservative
            tickers = list(
                config.get("universe", {})
                .get("tickers_by_strategy", {})
                .get("e1_conservative", [])
            )
    
    if not tickers:
        return "No tickers to train for E1 Simple strategy"
    
    raw_dir = root / "data/raw/daily"
    
    # Output dir con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "runs/e1_simple" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for i, ticker in enumerate(tickers, 1):
        with mlflow.start_run(run_name=f"E1Simple_{ticker}_{timestamp}"):
            # Extraer configuración del modelo
            e1_simple_config = config.get("strategies", {}).get("e1_simple", {})
            model_config = e1_simple_config.get("model", {})

            print(
                f"[{i}/{len(tickers)}] Entrenando {ticker} | lookback={e1_simple_config.get('lookback_days', 360)} "
                f"horizon={e1_simple_config.get('horizon_days', 90)} | gru_units={model_config.get('gru_units', [64])} "
                f"lr={model_config.get('learning_rate', 0.001)} batch={model_config.get('batch_size', 64)}"
            )
            
            # Log parámetros de estrategia
            mlflow.log_params({
                "strategy": "e1_simple",
                "ticker": ticker,
                "lookback_days": e1_simple_config.get("lookback_days", 360),
                "horizon_days": e1_simple_config.get("horizon_days", 90),
                "model_type": "GRU",
                "walk_forward": False,
            })
            
            # Log hiperparámetros del modelo
            mlflow.log_params({
                "gru_units": str(model_config.get("gru_units", [64])),
                "dropout": model_config.get("dropout", 0.2),
                "dense_units": model_config.get("dense_units", 16),
                "learning_rate": model_config.get("learning_rate", 0.001),
                "batch_size": model_config.get("batch_size", 64),
                "max_epochs": model_config.get("max_epochs", 100),
                "early_stopping_patience": model_config.get("early_stopping_patience", 10),
                "loss": model_config.get("loss", "huber"),
                "huber_delta": model_config.get("huber_delta", 1.0),
                "seed": config.get("project", {}).get("seed", 42),
            })
            
            try:
                result = run_e1_simple_for_ticker(
                    config=config,
                    ticker=ticker,
                    raw_dir=raw_dir,
                    out_dir=out_dir,
                )
                
                # Log métricas ML a MLflow
                mlflow.log_metrics({
                    "mae": sanitize_metric(result.get("ml_mae", 0)),
                    "rmse": sanitize_metric(result.get("ml_rmse", 0)),
                    "ic": sanitize_metric(result.get("ml_ic", 0)),
                    "directional_accuracy": sanitize_metric(result.get("ml_directional_accuracy", 0)),
                })
                
                # Log métricas de backtesting a MLflow
                mlflow.log_metrics({
                    "bt_sharpe": sanitize_metric(result.get("bt_sharpe", 0)),
                    "bt_sortino": sanitize_metric(result.get("bt_sortino", 0)),
                    "bt_cagr": sanitize_metric(result.get("bt_cagr", 0)),
                    "bt_max_drawdown": sanitize_metric(result.get("bt_max_drawdown", 0)),
                    "bt_calmar": sanitize_metric(result.get("bt_calmar", 0)),
                    "bt_profit_factor": sanitize_metric(result.get("bt_profit_factor", 0)),
                    "bt_win_rate": sanitize_metric(result.get("bt_win_rate", 0)),
                    "bt_hit_rate": sanitize_metric(result.get("bt_hit_rate", result.get("bt_win_rate", 0))),
                    "bt_num_trades": sanitize_metric(result.get("bt_num_trades", 0)),
                })

                # Log modelo como artifact
                model_path = out_dir / ticker / f"{ticker}_model.pth"
                if model_path.exists():
                    mlflow.log_artifact(str(model_path), artifact_path="models")
                
                # Log predicciones como artifact
                pred_path = out_dir / ticker / f"{ticker}_predictions.csv"
                if pred_path.exists():
                    mlflow.log_artifact(str(pred_path), artifact_path="predictions")
                
                # Log backtest como artifact
                backtest_path = out_dir / ticker / f"{ticker}_backtest.csv"
                if backtest_path.exists():
                    mlflow.log_artifact(str(backtest_path), artifact_path="backtest")
                
                results.append({
                    "ticker": ticker,
                    "status": "success",
                    "mae": result.get("ml_mae"),
                    "rmse": result.get("ml_rmse"),
                    "ic": result.get("ml_ic"),
                    "directional_accuracy": result.get("ml_directional_accuracy"),
                    "sharpe": result.get("bt_sharpe"),
                    "max_dd": result.get("bt_max_drawdown"),
                    "cagr": result.get("bt_cagr"),
                })
                
            except Exception as e:
                mlflow.log_param("error", str(e))
                results.append({
                    "ticker": ticker,
                    "status": "failed",
                    "error": str(e),
                })
    
    # Guardar y loggear resumen
    summary_df = pd.DataFrame(results)
    summary_path = out_dir / "summary_all.csv"
    summary_df.to_csv(summary_path, index=False)
    
    # Calcular métricas agregadas
    successful_results = [r for r in results if r["status"] == "success"]
    ic_values = [r["ic"] for r in successful_results if isinstance(r.get("ic"), (int, float))]
    sharpe_values = [r.get("sharpe") for r in successful_results if isinstance(r.get("sharpe"), (int, float))]
    
    
    # Targets/thresholds desde config
    decision_cfg = config.get("decision", {})
    default_targets = {
        'ic_min': 0.05,
        'directional_accuracy_min': 0.55,
        'sharpe_min': 1.0,
        'mae_max': 0.03,
        'rmse_max': 0.05,
    }
    targets = {**default_targets, **decision_cfg.get("targets", {})}
    
    # Log resumen como artifact en MLflow
    with mlflow.start_run(run_name=f"E1Simple_Summary_{timestamp}"):
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
            mlflow.log_param("ic_target_min", targets['ic_min'])
        
        # Métricas agregadas de Sharpe
        if sharpe_values:
            mlflow.log_metric("sharpe_mean", float(sum(sharpe_values) / len(sharpe_values)))
            mlflow.log_metric("sharpe_median", float(sorted(sharpe_values)[len(sharpe_values) // 2]))
            mlflow.log_metric("sharpe_min", float(min(sharpe_values)))
            mlflow.log_metric("sharpe_max", float(max(sharpe_values)))
            mlflow.log_metric("sharpe_above_threshold", sum(1 for s in sharpe_values if s > targets['sharpe_min']))
            mlflow.log_param("sharpe_target_min", targets['sharpe_min'])
        
    context['task_instance'].xcom_push(key='run_dir', value=str(out_dir))
    return f"Entrenados {len(successful_results)}/{len(tickers)} modelos E1 Simple"


def notify_api_model_ready(**context):
    """Task 3: Notificar a FastAPI que nuevos modelos están listos."""
    import requests
    
    run_dir = context['task_instance'].xcom_pull(task_ids='train_e1_simple_models', key='run_dir')
    
    if not run_dir:
        return "Sin run_dir, notificación omitida"
    
    try:
        response = requests.post(
            "http://fastapi:8800/models/register",
            json={
                "strategy": "e1_simple",
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
    task_id='train_e1_simple_models',
    python_callable=train_e1_simple_with_mlflow,
    dag=dag,
)

task_notify = PythonOperator(
    task_id='notify_api',
    python_callable=notify_api_model_ready,
    dag=dag,
)

# Flujo del DAG: Download → Clean → Train → Notify
task_download >> task_clean >> task_train >> task_notify
