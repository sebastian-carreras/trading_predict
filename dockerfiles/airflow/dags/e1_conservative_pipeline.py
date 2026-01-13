"""
DAG de Airflow para Estrategia E1 - Conservadora (GRU)

Flujo:
1. Descargar datos diarios → data/raw/daily/
2. Calcular features E1 (27 indicadores)
3. Entrenar modelo GRU con MLflow tracking
4. Guardar predicciones y métricas
5. Notificar a FastAPI (modelo disponible)
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
    'e1_conservative_pipeline',
    default_args=default_args,
    description='Pipeline E1: Predicción de retornos a 90 días (GRU)',
    schedule_interval='0 2 * * 1',  # Lunes 2 AM (semanal, para actualizar datos)
    catchup=False,
    tags=['trading', 'e1', 'conservative', 'gru'],
    params={
        'tickers': 'YPFD.BA, GGAL.BA, PAMP.BA, BYMA.BA, CEPU.BA, AAPL, MSFT, JNJ, PG, V',  # Comma-separated tickers: "AAPL,MSFT,GOOGL,META,NVDA", vacío = todos del config
        'use_tuned_params': 'False',  # True = aplicar overrides por ticker desde YAML (Optuna)
        'tuned_params_path': 'reports/hyperparameter_optimization/e1_tuned_params_by_ticker.yaml',
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
        # Usar todos los tickers del config
        tickers = list(config.get("universe", {}).get("tickers", []))
    
    out_dir = root / "data/raw/daily"
    
    written = download_daily_ohlcv(
        tickers, 
        out_dir=out_dir, 
        period="10y",
        skip_existing=True,  # No re-descargar si fresco
        min_days_fresh=1,    # Considerar fresco si < 1 día
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
    
    root = Path("/opt/airflow")
    raw_dir = root / "data/raw/daily"
    clean_dir = root / "data/clean"
    
    print("Ejecutando limpieza y validación de datos...")
    reports = process_daily_data_with_cleaning(
        raw_dir=raw_dir,
        clean_dir=clean_dir,
        strategy="forward_fill",  # Conservador: propaga último valor válido
        min_days=252,  # Mínimo 1 año de datos después de limpieza
        remove_zero_volume=True,  # Eliminar días con volumen=0 (sin trading real)
        verbose=True,
    )
    
    # Resumen para XCom
    cleaned_count = sum(1 for r in reports.values() if r.get("status") == "cleaned")
    rejected_count = sum(1 for r in reports.values() if r.get("status") == "rejected")
    
    context['task_instance'].xcom_push(key='cleaned_tickers', value=cleaned_count)
    context['task_instance'].xcom_push(key='rejected_tickers', value=rejected_count)
    
    return f"Limpiados {cleaned_count} tickers, rechazados {rejected_count}"


def train_e1_with_mlflow(**context):
    """Task 2: Entrenar modelos E1 con tracking MLflow."""
    import sys
    sys.path.insert(0, '/opt/airflow')
    
    from src.train_e1_pipeline import run_e1_for_ticker, load_ohlcv_csv
    from src.utils import load_yaml
    from pathlib import Path
    import pandas as pd
    
    mlflow.set_experiment("E1_Conservative_Strategy")
    
    root = Path("/opt/airflow")
    config = load_yaml(root / "src/config/base.yaml")

    # Elegir baseline vs tuned params (Optuna) desde la UI del DAG.
    use_tuned_params = _as_bool(context['params'].get('use_tuned_params'))
    tuned_params_path = str(context['params'].get('tuned_params_path') or '').strip()

    tuned_map = None
    if use_tuned_params:
        if not tuned_params_path:
            tuned_params_path = 'reports/hyperparameter_optimization/e1_tuned_params_by_ticker.yaml'

        resolved_tuned_path = root / tuned_params_path
        os.environ['E1_TUNED_PARAMS_PATH'] = str(resolved_tuned_path)

        if resolved_tuned_path.exists():
            try:
                tuned_map = load_yaml(resolved_tuned_path)
            except Exception as e:
                print(f"⚠️ No se pudo cargar tuned params YAML: {resolved_tuned_path}: {e}")
                tuned_map = None
        else:
            print(f"⚠️ use_tuned_params=True pero no existe: {resolved_tuned_path}")
    else:
        # Asegurar baseline aunque haya quedado una env var de ejecuciones previas.
        os.environ.pop('E1_TUNED_PARAMS_PATH', None)
        os.environ.pop('TUNED_PARAMS_PATH', None)
    
    # Obtener tickers del XCom (pasados desde download_daily_data)
    tickers_from_download = context['task_instance'].xcom_pull(
        task_ids='download_daily_data', 
        key='tickers_to_train'
    )
    
    if tickers_from_download:
        # Filtrar solo los tickers de E1 que fueron descargados
        e1_tickers_config = set(
            config.get("universe", {})
            .get("tickers_by_strategy", {})
            .get("e1_conservative", [])
        )
        tickers = [t for t in tickers_from_download if t in e1_tickers_config]
    else:
        # Usar todos los tickers E1 del config
        tickers = list(
            config.get("universe", {})
            .get("tickers_by_strategy", {})
            .get("e1_conservative", [])
        )
    
    if not tickers:
        return "No tickers to train for E1 strategy"
    
    # Benchmark
    benchmark = config.get("universe", {}).get("benchmark", "SPY")
    raw_dir = root / "data/raw/daily"
    benchmark_path = raw_dir / f"{benchmark}_daily.csv"
    
    if benchmark_path.exists():
        benchmark_df = load_ohlcv_csv(benchmark_path)
    else:
        benchmark_df = None
    
    # Output dir con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "runs/e1_conservative" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for ticker in tickers:
        with mlflow.start_run(run_name=f"E1_{ticker}_{timestamp}"):
            mlflow.log_param("use_tuned_params", use_tuned_params)
            mlflow.log_param("tuned_params_path", tuned_params_path if use_tuned_params else "")
            if use_tuned_params and isinstance(tuned_map, dict):
                overrides_for_ticker = tuned_map.get(ticker) or {}
                # Guardar un snapshot (string) para reproducibilidad
                mlflow.log_param(
                    "tuned_overrides",
                    json.dumps(overrides_for_ticker, sort_keys=True, ensure_ascii=False),
                )

            # Extraer configuración del modelo
            e1_config = config["strategies"]["e1_conservative"]
            model_config = e1_config.get("model", {})
            
            # Log parámetros de estrategia
            mlflow.log_params({
                "strategy": "e1_conservative",
                "ticker": ticker,
                "lookback_days": e1_config["lookback_days"],
                "horizon_days": e1_config["horizon_days"],
                "model_type": "GRU",
            })
            
            # Log hiperparámetros del modelo
            mlflow.log_params({
                "gru_units": str(model_config.get("gru_units", [128, 64])),
                "dropout": model_config.get("dropout", 0.2),
                "dense_units": model_config.get("dense_units", 32),
                "learning_rate": model_config.get("learning_rate", 1e-3),
                "batch_size": model_config.get("batch_size", 64),
                "max_epochs": model_config.get("max_epochs", 200),
                "early_stopping_patience": model_config.get("early_stopping_patience", 15),
                "loss": model_config.get("loss", "huber"),
                "huber_delta": model_config.get("huber_delta", 1.0),
                "seed": config.get("project", {}).get("seed", 42),
            })
            
            try:
                result = run_e1_for_ticker(
                    config=config,
                    ticker=ticker,
                    raw_dir=raw_dir,
                    out_dir=out_dir,
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
                    "bt_sortino": result.get("bt_sortino", 0),
                    "bt_cagr": result.get("bt_cagr", 0),
                    "bt_max_drawdown": result.get("bt_max_drawdown", 0),
                    "bt_calmar": result.get("bt_calmar", 0),
                    "bt_profit_factor": result.get("bt_profit_factor", 0),
                    "bt_win_rate": result.get("bt_win_rate", 0),
                    "bt_hit_rate": result.get("bt_hit_rate", result.get("bt_win_rate", 0)),
                    "bt_num_trades": result.get("bt_num_trades", 0),
                })

                if result.get("decision_profile"):
                    mlflow.log_param("decision_profile", result.get("decision_profile"))

                if result.get("decision_score") is not None:
                    mlflow.log_metric("decision_score", result.get("decision_score", 0))
                for k, v in (result or {}).items():
                    if not isinstance(k, str) or not k.startswith("decision_component_"):
                        continue
                    if v is None:
                        continue
                    mlflow.log_metric(k, v)
                if result.get("decision_signal"):
                    mlflow.log_param("decision_signal", result.get("decision_signal"))
                
                # Log modelo GRU como artifact
                model_path = out_dir / ticker / f"{ticker}_model.pth"
                if model_path.exists():
                    mlflow.log_artifact(str(model_path), artifact_path="models")
                
                # Log backtest como artifact
                backtest_path = out_dir / ticker / f"{ticker}_backtest.csv"
                if backtest_path.exists():
                    mlflow.log_artifact(str(backtest_path), artifact_path="backtest")
                
                results.append({
                    "ticker": ticker,
                    "status": "success",
                    "mae": result.get("ml_mae"),
                    "ic": result.get("ml_ic"),
                    "sharpe": result.get("bt_sharpe"),
                    "max_dd": result.get("bt_max_drawdown"),
                    "decision_score": result.get("decision_score"),
                    "decision_signal": result.get("decision_signal"),
                    "decision_profile": result.get("decision_profile"),
                    "decision_component_directional_accuracy": result.get(
                        "decision_component_directional_accuracy"
                    ),
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
    
    # Calcular métricas agregadas de IC
    successful_results = [r for r in results if r["status"] == "success" and r.get("ic") is not None]
    ic_values = [r["ic"] for r in successful_results]
    decision_scores = [r.get("decision_score") for r in successful_results if isinstance(r.get("decision_score"), (int, float))]
    buy_signals = sum(1 for r in successful_results if r.get("decision_signal") == "buy")
    
    # Log resumen como artifact en MLflow
    with mlflow.start_run(run_name=f"E1_Summary_{timestamp}"):
        mlflow.log_artifact(str(summary_path))
        mlflow.log_metric("total_tickers", len(tickers))
        mlflow.log_metric("successful_tickers", len(successful_results))
        mlflow.log_metric("decision_buy_signals", float(buy_signals))
        
        # Métricas agregadas de IC
        if ic_values:
            mlflow.log_metric("ic_mean", float(sum(ic_values) / len(ic_values)))
            mlflow.log_metric("ic_median", float(sorted(ic_values)[len(ic_values) // 2]))
            mlflow.log_metric("ic_min", float(min(ic_values)))
            mlflow.log_metric("ic_max", float(max(ic_values)))
            mlflow.log_metric("ic_positive_count", sum(1 for ic in ic_values if ic > 0))
            mlflow.log_metric("ic_above_threshold", sum(1 for ic in ic_values if ic > 0.05))  # IC > 5%
        if decision_scores:
            mlflow.log_metric("decision_score_mean", float(sum(decision_scores) / len(decision_scores)))
            mlflow.log_metric("decision_score_min", float(min(decision_scores)))
            mlflow.log_metric("decision_score_max", float(max(decision_scores)))
    
    context['task_instance'].xcom_push(key='run_dir', value=str(out_dir))
    return f"Entrenados {len(results)} modelos"


def notify_api_model_ready(**context):
    """Task 3: Notificar a FastAPI que nuevos modelos están listos."""
    import requests
    
    run_dir = context['task_instance'].xcom_pull(task_ids='train_e1_models', key='run_dir')
    
    try:
        response = requests.post(
            "http://fastapi:8800/models/register",
            json={
                "strategy": "e1_conservative",
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
    task_id='train_e1_models',
    python_callable=train_e1_with_mlflow,
    dag=dag,
)

task_notify = PythonOperator(
    task_id='notify_api',
    python_callable=notify_api_model_ready,
    dag=dag,
)

# Flujo del DAG: Download → Clean → Train → Notify
task_download >> task_clean >> task_train >> task_notify
