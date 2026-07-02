"""
DAG de Airflow para Estrategia E1 - Conservadora (GRU)

Flujo:
1. Descargar datos diarios → data/raw/daily/
2. Calcular features E1 (15 indicadores)
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
        'use_tuned_params': 'True',  # True = aplicar overrides por ticker desde YAML (Optuna)
        'tuned_params_path': 'reports/hyperparameter_optimization/e1_tuned_params_by_ticker.yaml',
        'train_with_new_data': 'False',  # False = descargar a staging sin usarlo; True = refrescar datos canónicos y entrenar con ellos
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
    from src.data.ingest import resolve_download_settings
    from src.utils import load_yaml
    from pathlib import Path

    root = Path("/opt/airflow")
    config = load_yaml(root / "src/config/base.yaml")
    
    # Obtener tickers del parámetro del DAG o del config
    tickers_param = context['params'].get('tickers', '').strip()
    if tickers_param:
        tickers = [t.strip() for t in tickers_param.split(',') if t.strip()]
    else:
        # Usar todos los tickers del config
        tickers = list(config.get("universe", {}).get("tickers", []))
    
    # train_with_new_data controla si los datos recién descargados se USAN para
    # entrenar. Por defecto (False) se descargan a un directorio de staging y los
    # datos canónicos (data/raw/daily) que consume el entrenamiento NO se tocan.
    train_with_new_data = _as_bool(context['params'].get('train_with_new_data'))
    canonical_dir = root / "data/raw/daily"
    staging_dir = root / "data/raw/daily_incoming"
    out_dir = canonical_dir if train_with_new_data else staging_dir

    print(
        f"[E1][download] train_with_new_data={train_with_new_data} -> destino={out_dir} "
        f"({'se usará para entrenar' if train_with_new_data else 'STAGING: no se usará para entrenar'})"
    )

    # Config-driven: period + incremental desde data.download (base.yaml), igual que la CLI.
    dl = resolve_download_settings(config, granularity="daily")
    written = download_daily_ohlcv(
        tickers,
        out_dir=out_dir,
        period=dl["period"],
        skip_existing=False,  # Siempre traer datos frescos (merge + dedupe por timestamp)
        incremental=dl["incremental"],
        overlap_days=dl["overlap_days"],
    )

    context['task_instance'].xcom_push(key='files_downloaded', value=len(written))
    context['task_instance'].xcom_push(key='tickers_to_train', value=tickers)
    context['task_instance'].xcom_push(key='train_with_new_data', value=train_with_new_data)
    context['task_instance'].xcom_push(key='download_dir', value=str(out_dir))
    return (
        f"Descargados {len(written)} archivos a {out_dir.name} "
        f"(train_with_new_data={train_with_new_data})"
    )


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
    
    from src.e1.train_pipeline import run_e1_for_ticker, load_ohlcv_csv
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
    
    mlflow.set_experiment("E1_Conservative_Strategy")

    root = Path("/opt/airflow")
    config = load_yaml(root / "src/config/base.yaml")

    # Único flag que decide si se entrena con datos nuevos:
    #   False (default) -> usa la ventana congelada en config (data.training_window)
    #   True            -> extiende la ventana hasta hoy (use_latest_data) y entrena
    #                      con los datos frescos que la descarga dejó en data/raw/daily
    train_with_new_data = _as_bool(context['params'].get('train_with_new_data'))

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
    
    raw_dir = root / "data/raw/daily"
    
    # Output dir con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "runs/e1_conservative" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for ticker in tickers:
        with mlflow.start_run(run_name=f"E1_{ticker}_{timestamp}"):
            mlflow.log_param("use_tuned_params", use_tuned_params)
            mlflow.log_param("tuned_params_path", tuned_params_path if use_tuned_params else "")
            mlflow.log_param("train_with_new_data", train_with_new_data)
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
                    use_latest_data=train_with_new_data,
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
    with mlflow.start_run(run_name=f"E1_Summary_{timestamp}"):
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
            mlflow.log_metric("ic_above_threshold", sum(1 for ic in ic_values if ic > targets['ic_min']))  # IC > target
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
    return f"Entrenados {len(results)} modelos"


def notify_api_model_ready(**context):
    """Task 4: Notificar a FastAPI que nuevos modelos están listos.

    TODO (post-presentación): esta task NO funciona todavía. Hace POST a
    `http://fastapi:8800/models/register`, pero ese endpoint NO existe en la API
    (solo hay `/`, `/health`, `/models/status`, `/predict`). El POST falla y el
    `try/except` lo silencia ("API notification skipped"), así que el DAG marca
    success pero la notificación nunca ocurre.

    Por qué hace falta: la API carga `registry.json` UNA sola vez al arrancar y lo
    cachea en memoria (app.py: `REGISTRY = ModelRegistry(...)`); no relee el archivo
    por request. Sin un aviso, sigue sirviendo el registro viejo hasta reiniciar el
    contenedor.

    Para dejarlo funcional (trabajo futuro):
      1. Agregar endpoint `POST /models/reload` en dockerfiles/fastapi/app.py que
         haga `REGISTRY.data = REGISTRY._load()` (releer el JSON de disco) y apuntar
         este POST ahí.
      2. Para que el cambio sea VISIBLE en /predict y /models/status (que solo leen
         champions), agregar además una task de promoción candidate->champion en
         este DAG; hoy solo se registran candidatos, el champion no cambia solo.
    Se deja en el flujo a propósito como recordatorio de esta deuda técnica.
    """
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


def promote_champions(**context):
    """Task: evaluar cada candidato entrenado vs el champion y promover si es mejor.

    Usa lifecycle.promotion del config (min_improvement=0.0 => promueve si el
    candidato es al menos tan bueno como el champion; si no hay champion, bootstrap).
    Loggea UNA línea por ticker indicando si se promovió o no y el porqué.
    """
    import sys
    sys.path.insert(0, '/opt/airflow')
    from pathlib import Path
    from src.lifecycle.registry import ModelRegistry
    from src.lifecycle.promotion import evaluate_and_promote
    from src.utils import load_yaml

    run_dir = context['task_instance'].xcom_pull(task_ids='train_e1_models', key='run_dir')
    if not run_dir:
        return "No run_dir disponible; se omite promoción"

    root = Path("/opt/airflow")
    config = load_yaml(root / "src/config/base.yaml")
    promo_cfg = config.get("lifecycle", {}).get("promotion", {})
    registry = ModelRegistry(root / "models" / "registry.json")

    promoted, kept = [], []
    print("=" * 60)
    print("PROMOCIÓN E1: candidato vs champion")
    print("=" * 60)
    for ticker_dir in sorted(Path(run_dir).iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        try:
            d = evaluate_and_promote(registry, "e1", ticker, promo_cfg)
            scores = f"cand={d.candidate_score:.4f} vs champ={d.champion_score:.4f}"
            if d.promoted:
                print(f"  ★ PROMOVIDO   {ticker}: {d.reason} [{scores}, mejora={d.improvement_pct * 100:+.1f}%]")
                promoted.append(ticker)
            else:
                print(f"  ↳ NO promovido {ticker}: {d.reason} [{scores}]")
                kept.append(ticker)
        except Exception as e:
            print(f"  ⚠️  Error evaluando {ticker}: {e}")
    print("-" * 60)
    print(f"Resumen: {len(promoted)} promovidos, {len(kept)} mantenidos")
    return f"Promovidos {len(promoted)}, mantenidos {len(kept)}"


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

task_promote = PythonOperator(
    task_id='promote_champions',
    python_callable=promote_champions,
    dag=dag,
)

task_notify = PythonOperator(
    task_id='notify_api',
    python_callable=notify_api_model_ready,
    dag=dag,
)

# Flujo del DAG: Download → Clean → Train (registra candidato + guardrails) → Promote → Notify
task_download >> task_clean >> task_train >> task_promote >> task_notify
