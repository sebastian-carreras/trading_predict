"""
DAG de Airflow para E1 Baseline - Regresión Lineal Simple

Flujo:
1. Descargar datos diarios → data/raw/daily/
2. Calcular features E1 (27 indicadores)
3. Entrenar modelo Baseline (Regresión Lineal) con MLflow tracking
4. Guardar predicciones y métricas
5. (Opcional) Comparar automáticamente con GRU

Este DAG ejecuta el baseline de Regresión Lineal para comparar contra el GRU.
Se puede ejecutar en paralelo con e1_conservative_pipeline o de forma independiente.
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
    'e1_baseline_linear_regression',
    default_args=default_args,
    description='Pipeline E1 Baseline: Regresión Lineal para comparación con GRU',
    schedule_interval='0 3 * * 1',  # Lunes 3 AM (1 hora después del GRU)
    catchup=False,
    tags=['trading', 'e1', 'baseline', 'linear_regression', 'comparison'],
    params={
        'tickers': 'YPFD.BA, GGAL.BA, PAMP.BA, BYMA.BA, CEPU.BA, AAPL, MSFT, JNJ, PG, V',  # Comma-separated, vacío = todos del config
        'auto_compare_with_gru': 'True',  # Auto-comparar con último run de GRU
    },
)


def _as_bool(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def download_daily_data(**context):
    """Task 1: Descargar datos diarios (compartido con GRU)."""
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
    else:
        # Usar todos los tickers E1 del config
        e1_tickers = list(
            config.get("universe", {})
            .get("tickers_by_strategy", {})
            .get("e1_conservative", [])
        )
        tickers = e1_tickers
    
    out_dir = root / "data/raw/daily"
    
    written = download_daily_ohlcv(
        tickers, 
        out_dir=out_dir, 
        period="10y",
        skip_existing=True,  # No re-descargar si fresco
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
    
    cleaned_count = sum(1 for r in reports.values() if r.get("status") == "cleaned")
    rejected_count = sum(1 for r in reports.values() if r.get("status") == "rejected")
    
    context['task_instance'].xcom_push(key='cleaned_tickers', value=cleaned_count)
    context['task_instance'].xcom_push(key='rejected_tickers', value=rejected_count)
    
    return f"Limpiados {cleaned_count} tickers, rechazados {rejected_count}"


def train_baseline_with_mlflow(**context):
    """Task 2: Entrenar Baseline (Regresión Lineal) con tracking MLflow."""
    import sys
    sys.path.insert(0, '/opt/airflow')
    
    from src.e1.train_baseline import run_baseline_for_ticker
    from src.utils import load_yaml
    from pathlib import Path
    import pandas as pd
    
    mlflow.set_experiment("E1_Baseline_LinearRegression")
    
    root = Path("/opt/airflow")
    config = load_yaml(root / "src/config/base.yaml")
    
    # Obtener tickers del XCom (pasados desde download_daily_data)
    tickers_from_download = context['task_instance'].xcom_pull(
        task_ids='download_daily_data', 
        key='tickers_to_train'
    )
    
    if tickers_from_download:
        # Filtrar solo los tickers de E1
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
        return "No tickers to train for E1 baseline"
    
    raw_dir = root / "data/clean"  # Baseline usa datos limpios
    
    # Output dir con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "runs/e1_baseline" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar config usada
    import yaml
    with open(out_dir / "config_used.yaml", "w") as f:
        yaml.dump(config, f)
    
    results = []
    for ticker in tickers:
        with mlflow.start_run(run_name=f"E1_Baseline_{ticker}_{timestamp}"):
            # Extraer configuración de E1
            e1_config = config["strategies"]["e1_conservative"]
            
            # Log parámetros de estrategia
            mlflow.log_params({
                "strategy": "e1_conservative",
                "model": "LinearRegression_Baseline",
                "ticker": ticker,
                "lookback_days": e1_config["lookback_days"],
                "horizon_days": e1_config["horizon_days"],
                "model_type": "sklearn.LinearRegression",
            })
            
            try:
                result = run_baseline_for_ticker(
                    config=config,
                    ticker=ticker,
                    raw_dir=raw_dir,
                    out_dir=out_dir / ticker,
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
                
                # Log artifacts
                ticker_dir = out_dir / ticker
                
                for fname in [
                    f"{ticker}_baseline_predictions.csv",
                    f"{ticker}_baseline_summary.csv",
                    f"{ticker}_baseline_backtest.csv",
                    f"{ticker}_baseline_folds.csv",
                ]:
                    artifact_path = ticker_dir / fname
                    if artifact_path.exists():
                        if "pred" in fname:
                            mlflow.log_artifact(str(artifact_path), artifact_path="predictions")
                        elif "backtest" in fname:
                            mlflow.log_artifact(str(artifact_path), artifact_path="backtest")
                        elif "folds" in fname:
                            mlflow.log_artifact(str(artifact_path), artifact_path="folds")
                        else:
                            mlflow.log_artifact(str(artifact_path), artifact_path="summary")
                
                results.append({
                    "ticker": ticker,
                    "status": "success",
                    "mae": result.get("ml_mae"),
                    "rmse": result.get("ml_rmse"),
                    "ic": result.get("ml_ic"),
                    "directional_accuracy": result.get("ml_directional_accuracy"),
                    "sharpe": result.get("bt_sharpe"),
                    "max_dd": result.get("bt_max_drawdown"),
                })
                
            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()
                mlflow.log_param("error", str(e))
                mlflow.log_text(error_traceback, "error_traceback.txt")
                results.append({
                    "ticker": ticker,
                    "status": "failed",
                    "error": str(e),
                })
                print(f"✗ Error en {ticker}: {e}")
                print(f"\n{'='*60}")
                print(f"TRACEBACK COMPLETO para {ticker}:")
                print(f"{'='*60}")
                print(error_traceback)
                print(f"{'='*60}\n")
    
    # Guardar resumen consolidado
    summary_df = pd.DataFrame(results)
    summary_path = out_dir / "baseline_summary_all.csv"
    summary_df.to_csv(summary_path, index=False)
    
    # Calcular métricas agregadas
    successful_results = [r for r in results if r["status"] == "success"]
    ic_values = [r.get("ic") for r in successful_results if r.get("ic") is not None]
    mae_values = [r.get("mae") for r in successful_results if r.get("mae") is not None]
    
    # Targets desde config
    decision_cfg = config.get("decision", {})
    default_targets = {
        'ic_min': 0.05,
        'mae_max': 0.03,
    }
    targets = {**default_targets, **decision_cfg.get("targets", {})}
    
    # Log resumen agregado en MLflow
    with mlflow.start_run(run_name=f"E1_Baseline_Summary_{timestamp}"):
        mlflow.log_artifact(str(summary_path))
        mlflow.log_metric("total_tickers", len(tickers))
        mlflow.log_metric("successful_tickers", len(successful_results))
        
        if ic_values:
            mlflow.log_metric("ic_mean", float(sum(ic_values) / len(ic_values)))
            mlflow.log_metric("ic_median", float(sorted(ic_values)[len(ic_values) // 2]))
            mlflow.log_metric("ic_positive_count", sum(1 for ic in ic_values if ic > 0))
            mlflow.log_metric("ic_above_threshold", sum(1 for ic in ic_values if ic > targets['ic_min']))
            mlflow.log_param("ic_target_min", targets['ic_min'])
        
        if mae_values:
            mlflow.log_metric("mae_mean", float(sum(mae_values) / len(mae_values)))
            mlflow.log_metric("mae_median", float(sorted(mae_values)[len(mae_values) // 2]))
            mlflow.log_param("mae_target_max", targets.get('mae_max', 0.03))
    
    context['task_instance'].xcom_push(key='run_dir', value=str(out_dir))
    context['task_instance'].xcom_push(key='successful_count', value=len(successful_results))
    
    return f"Entrenados {len(successful_results)}/{len(tickers)} modelos baseline exitosamente"


def compare_with_gru(**context):
    """Task 3: Comparar Baseline con GRU (opcional)."""
    import sys
    sys.path.insert(0, '/opt/airflow')
    
    from pathlib import Path
    import pandas as pd
    
    # Verificar si se debe ejecutar comparación
    auto_compare = _as_bool(context['params'].get('auto_compare_with_gru'))
    if not auto_compare:
        return "Comparación deshabilitada por parámetro auto_compare_with_gru=False"
    
    root = Path("/opt/airflow")
    
    # Obtener run del baseline desde XCom
    baseline_run_dir = context['task_instance'].xcom_pull(
        task_ids='train_baseline_models',
        key='run_dir'
    )
    
    if not baseline_run_dir:
        return "No se pudo obtener baseline run_dir del XCom"
    
    baseline_run_path = Path(baseline_run_dir)
    
    # Buscar último run de GRU
    gru_runs_dir = root / "runs/e1_conservative"
    if not gru_runs_dir.exists():
        return "No hay runs previos de GRU para comparar"
    
    gru_runs = sorted([d for d in gru_runs_dir.iterdir() if d.is_dir()])
    if not gru_runs:
        return "No hay runs previos de GRU para comparar"
    
    gru_run_path = gru_runs[-1]
    
    # Cargar summaries
    baseline_summary_path = baseline_run_path / "baseline_summary_all.csv"
    gru_summary_path = gru_run_path / "summary_all.csv"
    
    if not baseline_summary_path.exists():
        return f"No se encontró summary del baseline: {baseline_summary_path}"
    
    if not gru_summary_path.exists():
        return f"No se encontró summary del GRU: {gru_summary_path}"
    
    df_baseline = pd.read_csv(baseline_summary_path)
    df_gru = pd.read_csv(gru_summary_path)
    
    # Debug: mostrar columnas disponibles
    print("\n🔍 DEBUG: Columnas disponibles")
    print(f"Baseline columns: {list(df_baseline.columns)}")
    print(f"GRU columns: {list(df_gru.columns)}")
    print(f"Baseline shape: {df_baseline.shape}")
    print(f"GRU shape: {df_gru.shape}")
    
    # Comparación completa (ML + Trading)
    print("\n" + "="*80)
    print("COMPARACIÓN COMPLETA: Baseline vs GRU")
    print("="*80)
    
    # Métricas ML + Trading a comparar
    metrics_to_compare = [
        ("ml_mae", "lower_better"),
        ("ml_rmse", "lower_better"),
        ("ml_ic", "higher_better"),
        ("ml_directional_accuracy", "higher_better"),
        ("bt_sharpe", "higher_better"),
        ("bt_sortino", "higher_better"),
        ("bt_cagr", "higher_better"),
        ("bt_max_drawdown", "lower_better"),
        ("bt_calmar", "higher_better"),
        ("bt_profit_factor", "higher_better"),
        ("bt_hit_rate", "higher_better"),
    ]
    
    comparison_results = []
    gru_wins = 0
    total_compared = 0
    metrics_not_found = []
    
    for metric, direction in metrics_to_compare:
        if metric not in df_baseline.columns:
            metrics_not_found.append(f"{metric} (baseline)")
            continue
        if metric not in df_gru.columns:
            metrics_not_found.append(f"{metric} (gru)")
            continue
            
        baseline_mean = df_baseline[metric].mean()
        gru_mean = df_gru[metric].mean()
        
        # Validar valores válidos
        if pd.isna(baseline_mean) or pd.isna(gru_mean):
            print(f"⚠️  Skipping {metric}: valores NaN (baseline={baseline_mean}, gru={gru_mean})")
            continue
        
        diff = gru_mean - baseline_mean
        
        # Calcular mejora porcentual
        if baseline_mean != 0:
            if direction == "higher_better":
                improvement_pct = (diff / abs(baseline_mean)) * 100
            else:  # lower_better
                improvement_pct = (-diff / abs(baseline_mean)) * 100
        else:
            improvement_pct = 0.0
        
        # Determinar si GRU es mejor
        is_gru_better = (diff > 0 and direction == "higher_better") or \
                       (diff < 0 and direction == "lower_better")
        
        if is_gru_better:
            gru_wins += 1
        total_compared += 1
        
        comparison_results.append({
            "metric": metric,
            "baseline": baseline_mean,
            "gru": gru_mean,
            "difference": diff,
            "improvement_pct": improvement_pct,
            "gru_better": is_gru_better,
        })
        
        status = "✓" if is_gru_better else "✗"
        print(f"{metric:30} Baseline: {baseline_mean:>10.4f}  GRU: {gru_mean:>10.4f}  "
              f"Δ: {diff:>+10.4f}  {improvement_pct:>+7.2f}% {status}")
    
    # Reportar métricas no encontradas
    if metrics_not_found:
        print(f"\n⚠️  Métricas no encontradas: {', '.join(metrics_not_found)}")
    
    if total_compared == 0:
        print("\n❌ No se encontraron métricas para comparar!")
        return "No se pudieron comparar métricas (columnas no encontradas)"
    
    # Resumen de victorias
    baseline_wins = total_compared - gru_wins
    gru_win_rate = (gru_wins / total_compared * 100) if total_compared > 0 else 0
    
    print("="*80)
    print(f"RESUMEN: GRU superior en {gru_wins}/{total_compared} métricas ({gru_win_rate:.1f}%)")
    print("="*80)
    
    # Guardar comparación
    comparison_df = pd.DataFrame(comparison_results)
    comparison_path = baseline_run_path / "comparison_vs_gru.csv"
    comparison_df.to_csv(comparison_path, index=False)
    
    # Log completo a MLflow
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"\n📊 Guardando {len(comparison_results)} métricas en MLflow...")
    
    with mlflow.start_run(run_name=f"Comparison_Baseline_vs_GRU_{timestamp_str}"):
        # Tags para búsqueda y organización
        mlflow.set_tags({
            "comparison_type": "baseline_vs_gru",
            "strategy": "e1_conservative",
            "model_baseline": "LinearRegression",
            "model_gru": "GRU",
            "baseline_run": baseline_run_path.name,
            "gru_run": gru_run_path.name,
        })
        
        # Metadata y contexto
        mlflow.log_params({
            "baseline_run_dir": str(baseline_run_path),
            "gru_run_dir": str(gru_run_path),
            "baseline_timestamp": baseline_run_path.name,
            "gru_timestamp": gru_run_path.name,
            "n_tickers_compared": len(df_baseline),
            "n_metrics_compared": total_compared,
        })
        
        # Resumen de victorias
        print(f"  → Guardando resumen de victorias...")
        mlflow.log_metrics({
            "gru_wins": float(gru_wins),
            "baseline_wins": float(baseline_wins),
            "gru_win_rate": gru_win_rate,
            "total_metrics": float(total_compared),
        })
        
        # Métricas individuales con mejora porcentual
        print(f"  → Guardando {total_compared} métricas individuales...")
        for idx, row in enumerate(comparison_results):
            metric_name = row['metric']
            try:
                mlflow.log_metric(f"baseline_{metric_name}", float(row['baseline']))
                mlflow.log_metric(f"gru_{metric_name}", float(row['gru']))
                mlflow.log_metric(f"diff_{metric_name}", float(row['difference']))
                mlflow.log_metric(f"improvement_pct_{metric_name}", float(row['improvement_pct']))
                print(f"    ✓ {metric_name}")
            except Exception as e:
                print(f"    ✗ Error guardando {metric_name}: {e}")
        
        # Artifact: CSV de comparación
        print(f"  → Guardando artifact CSV...")
        mlflow.log_artifact(str(comparison_path))
        
        print(f"✅ Comparación guardada en MLflow (run: Comparison_Baseline_vs_GRU_{timestamp_str})")
    
    print("="*80)
    
    context['task_instance'].xcom_push(key='comparison_path', value=str(comparison_path))
    
    return f"Comparación completada. Guardada en: {comparison_path}"


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
    task_id='train_baseline_models',
    python_callable=train_baseline_with_mlflow,
    dag=dag,
)

task_compare = PythonOperator(
    task_id='compare_with_gru',
    python_callable=compare_with_gru,
    dag=dag,
)

# Flujo del DAG: Download → Clean → Train → Compare
task_download >> task_clean >> task_train >> task_compare
