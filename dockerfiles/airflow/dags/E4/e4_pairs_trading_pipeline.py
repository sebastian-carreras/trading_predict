"""
DAG de Airflow para Estrategia E4 - Pairs Trading (Cointegración + k-NN)

Flujo:
1. Descargar datos diarios para todos los activos en pares
2. Limpiar datos (forward fill, remove zero volume)
3. Validar cointegración de pares
4. Construir spreads y calcular parámetros OU
5. Entrenar k-NN (opcional) para confirmación
6. Generar señales y backtest
7. Registrar métricas en MLflow
8. Guardar resultados por par
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
    'start_date': datetime(2026, 1, 9),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'e4_pairs_trading_pipeline',
    default_args=default_args,
    description='Pipeline E4: Pairs Trading con descubrimiento automático y cointegración',
    schedule_interval='0 4 * * 1',  # Lunes 4 AM (después de E1 y E2)
    catchup=False,
    tags=['trading', 'e4', 'pairs', 'cointegration', 'market-neutral', 'discovery'],
    params={
        'use_discovery': 'True',  # Activar descubrimiento automático
        'universe': '',  # Universo custom (vacío = usar config)
        'use_knn': 'True',  # Activar confirmación k-NN
        'entry_z': '2.0',   # Umbral de entrada (z-score)
        'exit_z': '0.25',   # Umbral de salida
        'stop_z': '3.0',    # Stop-loss
        'output_tag': '',   # Tag custom para el run (vacío = timestamp)
    },
)


def _as_bool(value) -> bool:
    """Convierte valor a bool."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _as_float(value, default: float) -> float:
    """Convierte valor a float con default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def download_pairs_data(**context):
    """Task 1: Descargar datos para el universo de tickers."""
    import sys
    sys.path.insert(0, '/opt/airflow')
    
    from src.data.download_daily import download_daily_ohlcv
    from src.utils import load_yaml
    from pathlib import Path
    
    root = Path("/opt/airflow")
    config = load_yaml(root / "src/config/base.yaml")
    e4_config = config.get("strategies", {}).get("e4_pairs", {})
    
    # Determinar si usar descubrimiento automático
    use_discovery = _as_bool(context['params'].get('use_discovery'))
    
    if use_discovery or e4_config.get("discovery", {}).get("enabled", False):
        # Modo descubrimiento: descargar universo completo
        tickers = e4_config.get("universe", [])
        print(f"Modo descubrimiento: descargando {len(tickers)} tickers del universo")
    else:
        # Modo manual: usar pares fijos
        pairs = e4_config.get("pairs", [])
        pairs = [(p[0], p[1]) for p in pairs]
        
        # Extraer tickers únicos
        tickers = set()
        for ticker_a, ticker_b in pairs:
            tickers.add(ticker_a)
            tickers.add(ticker_b)
        tickers = sorted(list(tickers))
        print(f"Modo manual: descargando {len(tickers)} tickers de {len(pairs)} pares fijos")
    
    out_dir = root / "data/raw/daily"
    
    written = download_daily_ohlcv(
        tickers,
        out_dir=out_dir,
        period="10y",  # Necesitamos histórico largo para cointegración
        skip_existing=True,
        min_days_fresh=1,
    )
    
    # Extraer tickers exitosamente descargados de los archivos creados
    tickers_downloaded = sorted([p.stem.replace("_daily", "") for p in written])
    
    context['task_instance'].xcom_push(key='files_downloaded', value=len(written))
    context['task_instance'].xcom_push(key='tickers_downloaded', value=tickers_downloaded)
    context['task_instance'].xcom_push(key='use_discovery', value=use_discovery)
    
    return f"Descargados {len(written)} archivos para {len(tickers_downloaded)} tickers (de {len(tickers)} solicitados)"


def clean_pairs_data(**context):
    """Task 2: Limpiar datos de los activos."""
    import sys
    sys.path.insert(0, '/opt/airflow')
    
    from src.data.clean_daily import process_daily_data_with_cleaning
    from pathlib import Path
    
    root = Path("/opt/airflow")
    raw_dir = root / "data/raw/daily"
    clean_dir = root / "data/clean"
    
    print("Ejecutando limpieza de datos para pairs trading...")
    reports = process_daily_data_with_cleaning(
        raw_dir=raw_dir,
        clean_dir=clean_dir,
        strategy="forward_fill",
        min_days=252,  # Mínimo 1 año para cointegración
        remove_zero_volume=True,
        verbose=True,
    )
    
    cleaned_count = sum(1 for r in reports.values() if r.get("status") in ["ok", "cleaned"])
    rejected_count = sum(1 for r in reports.values() if r.get("status") == "rejected")
    
    context['task_instance'].xcom_push(key='cleaned_tickers', value=cleaned_count)
    context['task_instance'].xcom_push(key='rejected_tickers', value=rejected_count)
    
    return f"Limpiados {cleaned_count} tickers, rechazados {rejected_count}"


def discover_cointegrated_pairs(**context):
    """Task 3: Descubrir automáticamente pares cointegrados y registrar en MLflow."""
    import sys
    sys.path.insert(0, '/opt/airflow')
    
    from src.e4.pairs.discover_pairs import discover_cointegrated_pairs, filter_best_pairs, pairs_to_list
    from src.utils import load_yaml
    from pathlib import Path
    import pandas as pd
    
    mlflow.set_experiment("E4_Pairs_Discovery")
    
    root = Path("/opt/airflow")
    config = load_yaml(root / "src/config/base.yaml")
    e4_config = config.get("strategies", {}).get("e4_pairs", {})
    
    # Verificar si usar descubrimiento
    use_discovery = context['task_instance'].xcom_pull(
        task_ids='download_pairs_data',
        key='use_discovery'
    )
    
    if not use_discovery:
        # Modo manual: usar pares fijos
        pairs = e4_config.get("pairs", [])
        pairs = [(p[0], p[1]) for p in pairs]
        
        print(f"Modo manual: usando {len(pairs)} pares fijos del config")
        
        context['task_instance'].xcom_push(key='cointegrated_pairs', value=pairs)
        context['task_instance'].xcom_push(key='total_pairs', value=len(pairs))
        
        return f"Modo manual: {len(pairs)} pares fijos"
    
    # Modo descubrimiento
    data_dir = root / "data/clean"
    
    # Usar tickers que realmente existen en data/clean (post-limpieza)
    # en lugar de confiar en tickers_downloaded que puede tener tickers fallidos
    import glob
    clean_files = glob.glob(str(data_dir / "*_daily.csv"))
    tickers = sorted([Path(f).stem.replace("_daily", "") for f in clean_files])
    
    print(f"Tickers disponibles en data/clean: {len(tickers)}")
    
    discovery_config = e4_config.get("discovery", {})
    
    with mlflow.start_run(run_name="Pair_Discovery"):
        # Log parámetros de descubrimiento
        mlflow.log_param("num_tickers", len(tickers))
        mlflow.log_param("pvalue_max", discovery_config.get("pvalue_max", 0.10))
        mlflow.log_param("half_life_max", discovery_config.get("half_life_max", 500))
        mlflow.log_param("correlation_min", discovery_config.get("correlation_min", 0.3))
        
        # Descubrir pares
        print(f"Descubriendo pares cointegrados de {len(tickers)} tickers...")
        
        pairs_df = discover_cointegrated_pairs(
            tickers,
            data_dir,
            pvalue_max=discovery_config.get("pvalue_max", 0.10),
            half_life_min=discovery_config.get("half_life_min", 5.0),
            half_life_max=discovery_config.get("half_life_max", 500.0),
            correlation_min=discovery_config.get("correlation_min", 0.3),
            method="engle-granger",
            verbose=True,
        )
        
        # Guardar todos los pares descubiertos
        results_dir = root / "runs/e4_pairs/discovery"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        discovery_csv = results_dir / f"discovered_pairs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pairs_df.to_csv(discovery_csv, index=False)
        
        # Log métricas
        mlflow.log_metric("pairs_discovered", len(pairs_df))
        
        if len(pairs_df) > 0:
            mlflow.log_metric("avg_pvalue", pairs_df['pvalue'].mean())
            mlflow.log_metric("avg_half_life", pairs_df['ou_half_life'].mean())
            mlflow.log_metric("avg_correlation", pairs_df['correlation'].mean())
            mlflow.log_metric("best_quality_score", pairs_df['quality_score'].max())
            
            # Log artifact
            mlflow.log_artifact(str(discovery_csv))
            
            # Filtrar mejores pares
            best_pairs_df = filter_best_pairs(
                pairs_df,
                max_pairs=discovery_config.get("max_pairs", 20),
                min_quality_score=discovery_config.get("min_quality_score", 0.3),
                max_half_life=discovery_config.get("half_life_max", 500.0),
            )
            
            mlflow.log_metric("pairs_selected", len(best_pairs_df))
            
            # Convertir a lista de tuplas
            pairs_list = pairs_to_list(best_pairs_df)
            
            # Guardar pares seleccionados
            selected_csv = results_dir / f"selected_pairs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            best_pairs_df.to_csv(selected_csv, index=False)
            mlflow.log_artifact(str(selected_csv))
            
            # Guardar listado de pares en MLflow
            # 1. Como JSON con detalles
            pairs_details = []
            for _, row in best_pairs_df.iterrows():
                pairs_details.append({
                    "pair": f"{row['ticker_a']}-{row['ticker_b']}",
                    "ticker_a": row['ticker_a'],
                    "ticker_b": row['ticker_b'],
                    "pvalue": float(row['pvalue']),
                    "half_life": float(row['ou_half_life']),
                    "correlation": float(row['correlation']),
                    "quality_score": float(row['quality_score'])
                })
            
            pairs_json = results_dir / "selected_pairs_list.json"
            with open(pairs_json, 'w') as f:
                json.dump(pairs_details, f, indent=2)
            mlflow.log_artifact(str(pairs_json))
            
            # 2. Como texto simple (fácil de leer en UI)
            pairs_txt = results_dir / "selected_pairs_list.txt"
            with open(pairs_txt, 'w') as f:
                f.write("PARES SELECCIONADOS PARA TRADING\n")
                f.write("=" * 80 + "\n\n")
                for i, (ta, tb) in enumerate(pairs_list, 1):
                    row = best_pairs_df[(best_pairs_df['ticker_a'] == ta) & (best_pairs_df['ticker_b'] == tb)].iloc[0]
                    f.write(f"{i:2d}. {ta:12s} - {tb:12s} | ")
                    f.write(f"p={row['pvalue']:.4f} | ")
                    f.write(f"hl={row['ou_half_life']:6.1f}d | ")
                    f.write(f"corr={row['correlation']:.3f} | ")
                    f.write(f"quality={row['quality_score']:.3f}\n")
            mlflow.log_artifact(str(pairs_txt))
            
            # 3. Como parámetro (solo los nombres)
            pairs_names = [f"{ta}-{tb}" for ta, tb in pairs_list]
            mlflow.log_param("selected_pairs", ", ".join(pairs_names[:10]))  # Primeros 10
            if len(pairs_names) > 10:
                mlflow.log_param("total_selected_pairs", len(pairs_names))
            
            context['task_instance'].xcom_push(key='cointegrated_pairs', value=pairs_list)
            context['task_instance'].xcom_push(key='total_pairs', value=len(pairs_list))
            context['task_instance'].xcom_push(key='pairs_dataframe', value=best_pairs_df.to_dict('records'))
            
            print(f"✓ Descubiertos {len(pairs_df)} pares, seleccionados {len(pairs_list)} mejores")
            
            return f"Descubiertos {len(pairs_df)} pares, seleccionados {len(pairs_list)} para trading"
        else:
            print("⚠ No se encontraron pares cointegrados")
            context['task_instance'].xcom_push(key='cointegrated_pairs', value=[])
            context['task_instance'].xcom_push(key='total_pairs', value=0)
            
            return "No se encontraron pares cointegrados"


def process_pairs_with_mlflow(**context):
    """Task 4: Procesar cada par válido y registrar en MLflow."""
    import sys
    sys.path.insert(0, '/opt/airflow')
    
    from src.e4.train_pipeline import process_pair
    from src.utils import load_yaml, ensure_dir
    from pathlib import Path
    import pandas as pd
    
    mlflow.set_experiment("E4_Pairs_Trading_Strategy")
    
    root = Path("/opt/airflow")
    config = load_yaml(root / "src/config/base.yaml")
    e4_config = config.get("strategies", {}).get("e4_pairs", {})
    
    # Obtener pares válidos desde XCom
    valid_pairs = context['task_instance'].xcom_pull(
        task_ids='discover_cointegrated_pairs',
        key='cointegrated_pairs'
    )
    
    if not valid_pairs:
        print("No hay pares cointegrados válidos para procesar")
        return "0 pares procesados"
    
    # Aplicar parámetros del DAG si fueron especificados
    params = context['params']
    if _as_bool(params.get('use_knn')):
        e4_config['knn']['enabled'] = True
    else:
        e4_config['knn']['enabled'] = False
    
    # Override de umbrales si fueron especificados
    entry_z = _as_float(params.get('entry_z'), e4_config['entry_exit']['entry_z'])
    exit_z = _as_float(params.get('exit_z'), e4_config['entry_exit']['exit_z'])
    stop_z = _as_float(params.get('stop_z'), e4_config['entry_exit']['stop_z'])
    
    e4_config['entry_exit']['entry_z'] = entry_z
    e4_config['entry_exit']['exit_z'] = exit_z
    e4_config['entry_exit']['stop_z'] = stop_z
    
    # Output tag
    output_tag = (params.get('output_tag') or '').strip()
    if not output_tag:
        output_tag = f"airflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    data_dir = root / "data/clean"
    output_dir = root / "runs/e4_pairs" / output_tag
    ensure_dir(output_dir)
    
    all_results = []
    
    for ticker_a, ticker_b in valid_pairs:
        pair_name = f"{ticker_a}-{ticker_b}"
        
        with mlflow.start_run(run_name=pair_name):
            # Log parámetros
            mlflow.log_param("pair", pair_name)
            mlflow.log_param("ticker_a", ticker_a)
            mlflow.log_param("ticker_b", ticker_b)
            mlflow.log_param("entry_z", entry_z)
            mlflow.log_param("exit_z", exit_z)
            mlflow.log_param("stop_z", stop_z)
            mlflow.log_param("use_knn", e4_config['knn']['enabled'])
            
            try:
                # Procesar par
                result = process_pair(
                    ticker_a, ticker_b, data_dir, e4_config, output_dir
                )
                
                if result is not None:
                    # Log métricas en MLflow
                    mlflow.log_metric("total_return", result['total_return'])
                    mlflow.log_metric("cagr", result['cagr'])
                    mlflow.log_metric("sharpe_ratio", result['sharpe'])
                    mlflow.log_metric("max_drawdown", result['max_drawdown'])
                    mlflow.log_metric("num_trades", result['num_trades'])
                    mlflow.log_metric("win_rate", result['win_rate'])
                    mlflow.log_metric("avg_holding_days", result['avg_holding_days'])
                    mlflow.log_metric("net_exposure_mean", result['net_exposure_mean'])
                    mlflow.log_metric("cointegration_pvalue", result['cointegration_pvalue'])
                    mlflow.log_metric("ou_half_life", result['ou_half_life'])
                    
                    # Log artifacts (archivos del par)
                    pair_dir = output_dir / f"{ticker_a}_{ticker_b}"
                    if pair_dir.exists():
                        mlflow.log_artifacts(str(pair_dir), artifact_path=f"pairs/{pair_name}")
                    
                    all_results.append(result)
                    
                    print(f"✓ {pair_name}: Sharpe={result['sharpe']:.2f}, Return={result['total_return']:.2%}")
                else:
                    print(f"✗ {pair_name}: Procesamiento fallido")
                    
            except Exception as e:
                print(f"✗ {pair_name}: ERROR - {e}")
                mlflow.log_param("error", str(e))
                continue
    
    # Guardar resumen agregado
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_csv = output_dir / "summary_all_pairs.csv"
        summary_df.to_csv(summary_csv, index=False)
        
        # Log resumen agregado en MLflow
        with mlflow.start_run(run_name="E4_Aggregate_Summary"):
            mlflow.log_metric("pairs_processed", len(all_results))
            mlflow.log_metric("avg_sharpe", summary_df['sharpe'].mean())
            mlflow.log_metric("avg_return", summary_df['total_return'].mean())
            mlflow.log_metric("avg_win_rate", summary_df['win_rate'].mean())
            mlflow.log_metric("total_trades", summary_df['num_trades'].sum())
            
            # Log targets/thresholds
            from src.utils import load_yaml
            from pathlib import Path
            root = Path("/opt/airflow")
            config = load_yaml(root / "src/config/base.yaml")
            decision_cfg = config.get("decision", {})
            default_targets = {
                'sharpe_min': 1.0,
            }
            targets = {**default_targets, **decision_cfg.get("targets", {})}
            mlflow.log_param("sharpe_target_min", targets['sharpe_min'])
            
            # Log el resumen como artifact
            mlflow.log_artifact(str(summary_csv))
        
        context['task_instance'].xcom_push(key='pairs_processed', value=len(all_results))
        context['task_instance'].xcom_push(key='avg_sharpe', value=float(summary_df['sharpe'].mean()))
        
        return f"Procesados {len(all_results)}/{len(valid_pairs)} pares - Avg Sharpe: {summary_df['sharpe'].mean():.2f}"
    else:
        return "0 pares procesados exitosamente"


def generate_report(**context):
    """Task 5: Generar reporte consolidado de resultados."""
    import sys
    sys.path.insert(0, '/opt/airflow')
    
    from pathlib import Path
    import pandas as pd
    
    root = Path("/opt/airflow")
    
    # Obtener métricas desde XCom
    pairs_processed = context['task_instance'].xcom_pull(
        task_ids='process_pairs_with_mlflow',
        key='pairs_processed'
    ) or 0
    
    avg_sharpe = context['task_instance'].xcom_pull(
        task_ids='process_pairs_with_mlflow',
        key='avg_sharpe'
    ) or 0.0
    
    # Obtener número de pares descubiertos y seleccionados
    cointegrated_pairs = context['task_instance'].xcom_pull(
        task_ids='discover_cointegrated_pairs',
        key='total_pairs'
    ) or 0
    
    # Generar reporte
    report = f"""
    ========================================
    E4 PAIRS TRADING PIPELINE - RESUMEN
    ========================================
    
    Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    DATOS:
    - Pares descubiertos y seleccionados: {cointegrated_pairs}
    
    PROCESAMIENTO:
    - Pares procesados: {pairs_processed}
    - Sharpe promedio: {avg_sharpe:.2f}
    
    PARÁMETROS:
    - Entry Z-score: {context['params'].get('entry_z', '2.0')}
    - Exit Z-score: {context['params'].get('exit_z', '0.25')}
    - Stop Z-score: {context['params'].get('stop_z', '3.0')}
    - k-NN enabled: {context['params'].get('use_knn', 'True')}
    
    ========================================
    """
    
    print(report)
    
    # Guardar reporte
    reports_dir = root / "reports/e4_pairs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = reports_dir / f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    return report


# ============================
# Definir Tasks
# ============================

task_download = PythonOperator(
    task_id='download_pairs_data',
    python_callable=download_pairs_data,
    dag=dag,
)

task_clean = PythonOperator(
    task_id='clean_pairs_data',
    python_callable=clean_pairs_data,
    dag=dag,
)

task_discover = PythonOperator(
    task_id='discover_cointegrated_pairs',
    python_callable=discover_cointegrated_pairs,
    dag=dag,
)

task_process = PythonOperator(
    task_id='process_pairs_with_mlflow',
    python_callable=process_pairs_with_mlflow,
    dag=dag,
)

task_report = PythonOperator(
    task_id='generate_report',
    python_callable=generate_report,
    dag=dag,
)

# ============================
# Definir Flujo
# ============================

task_download >> task_clean >> task_discover >> task_process >> task_report
