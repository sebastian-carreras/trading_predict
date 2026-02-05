"""
DAG de Airflow para Re-calibración Mensual de E4 Pairs Trading

Objetivo: Re-estimar parámetros OU y validar cointegración mensualmente
para detectar breakdown de relaciones y pausar pares inestables.

Flujo:
1. Cargar pares activos existentes
2. Re-validar cointegración (p-value < 0.05)
3. Re-estimar parámetros OU (θ, μ, σ)
4. Calcular nuevos half-life
5. Detectar breakdown (p-value > 0.05 o half-life > 30 días)
6. Actualizar lista de pares activos/pausados
7. Registrar cambios en MLflow
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import mlflow
import os

# Configuración MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

default_args = {
    'owner': 'trading_predict',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 15),  # Primer recalibración el 15 de cada mes
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
}

dag = DAG(
    'e4_monthly_recalibration',
    default_args=default_args,
    description='Re-calibración mensual de parámetros OU y validación de cointegración',
    schedule_interval='0 5 15 * *',  # Día 15 de cada mes a las 5 AM
    catchup=False,
    tags=['trading', 'e4', 'pairs', 'recalibration', 'maintenance'],
)


def download_latest_data(**context):
    """Task 1: Descargar últimos datos para pares activos."""
    import sys
    sys.path.insert(0, '/opt/airflow')
    
    from src.data.download_daily import download_daily_ohlcv
    from src.utils import load_yaml
    from pathlib import Path
    
    root = Path("/opt/airflow")
    config = load_yaml(root / "src/config/base.yaml")
    e4_config = config.get("strategies", {}).get("e4_pairs", {})
    
    # Obtener todos los tickers de los pares
    pairs = e4_config.get("pairs", [])
    tickers = set()
    for pair in pairs:
        tickers.add(pair[0])
        tickers.add(pair[1])
    
    tickers = sorted(list(tickers))
    
    print(f"Descargando últimos datos para {len(tickers)} tickers")
    
    out_dir = root / "data/raw/daily"
    
    # Forzar descarga (sin skip_existing) para tener datos frescos
    written = download_daily_ohlcv(
        tickers,
        out_dir=out_dir,
        period="5y",  # Últimos 5 años para recalibración
        skip_existing=False,  # Forzar actualización
    )
    
    return f"Actualizados {len(written)} archivos"


def clean_latest_data(**context):
    """Task 2: Limpiar datos actualizados."""
    import sys
    sys.path.insert(0, '/opt/airflow')
    
    from src.data.clean_daily import process_daily_data_with_cleaning
    from pathlib import Path
    
    root = Path("/opt/airflow")
    raw_dir = root / "data/raw/daily"
    clean_dir = root / "data/clean"
    
    reports = process_daily_data_with_cleaning(
        raw_dir=raw_dir,
        clean_dir=clean_dir,
        strategy="forward_fill",
        min_days=252,
        remove_zero_volume=True,
        verbose=True,
    )
    
    cleaned = sum(1 for r in reports.values() if r.get("status") in ["ok", "cleaned"])
    return f"Limpiados {cleaned} archivos"


def recalibrate_ou_parameters(**context):
    """Task 3: Re-calibrar parámetros OU para cada par."""
    import sys
    sys.path.insert(0, '/opt/airflow')
    
    from src.pairs.select_pairs import test_pair_cointegration, load_pair_prices
    from src.pairs.build_spread import build_pair_features
    from src.pairs.ou_process import estimate_ou_parameters, test_stationarity
    from src.utils import load_yaml
    from pathlib import Path
    import pandas as pd
    import json
    
    mlflow.set_experiment("E4_Monthly_Recalibration")
    
    root = Path("/opt/airflow")
    config = load_yaml(root / "src/config/base.yaml")
    e4_config = config.get("strategies", {}).get("e4_pairs", {})
    data_dir = root / "data/clean"
    
    pairs = e4_config.get("pairs", [])
    pairs = [(p[0], p[1]) for p in pairs]
    
    results = []
    active_pairs = []
    paused_pairs = []
    
    with mlflow.start_run(run_name=f"Recalibration_{datetime.now().strftime('%Y%m')}"):
        mlflow.log_param("calibration_date", datetime.now().strftime('%Y-%m-%d'))
        mlflow.log_param("total_pairs", len(pairs))
        
        for ticker_a, ticker_b in pairs:
            pair_name = f"{ticker_a}-{ticker_b}"
            print(f"\nRecalibrando {pair_name}...")
            
            try:
                # Cargar datos
                price_a, price_b = load_pair_prices(data_dir, ticker_a, ticker_b)
                
                # 1. Re-validar cointegración
                coint_result = test_pair_cointegration(price_a, price_b)
                is_cointegrated = coint_result['is_cointegrated']
                coint_pvalue = coint_result['pvalue']
                
                # 2. Construir spread
                spread_features = build_pair_features(
                    price_a, price_b,
                    beta_window=120,
                    zscore_window=60,
                )
                spread = spread_features['spread']
                
                # 3. Re-estimar parámetros OU
                ou_params = estimate_ou_parameters(spread)
                theta = ou_params['theta']
                mu = ou_params['mu']
                sigma = ou_params['sigma']
                half_life = ou_params['half_life']
                
                # 4. Test de estacionariedad
                stationarity = test_stationarity(spread)
                is_stationary = stationarity['is_stationary']
                
                # 5. Detectar breakdown
                max_pvalue = e4_config.get('filters', {}).get('cointegration_pvalue_max', 0.05)
                max_half_life = e4_config.get('filters', {}).get('half_life_days_max', 20)
                extended_max_half_life = 30  # Para recalibración, más permisivo
                
                breakdown = (
                    not is_cointegrated or
                    coint_pvalue > max_pvalue or
                    half_life > extended_max_half_life or
                    not is_stationary
                )
                
                status = "PAUSED" if breakdown else "ACTIVE"
                
                if breakdown:
                    paused_pairs.append(pair_name)
                    reason = []
                    if not is_cointegrated:
                        reason.append("no_cointegration")
                    if coint_pvalue > max_pvalue:
                        reason.append(f"high_pvalue={coint_pvalue:.4f}")
                    if half_life > extended_max_half_life:
                        reason.append(f"long_half_life={half_life:.1f}")
                    if not is_stationary:
                        reason.append("not_stationary")
                    
                    print(f"⚠️  {pair_name}: {status} - {', '.join(reason)}")
                else:
                    active_pairs.append(pair_name)
                    print(f"✓ {pair_name}: {status} - half_life={half_life:.1f}d, pvalue={coint_pvalue:.4f}")
                
                # Registrar resultados
                result = {
                    'pair': pair_name,
                    'ticker_a': ticker_a,
                    'ticker_b': ticker_b,
                    'status': status,
                    'cointegration_pvalue': coint_pvalue,
                    'is_cointegrated': is_cointegrated,
                    'is_stationary': is_stationary,
                    'ou_theta': theta,
                    'ou_mu': mu,
                    'ou_sigma': sigma,
                    'half_life_days': half_life,
                    'calibration_date': datetime.now().strftime('%Y-%m-%d'),
                }
                results.append(result)
                
            except Exception as e:
                print(f"✗ {pair_name}: ERROR - {e}")
                results.append({
                    'pair': pair_name,
                    'ticker_a': ticker_a,
                    'ticker_b': ticker_b,
                    'status': 'ERROR',
                    'error': str(e),
                })
                paused_pairs.append(pair_name)
        
        # Guardar resultados
        results_dir = root / "runs/e4_pairs/recalibration"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        df_results = pd.DataFrame(results)
        csv_path = results_dir / f"recalibration_{datetime.now().strftime('%Y%m%d')}.csv"
        df_results.to_csv(csv_path, index=False)
        
        # Guardar lista de pares activos/pausados
        status_file = results_dir / "pairs_status_current.json"
        status_data = {
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'active_pairs': active_pairs,
            'paused_pairs': paused_pairs,
            'active_count': len(active_pairs),
            'paused_count': len(paused_pairs),
        }
        
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        # Log en MLflow
        mlflow.log_metric("active_pairs", len(active_pairs))
        mlflow.log_metric("paused_pairs", len(paused_pairs))
        mlflow.log_metric("breakdown_rate", len(paused_pairs) / len(pairs) if pairs else 0)
        # Log targets configurados (para referencia en summary)
        max_pvalue = e4_config.get('filters', {}).get('cointegration_pvalue_max', 0.05)
        max_half_life = e4_config.get('filters', {}).get('half_life_days_max', 20)
        mlflow.log_param("cointegration_pvalue_max", float(max_pvalue))
        mlflow.log_param("half_life_days_max", float(max_half_life))
        mlflow.log_artifact(str(csv_path))
        mlflow.log_artifact(str(status_file))
        
        # Calcular métricas agregadas de pares activos
        active_results = df_results[df_results['status'] == 'ACTIVE']
        if len(active_results) > 0:
            mlflow.log_metric("avg_half_life", active_results['half_life_days'].mean())
            mlflow.log_metric("avg_coint_pvalue", active_results['cointegration_pvalue'].mean())
        
        context['task_instance'].xcom_push(key='active_pairs', value=active_pairs)
        context['task_instance'].xcom_push(key='paused_pairs', value=paused_pairs)
        context['task_instance'].xcom_push(key='results_csv', value=str(csv_path))
    
    return f"Recalibración completa: {len(active_pairs)} activos, {len(paused_pairs)} pausados"


def send_recalibration_alert(**context):
    """Task 4: Enviar alerta si hay breakdown significativo."""
    import sys
    sys.path.insert(0, '/opt/airflow')
    
    active_pairs = context['task_instance'].xcom_pull(
        task_ids='recalibrate_ou_parameters',
        key='active_pairs'
    )
    
    paused_pairs = context['task_instance'].xcom_pull(
        task_ids='recalibrate_ou_parameters',
        key='paused_pairs'
    )
    
    total_pairs = len(active_pairs) + len(paused_pairs)
    breakdown_rate = len(paused_pairs) / total_pairs if total_pairs > 0 else 0
    
    # Alert si más del 30% de pares tienen breakdown
    if breakdown_rate > 0.30:
        alert_msg = f"""
        ⚠️  ALERTA: Alta tasa de breakdown en pairs trading
        
        Pares pausados: {len(paused_pairs)}/{total_pairs} ({breakdown_rate:.1%})
        
        Pares con problemas:
        {chr(10).join('- ' + p for p in paused_pairs)}
        
        Acción requerida: Revisar condiciones de mercado y considerar
        rebalanceo del portfolio de pares.
        """
        
        print(alert_msg)
        
        # Aquí se podría integrar con Slack, email, etc.
        # Por ahora solo log
        
        return f"ALERT: {breakdown_rate:.1%} breakdown rate"
    else:
        return f"OK: {breakdown_rate:.1%} breakdown rate (normal)"


# ============================
# Definir Tasks
# ============================

task_download = PythonOperator(
    task_id='download_latest_data',
    python_callable=download_latest_data,
    dag=dag,
)

task_clean = PythonOperator(
    task_id='clean_latest_data',
    python_callable=clean_latest_data,
    dag=dag,
)

task_recalibrate = PythonOperator(
    task_id='recalibrate_ou_parameters',
    python_callable=recalibrate_ou_parameters,
    dag=dag,
)

task_alert = PythonOperator(
    task_id='send_recalibration_alert',
    python_callable=send_recalibration_alert,
    dag=dag,
)

# ============================
# Definir Flujo
# ============================

task_download >> task_clean >> task_recalibrate >> task_alert
