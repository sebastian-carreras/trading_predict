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
import pandas as pd

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
    # Solo manual: E3 está "needs rework" (Sharpe negativo, no le gana a los
    # costos intradiarios — ver PORTFOLIO.md). No tiene sentido gastarle cómputo
    # a diario ni exponerlo a colisiones de registry.json con E1/E2 mientras no
    # esté en uso. Disparar a mano: `airflow dags trigger e3_intraday_pipeline`.
    schedule_interval=None,
    catchup=False,
    tags=['trading', 'e3', 'intraday', 'lstm', 'ensemble'],
    params={
        "tickers": "",
        # False = descargar a staging sin usarlo; True = refrescar datos canónicos y entrenar con ellos
        "train_with_new_data": "False",
    },
)


def _parse_tickers_param(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [t.strip() for t in value.split(",") if t.strip()]
    if isinstance(value, (list, tuple)):
        return [str(t).strip() for t in value if str(t).strip()]
    return []


def _as_bool(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def download_intraday_data(**context):
    """Task 1: Descargar datos 5-min."""
    import sys
    sys.path.insert(0, '/opt/airflow')

    from src.e3.intraday_data import download_ohlcv_5m
    from src.data.ingest import resolve_download_settings
    from src.utils import load_yaml
    from pathlib import Path

    root = Path("/opt/airflow")
    config = load_yaml(root / "src/config/base.yaml")

    print("[E3][download] Iniciando descarga intraday...")

    # Tickers E3 (params > dag_run.conf > base.yaml)
    dag_run = context.get("dag_run")
    conf = dag_run.conf if dag_run else {}
    params = context.get("params", {})
    tickers = _parse_tickers_param(conf.get("tickers") or params.get("tickers"))
    if not tickers:
        tickers = list(
            config.get("universe", {})
            .get("tickers_by_strategy", {})
            .get("e3_intraday", [])
        )
    if not tickers:
        raise ValueError("No se especificaron tickers para E3")

    # train_with_new_data controla si los datos recién descargados se USAN para
    # entrenar. Por defecto (False) se descargan a un directorio de staging y los
    # datos canónicos (data/raw/intraday) que consume el entrenamiento NO se tocan.
    train_with_new_data = _as_bool(conf.get("train_with_new_data") if conf else None) \
        or _as_bool(params.get("train_with_new_data"))
    canonical_dir = root / "data/raw/intraday"
    staging_dir = root / "data/raw/intraday_incoming"
    out_dir = canonical_dir if train_with_new_data else staging_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Config-driven: period + incremental desde data.download (base.yaml), igual que la CLI.
    dl = resolve_download_settings(config, granularity="intraday")

    print(f"[E3][download] Tickers seleccionados: {tickers}")
    print(
        f"[E3][download] train_with_new_data={train_with_new_data} -> destino={out_dir} "
        f"({'se usará para entrenar' if train_with_new_data else 'STAGING: no se usará para entrenar'})"
    )

    # Download all tickers at once
    try:
        written_paths = download_ohlcv_5m(
            tickers=tickers,
            out_dir=out_dir,
            period=dl["period"],
            interval="5m",
            incremental=dl["incremental"],
            overlap_days=dl["overlap_days"],
        )
        downloaded = [p.stem.replace("_5m", "") for p in written_paths]
        print(f"✓ Downloaded {len(downloaded)} tickers: {downloaded}")
    except Exception as e:
        print(f"⚠️ Error descargando tickers: {e}")
        downloaded = []

    context['task_instance'].xcom_push(key='tickers_selected', value=tickers)
    context['task_instance'].xcom_push(key='tickers_downloaded', value=downloaded)
    context['task_instance'].xcom_push(key='train_with_new_data', value=train_with_new_data)
    context['task_instance'].xcom_push(key='download_dir', value=str(out_dir))
    return f"Descargados {len(downloaded)} tickers 5-min a {out_dir.name} (train_with_new_data={train_with_new_data})"


def clean_intraday_data(**context):
    """Task 2: Limpiar datos 5-min."""
    import sys
    sys.path.insert(0, '/opt/airflow')

    from src.utils import load_yaml
    from pathlib import Path

    print("[E3][clean] Iniciando limpieza intraday...")

    root = Path("/opt/airflow")
    config = load_yaml(root / "src/config/base.yaml")

    tickers_selected = context['task_instance'].xcom_pull(task_ids='download_intraday', key='tickers_selected')
    tickers_downloaded = context['task_instance'].xcom_pull(task_ids='download_intraday', key='tickers_downloaded')
    tickers = tickers_downloaded or tickers_selected
    if not tickers:
        return "No tickers to clean"

    raw_dir = root / "data/raw/intraday"
    clean_dir = root / "data/clean/intraday"
    clean_dir.mkdir(parents=True, exist_ok=True)

    required = {"open", "high", "low", "close", "volume"}
    cleaned = []
    skipped = []

    print(f"[E3][clean] Tickers a limpiar: {tickers}")
    print(f"[E3][clean] Directorio clean: {clean_dir}")

    for ticker in tickers:
        raw_path = raw_dir / f"{ticker}_5m.csv"
        if not raw_path.exists():
            print(f"⚠️  [E3][clean] Raw no encontrado: {raw_path.name}")
            skipped.append(ticker)
            continue

        try:
            df = pd.read_csv(raw_path)
            if "timestamp" not in df.columns:
                df = df.rename(columns={df.columns[0]: "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601', utc=True)
            df = df.sort_values("timestamp")
            df = df.drop_duplicates(subset=["timestamp"], keep="last")

            # Normalizar columnas
            rename_map = {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
            df = df.rename(columns=rename_map)

            missing = required.difference(df.columns)
            if missing:
                print(f"⚠️  [E3][clean] Faltan columnas {sorted(missing)} en {ticker}")
                skipped.append(ticker)
                continue

            # Limpieza simple: forward-fill y drop NaNs residuales
            df = df.set_index("timestamp")
            df["volume"] = df["volume"].fillna(0)
            df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill()
            df = df.dropna(subset=["open", "high", "low", "close", "volume"])

            out_path = clean_dir / f"{ticker}_5m.csv"
            df.reset_index().to_csv(out_path, index=False)
            cleaned.append(ticker)
            print(f"✓ [E3][clean] Limpiado {ticker}: {len(df)} filas")
        except Exception as exc:
            print(f"⚠️  [E3][clean] Error limpiando {ticker}: {exc}")
            skipped.append(ticker)

    context['task_instance'].xcom_push(key='tickers_cleaned', value=cleaned)
    return f"Limpiados {len(cleaned)} tickers; skipped={len(skipped)}"


def train_e3_with_mlflow(**context):
    """Task 3: Entrenar ensemble E3 con MLflow."""
    import sys
    sys.path.insert(0, '/opt/airflow')

    from src.e3.train_pipeline import run_for_ticker
    from src.utils import load_yaml
    from pathlib import Path

    mlflow.set_experiment("E3_Intraday_Strategy")

    root = Path("/opt/airflow")
    config = load_yaml(root / "src/config/base.yaml")

    # Único flag que decide si se entrena con datos nuevos:
    #   False (default) -> usa la ventana congelada en config (data.training_window.intraday)
    #   True            -> extiende la ventana hasta hoy (use_latest_data) y entrena
    #                      con los datos frescos que la descarga dejó en data/raw/intraday
    train_with_new_data = bool(context['task_instance'].xcom_pull(
        task_ids='download_intraday', key='train_with_new_data'
    ))

    tickers_cleaned = context['task_instance'].xcom_pull(task_ids='clean_intraday', key='tickers_cleaned')
    tickers_downloaded = context['task_instance'].xcom_pull(task_ids='download_intraday', key='tickers_downloaded')
    tickers_selected = context['task_instance'].xcom_pull(task_ids='download_intraday', key='tickers_selected')
    tickers = tickers_cleaned or tickers_downloaded or tickers_selected
    if not tickers:
        return "No tickers to train"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = root / "runs/e3_intraday" / timestamp
    out_base.mkdir(parents=True, exist_ok=True)

    # Set MLflow environment variables for the training pipeline
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "E3_Intraday_Strategy"

    e3_cfg = config.get("strategies", {}).get("e3_intraday", {})
    model_cfg = e3_cfg.get("model", {})

    def _sanitize(v):
        """NaN -> 0, Inf -> ±1e8 para que MLflow no rechace la métrica."""
        import math
        if isinstance(v, (int, float)):
            if math.isnan(v):
                return 0.0
            if math.isinf(v):
                return 1e8 if v > 0 else -1e8
        return v

    results = []
    print(f"[E3][train] Tickers a entrenar: {tickers}")
    for ticker in tickers:
        ticker_out = out_base / ticker
        print(f"\n{'='*60}")
        print(f"Training E3 ensemble for {ticker}")
        print(f"{'='*60}")

        # Un run MLflow por ticker (alineado con E1/E2): así el modelo y los
        # backtests/predicciones quedan como artifacts en S3, no solo en disco.
        with mlflow.start_run(run_name=f"E3_{ticker}_{timestamp}"):
            mlflow.log_params({
                "strategy": "e3_intraday",
                "ticker": ticker,
                "model_type": "LSTM_Ensemble",
                "lookback_bars": e3_cfg.get("lookback_bars", 96),
                "horizon_bars": e3_cfg.get("horizon_bars", 6),
                "ensemble_size": model_cfg.get("ensemble_size", 3),
                "timestamp": timestamp,
                "train_with_new_data": train_with_new_data,
            })

            try:
                summary = run_for_ticker(
                    config=config,
                    ticker=ticker,
                    raw_dir=root / "data/clean/intraday",
                    out_dir=ticker_out,
                    use_latest_data=train_with_new_data,
                )

                print(f"\n✓ {ticker} complete:")
                print(f"  MAE: {summary.get('ml_mae', 0):.6f}")
                print(f"  RMSE: {summary.get('ml_rmse', 0):.6f}")
                print(f"  IC: {summary.get('ml_ic', 0):.4f}")
                print(f"  Dir Acc: {summary.get('ml_directional_accuracy', 0):.3f}")
                print(f"  Profit Factor: {summary.get('tr_profit_factor', 0):.3f}")
                print(f"  Max DD: {summary.get('tr_max_drawdown', 0):.3%}")

                # Métricas ML + trading a MLflow
                mlflow.log_metrics({
                    "mae": _sanitize(summary.get("ml_mae", 0)),
                    "rmse": _sanitize(summary.get("ml_rmse", 0)),
                    "ic": _sanitize(summary.get("ml_ic", 0)),
                    "directional_accuracy": _sanitize(summary.get("ml_directional_accuracy", 0)),
                    "tr_profit_factor": _sanitize(summary.get("tr_profit_factor", 0)),
                    "tr_sharpe": _sanitize(summary.get("tr_sharpe", 0)),
                    "tr_max_drawdown": _sanitize(summary.get("tr_max_drawdown", 0)),
                })

                # Modelo (ensemble) como artifact
                model_path = ticker_out / f"{ticker}_model.pth"
                if model_path.exists():
                    mlflow.log_artifact(str(model_path), artifact_path="models")

                # Backtests / predicciones como artifacts
                for csv_path in sorted(ticker_out.glob("*.csv")):
                    mlflow.log_artifact(str(csv_path), artifact_path="backtest")

                results.append({
                    "ticker": ticker,
                    "status": "success",
                    "profit_factor": summary.get("tr_profit_factor"),
                    "ic": summary.get("ml_ic"),
                })

            except Exception as e:
                mlflow.log_param("error", str(e))
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
        mlflow.log_param("train_with_new_data", train_with_new_data)
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

    run_dir = context['task_instance'].xcom_pull(task_ids='train_e3_ensemble', key='run_dir')
    if not run_dir:
        return "No run_dir disponible; se omite promoción"

    root = Path("/opt/airflow")
    config = load_yaml(root / "src/config/base.yaml")
    promo_cfg = config.get("lifecycle", {}).get("promotion", {})
    registry = ModelRegistry(root / "models" / "registry.json")

    promoted, kept = [], []
    print("=" * 60)
    print("PROMOCIÓN E3: candidato vs champion")
    print("=" * 60)
    for ticker_dir in sorted(Path(run_dir).iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        try:
            d = evaluate_and_promote(registry, "e3", ticker, promo_cfg)
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
    task_id='download_intraday',
    python_callable=download_intraday_data,
    dag=dag,
)

task_clean = PythonOperator(
    task_id='clean_intraday',
    python_callable=clean_intraday_data,
    dag=dag,
)

task_train = PythonOperator(
    task_id='train_e3_ensemble',
    python_callable=train_e3_with_mlflow,
    dag=dag,
)

task_promote = PythonOperator(
    task_id='promote_champions',
    python_callable=promote_champions,
    dag=dag,
)

# Flujo
task_download >> task_clean >> task_train >> task_promote
