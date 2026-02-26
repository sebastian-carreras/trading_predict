"""
Dashboard health checker — CLI tool that queries MLflow and the timing log
to produce per-strategy metrics with alert colors.

Usage:
    # Vista summary (default) — promedio y último valor por estrategia
    python -m src.dashboard.checker [--strategy e1_conservative]

    # Vista per-ticker — último entrenamiento por ticker
    python -m src.dashboard.checker --view ticker --strategy e1_conservative

    # Vista per-ticker — promedio de últimos 3 entrenamientos
    python -m src.dashboard.checker --view ticker --strategy e1_conservative --last 3

    # Vista historial — últimos 10 entrenamientos de un ticker
    python -m src.dashboard.checker --view history --strategy e1_conservative --ticker AAPL

    # Historial con menos runs
    python -m src.dashboard.checker --view history --strategy e1_conservative --ticker AAPL --last 5

The output is a colour-coded table printed to stdout **and** saved as CSV
in ``reports/dashboard/``.
"""

from __future__ import annotations  # Permite usar sintaxis de tipos moderna (ej: str | None) en Python 3.9+

import argparse   # Para parsear argumentos de línea de comandos (--strategy, --save, etc.)
import json       # Para leer el archivo de timing que está en formato JSONL
import math       # Para chequear NaN e infinitos al formatear valores numéricos
import os         # Para leer variables de entorno (MLFLOW_TRACKING_URI)
import socket     # Para chequear conectividad al servidor MLflow remoto
from datetime import datetime, timezone  # Para manejar timestamps UTC de MLflow y reportes
from pathlib import Path                 # Para manejar rutas de archivos de forma cross-platform
from typing import Any                   # Tipo genérico para diccionarios con valores mixtos
from urllib.parse import urlparse        # Para parsear la URI de MLflow y extraer host/puerto

import numpy as np    # (importado pero no usado directamente — posiblemente usado indirectamente por pandas)
import pandas as pd   # Para manipular datos tabulares (DataFrames) en todo el módulo

# Importaciones internas del proyecto
from ..utils import ensure_dir, load_yaml, project_root  # Utilidades: crear dirs, cargar YAML, raíz del proyecto
from ..lifecycle.promotion import (  # Scoring de promoción champion/challenger
    compute_score,                    # Calcula score compuesto ponderado
    get_strategy_promotion_config,    # Resuelve weights por estrategia
)
from . import (
    evaluate_alert,          # Evalúa si una métrica ML/trading está en zona verde/amarilla/roja
    evaluate_timing_alert,   # Evalúa si el tiempo de ejecución está en zona verde/amarilla/roja
    get_strategy_thresholds, # Obtiene los umbrales configurados para una estrategia específica
)


# ── Timing log reader ──────────────────────────────────────
# Lee el archivo de log de tiempos de ejecución (cuánto tarda cada entrenamiento/predicción)

def read_timing_log(log_path: Path | None = None) -> pd.DataFrame:
    """Read the JSONL timing log into a DataFrame."""
    # Si no se pasa una ruta explícita, usa la ruta por defecto del proyecto
    if log_path is None:
        log_path = project_root() / "reports" / "timing" / "timing_log.jsonl"

    # Si el archivo no existe, retorna un DataFrame vacío (no hay datos de timing aún)
    if not log_path.exists():
        return pd.DataFrame()

    # Lee el archivo JSONL línea por línea (cada línea es un JSON independiente)
    records: list[dict] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # Elimina espacios y saltos de línea
            if line:  # Ignora líneas vacías
                try:
                    records.append(json.loads(line))  # Parsea el JSON de cada línea
                except json.JSONDecodeError:
                    continue  # Si una línea está corrupta, la ignora y sigue

    # Si no se pudo leer ningún registro válido, retorna DataFrame vacío
    if not records:
        return pd.DataFrame()

    # Convierte la lista de diccionarios a DataFrame
    df = pd.DataFrame(records)

    # Convierte la columna "ended_at" a tipo datetime UTC (si existe)
    if "ended_at" in df.columns:
        df["ended_at"] = pd.to_datetime(df["ended_at"], utc=True, errors="coerce")

    return df


def timing_summary_for_strategy(
    df_timing: pd.DataFrame,
    strategy: str,
) -> dict[str, Any]:
    """Compute avg and last timing for train/predict phases."""
    result: dict[str, Any] = {}

    # Si no hay datos de timing, retorna diccionario vacío
    if df_timing.empty:
        return result

    # Filtra solo los registros de la estrategia solicitada (ej: "e1_simple", "e2_moderate")
    subset = df_timing[df_timing["strategy"] == strategy].copy()
    if subset.empty:
        return result

    # Calcula estadísticas para cada fase: "train" (entrenamiento) y "predict" (predicción)
    for phase in ("train", "predict"):
        # Filtra los registros de esta fase específica
        phase_df = subset[subset["phase"] == phase]
        if phase_df.empty:
            continue

        # Extrae las duraciones en segundos y calcula estadísticas
        durations = phase_df["duration_seconds"].astype(float)
        result[f"{phase}_avg_seconds"] = float(durations.mean())     # Promedio de duración
        result[f"{phase}_p50_seconds"] = float(durations.median())   # Mediana (percentil 50)
        # Percentil 95 — si hay solo 1 dato, usa ese valor directamente
        result[f"{phase}_p95_seconds"] = float(durations.quantile(0.95)) if len(durations) > 1 else float(durations.iloc[0])

        # Busca la última ejecución (la más reciente)
        if "ended_at" in phase_df.columns and phase_df["ended_at"].notna().any():
            # Si hay timestamps, ordena por fecha y toma la última
            last_row = phase_df.sort_values("ended_at").iloc[-1]
        else:
            # Si no hay timestamps, simplemente toma la última fila del DataFrame
            last_row = phase_df.iloc[-1]

        # Guarda la duración y timestamp de la última ejecución
        result[f"{phase}_last_seconds"] = float(last_row["duration_seconds"])
        if "ended_at" in last_row and pd.notna(last_row["ended_at"]):
            result[f"{phase}_last_at"] = str(last_row["ended_at"])

    return result


# ── MLflow reader ───────────────────────────────────────────
# Lee métricas de entrenamiento/evaluación desde el servidor MLflow
# Usa la misma lógica de fallback que el training:
#   1. Servidor remoto (MLFLOW_TRACKING_URI) si está accesible
#   2. SQLite local (runs/mlflow_local/mlflow.db)
#   3. File store por defecto (mlruns/ — solo si existe y no está corrupto)
#
# IMPORTANTE: Usa MlflowClient(tracking_uri=...) con URI explícita
# en vez de mlflow.set_tracking_uri() para evitar contaminar el estado
# global de MLflow.

# Cache del cliente MLflow resuelto (se resuelve una sola vez por ejecución)
_mlflow_client_cache: dict[str, Any] = {}


def _is_tracking_uri_reachable(uri: str, timeout_seconds: float = 1.5) -> tuple[bool, str | None]:
    """Chequeo rápido de conectividad para URIs HTTP/HTTPS.
    URIs locales (file/sqlite) se consideran siempre alcanzables."""
    parsed = urlparse(uri)
    if parsed.scheme not in {"http", "https"}:
        return True, None

    host = parsed.hostname
    if not host:
        return True, None

    if parsed.port is not None:
        port = parsed.port
    elif parsed.scheme == "https":
        port = 443
    else:
        port = 80

    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True, None
    except OSError as exc:
        return False, str(exc)


def _get_mlflow_client() -> Any | None:
    """Obtiene un MlflowClient conectado usando la cascada de fallback:
    remoto -> SQLite local -> file store.
    Cachea el resultado para no repetir la resolución en cada llamada.
    Retorna None si MLflow no está disponible."""
    # Si ya se resolvió, retorna el cache
    if "client" in _mlflow_client_cache:
        return _mlflow_client_cache["client"]

    try:
        from mlflow import MlflowClient  # type: ignore
    except ImportError:
        _mlflow_client_cache["client"] = None
        return None

    root = project_root()

    # Lista de URIs a intentar en orden de prioridad (misma cascada que train_pipeline)
    candidates: list[str] = []

    # 1. Servidor remoto si está configurado
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if tracking_uri:
        remote_timeout = float(os.getenv("MLFLOW_REMOTE_CHECK_TIMEOUT_SECONDS", "1.5"))
        reachable, _ = _is_tracking_uri_reachable(tracking_uri, remote_timeout)
        if reachable:
            candidates.append(tracking_uri)

    # 2. SQLite local (misma DB que usa el training como fallback)
    local_sqlite_db = root / "runs" / "mlflow_local" / "mlflow.db"
    if local_sqlite_db.exists():
        candidates.append(f"sqlite:///{local_sqlite_db}")

    # 3. File store por defecto (mlruns/), solo si no está corrupto
    local_mlruns = root / "mlruns"
    if local_mlruns.exists():
        has_malformed = any(
            not (exp_dir / "meta.yaml").exists()
            for exp_dir in local_mlruns.iterdir()
            if exp_dir.is_dir() and exp_dir.name.isdigit()
        )
        if not has_malformed:
            candidates.append(f"file://{local_mlruns}")

    # Intenta cada URI en orden hasta encontrar una que funcione
    for uri in candidates:
        try:
            client = MlflowClient(tracking_uri=uri)
            client.search_experiments()  # Validación rápida de que funciona
            _mlflow_client_cache["client"] = client
            return client
        except Exception:
            continue

    _mlflow_client_cache["client"] = None
    return None


def _mlflow_runs_for_experiment(
    experiment_name: str,
    max_results: int = 200,
) -> pd.DataFrame:
    """Query MLflow for runs in an experiment."""
    try:
        # Obtiene el cliente ya resuelto (con la URI correcta)
        client = _get_mlflow_client()
        if client is None:
            return pd.DataFrame()

        # Busca el experimento por nombre (ej: "E1_Conservative_Strategy")
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return pd.DataFrame()

        # Busca las ejecuciones (runs) del experimento, ordenadas por más reciente primero
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=max_results,
        )
        if not runs:
            return pd.DataFrame()

        # Convierte cada run de MLflow a un diccionario plano
        rows = []
        for run in runs:
            row: dict[str, Any] = {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name or "",
                "status": run.info.status,
                "start_time": datetime.fromtimestamp(
                    run.info.start_time / 1000, tz=timezone.utc
                )
                if run.info.start_time
                else None,
            }
            row.update(run.data.params)
            row.update(run.data.metrics)
            for k, v in run.data.tags.items():
                row[f"tag.{k}"] = v
            rows.append(row)

        return pd.DataFrame(rows)

    except Exception:
        return pd.DataFrame()


def mlflow_metrics_summary(
    experiment_name: str,
    metric_keys: list[str],
    max_results: int = 200,
) -> dict[str, Any]:
    """Compute avg and last-run values for requested metric keys."""
    # Obtiene todos los runs del experimento desde MLflow
    df = _mlflow_runs_for_experiment(experiment_name, max_results)
    if df.empty:
        return {}

    # Filtra solo los runs que terminaron exitosamente (descarta FAILED, RUNNING, etc.)
    if "status" in df.columns:
        df = df[df["status"] == "FINISHED"]
    if df.empty:
        return {}

    # Descarta runs de prueba/inicialización (tienen "_init_test" en el nombre)
    if "run_name" in df.columns:
        df = df[~df["run_name"].str.contains("_init_test", na=False)]
    if df.empty:
        return {}

    result: dict[str, Any] = {}

    # Ordena los runs por fecha de inicio (más reciente primero)
    if "start_time" in df.columns:
        df_sorted = df.sort_values("start_time", ascending=False)
    else:
        df_sorted = df

    # Busca el último run que tenga al menos una de las métricas solicitadas
    # (algunos runs pueden haber terminado sin loguear métricas)
    last = df_sorted.iloc[0]  # Por defecto, el más reciente
    for _, candidate in df_sorted.iterrows():
        has_any = any(
            key in candidate and pd.notna(candidate.get(key))
            for key in metric_keys
        )
        if has_any:
            last = candidate  # Encontramos un run con métricas válidas
            break

    # Guarda metadatos del último run válido
    result["last_run_name"] = last.get("run_name", "")
    result["last_run_time"] = str(last.get("start_time", ""))
    result["total_runs"] = len(df)  # Cantidad total de runs válidos

    # Para cada métrica solicitada, calcula promedio histórico y último valor
    for key in metric_keys:
        if key in df.columns:
            vals = df[key].dropna().astype(float)  # Descarta NaNs
            if not vals.empty:
                result[f"{key}_avg"] = float(vals.mean())                  # Promedio de todos los runs
                result[f"{key}_last"] = float(last.get(key, float("nan"))) # Valor del último run
            else:
                result[f"{key}_avg"] = float("nan")   # No hay datos → NaN
                result[f"{key}_last"] = float("nan")
        else:
            # La métrica no existe en ningún run
            result[f"{key}_avg"] = float("nan")
            result[f"{key}_last"] = float("nan")

    return result


def mlflow_ticker_metrics(
    experiment_name: str,
    max_results: int = 500,
) -> pd.DataFrame:
    """Obtiene runs per-ticker desde MLflow.

    Filtra Summary runs e init_test. Retorna DataFrame con columnas:
    ticker, start_time, run_name, + todas las métricas logueadas.
    """
    df = _mlflow_runs_for_experiment(experiment_name, max_results)
    if df.empty:
        return pd.DataFrame()

    # Solo runs terminados
    if "status" in df.columns:
        df = df[df["status"] == "FINISHED"]
    if df.empty:
        return pd.DataFrame()

    # Descartar runs de test y Summary (que agregan métricas de todos los tickers)
    if "run_name" in df.columns:
        df = df[~df["run_name"].str.contains("_init_test|Summary", na=False, regex=True)]
    if df.empty:
        return pd.DataFrame()

    # Necesitamos la columna ticker (viene como param de MLflow)
    if "ticker" not in df.columns:
        return pd.DataFrame()

    # Ordenar por fecha descendente
    if "start_time" in df.columns:
        df = df.sort_values("start_time", ascending=False)

    return df.reset_index(drop=True)


# ── Ticker view ─────────────────────────────────────────────
# Vista per-ticker: último entrenamiento o promedio de últimos N

def _resolve_scoring_weights(strategy_key: str) -> dict[str, float]:
    """Obtiene los weights de scoring para la estrategia desde base.yaml."""
    cfg_path = project_root() / "src" / "config" / "base.yaml"
    if not cfg_path.exists():
        return {}
    base_cfg = load_yaml(cfg_path)
    promo_cfg = base_cfg.get("lifecycle", {}).get("promotion")
    resolved = get_strategy_promotion_config(promo_cfg, strategy_key)
    return resolved.get("scoring_weights", {})


def _worst_alert(*alerts: str) -> str:
    """Retorna la peor alerta de una lista (🔴 > 🟡 > 🟢 > ⚪)."""
    priority = {"🔴": 3, "🟡": 2, "🟢": 1, "⚪": 0}
    worst = "⚪"
    for a in alerts:
        if priority.get(a, 0) > priority.get(worst, 0):
            worst = a
    return worst


def _categorize_metric(metric_name: str) -> str:
    """Clasifica una métrica como Trading o ML."""
    if any(p in metric_name for p in ("bt_", "tr_", "sharpe", "cagr", "drawdown", "profit", "return", "win_rate")):
        return "trading"
    return "ml"


# Nombres cortos para columnas de la tabla — orden no importa aquí,
# el orden lo define dashboard_thresholds.yaml.
_METRIC_LABELS: dict[str, str] = {
    "ml_mae": "MAE",
    "ml_rmse": "RMSE",
    "ml_ic": "IC",
    "ml_directional_accuracy": "Dir Acc",
    "bt_sharpe": "Sharpe",
    "bt_sortino": "Sortino",
    "bt_cagr": "CAGR",
    "bt_max_drawdown": "Max DD",
    "bt_calmar": "Calmar",
    "bt_profit_factor": "Prof Fac",
    # E3/E4 metrics
    "tr_profit_factor": "Prof Fac",
    "tr_max_drawdown": "Max DD",
    "sharpe": "Sharpe",
    "max_drawdown": "Max DD",
    "win_rate": "Win Rate",
    "total_return": "Tot Ret",
}


def _label(metric_name: str) -> str:
    """Retorna el label corto para una métrica."""
    return _METRIC_LABELS.get(metric_name, metric_name)


def build_ticker_report(
    strategy_key: str,
    last: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Construye reporte per-ticker para una estrategia.

    Parameters
    ----------
    strategy_key : str
        Clave de la estrategia (ej: "e1_conservative").
    last : int | None
        Si None, usa solo el último entrenamiento por ticker.
        Si N, promedia los últimos N entrenamientos por ticker.

    Returns
    -------
    (DataFrame, metadata)
        DataFrame con una fila por ticker, metadata con info de contexto.
    """
    thr = get_strategy_thresholds(strategy_key)
    if not thr:
        return pd.DataFrame(), {"error": f"Strategy '{strategy_key}' not found in dashboard_thresholds.yaml"}

    experiment_name = thr.get("experiment_name", strategy_key)
    metric_keys = list(thr.get("metrics", {}).keys())
    weights = _resolve_scoring_weights(strategy_key)

    df = mlflow_ticker_metrics(experiment_name)
    if df.empty:
        return pd.DataFrame(), {"error": f"No MLflow runs found for experiment '{experiment_name}'"}

    rows: list[dict[str, Any]] = []

    for ticker, group in df.groupby("ticker"):
        # Tomar el último o promediar los últimos N
        if last is not None and last > 0:
            subset = group.head(last)  # Ya está ordenado desc por start_time
            metrics: dict[str, Any] = {}
            for key in metric_keys:
                if key in subset.columns:
                    vals = subset[key].dropna().astype(float)
                    metrics[key] = float(vals.mean()) if not vals.empty else float("nan")
                else:
                    metrics[key] = float("nan")
            date_label = f"avg últimos {len(subset)}"
        else:
            latest_run = group.iloc[0]  # Primer row = más reciente
            metrics = {}
            for key in metric_keys:
                val = latest_run.get(key)
                metrics[key] = float(val) if pd.notna(val) else float("nan")
            date_label = str(latest_run.get("start_time", ""))[:10]

        # Calcular score compuesto usando la misma función de promoción
        score, detail = compute_score(metrics, weights)

        # Evaluar alertas per categoría (ML y Trading por separado)
        ml_alerts = []
        trading_alerts = []
        for key in metric_keys:
            val = metrics.get(key, float("nan"))
            if isinstance(val, float) and math.isfinite(val):
                alert = evaluate_alert(key, val, strategy_key)
                if _categorize_metric(key) == "ml":
                    ml_alerts.append(alert)
                else:
                    trading_alerts.append(alert)
        ml_alert = _worst_alert(*ml_alerts) if ml_alerts else "⚪"
        trading_alert = _worst_alert(*trading_alerts) if trading_alerts else "⚪"

        row: dict[str, Any] = {
            "ticker": ticker,
            "date": date_label,
            "n_runs": len(group),
            "score": round(score, 4),
            "ml_alert": ml_alert,
            "trading_alert": trading_alert,
        }
        for key in metric_keys:
            row[key] = metrics.get(key, float("nan"))
        rows.append(row)

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values("score", ascending=False).reset_index(drop=True)

    # Metadata para el header del reporte
    weight_desc = " + ".join(f"{int(w*100)}% {k}" for k, w in weights.items())
    meta = {
        "strategy_key": strategy_key,
        "display_name": thr.get("display_name", strategy_key),
        "description": thr.get("description", ""),
        "weight_desc": weight_desc,
        "last": last,
        "total_tickers": len(result_df),
    }

    return result_df, meta


def print_ticker_report(df: pd.DataFrame, meta: dict[str, Any]) -> None:
    """Imprime el reporte per-ticker en formato tabla."""
    if "error" in meta:
        print(f"⚠️  {meta['error']}")
        return
    if df.empty:
        print("No data available.")
        return

    strategy_key = meta["strategy_key"]
    thr = get_strategy_thresholds(strategy_key)
    metric_cfgs = thr.get("metrics", {})
    metric_keys = list(metric_cfgs.keys())
    ml_keys = [k for k in metric_keys if _categorize_metric(k) == "ml"]
    trading_keys = [k for k in metric_keys if _categorize_metric(k) == "trading"]

    # Header
    mode = f"promedio últimos {meta['last']}" if meta.get("last") else "último entrenamiento"
    print(f"\n{'='*80}")
    print(f"  {meta['display_name']} — Per-Ticker ({mode})")
    if meta.get("weight_desc"):
        print(f"  Score = {meta['weight_desc']}")
    print(f"{'='*80}")

    # ML metrics table
    if ml_keys:
        print(f"\n  🤖 ML")
        print(f"  {'─'*74}")
        header = f"  {'Ticker':<14}"
        for k in ml_keys:
            header += f" {_label(k):>10}"
        header += f" {'':>4}"
        print(header)
        print(f"  {'─'*74}")
        for _, row in df.iterrows():
            line = f"  {row['ticker']:<14}"
            for k in ml_keys:
                val = row.get(k, float("nan"))
                line += f" {_fmt(val, 4):>10}"
            line += f"  {row['ml_alert']}"
            print(line)

    # Trading metrics table
    if trading_keys:
        print(f"\n  📈 Trading")
        print(f"  {'─'*74}")
        header = f"  {'Ticker':<14}"
        for k in trading_keys:
            header += f" {_label(k):>10}"
        header += f" {'Score':>8} {'':>4}"
        print(header)
        print(f"  {'─'*74}")
        for _, row in df.iterrows():
            line = f"  {row['ticker']:<14}"
            for k in trading_keys:
                val = row.get(k, float("nan"))
                line += f" {_fmt(val, 4):>10}"
            line += f" {_fmt(row['score'], 4):>8}  {row['trading_alert']}"
            print(line)

    # Summary per category
    def _count_alerts(col: str) -> tuple[int, int, int]:
        g = sum(1 for _, r in df.iterrows() if r[col] == "🟢")
        y = sum(1 for _, r in df.iterrows() if r[col] == "🟡")
        rd = sum(1 for _, r in df.iterrows() if r[col] == "🔴")
        return g, y, rd

    total = len(df)
    ml_g, ml_y, ml_r = _count_alerts("ml_alert")
    tr_g, tr_y, tr_r = _count_alerts("trading_alert")
    print(f"\n  📊 Resumen ML:      {ml_g}🟢 {ml_y}🟡 {ml_r}🔴 de {total} tickers")
    print(f"  📊 Resumen Trading: {tr_g}🟢 {tr_y}🟡 {tr_r}🔴 de {total} tickers")

    # US vs AR comparison
    ar_tickers = df[df["ticker"].str.contains(r"\.BA$", na=False)]
    us_tickers = df[~df["ticker"].str.contains(r"\.BA$", na=False)]
    if not ar_tickers.empty and not us_tickers.empty:
        ar_avg = ar_tickers["score"].mean()
        us_avg = us_tickers["score"].mean()
        print(f"  📊 US avg Score: {us_avg:.4f} | AR avg Score: {ar_avg:.4f}")
    print()


# ── History view ────────────────────────────────────────────
# Historial de entrenamientos para un ticker específico

def build_history_report(
    strategy_key: str,
    ticker: str,
    last: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Construye historial de entrenamientos para un ticker.

    Parameters
    ----------
    strategy_key : str
        Clave de la estrategia.
    ticker : str
        Ticker a consultar (ej: "AAPL", "GGAL.BA").
    last : int
        Cantidad de entrenamientos a mostrar (default 10).

    Returns
    -------
    (DataFrame, metadata)
        DataFrame con una fila por entrenamiento, metadata con info de contexto.
    """
    thr = get_strategy_thresholds(strategy_key)
    if not thr:
        return pd.DataFrame(), {"error": f"Strategy '{strategy_key}' not found in dashboard_thresholds.yaml"}

    experiment_name = thr.get("experiment_name", strategy_key)
    metric_keys = list(thr.get("metrics", {}).keys())
    weights = _resolve_scoring_weights(strategy_key)

    df = mlflow_ticker_metrics(experiment_name)
    if df.empty:
        return pd.DataFrame(), {"error": f"No MLflow runs found for experiment '{experiment_name}'"}

    # Filtrar por ticker (case-insensitive match)
    ticker_df = df[df["ticker"].str.upper() == ticker.upper()]
    if ticker_df.empty:
        available = sorted(df["ticker"].unique())
        return pd.DataFrame(), {
            "error": f"Ticker '{ticker}' not found. Available: {', '.join(available)}"
        }

    # Tomar los últimos N (ya ordenado desc por start_time)
    ticker_df = ticker_df.head(last).copy()

    rows: list[dict[str, Any]] = []
    for _, run in ticker_df.iterrows():
        metrics: dict[str, Any] = {}
        for key in metric_keys:
            val = run.get(key)
            metrics[key] = float(val) if pd.notna(val) else float("nan")

        score, _ = compute_score(metrics, weights)

        # Alertas per categoría
        ml_alerts = []
        trading_alerts = []
        for key in metric_keys:
            val = metrics.get(key, float("nan"))
            if isinstance(val, float) and math.isfinite(val):
                alert = evaluate_alert(key, val, strategy_key)
                if _categorize_metric(key) == "ml":
                    ml_alerts.append(alert)
                else:
                    trading_alerts.append(alert)
        ml_alert = _worst_alert(*ml_alerts) if ml_alerts else "⚪"
        trading_alert = _worst_alert(*trading_alerts) if trading_alerts else "⚪"

        row: dict[str, Any] = {
            "date": str(run.get("start_time", ""))[:10],
            "run_name": run.get("run_name", ""),
            "score": round(score, 4),
            "ml_alert": ml_alert,
            "trading_alert": trading_alert,
        }
        for key in metric_keys:
            row[key] = metrics.get(key, float("nan"))
        rows.append(row)

    result_df = pd.DataFrame(rows)

    weight_desc = " + ".join(f"{int(w*100)}% {k}" for k, w in weights.items())
    # Usar el ticker tal como aparece en los datos
    real_ticker = ticker_df["ticker"].iloc[0]
    meta = {
        "strategy_key": strategy_key,
        "display_name": thr.get("display_name", strategy_key),
        "ticker": real_ticker,
        "weight_desc": weight_desc,
        "last": last,
        "total_runs": len(result_df),
    }

    return result_df, meta


def print_history_report(df: pd.DataFrame, meta: dict[str, Any]) -> None:
    """Imprime el historial de entrenamientos para un ticker."""
    if "error" in meta:
        print(f"⚠️  {meta['error']}")
        return
    if df.empty:
        print("No data available.")
        return

    strategy_key = meta["strategy_key"]
    thr = get_strategy_thresholds(strategy_key)
    # Usa el orden del YAML directamente (mismo orden que ticker view)
    display_keys = list(thr.get("metrics", {}).keys())

    print(f"\n{'='*80}")
    print(f"  {meta['display_name']} — {meta['ticker']} (últimos {meta['total_runs']} entrenamientos)")
    if meta.get("weight_desc"):
        print(f"  Score = {meta['weight_desc']}")
    print(f"{'='*80}")

    # Tabla — calcular ancho dinámico
    header = f"  {'Fecha':<12}"
    for k in display_keys:
        header += f" {_label(k):>8}"
    header += f" {'Score':>8} {'ML':>4} {'Trad':>4}"
    sep_width = max(len(header), 80)
    print(f"\n  {'─' * (sep_width - 2)}")
    print(header)
    print(f"  {'─' * (sep_width - 2)}")

    for _, row in df.iterrows():
        line = f"  {row['date']:<12}"
        for k in display_keys:
            val = row.get(k, float("nan"))
            line += f" {_fmt(val, 4):>8}"
        line += f" {_fmt(row['score'], 4):>8}  {row['ml_alert']} {row['trading_alert']}"
        print(line)

    # Tendencia: comparar primer y último score
    if len(df) >= 2:
        first_score = df.iloc[-1]["score"]  # Más antiguo
        last_score = df.iloc[0]["score"]    # Más reciente
        if first_score > 0:
            trend = (last_score - first_score) / first_score
            arrow = "↑" if trend > 0.05 else "↓" if trend < -0.05 else "→"
            print(f"\n  📈 Tendencia score: {first_score:.4f} → {last_score:.4f} ({trend:+.1%}) {arrow}")
    print()


# ── Report builder ──────────────────────────────────────────
# Construye el reporte completo combinando datos de timing y MLflow

def build_dashboard_report(
    strategies: list[str] | None = None,
) -> pd.DataFrame:
    """Build the full dashboard report as a DataFrame.

    Each row is a (strategy, metric) pair with avg/last values and alert color.
    """
    # Carga la configuración de umbrales desde el YAML del dashboard
    cfg_path = project_root() / "src" / "config" / "dashboard_thresholds.yaml"
    if not cfg_path.exists():
        print("⚠️  dashboard_thresholds.yaml not found")
        return pd.DataFrame()

    cfg = load_yaml(cfg_path)
    all_strategies = cfg.get("strategies", {})  # Dict de todas las estrategias configuradas

    # Si se pidió una estrategia específica, filtra solo esa
    if strategies:
        all_strategies = {k: v for k, v in all_strategies.items() if k in strategies}

    # Lee el log de timing una sola vez (para todas las estrategias)
    df_timing = read_timing_log()

    rows: list[dict[str, Any]] = []  # Acumula las filas del reporte final

    # Itera sobre cada estrategia configurada
    for strat_key, strat_cfg in all_strategies.items():
        display = strat_cfg.get("display_name", strat_key)     # Nombre para mostrar (ej: "E1 Simple")
        experiment = strat_cfg.get("experiment_name", strat_key) # Nombre del experimento en MLflow

        # --- Timing ---
        # Obtiene el resumen de tiempos para esta estrategia
        timing = timing_summary_for_strategy(df_timing, strat_key)

        # Agrega filas de timing para las fases "train" y "predict"
        for phase in ("train", "predict"):
            # Construye las claves para acceder a los valores del resumen de timing
            avg_key = f"{phase}_avg_seconds"
            last_key = f"{phase}_last_seconds"
            p50_key = f"{phase}_p50_seconds"
            p95_key = f"{phase}_p95_seconds"

            # Obtiene los valores; si no existen, usa NaN como valor por defecto
            avg_val = timing.get(avg_key, float("nan"))
            last_val = timing.get(last_key, float("nan"))
            p50_val = timing.get(p50_key, float("nan"))
            p95_val = timing.get(p95_key, float("nan"))

            # Evalúa la alerta de timing (🟢 verde, 🟡 amarillo, 🔴 rojo) según umbrales
            alert = evaluate_timing_alert(phase, last_val, strat_key)

            # Agrega la fila al reporte
            rows.append({
                "strategy": display,          # Nombre visible de la estrategia
                "strategy_key": strat_key,    # Clave interna (ej: "e1_conservative")
                "category": "⏱ Timing",      # Categoría para agrupar en el reporte
                "metric": f"{phase}_seconds", # Nombre de la métrica
                "avg": _fmt(avg_val, 1),      # Promedio formateado (1 decimal)
                "p50": _fmt(p50_val, 1),      # Mediana formateada
                "p95": _fmt(p95_val, 1),      # Percentil 95 formateado
                "last": _fmt(last_val, 1),    # Último valor formateado
                "alert": alert,               # Emoji de alerta (🟢/🟡/🔴)
            })

        # --- ML / Trading metrics ---
        # Obtiene las claves de métricas configuradas para esta estrategia
        metric_keys = list(strat_cfg.get("metrics", {}).keys())
        # Consulta MLflow para obtener los valores de esas métricas
        mlflow_data = mlflow_metrics_summary(experiment, metric_keys)

        # Agrega una fila por cada métrica configurada
        for metric_name, metric_cfg in strat_cfg.get("metrics", {}).items():
            avg_val = mlflow_data.get(f"{metric_name}_avg", float("nan"))   # Promedio histórico
            last_val = mlflow_data.get(f"{metric_name}_last", float("nan")) # Último valor
            # Evalúa la alerta para esta métrica según los umbrales configurados
            alert = evaluate_alert(metric_name, last_val, strat_key)

            # Clasifica la métrica como "Trading" o "ML" según palabras clave en el nombre
            cat = "📈 Trading" if any(
                p in metric_name for p in ("bt_", "tr_", "sharpe", "cagr", "drawdown", "profit", "return", "win_rate")
            ) else "🤖 ML"

            rows.append({
                "strategy": display,
                "strategy_key": strat_key,
                "category": cat,
                "metric": metric_name,
                "avg": _fmt(avg_val, 4),    # 4 decimales para métricas ML/trading
                "p50": "",                  # No se calcula p50/p95 para métricas de MLflow
                "p95": "",
                "last": _fmt(last_val, 4),
                "alert": alert,
            })

        # Agrega una fila informativa con el total de runs y la fecha del último run
        total = mlflow_data.get("total_runs", 0)
        last_time = mlflow_data.get("last_run_time", "")
        rows.append({
            "strategy": display,
            "strategy_key": strat_key,
            "category": "ℹ️  Info",
            "metric": "total_mlflow_runs",
            "avg": str(total),
            "p50": "",
            "p95": "",
            "last": str(last_time)[:19] if last_time else "",  # Trunca a YYYY-MM-DD HH:MM:SS
            "alert": "⚪",  # Blanco = solo informativo, sin alerta
        })

    # Retorna el DataFrame completo con todas las filas del reporte
    return pd.DataFrame(rows)


def _fmt(val: float, decimals: int = 4) -> str:
    """Formatea un valor numérico a string con N decimales.
    Si es NaN o infinito, retorna '—' (guión largo) como placeholder."""
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return "—"
    return f"{val:.{decimals}f}"


# ── Pretty print ────────────────────────────────────────────
# Imprime el reporte en formato tabla legible en la terminal

def print_report(df: pd.DataFrame) -> None:
    """Print a formatted dashboard report to stdout."""
    if df.empty:
        print("No data available for dashboard report.")
        return

    # Obtiene la lista de estrategias únicas en el reporte
    strategies = df["strategy"].unique()

    # Imprime una sección por cada estrategia
    for strat in strategies:
        sdf = df[df["strategy"] == strat]           # Filtra filas de esta estrategia
        strat_key = sdf["strategy_key"].iloc[0]      # Obtiene la clave interna

        # Imprime encabezado de la estrategia con separador visual
        print(f"\n{'='*70}")
        print(f"  {strat}")
        # Busca la descripción de la estrategia en los umbrales configurados
        thr = get_strategy_thresholds(strat_key)
        if thr.get("description"):
            print(f"  {thr['description']}")
        print(f"{'='*70}")

        # Agrupa las métricas por categoría (Timing, ML, Trading, Info)
        for cat in sdf["category"].unique():
            cat_df = sdf[sdf["category"] == cat]
            print(f"\n  {cat}")
            print(f"  {'─'*64}")
            # Encabezados de las columnas
            print(f"  {'Metric':<30} {'Avg':>8} {'Last':>8} {'Alert':>6}")
            print(f"  {'─'*64}")
            # Imprime cada fila de métricas con sus valores y alerta
            for _, row in cat_df.iterrows():
                metric = row["metric"]
                avg = row["avg"]
                last = row["last"]
                alert = row["alert"]
                print(f"  {metric:<30} {avg:>8} {last:>8}  {alert}")

    print()  # Línea final en blanco para separación


# ── Save report ─────────────────────────────────────────────
# Guarda el reporte como archivo CSV

def save_report(df: pd.DataFrame) -> Path:
    """Save report to CSV."""
    # Directorio de salida para reportes del dashboard
    out_dir = project_root() / "reports" / "dashboard"
    ensure_dir(out_dir)  # Crea el directorio si no existe

    # Genera nombre con timestamp para tener histórico de reportes
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"dashboard_report_{ts}.csv"
    df.to_csv(path, index=False)  # Guarda con timestamp

    # También guarda como "latest" para acceso rápido al más reciente
    latest = out_dir / "dashboard_report_latest.csv"
    df.to_csv(latest, index=False)

    return path


# ── CLI ─────────────────────────────────────────────────────
# Punto de entrada cuando se ejecuta desde la línea de comandos

def main() -> None:
    # Configura el parser de argumentos de CLI
    parser = argparse.ArgumentParser(
        description="Dashboard health checker — per-strategy metrics + alerts"
    )
    parser.add_argument(
        "--view",
        type=str,
        choices=["summary", "ticker", "history"],
        default="summary",
        help="Vista a mostrar: summary (default), ticker (per-ticker), history (historial de un ticker).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Strategy key e.g. e1_conservative. Required for ticker/history views. Defined in src/config/dashboard_thresholds.yaml",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Ticker symbol e.g. AAPL, GGAL.BA. Required for history view.",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        help="ticker view: promedia últimos N entrenamientos (default: solo el último). "
             "history view: cantidad de entrenamientos a listar (default: 10).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save report CSV (default: True). Only applies to summary view.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout output.",
    )
    args = parser.parse_args()

    # ── Vista ticker ──
    if args.view == "ticker":
        if not args.strategy:
            parser.error("--strategy is required for --view ticker")
        df, meta = build_ticker_report(args.strategy, last=args.last)
        if not args.quiet:
            print_ticker_report(df, meta)
        if args.save and not df.empty:
            out_dir = project_root() / "reports" / "dashboard"
            ensure_dir(out_dir)
            path = out_dir / f"ticker_report_{args.strategy}_latest.csv"
            df.to_csv(path, index=False)
            print(f"✓ Report saved: {path.relative_to(project_root())}")
        return

    # ── Vista history ──
    if args.view == "history":
        if not args.strategy:
            parser.error("--strategy is required for --view history")
        if not args.ticker:
            parser.error("--ticker is required for --view history")
        last_n = args.last if args.last is not None else 10
        df, meta = build_history_report(args.strategy, args.ticker, last=last_n)
        if not args.quiet:
            print_history_report(df, meta)
        if args.save and not df.empty:
            out_dir = project_root() / "reports" / "dashboard"
            ensure_dir(out_dir)
            safe_ticker = args.ticker.replace(".", "_")
            path = out_dir / f"history_report_{args.strategy}_{safe_ticker}_latest.csv"
            df.to_csv(path, index=False)
            print(f"✓ Report saved: {path.relative_to(project_root())}")
        return

    # ── Vista summary (default) ──
    strategies = [args.strategy] if args.strategy else None
    df = build_dashboard_report(strategies)

    if not args.quiet:
        print_report(df)

    if args.save and not df.empty:
        path = save_report(df)
        print(f"✓ Report saved: {path.relative_to(project_root())}")


# Si se ejecuta directamente (python -m src.dashboard.checker), llama a main()
if __name__ == "__main__":
    main()
