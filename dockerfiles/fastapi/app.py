"""
FastAPI Service - Trading Predict

Sirve la ÚLTIMA predicción guardada del modelo champion para un (strategy, ticker)
y la convierte en una señal accionable BUY/SELL/HOLD. No hace inferencia en vivo:
lee la predicción precalculada del walk-forward del champion registrado en
``models/registry.json``.

Endpoints:
- GET /                          - Health check básico
- GET /health                    - Health check detallado
- GET /models/status             - Champions registrados por estrategia
- GET /predict/{strategy}/{ticker} - Predicción + señal del champion (e1|e2|e3)
- GET /leaderboard               - Ranking de champions (mismo cálculo que el CLI)

La superficie del demo es la UI de Swagger en /docs.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Response

# La API reutiliza DIRECTAMENTE el script del CLI: cualquier cambio en la lógica
# de leaderboard.py (scoring, deltas, columnas) se refleja acá sin tocar la API.
from scripts.evaluation.leaderboard import add_deltas, build_leaderboard
from src.lifecycle.registry import ModelRegistry
from src.utils import get_nested, load_yaml, project_root

app = FastAPI(
    title="Trading Predict API",
    description="API para predicciones de trading con ML (modelos champion del lifecycle)",
    version="1.0.0",
)

# Registry como única fuente de verdad de los champions.
_REGISTRY_PATH = project_root() / "models" / "registry.json"
REGISTRY = ModelRegistry(_REGISTRY_PATH)

# Config centralizada: provee umbrales por defecto por variant cuando el
# champion no los tiene en el registry (ej. tickers entrenados sin Optuna).
_CONFIG_PATH = project_root() / "src" / "config" / "base.yaml"
CONFIG = load_yaml(_CONFIG_PATH) if _CONFIG_PATH.exists() else {}

# Estrategias soportadas en el demo.
SUPPORTED_STRATEGIES = ("e1", "e2", "e3")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _resolve_run_dir(run_dir: str) -> Path:
    """Resolver run_dir (relativo en el registry) contra la raíz del proyecto."""
    path = Path(run_dir)
    if not path.is_absolute():
        path = project_root() / path
    return path


def _latest_prediction(run_dir: Path, ticker: str) -> dict[str, Any]:
    """Leer la última fila del CSV de predicciones walk-forward del champion.

    Devuelve un dict con: prediction_date, predicted_return y, si existe (E3),
    pred_std. Lanza FileNotFoundError si no hay CSV de predicciones.
    """
    candidates = list(run_dir.glob(f"{ticker}*_walkforward_predictions.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No se encontró CSV de predicciones en {run_dir} para {ticker}"
        )

    df = pd.read_csv(candidates[0])
    if df.empty:
        raise FileNotFoundError(f"CSV de predicciones vacío: {candidates[0]}")

    last = df.iloc[-1]
    result: dict[str, Any] = {
        "prediction_date": str(last["timestamp"]),
        "predicted_return": float(last["y_pred"]),
    }
    # E3 (ensemble) incluye la desviación estándar del ensemble como confianza.
    if "pred_std" in df.columns and pd.notna(last["pred_std"]):
        result["pred_std"] = float(last["pred_std"])
    return result


def _to_signal(y_pred: float, tau_buy: float, tau_sell: float) -> str:
    """Convertir un retorno predicho en señal accionable."""
    if y_pred >= tau_buy:
        return "BUY"
    if y_pred <= -tau_sell:
        return "SELL"
    return "HOLD"


def _thresholds(metrics: dict[str, Any], variant: str | None) -> tuple[float, float]:
    """Umbrales del champion: primero el registry (Optuna por ticker, claves
    ``tau_*`` en E3 o ``hp_tau_*`` en E1/E2); si no, los defaults de ``base.yaml``
    según el variant (ej. ``e1_conservative``).
    """
    tau_buy = metrics.get("tau_buy", metrics.get("hp_tau_buy"))
    tau_sell = metrics.get("tau_sell", metrics.get("hp_tau_sell"))

    if (tau_buy is None or tau_sell is None) and variant:
        defaults = get_nested(CONFIG, ["strategies", variant, "thresholds"], default={})
        if tau_buy is None:
            tau_buy = defaults.get("tau_buy")
        if tau_sell is None:
            tau_sell = defaults.get("tau_sell")

    if tau_buy is None or tau_sell is None:
        raise HTTPException(
            status_code=500,
            detail=f"No se encontraron umbrales (tau_buy/tau_sell) para variant '{variant}'.",
        )
    return float(tau_buy), float(tau_sell)


def _champion_metrics_summary(metrics: dict[str, Any], variant: str | None = None) -> dict[str, Any]:
    """Resumen de métricas del champion para mostrar en el demo.

    horizon_days/horizon_bars son parámetros ESTÁTICOS de configuración, no
    métricas aprendidas. Si el champion no los tiene guardados (modelos viejos),
    se completan desde el config según el variant: E1/E2 -> horizon_days, E3 ->
    horizon_bars. Así el valor es consistente y nunca queda null por un gap de
    persistencia del entrenamiento.
    """
    horizon_days = metrics.get("horizon_days")
    horizon_bars = metrics.get("horizon_bars")
    if variant:
        strat_cfg = get_nested(CONFIG, ["strategies", variant], default={}) or {}
        if horizon_days is None:
            horizon_days = strat_cfg.get("horizon_days")
        if horizon_bars is None:
            horizon_bars = strat_cfg.get("horizon_bars")
    return {
        "bt_sharpe": metrics.get("bt_sharpe"),
        "bt_total_return": metrics.get("bt_total_return"),
        "bt_cagr": metrics.get("bt_cagr"),  # retorno anualizado (E1/E2; null en E3 intradía)
        "ml_directional_accuracy": metrics.get("ml_directional_accuracy"),
        "ml_ic": metrics.get("ml_ic"),
        # E1/E2 reportan horizon_days; E3 (intradía) reporta horizon_bars.
        "horizon_days": horizon_days,
        "horizon_bars": horizon_bars,
    }


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------

@app.get("/")
def read_root():
    """Health check básico."""
    return {
        "service": "Trading Predict API",
        "status": "running",
        "strategies": list(SUPPORTED_STRATEGIES),
        "endpoints": ["/health", "/models/status", "/predict/{strategy}/{ticker}", "/leaderboard"],
        "docs": "/docs",
    }


@app.get("/health")
def health_check(response: Response):
    """Health check de readiness.

    Relee el registry en disco (la instancia global se cachea al arrancar) para
    reflejar el estado actual y validar que sea parseable. Devuelve HTTP 503 si el
    registry no existe, no se puede leer/parsear, o no hay champions. No expone
    rutas internas del filesystem.
    """
    registry_exists = _REGISTRY_PATH.exists()
    last_updated: str | None = None
    champions_count = 0
    ok = registry_exists
    if registry_exists:
        try:
            fresh = ModelRegistry(_REGISTRY_PATH)
            champions_count = len(fresh.list_all(stage="champion"))
            last_updated = fresh.data.get("last_updated")
        except Exception:
            ok = False

    ok = ok and champions_count > 0
    if not ok:
        response.status_code = 503

    return {
        "status": "healthy" if ok else "unhealthy",
        "version": app.version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "registry_exists": registry_exists,
        "registry_last_updated": last_updated,
        "champions_count": champions_count,
    }


@app.get("/models/status")
def get_models_status():
    """Champions registrados, agrupados por estrategia."""
    champions = REGISTRY.list_all(stage="champion")
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for entry in champions:
        strategy = entry["strategy"]
        by_strategy.setdefault(strategy, []).append(
            {
                "ticker": entry["ticker"],
                "variant": entry.get("variant"),
                "promoted_at": entry.get("promoted_at"),
            }
        )
    return {
        "champions_count": len(champions),
        "strategies": by_strategy,
    }


@app.get("/predict/{strategy}/{ticker}")
def predict(strategy: str, ticker: str):
    """Predicción + señal del champion para una estrategia/ticker.

    Args:
        strategy: e1 (90d) | e2 (20d) | e3 (intradía 30min)
        ticker: símbolo del activo (ej: AAPL, GGAL.BA, SPY)

    Returns:
        Predicción guardada más reciente del walk-forward del champion,
        convertida en señal BUY/SELL/HOLD, con métricas del champion.
    """
    strategy = strategy.lower()
    if strategy not in SUPPORTED_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Estrategia '{strategy}' no soportada. Use: {SUPPORTED_STRATEGIES}",
        )

    # Registry fresco en cada request (igual que /health y /leaderboard): la
    # instancia global REGISTRY se cachea al arrancar el proceso y quedaría
    # desactualizada tras una promoción/reentrenamiento posterior.
    registry = ModelRegistry(_REGISTRY_PATH)
    champion = registry.get_champion(strategy, ticker)
    if champion is None:
        raise HTTPException(
            status_code=404,
            detail=f"No hay champion para {strategy.upper()} / {ticker}.",
        )

    metrics = champion.get("metrics", {})
    run_dir = _resolve_run_dir(champion["run_dir"])

    try:
        pred = _latest_prediction(run_dir, ticker)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    tau_buy, tau_sell = _thresholds(metrics, champion.get("variant"))
    signal = _to_signal(pred["predicted_return"], tau_buy, tau_sell)

    response: dict[str, Any] = {
        "ticker": ticker,
        "strategy": strategy,
        "variant": champion.get("variant"),
        "signal": signal,
        "predicted_return": pred["predicted_return"],
        "prediction_date": pred["prediction_date"],
        "thresholds": {"tau_buy": tau_buy, "tau_sell": tau_sell},
        "champion_metrics": _champion_metrics_summary(metrics, champion.get("variant")),
        "note": "predicted_return es la última predicción walk-forward del champion "
        "(fecha = prediction_date), no una inferencia en tiempo real.",
    }
    if "pred_std" in pred:
        response["pred_std"] = pred["pred_std"]
    return response


def _json_safe(value: Any) -> Any:
    """Convertir escalares de pandas/numpy a tipos nativos JSON-serializables.

    - numpy scalars (np.int64/np.float64) -> int/float de Python.
    - NaN -> None (JSON no admite NaN; aparece cuando falta una métrica o delta).
    """
    if value is None:
        return None
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _leaderboard_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame del leaderboard -> lista de dicts JSON-safe (todas las columnas)."""
    return [
        {key: _json_safe(val) for key, val in row.items()}
        for row in df.to_dict(orient="records")
    ]


@app.get("/leaderboard")
def leaderboard(strategies: str = "e1,e2", top: int | None = None):
    """Ranking de champions — mismo cálculo que ``python -m scripts.evaluation.leaderboard``.

    Reutiliza directamente ``build_leaderboard`` + ``add_deltas`` del script CLI, por lo
    que rankea por el score compuesto de la promoción y agrega el delta de score/rank
    contra el snapshot previo (``reports/dashboard/leaderboard_history.jsonl``). Es de
    solo lectura: NO escribe el CSV ni el history (a diferencia del CLI, que persiste).

    Args:
        strategies: estrategias a rankear, separadas por coma (default: ``e1,e2``).
        top: si se indica, devuelve solo las primeras N filas (el resumen usa el total).

    Returns:
        ``rows`` con todas las columnas del leaderboard (rank, strategy, ticker, variant,
        score, deltas, trend, y las métricas bt_/ml_) + un ``summary`` de tendencias.
    """
    requested = [s.strip().lower() for s in strategies.split(",") if s.strip()]
    if not requested:
        raise HTTPException(
            status_code=400, detail="Indicá al menos una estrategia (ej: strategies=e1,e2)."
        )
    unknown = [s for s in requested if s not in SUPPORTED_STRATEGIES]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Estrategia(s) no soportada(s): {unknown}. Use: {list(SUPPORTED_STRATEGIES)}",
        )
    if top is not None and top < 1:
        raise HTTPException(status_code=400, detail="top debe ser >= 1.")

    # Registry fresco en cada request para reflejar promociones recientes
    # (la instancia global REGISTRY se cachea al arrancar el proceso).
    registry = ModelRegistry(_REGISTRY_PATH)
    df = build_leaderboard(registry, CONFIG, requested)
    df = add_deltas(df)

    generated_at = datetime.now(timezone.utc).isoformat()
    if df.empty:
        return {
            "strategies": requested,
            "generated_at": generated_at,
            "count": 0,
            "total_champions": 0,
            "rows": [],
            "summary": {"improving": 0, "worsening": 0, "flat": 0, "new": 0},
        }

    view = df.head(top) if top else df
    summary = {
        "improving": int((df["trend"] == "up").sum()),
        "worsening": int((df["trend"] == "down").sum()),
        "flat": int((df["trend"] == "flat").sum()),
        "new": int((df["trend"] == "new").sum()),
    }
    return {
        "strategies": requested,
        "generated_at": generated_at,
        "count": int(len(view)),
        "total_champions": int(len(df)),
        "rows": _leaderboard_records(view),
        "summary": summary,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8800)
