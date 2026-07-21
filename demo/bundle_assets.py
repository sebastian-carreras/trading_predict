"""Prepara ``demo/assets/`` self-contained para la demo Streamlit.

Corre DENTRO del repo completo (reutiliza ``ModelRegistry`` + config), pero NO
necesita torch/mlflow: solo lee el registry, las predicciones walk-forward ya
guardadas del champion, el leaderboard ya generado y las figuras curadas.

La app (``demo/app.py``) después lee SOLO ``demo/assets/`` — nada de ``src/``,
nada de ``runs/``. Así el bundle es desplegable tal cual a Hugging Face Spaces.

Uso:
    python -m demo.bundle_assets
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.lifecycle.registry import ModelRegistry
from src.utils import get_nested, load_yaml

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "demo" / "assets"
REGISTRY_PATH = ROOT / "models" / "registry.json"
CONFIG_PATH = ROOT / "src" / "config" / "base.yaml"
LEADERBOARD_SRC = ROOT / "reports" / "dashboard" / "leaderboard_latest.csv"
DRIFT_LOG_SRC = ROOT / "models" / "drift_log.jsonl"

STRATEGIES = ("e1", "e2", "e3")

# Ticker sintético usado en tests de promoción; nunca debe aparecer en la demo.
EXCLUDE_TICKERS = {"AAA"}

# Métricas para las medianas del Overview (recalculadas en vivo desde el registry).
HEADLINE_METRICS = ("bt_sharpe", "ml_directional_accuracy", "ml_ic")

# Figuras curadas a copiar (origen relativo a ROOT). Se copian si existen.
FIGURE_SOURCES = [
    # Overview (heroes)
    "reports/conclusiones/figuras/fig_sharpe_boxplot_by_strategy.png",
    "reports/conclusiones/figuras/fig_risk_return_scatter.png",
    # Strategy performance
    "reports/conclusiones/figuras/fig_directional_accuracy_heatmap.png",
    "reports/conclusiones/figuras/fig_ic_vs_sharpe.png",
    "reports/conclusiones/figuras/fig_conclusions_cumulative_return_us.png",
    "reports/conclusiones/figuras/fig_conclusions_cumulative_return_ar.png",
    "reports/conclusiones/figuras/fig_conclusions_monthly_heatmap_e1.png",
    "reports/conclusiones/figuras/fig_conclusions_monthly_heatmap_e2.png",
    "reports/conclusiones/figuras/fig_conclusions_monthly_heatmap_e3.png",
    "reports/conclusiones/figuras/fig_conclusions_underwater.png",
    # Champion vs baseline
    "reports/conclusiones/comparaciones/fig_e1_vs_baseline_ml.png",
    "reports/conclusiones/comparaciones/fig_e1_vs_baseline_trading_retorno.png",
    "reports/conclusiones/comparaciones/fig_e1_vs_baseline_trading_riesgo.png",
    "reports/conclusiones/comparaciones/fig_e2_vs_baseline_ml.png",
    "reports/conclusiones/comparaciones/fig_e2_vs_baseline_trading_retorno.png",
    "reports/conclusiones/comparaciones/fig_e2_vs_baseline_trading_riesgo.png",
    "reports/conclusiones/comparaciones/fig_e3_vs_baseline_ml.png",
    "reports/conclusiones/comparaciones/fig_e3_vs_baseline_trading_retorno.png",
    # Drift & stability
    "reports/conclusiones/figuras/fig_champion_stability_sharpe.png",
    "reports/conclusiones/figuras/fig_champion_stability_ic.png",
    # Methodology
    "reports/conclusiones/figuras/fig_conclusions_walkforward_stability.png",
    "reports/conclusiones/figuras/fig_conclusions_roc.png",
    "reports/conclusiones/figuras/fig_conclusions_calibration.png",
    "reports/conclusiones/figuras/fig_conclusions_rolling_ic.png",
    "reports/hyperparameter_optimization/figures/param_importances.png",
    "reports/hyperparameter_optimization/figures/optimization_history.png",
]

# Diagrama de arquitectura (raíz del repo) → nombre estable en el bundle.
ARCH_SRC = ROOT / "docs" / "architecture-mlops.png"
ARCH_DST_NAME = "architecture.png"


def _thresholds(metrics: dict[str, Any], variant: str | None, config: dict[str, Any]) -> tuple[float | None, float | None]:
    """Umbrales del champion (misma lógica que dockerfiles/fastapi/app.py).

    Primero el registry (``tau_*`` en E3 o ``hp_tau_*`` en E1/E2); si faltan, los
    defaults de ``base.yaml`` según el variant.
    """
    tau_buy = metrics.get("tau_buy", metrics.get("hp_tau_buy"))
    tau_sell = metrics.get("tau_sell", metrics.get("hp_tau_sell"))
    if (tau_buy is None or tau_sell is None) and variant:
        defaults = get_nested(config, ["strategies", variant, "thresholds"], default={}) or {}
        if tau_buy is None:
            tau_buy = defaults.get("tau_buy")
        if tau_sell is None:
            tau_sell = defaults.get("tau_sell")
    tau_buy = float(tau_buy) if tau_buy is not None else None
    tau_sell = float(tau_sell) if tau_sell is not None else None
    return tau_buy, tau_sell


def _to_signal(y_pred: float, tau_buy: float | None, tau_sell: float | None) -> str:
    """Retorno predicho → señal accionable BUY/SELL/HOLD."""
    if tau_buy is None or tau_sell is None:
        return "N/A"
    if y_pred >= tau_buy:
        return "BUY"
    if y_pred <= -tau_sell:
        return "SELL"
    return "HOLD"


def _metrics_summary(metrics: dict[str, Any], variant: str | None, config: dict[str, Any]) -> dict[str, Any]:
    """Resumen de métricas del champion para las cards (maneja E1/E2 vs E3)."""
    horizon_days = metrics.get("horizon_days")
    horizon_bars = metrics.get("horizon_bars")
    if variant:
        strat_cfg = get_nested(config, ["strategies", variant], default={}) or {}
        if horizon_days is None:
            horizon_days = strat_cfg.get("horizon_days")
        if horizon_bars is None:
            horizon_bars = strat_cfg.get("horizon_bars")
    return {
        "bt_sharpe": metrics.get("bt_sharpe"),
        "bt_total_return": metrics.get("bt_total_return"),
        "bt_cagr": metrics.get("bt_cagr"),
        "bt_calmar": metrics.get("bt_calmar"),
        "bt_max_drawdown": metrics.get("bt_max_drawdown"),
        "ml_directional_accuracy": metrics.get("ml_directional_accuracy"),
        "ml_ic": metrics.get("ml_ic"),
        "horizon_days": horizon_days,
        "horizon_bars": horizon_bars,
    }


def _resolve_run_dir(run_dir: str) -> Path:
    path = Path(run_dir)
    return path if path.is_absolute() else ROOT / path


def _latest_prediction(pred_csv: Path) -> dict[str, Any]:
    """Última fila del CSV de predicciones walk-forward (fecha + y_pred)."""
    df = pd.read_csv(pred_csv)
    if df.empty:
        return {"prediction_date": None, "predicted_return": None}
    last = df.iloc[-1]
    result: dict[str, Any] = {
        "prediction_date": str(last.get("timestamp")),
        "predicted_return": float(last["y_pred"]) if pd.notna(last.get("y_pred")) else None,
    }
    if "pred_std" in df.columns and pd.notna(last.get("pred_std")):
        result["pred_std"] = float(last["pred_std"])
    return result


def _reset_assets() -> None:
    if ASSETS.exists():
        shutil.rmtree(ASSETS)
    (ASSETS / "predictions").mkdir(parents=True, exist_ok=True)
    (ASSETS / "figures").mkdir(parents=True, exist_ok=True)


def _copy_predictions(strategy: str, ticker: str, run_dir: Path) -> str | None:
    """Copia el CSV de predicciones walk-forward del champion al bundle plano.

    Devuelve el path relativo (dentro de assets) del CSV copiado, o None.
    """
    candidates = list(run_dir.glob(f"{ticker}*_walkforward_predictions.csv"))
    if not candidates:
        return None
    dst_dir = ASSETS / "predictions" / strategy
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{ticker}.csv"
    shutil.copyfile(candidates[0], dst)
    return f"predictions/{strategy}/{ticker}.csv"


def _copy_downscaled(src: Path, dst: Path, max_width: int = 1600) -> None:
    """Copia una imagen reduciéndola si es más ancha que ``max_width``.

    Usa Pillow si está disponible (para no inflar el bundle con el diagrama de
    arquitectura de ~5 MB); si no, copia tal cual.
    """
    try:
        from PIL import Image  # type: ignore

        with Image.open(src) as img:
            if img.width > max_width:
                ratio = max_width / img.width
                img = img.resize((max_width, int(img.height * ratio)))
            img.save(dst, optimize=True)
        return
    except Exception:
        shutil.copyfile(src, dst)


def _copy_backtest(strategy: str, ticker: str, run_dir: Path) -> str | None:
    """Copia una versión adelgazada del backtest walk-forward (curva de equity).

    Del CSV original ``timestamp,pos,signal,gross_ret,costs,net_ret,equity,turnover``
    quedan solo ``timestamp,equity,net_ret`` para no inflar el bundle. La equity es
    NETA de costos (curva real del backtest). Devuelve el path relativo o None.
    """
    candidates = list(run_dir.glob(f"{ticker}*_walkforward_backtest.csv"))
    if not candidates:
        return None
    df = pd.read_csv(candidates[0])
    if "equity" not in df.columns or "timestamp" not in df.columns:
        return None
    keep = [c for c in ("timestamp", "equity", "net_ret") if c in df.columns]
    dst_dir = ASSETS / "backtests" / strategy
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{ticker}.csv"
    df[keep].to_csv(dst, index=False)
    return f"backtests/{strategy}/{ticker}.csv"


def _copy_figures() -> list[str]:
    copied: list[str] = []
    for rel in FIGURE_SOURCES:
        src = ROOT / rel
        if src.exists():
            dst = ASSETS / "figures" / src.name
            shutil.copyfile(src, dst)
            copied.append(src.name)
    if ARCH_SRC.exists():
        _copy_downscaled(ARCH_SRC, ASSETS / "figures" / ARCH_DST_NAME)
        copied.append(ARCH_DST_NAME)
    return copied


def _copy_leaderboard() -> bool:
    """Copia leaderboard_latest.csv al bundle, filtrando tickers excluidos."""
    if not LEADERBOARD_SRC.exists():
        return False
    df = pd.read_csv(LEADERBOARD_SRC)
    if "ticker" in df.columns:
        df = df[~df["ticker"].isin(EXCLUDE_TICKERS)].reset_index(drop=True)
    df.to_csv(ASSETS / "leaderboard.csv", index=False)
    return True


def _median(values: list[float]) -> float | None:
    s = pd.Series([v for v in values if v is not None], dtype="float64").dropna()
    return round(float(s.median()), 4) if not s.empty else None


def build() -> None:
    config = load_yaml(CONFIG_PATH) if CONFIG_PATH.exists() else {}
    registry = ModelRegistry(REGISTRY_PATH)
    champions = [
        c for c in registry.list_all(stage="champion")
        if c.get("strategy") in STRATEGIES and c.get("ticker") not in EXCLUDE_TICKERS
    ]

    _reset_assets()

    champ_records: list[dict[str, Any]] = []
    headline_acc: dict[str, dict[str, list[float]]] = {
        s: {m: [] for m in HEADLINE_METRICS} for s in STRATEGIES
    }
    missing_preds: list[str] = []

    for champ in champions:
        strategy = champ["strategy"]
        ticker = champ["ticker"]
        variant = champ.get("variant")
        metrics = champ.get("metrics", {}) or {}
        run_dir = _resolve_run_dir(champ.get("run_dir", ""))

        pred_rel = _copy_predictions(strategy, ticker, run_dir)
        if pred_rel is None:
            missing_preds.append(f"{strategy}/{ticker}")
            pred_info: dict[str, Any] = {"prediction_date": None, "predicted_return": None}
        else:
            pred_info = _latest_prediction(ASSETS / pred_rel)

        backtest_rel = _copy_backtest(strategy, ticker, run_dir)

        tau_buy, tau_sell = _thresholds(metrics, variant, config)
        y_pred = pred_info.get("predicted_return")
        signal = _to_signal(y_pred, tau_buy, tau_sell) if y_pred is not None else "N/A"

        champ_records.append({
            "strategy": strategy,
            "ticker": ticker,
            "variant": variant,
            "signal": signal,
            "predicted_return": y_pred,
            "prediction_date": pred_info.get("prediction_date"),
            "pred_std": pred_info.get("pred_std"),
            "thresholds": {"tau_buy": tau_buy, "tau_sell": tau_sell},
            "promoted_at": champ.get("promoted_at"),
            "metrics": _metrics_summary(metrics, variant, config),
            "predictions_csv": pred_rel,
            "backtest_csv": backtest_rel,
        })

        for m in HEADLINE_METRICS:
            val = metrics.get(m)
            if isinstance(val, (int, float)):
                headline_acc[strategy][m].append(float(val))

    # headline.json: medianas recalculadas en vivo + conteos (nunca hardcodear el README)
    headline = {
        "generated_from": "models/registry.json",
        "total_champions": len(champ_records),
        "by_strategy": {},
    }
    for s in STRATEGIES:
        rows = [c for c in champ_records if c["strategy"] == s]
        headline["by_strategy"][s] = {
            "n_champions": len(rows),
            "median_sharpe": _median(headline_acc[s]["bt_sharpe"]),
            "median_directional_accuracy": _median(headline_acc[s]["ml_directional_accuracy"]),
            "median_ic": _median(headline_acc[s]["ml_ic"]),
        }

    (ASSETS / "champions.json").write_text(
        json.dumps(champ_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (ASSETS / "headline.json").write_text(
        json.dumps(headline, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lb_ok = _copy_leaderboard()
    figs = _copy_figures()

    drift_ok = False
    if DRIFT_LOG_SRC.exists():
        shutil.copyfile(DRIFT_LOG_SRC, ASSETS / "drift_log.jsonl")
        drift_ok = True

    # Resumen
    total_mb = sum(f.stat().st_size for f in ASSETS.rglob("*") if f.is_file()) / 1e6
    print("=" * 60)
    print("BUNDLE LISTO — demo/assets/")
    print("=" * 60)
    print(f"Champions:        {len(champ_records)} "
          f"(e1={headline['by_strategy']['e1']['n_champions']}, "
          f"e2={headline['by_strategy']['e2']['n_champions']}, "
          f"e3={headline['by_strategy']['e3']['n_champions']})")
    print(f"Predicciones:     {len(champ_records) - len(missing_preds)} copiadas"
          + (f", {len(missing_preds)} sin CSV ({', '.join(missing_preds)})" if missing_preds else ""))
    print(f"Leaderboard:      {'copiado' if lb_ok else 'NO ENCONTRADO'}")
    print(f"Figuras:          {len(figs)} copiadas")
    print(f"Drift log:        {'copiado' if drift_ok else 'no existe todavía (opcional)'}")
    print(f"Tamaño total:     {total_mb:.1f} MB")
    print(f"Medianas E1:      Sharpe={headline['by_strategy']['e1']['median_sharpe']}, "
          f"DirAcc={headline['by_strategy']['e1']['median_directional_accuracy']}, "
          f"IC={headline['by_strategy']['e1']['median_ic']}")


if __name__ == "__main__":
    build()
