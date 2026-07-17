"""Demo Gradio del flagship trading_predict — showcase de portfolio.

Lector FINO y self-contained: lee SOLO ``assets/`` (generado por
``demo/bundle_assets.py``). No importa ``src/``, no usa torch/mlflow, no necesita
secretos. Desplegable tal cual a Hugging Face Spaces (SDK: Gradio).

Correr local:   python demo/app.py
Regenerar data: python -m demo.bundle_assets
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import plotly.express as px

ASSETS = Path(__file__).parent / "assets"

STRATEGY_META = {
    "e1": {"name": "E1 · Conservadora", "desc": "GRU · horizonte 90 días · diario"},
    "e2": {"name": "E2 · Moderada", "desc": "LSTM · horizonte 20 días · diario"},
    "e3": {"name": "E3 · Intradía", "desc": "LSTM ensemble · 30 min · resultado negativo documentado"},
}
STRATS = ("e1", "e2", "e3")

# Región de las figuras de retorno acumulado por estrategia (AR = BYMA, US = US).
REGION_FIG = {"e1": "ar", "e2": "us", "e3": "us"}


# ----------------------------------------------------------------------
# Carga de assets
# ----------------------------------------------------------------------

def load_json(name: str) -> Any:
    path = ASSETS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@lru_cache(maxsize=256)
def load_csv(rel: str) -> pd.DataFrame | None:
    path = ASSETS / rel
    return pd.read_csv(path) if path.exists() else None


def fig_file(name: str) -> str | None:
    p = ASSETS / "figures" / name
    return str(p) if p.exists() else None


def fmt(value: Any, pct: bool = False, decimals: int = 2) -> str:
    """Formatea un número manejando None/NaN (los E3 tienen métricas nulas)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{v * 100:.1f}%" if pct else f"{v:.{decimals}f}"


champions: list[dict[str, Any]] = load_json("champions.json") or []
headline: dict[str, Any] = load_json("headline.json") or {}
leaderboard = load_csv("leaderboard.csv")


# ----------------------------------------------------------------------
# Helpers de layout
# ----------------------------------------------------------------------

def image_or_note(name: str, label: str) -> None:
    """Coloca una figura del bundle si existe; si no, una nota discreta."""
    path = fig_file(name)
    if path is not None:
        gr.Image(value=path, label=label)
    else:
        gr.Markdown(f"_(figura no disponible: {name})_")


# ----------------------------------------------------------------------
# Callbacks del tab Performance (drilldown interactivo)
# ----------------------------------------------------------------------

LB_COLS = ["rank", "ticker", "score", "bt_sharpe", "ml_ic",
           "ml_directional_accuracy", "bt_calmar", "bt_max_drawdown", "signal", "trend"]


def _leaderboard_from_champions(strat: str) -> pd.DataFrame:
    """Tabla de fallback desde champions.json (para estrategias que el leaderboard
    diario no calcula, p.ej. E3). Ordenada por Sharpe descendente."""
    rows = []
    for c in champions:
        if c["strategy"] != strat:
            continue
        m = c.get("metrics", {})
        rows.append({
            "ticker": c["ticker"],
            "bt_sharpe": m.get("bt_sharpe"),
            "ml_ic": m.get("ml_ic"),
            "ml_directional_accuracy": m.get("ml_directional_accuracy"),
            "bt_calmar": m.get("bt_calmar"),
            "bt_max_drawdown": m.get("bt_max_drawdown"),
            "signal": c.get("signal"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("bt_sharpe", ascending=False, na_position="last").reset_index(drop=True)
    return df


def _leaderboard_for(strat: str) -> pd.DataFrame:
    if leaderboard is not None:
        lb = leaderboard[leaderboard["strategy"] == strat]
        if not lb.empty:
            cols = [c for c in LB_COLS if c in lb.columns]
            return lb[cols].reset_index(drop=True)
    return _leaderboard_from_champions(strat)


def _equity_fig(champ: dict[str, Any]):
    bt = load_csv(champ["backtest_csv"]) if champ.get("backtest_csv") else None
    if bt is None or "equity" not in bt.columns:
        return None
    bt = bt.copy()
    bt["timestamp"] = pd.to_datetime(bt["timestamp"], errors="coerce")
    fig = px.line(bt, x="timestamp", y="equity",
                  title="Curva de equity (walk-forward, neta de costos)")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def _preds_fig(champ: dict[str, Any]):
    preds = load_csv(champ["predictions_csv"]) if champ.get("predictions_csv") else None
    if preds is None or not {"y_true", "y_pred"}.issubset(preds.columns):
        return None
    preds = preds.copy()
    preds["timestamp"] = pd.to_datetime(preds["timestamp"], errors="coerce")
    long = preds.melt(id_vars="timestamp", value_vars=["y_true", "y_pred"],
                      var_name="serie", value_name="retorno")
    fig = px.line(long, x="timestamp", y="retorno", color="serie",
                  title="Predicho vs real (retorno out-of-sample)")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.1))
    return fig


def _metrics_md(champ: dict[str, Any]) -> str:
    m = champ.get("metrics", {})
    thr = champ.get("thresholds", {})
    return (
        f"**{champ['ticker']}** · variant `{champ.get('variant')}`\n\n"
        "| Sharpe | Dir. acc. | IC | Calmar | Max DD | Ret. total | CAGR | Señal |\n"
        "|--------|-----------|----|--------|--------|------------|------|-------|\n"
        f"| {fmt(m.get('bt_sharpe'))} | {fmt(m.get('ml_directional_accuracy'), pct=True)} "
        f"| {fmt(m.get('ml_ic'), decimals=3)} | {fmt(m.get('bt_calmar'))} "
        f"| {fmt(m.get('bt_max_drawdown'), pct=True)} | {fmt(m.get('bt_total_return'), pct=True)} "
        f"| {fmt(m.get('bt_cagr'), pct=True)} | **{champ.get('signal', '—')}** |\n\n"
        f"_Última predicción walk-forward: **{fmt(champ.get('predicted_return'), pct=True)}** "
        f"al {str(champ.get('prediction_date'))[:10]} · "
        f"τ_buy={thr.get('tau_buy')} / τ_sell={thr.get('tau_sell')}. "
        "Es la última predicción del walk-forward, no una inferencia en tiempo real._"
    )


def ticker_view(strat: str, ticker: str | None):
    champ = next((c for c in champions if c["strategy"] == strat and c["ticker"] == ticker), None)
    if champ is None:
        return "_(sin champion)_", None, None
    return _metrics_md(champ), _equity_fig(champ), _preds_fig(champ)


def strategy_view(strat: str):
    """Actualiza tabla + opciones de ticker + drilldown + figuras de apoyo."""
    tickers = sorted(c["ticker"] for c in champions if c["strategy"] == strat)
    first = tickers[0] if tickers else None
    md, eq, pv = ticker_view(strat, first)
    heatmap = fig_file(f"fig_conclusions_monthly_heatmap_{strat}.png")
    cumret = fig_file(f"fig_conclusions_cumulative_return_{REGION_FIG[strat]}.png")
    return (
        _leaderboard_for(strat),
        gr.update(choices=tickers, value=first),
        md, eq, pv, heatmap, cumret,
    )


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

def build() -> gr.Blocks:
    by_strat = headline.get("by_strategy", {})

    with gr.Blocks(title="trading_predict — ML Trading MLOps Demo") as demo:
        gr.Markdown(
            "# 📈 trading_predict — sistema MLOps de trading con ML\n"
            "Predicción de retornos con deep learning (GRU/LSTM) sobre 3 estrategias, con registry "
            "de modelos, promoción champion/challenger, validación **walk-forward** anti-leakage y "
            "monitoreo de drift. Proyecto de portfolio (final CEIA-FIUBA)."
        )
        gr.Markdown(
            "> 🎓 **Demo de portfolio, no una herramienta de trading.** No ejecuta órdenes ni da "
            "consejo financiero. Todas las cifras son resultados **out-of-sample** del walk-forward "
            "de los modelos champion, leídas de artefactos pre-calculados."
        )

        with gr.Tabs():
            # ---- Overview ----
            with gr.Tab("Overview"):
                gr.Markdown(
                    f"### {headline.get('total_champions', len(champions))} modelos champion en producción\n"
                    "_Medianas recalculadas en vivo desde `models/registry.json` (no hardcodeadas)._"
                )
                rows = []
                for s in STRATS:
                    info = by_strat.get(s, {})
                    rows.append(
                        f"| **{STRATEGY_META[s]['name']}** | {info.get('n_champions', 0)} "
                        f"| {fmt(info.get('median_sharpe'))} "
                        f"| {fmt(info.get('median_directional_accuracy'), pct=True)} "
                        f"| {fmt(info.get('median_ic'), decimals=3)} |"
                    )
                gr.Markdown(
                    "| Estrategia | Champions | Sharpe (mediana) | Dir. acc. (mediana) | IC (mediana) |\n"
                    "|------------|-----------|------------------|---------------------|--------------|\n"
                    + "\n".join(rows)
                )
                with gr.Row():
                    image_or_note("fig_sharpe_boxplot_by_strategy.png", "Distribución de Sharpe por estrategia")
                    image_or_note("fig_risk_return_scatter.png", "Riesgo vs retorno por champion")
                with gr.Accordion("🏗️ Arquitectura MLOps del sistema", open=False):
                    image_or_note("architecture.png", "Pipeline: ingesta → features → walk-forward → registry → serving")

            # ---- Performance por estrategia ----
            with gr.Tab("Performance por estrategia"):
                strat_dd = gr.Dropdown(
                    choices=[(STRATEGY_META[s]["name"], s) for s in STRATS],
                    value="e1", label="Estrategia",
                )
                lb_table = gr.Dataframe(label="Leaderboard de champions", interactive=False, wrap=True)
                gr.Markdown("#### 🔎 Drilldown por ticker")
                ticker_dd = gr.Dropdown(label="Ticker", choices=[], value=None)
                metrics_md = gr.Markdown()
                with gr.Row():
                    equity_plot = gr.Plot(label="Equity")
                    preds_plot = gr.Plot(label="Predicho vs real")
                with gr.Row():
                    heatmap_img = gr.Image(label="Retornos mensuales")
                    cumret_img = gr.Image(label="Retorno acumulado")

                perf_outputs = [lb_table, ticker_dd, metrics_md, equity_plot, preds_plot, heatmap_img, cumret_img]
                strat_dd.change(strategy_view, inputs=strat_dd, outputs=perf_outputs)
                ticker_dd.change(ticker_view, inputs=[strat_dd, ticker_dd],
                                 outputs=[metrics_md, equity_plot, preds_plot])
                demo.load(strategy_view, inputs=strat_dd, outputs=perf_outputs)

            # ---- Champion vs Baseline ----
            with gr.Tab("Champion vs Baseline"):
                gr.Markdown(
                    "Cada champion se compara contra un **baseline fijo** (regresión lineal para E1) que "
                    "actúa como piso de sanidad y nunca se promueve. La promoción exige superar al champion "
                    "vigente por ≥5% en un score compuesto "
                    "(`0.35·Sharpe + 0.25·IC + 0.20·DirAcc + 0.20·Calmar`)."
                )
                for s in STRATS:
                    gr.Markdown(f"#### {STRATEGY_META[s]['name']}")
                    with gr.Row():
                        image_or_note(f"fig_{s}_vs_baseline_ml.png", "Calidad ML")
                        image_or_note(f"fig_{s}_vs_baseline_trading_retorno.png", "Retorno")
                        image_or_note(f"fig_{s}_vs_baseline_trading_riesgo.png", "Riesgo")

            # ---- Drift & Estabilidad ----
            with gr.Tab("Drift & Estabilidad"):
                gr.Markdown(
                    "El sistema monitorea si los champions **se degradan con el tiempo** (concept drift) y "
                    "si la distribución de las features **se corre** respecto del entrenamiento (data drift, "
                    "vía tests **KS/PSI** en `src/lifecycle/drift.py`)."
                )
                if leaderboard is not None and "trend" in leaderboard.columns:
                    counts = leaderboard["trend"].value_counts().to_dict()
                    gr.Markdown(
                        "| ▲ Mejorando | ▼ Empeorando | = Estables | ★ Nuevos |\n"
                        "|-------------|--------------|------------|----------|\n"
                        f"| {counts.get('up', 0)} | {counts.get('down', 0)} "
                        f"| {counts.get('flat', 0)} | {counts.get('new', 0)} |\n\n"
                        "_Δ del score compuesto de cada champion vs. el snapshot del día anterior._"
                    )
                with gr.Row():
                    image_or_note("fig_champion_stability_sharpe.png", "Evolución del Sharpe de los champions")
                    image_or_note("fig_champion_stability_ic.png", "Evolución del IC de los champions")
                if (ASSETS / "drift_log.jsonl").exists():
                    lines = [json.loads(l) for l in (ASSETS / "drift_log.jsonl").read_text().splitlines() if l.strip()]
                    if lines:
                        last = lines[-1]
                        gr.Markdown("#### Último reporte de data drift (KS/PSI)")
                        gr.JSON(value={k: last[k] for k in ("strategy", "ticker", "n_drifted", "worst_feature", "worst_psi") if k in last})
                else:
                    gr.Markdown(
                        "_El log de drift (`models/drift_log.jsonl`) se genera al correr el módulo KS/PSI "
                        "sobre datos recientes; todavía no está incluido en este bundle._"
                    )

            # ---- Metodología ----
            with gr.Tab("Metodología"):
                gr.Markdown(
                    "#### Validación honesta, anti-leakage\n"
                    "- **Walk-forward CV con embargo** (≥ horizonte de predicción) entre train y test: nunca "
                    "se evalúa con información que el modelo no podría haber tenido.\n"
                    "- **Z-score calculado solo sobre el fold de train** (las stats no ven val/test → sin data "
                    "snooping).\n"
                    "- **Loss de Huber** (robusta a outliers de precio), no MSE.\n"
                    "- **Registry como única fuente de verdad**, escrituras atómicas, promoción "
                    "champion/challenger con guardrails (rechaza NaN/Inf, Sharpe < 0, IC peor que baseline)."
                )
                e3 = by_strat.get("e3", {})
                gr.Markdown(
                    f"> ⚠️ **Resultado negativo documentado (E3 intradía):** Sharpe mediano "
                    f"**{fmt(e3.get('median_sharpe'))}** — la estrategia intradía **no supera los costos** de "
                    "transacción. Se reporta como tal (no se esconde): un resultado negativo honesto vale más "
                    "que uno maquillado."
                )
                gr.Markdown("#### Diagnósticos del modelo")
                with gr.Row():
                    image_or_note("fig_conclusions_walkforward_stability.png", "Estabilidad walk-forward")
                    image_or_note("fig_conclusions_rolling_ic.png", "IC rolling")
                    image_or_note("fig_conclusions_calibration.png", "Calibración")
                gr.Markdown("#### Optimización de hiperparámetros (Optuna)")
                with gr.Row():
                    image_or_note("param_importances.png", "Importancia de hiperparámetros")
                    image_or_note("optimization_history.png", "Historia de optimización")
                gr.Markdown(
                    "---\nCódigo: [github.com/sebastian-carreras/trading_predict]"
                    "(https://github.com/sebastian-carreras/trading_predict) · "
                    "Stack: PyTorch · scikit-learn · MLflow · Airflow · FastAPI · Docker"
                )

    return demo


if __name__ == "__main__":
    build().launch()
