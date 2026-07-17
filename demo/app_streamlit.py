"""Demo Streamlit del flagship trading_predict — showcase de portfolio.

Lector FINO y self-contained: lee SOLO ``assets/`` (generado por
``demo/bundle_assets.py``). No importa ``src/``, no usa torch/mlflow, no necesita
secretos. Desplegable tal cual a Hugging Face Spaces.

Correr local:   streamlit run demo/app.py
Regenerar data: python -m demo.bundle_assets
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

ASSETS = Path(__file__).parent / "assets"

STRATEGY_META = {
    "e1": {"name": "E1 · Conservadora", "desc": "GRU · horizonte 90 días · diario"},
    "e2": {"name": "E2 · Moderada", "desc": "LSTM · horizonte 20 días · diario"},
    "e3": {"name": "E3 · Intradía", "desc": "LSTM ensemble · 30 min · resultado negativo documentado"},
}

st.set_page_config(page_title="trading_predict — ML trading demo", page_icon="📈", layout="wide")


# ----------------------------------------------------------------------
# Carga de assets (cacheada)
# ----------------------------------------------------------------------

@st.cache_data
def load_json(name: str) -> Any:
    path = ASSETS / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_csv(rel: str) -> pd.DataFrame | None:
    path = ASSETS / rel
    if not path.exists():
        return None
    return pd.read_csv(path)


def fig_path(name: str) -> Path | None:
    p = ASSETS / "figures" / name
    return p if p.exists() else None


def show_fig(name: str, caption: str | None = None, width: str = "stretch") -> None:
    """Muestra una figura del bundle si existe; si no, un aviso discreto."""
    p = fig_path(name)
    if p is not None:
        st.image(str(p), caption=caption, use_container_width=(width == "stretch"))
    else:
        st.caption(f"_(figura no disponible: {name})_")


def fmt(value: Any, pct: bool = False, decimals: int = 2) -> str:
    """Formatea un número manejando None/NaN (los E3 tienen métricas nulas)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{v * 100:.1f}%" if pct else f"{v:.{decimals}f}"


# ----------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------

champions: list[dict[str, Any]] = load_json("champions.json") or []
headline: dict[str, Any] = load_json("headline.json") or {}
leaderboard = load_csv("leaderboard.csv")

if not champions:
    st.error("No se encontraron assets. Corré `python -m demo.bundle_assets` primero.")
    st.stop()


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------

st.title("📈 trading_predict — sistema MLOps de trading con ML")
st.markdown(
    "Sistema de **predicción de retornos** con deep learning (GRU/LSTM) sobre 3 estrategias, "
    "con registry de modelos, promoción champion/challenger, validación **walk-forward** "
    "anti-leakage y monitoreo de drift. "
    "Proyecto de portfolio (final CEIA-FIUBA)."
)
st.info(
    "🎓 **Demo de portfolio, no una herramienta de trading.** No ejecuta órdenes ni da consejo "
    "financiero. Todas las cifras son resultados **out-of-sample** del walk-forward de los modelos "
    "champion, leídas de artefactos pre-calculados.",
    icon="ℹ️",
)

tab_overview, tab_perf, tab_vs, tab_drift, tab_method = st.tabs(
    ["Overview", "Performance por estrategia", "Champion vs Baseline", "Drift & Estabilidad", "Metodología"]
)


# ----------------------------------------------------------------------
# Tab 1 — Overview
# ----------------------------------------------------------------------

with tab_overview:
    by_strat = headline.get("by_strategy", {})
    st.subheader(f"{headline.get('total_champions', len(champions))} modelos champion en producción")
    st.caption("Medianas recalculadas en vivo desde `models/registry.json` (no hardcodeadas).")

    cols = st.columns(3)
    for col, s in zip(cols, ("e1", "e2", "e3")):
        info = by_strat.get(s, {})
        with col:
            st.markdown(f"**{STRATEGY_META[s]['name']}**")
            st.caption(STRATEGY_META[s]["desc"])
            st.metric("Champions", info.get("n_champions", 0))
            st.metric("Sharpe (mediana)", fmt(info.get("median_sharpe")))
            st.metric("Directional accuracy (mediana)", fmt(info.get("median_directional_accuracy"), pct=True))

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        show_fig("fig_sharpe_boxplot_by_strategy.png", "Distribución de Sharpe por estrategia")
    with c2:
        show_fig("fig_risk_return_scatter.png", "Riesgo vs retorno por champion")

    with st.expander("🏗️ Arquitectura MLOps del sistema"):
        show_fig("architecture.png", "Pipeline: ingesta → features → walk-forward → registry → serving")


# ----------------------------------------------------------------------
# Tab 2 — Performance por estrategia (con drilldown interactivo)
# ----------------------------------------------------------------------

with tab_perf:
    strat = st.selectbox(
        "Estrategia",
        options=("e1", "e2", "e3"),
        format_func=lambda s: STRATEGY_META[s]["name"],
    )
    st.caption(STRATEGY_META[strat]["desc"])

    strat_champs = [c for c in champions if c["strategy"] == strat]

    # Tabla del leaderboard filtrada
    if leaderboard is not None:
        lb = leaderboard[leaderboard["strategy"] == strat].copy()
        show_cols = [c for c in
                     ["rank", "ticker", "score", "bt_sharpe", "ml_ic",
                      "ml_directional_accuracy", "bt_calmar", "bt_max_drawdown", "signal", "trend"]
                     if c in lb.columns]
        st.dataframe(
            lb[show_cols].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.markdown("#### 🔎 Drilldown por ticker")
    tickers = sorted(c["ticker"] for c in strat_champs)
    ticker = st.selectbox("Ticker", options=tickers)
    champ = next(c for c in strat_champs if c["ticker"] == ticker)
    m = champ.get("metrics", {})

    # Cards de métricas del champion
    cc = st.columns(4)
    cc[0].metric("Sharpe", fmt(m.get("bt_sharpe")))
    cc[1].metric("Directional acc.", fmt(m.get("ml_directional_accuracy"), pct=True))
    cc[2].metric("IC", fmt(m.get("ml_ic"), decimals=3))
    cc[3].metric("Calmar", fmt(m.get("bt_calmar")))
    cc2 = st.columns(4)
    cc2[0].metric("Retorno total", fmt(m.get("bt_total_return"), pct=True) if m.get("bt_total_return") is not None else "—")
    cc2[1].metric("CAGR", fmt(m.get("bt_cagr"), pct=True) if m.get("bt_cagr") is not None else "—")
    cc2[2].metric("Max drawdown", fmt(m.get("bt_max_drawdown"), pct=True))
    cc2[3].metric("Señal (última pred.)", champ.get("signal", "—"))

    st.caption(
        f"Última predicción walk-forward: **{fmt(champ.get('predicted_return'), pct=True)}** "
        f"al {str(champ.get('prediction_date'))[:10]} · umbrales "
        f"τ_buy={champ['thresholds'].get('tau_buy')} / τ_sell={champ['thresholds'].get('tau_sell')}. "
        "Es la última predicción del walk-forward, no una inferencia en tiempo real."
    )

    # Equity curve real (neta de costos) + pred vs actual
    g1, g2 = st.columns(2)
    with g1:
        bt = load_csv(champ["backtest_csv"]) if champ.get("backtest_csv") else None
        if bt is not None and "equity" in bt.columns:
            bt = bt.copy()
            bt["timestamp"] = pd.to_datetime(bt["timestamp"], errors="coerce")
            fig = px.line(bt, x="timestamp", y="equity", title="Curva de equity (walk-forward, neta de costos)")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("_(sin curva de equity para este ticker)_")
    with g2:
        preds = load_csv(champ["predictions_csv"]) if champ.get("predictions_csv") else None
        if preds is not None and {"y_true", "y_pred"}.issubset(preds.columns):
            preds = preds.copy()
            preds["timestamp"] = pd.to_datetime(preds["timestamp"], errors="coerce")
            long = preds.melt(
                id_vars="timestamp", value_vars=["y_true", "y_pred"],
                var_name="serie", value_name="retorno",
            )
            fig2 = px.line(long, x="timestamp", y="retorno", color="serie",
                           title="Predicho vs real (retorno out-of-sample)")
            fig2.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10),
                               legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.caption("_(sin predicciones para este ticker)_")

    # Figuras de apoyo por estrategia
    st.divider()
    region_fig = {"e1": "ar", "e2": "us", "e3": "us"}[strat]
    f1, f2 = st.columns(2)
    with f1:
        show_fig(f"fig_conclusions_monthly_heatmap_{strat}.png", "Retornos mensuales")
    with f2:
        show_fig(f"fig_conclusions_cumulative_return_{region_fig}.png", "Retorno acumulado")


# ----------------------------------------------------------------------
# Tab 3 — Champion vs Baseline
# ----------------------------------------------------------------------

with tab_vs:
    st.markdown(
        "Cada champion se compara contra un **baseline fijo** (regresión lineal para E1) que actúa "
        "como piso de sanidad y nunca se promueve. La promoción exige superar al champion vigente por "
        "≥5% en un score compuesto (`0.35·Sharpe + 0.25·IC + 0.20·DirAcc + 0.20·Calmar`)."
    )
    for s in ("e1", "e2", "e3"):
        st.markdown(f"#### {STRATEGY_META[s]['name']}")
        cols = st.columns(3)
        with cols[0]:
            show_fig(f"fig_{s}_vs_baseline_ml.png", "Calidad ML")
        with cols[1]:
            show_fig(f"fig_{s}_vs_baseline_trading_retorno.png", "Retorno")
        with cols[2]:
            show_fig(f"fig_{s}_vs_baseline_trading_riesgo.png", "Riesgo")


# ----------------------------------------------------------------------
# Tab 4 — Drift & Estabilidad
# ----------------------------------------------------------------------

with tab_drift:
    st.markdown(
        "El sistema monitorea si los champions **se degradan con el tiempo** (concept drift) y si la "
        "distribución de las features **se corre** respecto del entrenamiento (data drift, vía tests "
        "**KS/PSI** en `src/lifecycle/drift.py`)."
    )

    if leaderboard is not None and "trend" in leaderboard.columns:
        counts = leaderboard["trend"].value_counts().to_dict()
        d = st.columns(4)
        d[0].metric("▲ Mejorando", counts.get("up", 0))
        d[1].metric("▼ Empeorando", counts.get("down", 0))
        d[2].metric("= Estables", counts.get("flat", 0))
        d[3].metric("★ Nuevos", counts.get("new", 0))
        st.caption("Δ del score compuesto de cada champion vs. el snapshot del día anterior.")

    st.divider()
    s1, s2 = st.columns(2)
    with s1:
        show_fig("fig_champion_stability_sharpe.png", "Evolución del Sharpe de los champions")
    with s2:
        show_fig("fig_champion_stability_ic.png", "Evolución del IC de los champions")

    drift_log = ASSETS / "drift_log.jsonl"
    if drift_log.exists():
        st.divider()
        st.markdown("#### Último reporte de data drift (KS/PSI)")
        lines = [json.loads(l) for l in drift_log.read_text().splitlines() if l.strip()]
        if lines:
            last = lines[-1]
            st.json({k: last[k] for k in ("strategy", "ticker", "n_drifted", "worst_feature", "worst_psi") if k in last})
    else:
        st.caption(
            "_El log de drift (`models/drift_log.jsonl`) se genera al correr el módulo KS/PSI sobre "
            "datos recientes; todavía no está incluido en este bundle._"
        )


# ----------------------------------------------------------------------
# Tab 5 — Metodología
# ----------------------------------------------------------------------

with tab_method:
    st.markdown(
        """
#### Validación honesta, anti-leakage
- **Walk-forward CV con embargo** (≥ horizonte de predicción) entre train y test: nunca se
  evalúa con información que el modelo no podría haber tenido.
- **Z-score calculado solo sobre el fold de train** (las stats no ven val/test → sin data snooping).
- **Loss de Huber** (robusta a outliers de precio), no MSE.
- **Registry como única fuente de verdad**, escrituras atómicas, promoción champion/challenger
  con guardrails (rechaza NaN/Inf, Sharpe < 0, IC peor que baseline).
"""
    )

    e3 = headline.get("by_strategy", {}).get("e3", {})
    st.warning(
        f"**Resultado negativo documentado (E3 intradía):** Sharpe mediano "
        f"**{fmt(e3.get('median_sharpe'))}** — la estrategia intradía **no supera los costos** de "
        "transacción. Se reporta como tal (no se esconde): un resultado negativo honesto vale más "
        "que uno maquillado.",
        icon="⚠️",
    )

    st.markdown("#### Diagnósticos del modelo")
    mcols = st.columns(3)
    with mcols[0]:
        show_fig("fig_conclusions_walkforward_stability.png", "Estabilidad walk-forward")
    with mcols[1]:
        show_fig("fig_conclusions_rolling_ic.png", "IC rolling")
    with mcols[2]:
        show_fig("fig_conclusions_calibration.png", "Calibración")

    st.markdown("#### Optimización de hiperparámetros (Optuna)")
    ocols = st.columns(2)
    with ocols[0]:
        show_fig("param_importances.png", "Importancia de hiperparámetros")
    with ocols[1]:
        show_fig("optimization_history.png", "Historia de optimización")

    st.divider()
    st.caption(
        "Código: github.com/sebastian-carreras/trading_predict · "
        "Stack: PyTorch · scikit-learn · MLflow · Airflow · FastAPI · Docker"
    )
