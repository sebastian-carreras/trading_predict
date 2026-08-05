# Catálogo de Features — trading_predict

Referencia única de **todas las features** que consumen los modelos, por estrategia, más las **features exógenas** (macro / cross-asset / sentimiento) que se están incorporando por fases.

Audiencia: comité de tesis, revisión de portfolio y yo-futuro al retomar el código. Complementa a [`docs/METRICS.md`](METRICS.md) (métricas de evaluación) y [`docs/SPEC.md`](SPEC.md) (especificación).

> **Fuente de verdad del código**, no de este documento: el catálogo completo vive en `src/e{1,2,3}/build_features.py::compute_*_features`, y **qué se entrena hoy** lo decide `strategies.<estrategia>.features.active` en `src/config/base.yaml`. Si este README y el código difieren, gana el código.

---

## Conceptos

- **Una feature es por-ticker y se deriva del OHLCV** de ese activo. Cada modelo se entrena por `(estrategia, ticker)` y recibe las features como **secuencias** (ventana `lookback`), no como una matriz plana. Alineación: el target se ubica al **final** de cada ventana.
- **El target NO es una feature.** Es el retorno logarítmico forward `log(close[t+H]/close[t])`, construido por `make_target_*`. Ver [`docs/METRICS.md`](METRICS.md).
- **Catálogo vs activas** — dos conceptos distintos:
  - *Catálogo completo*: todo lo que calcula `compute_*_features`. **Nunca se borra** el cálculo de una feature mientras un champion vivo (`models/registry.json`) la use en su contrato — romperlo tira abajo la re-evaluación fair-window (`src/lifecycle/reevaluation.py`).
  - *Activas*: el subset que entrena modelos nuevos, listado en `features.active` de `base.yaml`. Desactivar = sacar de esa lista, **no** borrar del código.
- **La promoción es por ticker**, así que dos tickers de la misma estrategia pueden terminar con contratos de features distintos. Es esperable, no un bug.

## Convenciones

- **Estacionariedad**: se prefieren deltas, ratios, z-scores y percentiles sobre niveles crudos. El escalado final es **z-score calculado sólo con el fold de train** (sin data snooping).
- **Estados en las tablas**:
  - ✅ **activa** — entra a entrenamientos nuevos hoy.
  - 💤 **en catálogo, desactivada** — se calcula pero no se entrena (motivo entre paréntesis, típicamente multicolinealidad).
  - 🔜 **planificada** — feature exógena nueva, aún no implementada.
- **Point-in-time (PIT)** — aplica sólo a las features exógenas (macro): cada dato se sella con su fecha *known-as-of* (fecha de publicación, no de referencia) y se hace `ffill` al calendario del ticker. Usar el valor por fecha de referencia = filtrar futuro. Ver el plan de exógenas.

---

## E1 — Conservadora · GRU · H=90 días · lookback 360 · 12 activas

`src/e1/build_features.py::compute_e1_features` · config: `strategies.e1_conservative.features.active`

| Feature | Fórmula / definición | Categoría | Estado |
|---|---|---|---|
| `ret_1w` | `log(close).diff(5)` — momentum semanal | Retorno | ✅ |
| `ret_4w` | `log(close).diff(20)` — momentum mensual | Retorno | ✅ |
| `ret_13w` | `log(close).diff(60)` — momentum trimestral (alineado a H=90) | Retorno | ✅ |
| `vol_regime` | `vol_4w / vol_4w[-20] − 1` — expansión/contracción de volatilidad | Volatilidad | ✅ |
| `atr_14` | `mean(TrueRange,14) / close` — ATR normalizado | Volatilidad | ✅ |
| `vol_zscore_60` | z-score del volumen vs 60d — anomalías de volumen | Volumen | ✅ |
| `sma_50` | `mean(close,50)` — tendencia medio plazo | Tendencia | ✅ |
| `sma50_sma200_ratio` | `sma_50/sma_200 − 1` — Golden/Death Cross | Tendencia | ✅ |
| `macd_hist` | `MACD(12,26) − signal(9)` — momentum | Momentum | ✅ |
| `bb_pct_b` | `(close − BB_low)/(BB_up − BB_low)` — posición en Bollinger | Bollinger | ✅ |
| `bb_bandwidth` | `(BB_up − BB_low)/sma20` — ancho de bandas (squeeze) | Volatilidad | ✅ |
| `adx_14` | `mean(DX,14)` — fuerza de tendencia (no dirección) | Fuerza tend. | ✅ |
| `vol_4w` | `std(ret_1d,20)` | Volatilidad | 💤 (r≈0.85 con bb_bandwidth, 0.72 con atr_14) |
| `sma_200` | `mean(close,200)` | Tendencia | 💤 (r≈0.99 con sma_50) |
| `close_sma200_dist` | `close/sma_200 − 1` | Tendencia | 💤 (r≈0.78 sma50_ratio, 0.79 ret_13w) |

## E2 — Moderada · LSTM · H=20 días · lookback 60 · 12 activas

`src/e2/build_features.py::compute_e2_features` · config: `strategies.e2_moderate.features.active`

| Feature | Fórmula / definición | Categoría | Estado |
|---|---|---|---|
| `ret_1d` | `log(close).diff()` — retorno diario | Retorno | ✅ |
| `ret_20d` | `sum(ret_1d,20)` — momentum mensual (alineado a H=20) | Retorno | ✅ |
| `vol_20d` | `std(ret_1d,20)` — volatilidad realizada | Volatilidad | ✅ |
| `atr_14` | `mean(TrueRange,14) / close` | Volatilidad | ✅ |
| `vol_ratio_20d` | `volume / sma(volume,20)` — volumen relativo | Volumen | ✅ |
| `sma50_sma200_ratio` | `sma_50/sma_200 − 1` | Tendencia | ✅ |
| `close_sma50_dist` | `close/sma_50 − 1` — desvío vs tendencia medio plazo | Tendencia | ✅ |
| `macd_hist` | `MACD(12,26) − signal(9)` | Momentum | ✅ |
| `rsi_14` | RSI Wilder (vía EWM) — sobrecompra/sobreventa | Momentum | ✅ |
| `bb_pct_b` | `(close − BB_low)/(BB_up − BB_low)` | Bollinger | ✅ |
| `adx_14` | `mean(DX,14)` | Fuerza tend. | ✅ |
| `skew_ret_20d` | asimetría de `ret_1d` en 20d — riesgo de cola | Riesgo | ✅ |
| `ret_5d` | `sum(ret_1d,5)` | Retorno | 💤 (r≈0.7–0.85 con ret_1d/ret_20d) |
| `ret_10d` | `sum(ret_1d,10)` | Retorno | 💤 (r≈0.7–0.85 con ret_1d/ret_20d) |
| `close_sma200_dist` | `close/sma_200 − 1` | Tendencia | 💤 (r≈0.78) |
| `bb_bandwidth` | `(BB_up − BB_low)/sma20` | Volatilidad | 💤 (r≈0.85–0.95 con vol_20d) |

## E3 — Intradía · LSTM · H=6 barras (30 min) · lookback 96 · 12 features

`src/e3/build_features.py::compute_intraday_features` · estrategia `enabled: false` (resultado negativo documentado, ver `README.md`). Acá las features desactivadas están **comentadas** en el código, no en `features.active`.

| Feature | Fórmula / definición | Categoría | Estado |
|---|---|---|---|
| `ret_1` | `log(close).diff()` — retorno por barra | Retorno | ✅ |
| `ret_12m` | `sum(ret_1,12)` — momentum ~1h | Retorno | ✅ |
| `vol_24` | `std(ret_1,24)` — volatilidad ~2h | Volatilidad | ✅ |
| `range_pct` | `(high − low)/close` — rango de barra | Volatilidad | ✅ |
| `vol_z_96` | z-score de volumen vs 96 barras | Volumen | ✅ |
| `tod_sin`,`tod_cos` | hora del día (encoding cíclico) | Calendario | ✅ |
| `dow_sin`,`dow_cos` | día de la semana (encoding cíclico) | Calendario | ✅ |
| `vwap_dev` | `(close − VWAP_diario)/VWAP` — desvío del VWAP | Precio/valor | ✅ |
| `vol_ratio` | `vol_24 / vol_96` — régimen de volatilidad corto/largo | Volatilidad | ✅ |
| `rsi_14` | RSI Wilder | Momentum | ✅ |
| `vol_96`, `atr_14`, `macd_hist`, `bb_pct_b`, `close_sma78_dist` | — | varias | 💤 eliminadas por multicolinealidad (EDA §3.2) |

---

## Features exógenas (macro / cross-asset / sentimiento)

Contexto que antes **no existía** en el modelo: ni VIX, ni tasas, ni crédito, ni sector, ni commodities, ni sentimiento. A 10–90 días el régimen macro y el apetito por riesgo explican una fracción grande del retorno. Se incorporan **por fases** sobre **E2** primero, con PIT estricto y activación por ablación.

> **Estado (2026-08-02) — Fase 1 CERRADA, NO activada (`data.exog.enabled: false`).** Conclusión
> matizada: con hiperparámetros **re-tuneados para el set exógeno** (Optuna `--exog`), in-sample
> las exógenas **SÍ mejoran** E2 (ablación justa v3: +0.036 score, Sharpe/Calmar de − a +) — el
> "negativo" previo era un confound de thresholds del baseline. **Pero out-of-sample** (held-out
> multi-seed, train ≤2023 / eval 2024-2026) el beneficio **no generaliza per-ticker** (winners Δ
> mediana −0.134, sobrevivientes inestables entre seeds). → sin allowlist defendible, se deja
> desactivado. Los `.BA` nunca se beneficiaron → Fase 3. La **infraestructura queda** (fetch macro
> + PIT + guardrail cobertura + Optuna exog-aware + arnés ablación/held-out multi-seed) para Fase
> 2/3. Detalle: [reports/ablation/ablation_report.md](../reports/ablation/ablation_report.md).

- **Infra nueva**: `src/data/macro.py` (FRED vía `fredapi`/`pandas_datareader` + yfinance para índices/ETFs/futuros), cache en `data/macro/`, wiring en `src/data/ingest.py`, tabla ticker→sector/ADR en `src/config/sectors.yaml`.
- **Integración**: `compute_e2_features(df, exog=None)` (compatible hacia atrás) hace left-join del `exog` alineado por PIT; se pasa desde `src/e2/train_pipeline.py`.
- **Scope**: *global* = mismo valor para todos los tickers en una fecha (timea el régimen); *per-ticker* = específico del activo (rankea nombres); *.BA* = sólo tickers argentinos.
- ✅ marca el **núcleo** a activar primero; el resto entra por ablación.

### FASE 1 — Bloques A + B + C + D (foco actual)

**Bloque A — Cross-asset / apetito por riesgo** (global · yfinance + FRED)

| Feature | Definición / fuente | Núcleo |
|---|---|---|
| `vix_ts` | term structure `^VIX3M/^VIX − 1` (contango/backwardation) | ✅ |
| `vix_pctl_252` | percentil rolling 252d del `^VIX` | ✅ |
| `hy_oas_z` | z-score 120d del OAS high-yield (FRED `BAMLH0A0HYM2`); el crédito anticipa a la equity | ✅ |
| `curve_10y2y` | pendiente `T10Y2Y` (FRED) — señal recesiva multi-mes | ✅ |
| `dxy_mom_20` | momentum 20d del dólar (`DX-Y.NYB`/`UUP`) | ✅ |
| `spy_mom_20`, `spy_mom_60` | momentum del mercado (contexto de beta) | ✅ |
| `rate_shock_10y` | variación 20d del 10Y (`DGS10`) | |

**Bloque B — Fuerza relativa sectorial** (per-ticker · yfinance + `sectors.yaml`)

| Feature | Definición | Núcleo |
|---|---|---|
| `rel_strength_sector` | `ret_20d_ticker − ret_20d_ETFsector` — ¿lidera o rezaga a su grupo? | ✅ |
| `sector_rotation` | `ret_20d_ETFsector − ret_20d_SPY` — ¿el grupo está en favor? | ✅ |
| `beta_60` | beta rolling 60d del ticker vs SPY | |

ETFs sectoriales: XLK, XLF, XLV, XLE, XLY, XLP, XLI, XLC, XLU, XLB, XLRE.

**Bloque C — Macro fundamental** (global · FRED · diario/semanal, forward-looking, baja revisión)

| Feature | Definición / fuente | Núcleo |
|---|---|---|
| `breakeven_10y` | inflación esperada 10Y (`T10YIE`), diaria, sin lag → mejor que el CPI print | ✅ |
| `jobless_claims_z` | z-score 52w de initial claims (`ICSA`), semanal, leading, casi sin revisión | ✅ |
| `nfci` | National Financial Conditions Index (`NFCI`) — resume crédito+tasas+vol en 1 feature | ✅ |
| `cpi_yoy`, `unrate_delta` | prints mensuales — **sólo con PIT estricto**; alto riesgo de leakage, aporte marginal | |

**Bloque D — Commodities** (global/per-ticker · yfinance futuros/ETF)

| Feature | Definición / fuente | Núcleo |
|---|---|---|
| `oil_mom_20` | momentum WTI (`CL=F`/`USO`) — mueve energía y YPFD.BA/XOM/CVX/PAMP | ✅ |
| `copper_mom_20` | `HG=F` — "Dr. Copper", proxy de crecimiento global | |
| `gold_mom_20` | `GC=F` — proxy tasas reales / risk-off | |
| `soy_mom_20` | `ZS=F` — reservas/FX de Argentina | |

### Fases siguientes (reservadas hasta validar la Fase 1)

| Fase | Bloque | Features (resumen) | Nota |
|---|---|---|---|
| 2 | **E — Sentimiento de mercado** | `putcall_z` (put/call CBOE), `aaii_bull_bear` (survey semanal, contrarian) | Fear&Greed de CNN es reconstruible desde A+E → no scrapearlo |
| 2 | **F — Eventos / calendario** | `days_to_earnings`, `days_to_fomc`, `month_sin`/`month_cos` | Cero leakage: fechas conocidas de antemano |
| 3 | **G — Macro Argentina** (.BA) | `ccl_mom_20`/`ccl_brecha` (vía ratio ADR), `riesgo_pais_proxy` (bonos AL30/GD30), `merval_mom_20` | Track aparte; 0/neutral para tickers US |
| 4 | **H — News-sentiment NLP** | `news_sent_7d` (FinBERT sobre titulares), `news_vol_7d` | ⚠️ historia sólo ~2022+ → resolver hueco 2016–2021 |
| 4 | **I — Social / Trends** | `gtrends_interest` (pytrends), `wsb_mentions_z` (Reddit) | Experimental, ruidoso, más señal intradía |

---

## Fuentes de datos

| Fuente | Uso | Módulo |
|---|---|---|
| **yfinance** | OHLCV diario (todas), fallback intradía, ^VIX/ETFs/futuros (exógenas) | `src/data/download_daily.py` |
| **IOL (InvertirOnline)** | Fallback/complemento para activos argentinos `.BA` y bonos | `src/data/iol_api.py` |
| **Alpaca** | OHLCV intradía 5-min (E3) | `src/e3/intraday_data.py` |
| **FRED** 🔜 | Macro: breakevens, OAS, curva, claims, NFCI, put/call | `src/data/macro.py` (nuevo) |
| **Alpha Vantage / GDELT** 🔜 | News-sentiment (Bloque H) | `src/data/macro.py` (nuevo) |

## Cómo agregar una feature

1. Agregar el cálculo al catálogo en `src/e{1,2,3}/build_features.py::compute_*_features` (**nunca** borrar una existente si un champion vivo la usa).
2. Para exógenas: sumar la serie a `src/data/macro.py` con su offset de publicación (PIT) y su transform estacionario.
3. Activarla en `strategies.<estrategia>.features.active` de `src/config/base.yaml`.
4. Validar por ablación: entrenar (`python -m src.e2.train_pipeline`), registrar candidate, comparar vía `scripts/evaluation/leaderboard.py` contra el baseline congelado (score = 0.35·Sharpe + 0.25·IC + 0.20·DirAcc + 0.20·Calmar).
