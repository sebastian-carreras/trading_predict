# Plan: Mejora del set de features de E2 para incrementar IC

## Context

**Problema observado.** El IC (Spearman rank correlation) del modelo E2 (LSTM 20d) es bajo y mayormente negativo en el universo. Datos del run [runs/e2_moderate/20260413_112244/summary_all.csv](runs/e2_moderate/20260413_112244/summary_all.csv):

| Ticker     | ml_ic   | bt_sharpe |
|------------|---------|-----------|
| BBAR.BA    | +0.147  | 0.72      |
| LOMA.BA    | +0.060  | 1.17      |
| BMA.BA     | -0.026  | 0.32      |
| TGSU2.BA   | -0.033  | 1.17      |
| GOOGL      | -0.067  | 0.26      |
| NFLX       | -0.069  | -0.06     |
| EDN.BA     | -0.120  | 0.03      |
| NVDA       | -0.152  | 0.36      |
| META       | -0.191  | 0.28      |
| AMZN       | -0.228  | -0.25     |
| **mean**   | **-0.068** |        |

IC mean ≈ -0.07 implica que el modelo está rankeando **al revés** del objetivo en promedio. Que algunos tickers tengan Sharpe positivo a pesar del IC negativo se debe a que `tau_buy=0.025` (definido en [src/config/base.yaml:152-184](src/config/base.yaml#L152-L184)) actúa como filtro adicional, no a que las predicciones sean buenas. Para mejorar el IC hay que atacar el set de features.

**Diagnóstico.** Las 12 features actuales en [src/e2/build_features.py](src/e2/build_features.py) son técnicos univariados clásicos derivados solo de OHLCV del propio ticker. Faltan:
- Información **cross-sectional** (cómo se comporta el ticker vs su universo).
- Features de **régimen** (no solo nivel de volatilidad/momentum, sino su cambio).
- Features **risk-adjusted** y de **interacción** entre las existentes.
- Validación empírica de qué features tienen señal real antes de meterlas al LSTM.

**Outcome esperado.** Reemplazar el set actual por uno seleccionado por evidencia de IC individual + estabilidad temporal + control de multicolinealidad. Meta operativa: subir IC mean del modelo de -0.07 a >+0.03 (mover la masa de IC al rango positivo en al menos 6/10 tickers).

## Approach

Workflow en 3 fases (ninguna entrena LSTMs todavía — eso es trabajo posterior, fuera del scope de este plan):

```
[Fase A: Generación]    build_features_extended.py  → 28 candidatas
                                  ↓
[Fase B: Análisis]      feature_ic_analysis.py      → ranking IC + estabilidad
                                  ↓
[Fase C: Selección]     filtros (IC + estabilidad + |r|<0.85)  → set final ~14-22 features
```

**Fase A** crea un módulo paralelo (no toca `build_features.py`) con todas las features actuales más candidatas nuevas. Esto permite comparar sin romper nada.

**Fase B** corre un script que computa IC por (feature, ticker) sobre el universo E2 completo, en 3 sub-períodos para evaluar estabilidad temporal, y exporta CSV + heatmap + markdown.

**Fase C** documenta los criterios de filtrado. La selección concreta del set final se hace **leyendo el reporte** y editando una lista en `build_features_extended.py` (no es automática para mantener trazabilidad académica).

**Fuera de scope explícito** (siguiente iteración, no este plan):
- Re-entrenamiento del LSTM con el set nuevo.
- Comparación A/B contra champion vía `compare_e2_models.py`.
- Cambios al `train_pipeline.py` (el flag `--feature-set` se diseña pero no se implementa todavía).

## Catálogo de features candidatas (28 total)

Agrupadas por familia. **Negrita** = feature que ya existe en `build_features.py` (se mantiene como baseline de comparación). El resto son nuevas.

### A. Retornos y momentum (6)
1. **`ret_1d`** — log-retorno diario (baseline).
2. **`ret_20d`** — momentum 20d (baseline).
3. `ret_5d` — log-retorno semanal. Hipótesis: aporta granularidad intermedia que ret_1d/ret_20d pierden. Riesgo conocido: redundancia (EDA §2.2 lo descartó por r≈0.7-0.85). Lo re-evaluamos con IC.
4. `ret_60d` — momentum trimestral. Captura tendencia más larga que el horizonte; útil para distinguir mean-reversion (ret_20d alto + ret_60d bajo) vs continuación (ambos altos).
5. `vol_adj_ret_20d` — `ret_20d / vol_20d`. Information ratio rolling. Hipótesis: separa "rallies con convicción" (alto) de "rallies caóticos" (bajo).
6. `ret_20d_zscore_252` — z-score de ret_20d vs distribución del último año del propio ticker. Normaliza momentum por régimen idiosincrático.

### B. Volatilidad y régimen de volatilidad (4)
7. **`vol_20d`** — std de retornos 20d (baseline).
8. **`atr_14`** — Average True Range normalizado (baseline).
9. `vol_ratio_20_60` — `vol_20d / vol_60d`. >1 → expansión (alerta), <1 → contracción (squeeze, posible breakout). Equivalente conceptual a `vol_regime` de E1 en [src/e1/build_features.py:75-77](src/e1/build_features.py#L75-L77).
10. `realized_skew_60d` — skew de retornos en ventana 60d (skew_ret_20d ya existe pero es muy ruidoso a 20d). 60d da una estimación más estable.

### C. Volumen (3)
11. **`vol_ratio_20d`** — volumen / SMA(volumen, 20) (baseline).
12. `vol_zscore_60` — z-score de volumen vs 60d. Más robusto que ratio (controla por la dispersión, no solo media). Idéntico a la versión E1.
13. `vol_price_corr_20` — correlación rolling 20d entre `volume` y `|ret_1d|`. Confirmación volumen-precio: alta correlación → movimientos respaldados por flujo.

### D. Tendencia (3)
14. **`sma50_sma200_ratio`** — golden/death cross (baseline).
15. **`close_sma50_dist`** — desviación del precio vs SMA50 (baseline).
16. `dist_from_high_252` — `(close / max(close, 252)) - 1`. Distancia al máximo anual; los breakouts de máximos suelen continuar (drawdown desde ATH es señal cualitativa distinta a SMA).

### E. Momentum técnico (3)
17. **`macd_hist`** — histograma MACD (baseline).
18. **`rsi_14`** — RSI 14d (baseline).
19. `macd_hist_change_5d` — `macd_hist - macd_hist.shift(5)`. Aceleración del momentum (segunda derivada); a menudo más predictiva que el nivel.

### F. Bollinger e interacciones (3)
20. **`bb_pct_b`** — posición en bandas (baseline).
21. `bb_pct_b_x_rsi_centered` — `bb_pct_b * (rsi_14 - 50) / 50`. Interacción no-lineal explícita: amplifica señal cuando ambos indican mismo extremo. Las redes pueden aprender esto en teoría, pero darlo explícito mejora aprendizaje con pocos datos.
22. `bb_squeeze` — `bb_bandwidth / rolling_max(bb_bandwidth, 60)`. Detecta squeezes (compresión histórica), distinto de `vol_ratio_20_60`.

### G. Fuerza de tendencia (2)
23. **`adx_14`** — ADX (baseline).
24. `adx_14_change_10d` — cambio de ADX a 10d. Fortalecimiento/debilitamiento de tendencia; útil en interacción con dirección.

### H. Asimetría/cola (1)
25. **`skew_ret_20d`** — skewness 20d (baseline).

### I. Cross-sectional (3)
*Estas features requieren computar el universo entero junto. Se calculan agregando los retornos de los 10 tickers de E2 definidos en [src/config/base.yaml:11-22](src/config/base.yaml#L11-L22). El "índice del universo" es la mediana cross-sectional de retornos diarios — robusto a outliers y no requiere data externa.*

26. `rel_strength_20d` — `ret_20d_ticker - ret_20d_universe_median`. Alfa cross-sectional 20d. Hipótesis: tickers con momentum *relativo* persisten mejor que con momentum absoluto.
27. `beta_60d_universe` — beta del ticker vs índice del universo, ventana 60d. Captura sensibilidad sistémica; tickers high-beta tienen retornos más predecibles cuando se conoce el régimen del mercado.
28. `dispersion_20d` — std cross-sectional de ret_20d en el universo (mismo valor para todos los tickers en cada fecha). Régimen de mercado: dispersión alta → mercado discriminando (stock-picking funciona); baja → todos correlacionados (momentum macro).

**Nota AR vs US.** El universo mezcla 5 tickers US + 5 tickers AR (.BA). La mediana cross-sectional sobre el conjunto mezcla regímenes muy distintos. **Decisión:** computar dos versiones de las features cross-sectional — una con el universo completo y otra con el sub-universo geográfico al que pertenece cada ticker (US o AR). El script de IC reporta ambas y se elige la mejor por feature.

## Diseño del script de validación de IC

**Archivo nuevo:** `scripts/evaluation/feature_ic_analysis.py`

### Inputs
- Universo E2 desde [src/config/base.yaml](src/config/base.yaml) (`strategies.e2_moderate.tickers`).
- OHLCV vía el mismo data loader que `train_pipeline.py` (reusar `src/data/loader.py` o equivalente — confirmar al implementar).
- Features computadas vía nuevo `src/e2/build_features_extended.py` (función `compute_e2_features_extended(df_dict)` que recibe dict de DataFrames por ticker para soportar features cross-sectional).

### Cómputo
Para cada `(feature_i, ticker_t)`:

```python
from scipy.stats import spearmanr

# Alinear feature y target, descartar NaN
mask = feature_i.notna() & target.notna()
f, y = feature_i[mask], target[mask]

# IC global
ic, p_value = spearmanr(f, y)

# IC por sub-período (estabilidad temporal)
n = len(f)
splits = [(0, n//3), (n//3, 2*n//3), (2*n//3, n)]
ic_by_period = [spearmanr(f.iloc[a:b], y.iloc[a:b])[0] for a, b in splits]

# Métricas
ic_stability_std = np.std(ic_by_period)
ic_sign_flips = sum(1 for ic_p in ic_by_period if np.sign(ic_p) != np.sign(ic))
```

### Outputs
- `reports/feature_analysis/e2_feature_ic.csv` — una fila por (feature, ticker) con: `ic, p_value, n_obs, ic_period_1, ic_period_2, ic_period_3, ic_stability_std`.
- `reports/feature_analysis/e2_feature_ic_summary.csv` — agregado por feature: `ic_mean, ic_std_across_tickers, ic_median, pct_tickers_positive, pct_tickers_significant_p05, ic_stability_mean`.
- `reports/feature_analysis/e2_feature_ic_heatmap.png` — heatmap features (filas) × tickers (columnas), color = IC. Visualmente revela features con señal consistente vs ruidosas.
- `reports/feature_analysis/e2_feature_ic_summary.md` — markdown con:
  - Tabla top 15 features por `|ic_mean|`.
  - Tabla bottom 5 features por estabilidad (candidatas a descartar).
  - Matriz de correlación de las top 15 (clustering jerárquico).
  - Recomendación final: lista de features sugeridas para el set ampliado.

### Criterios de filtrado documentados (para Fase C manual)
1. **Filtro IC absoluto:** descartar features con `|ic_mean| < 0.02` Y `pct_tickers_significant_p05 < 30%`.
2. **Filtro estabilidad:** descartar features con `ic_sign_flips >= 2` (cambia signo en ≥2 sub-períodos respecto al global).
3. **Filtro multicolinealidad:** sobre las que pasan, calcular matriz de correlación de Spearman (entre features, no contra target). Para cada par con `|r| > 0.85`, conservar la de mayor `|ic_mean|`.

### Tiempo de ejecución estimado
< 2 minutos sobre los 10 tickers del universo (sin entrenar modelos, solo correlaciones).

## Critical files

**A crear:**
- `src/e2/build_features_extended.py` — función `compute_e2_features_extended(ohlcv_dict)` con las 28 candidatas. Recibe dict de DataFrames por ticker (necesario para cross-sectional).
- `scripts/evaluation/feature_ic_analysis.py` — script descrito arriba.

**A leer (no modificar):**
- [src/e2/build_features.py](src/e2/build_features.py) — referencia para implementación de las 12 baseline.
- [src/e1/build_features.py](src/e1/build_features.py) — referencia para `vol_regime` (línea 75-77) y `vol_zscore_60` (línea 116-117).
- [src/e2/train_pipeline.py:89-100](src/e2/train_pipeline.py#L89-L100) — fórmula exacta del IC del modelo (Spearman). El script de feature IC debe usar la misma para consistencia.
- [src/config/base.yaml:11-22](src/config/base.yaml#L11-L22) — universo E2.
- Loader de OHLCV usado por `train_pipeline.py` (a identificar al implementar; probablemente `src/data/` o similar).

**No modificar:**
- `build_features.py` actual (queda intacto para no romper el champion).
- `train_pipeline.py` actual.
- `models/registry.json`.

## Verification

### Test 1: smoke test del módulo extended
```bash
conda activate ia_ceia_18co
python -c "
from src.e2.build_features_extended import compute_e2_features_extended
import pandas as pd
import numpy as np

# OHLCV sintético para 1 ticker, 500 días
np.random.seed(42)
n = 500
idx = pd.date_range('2023-01-01', periods=n, freq='B')
close = 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, n)))
df = pd.DataFrame({
    'open': close * (1 + np.random.normal(0, 0.005, n)),
    'high': close * (1 + np.abs(np.random.normal(0, 0.01, n))),
    'low':  close * (1 - np.abs(np.random.normal(0, 0.01, n))),
    'close': close,
    'volume': np.random.lognormal(15, 0.5, n),
}, index=idx)

feats = compute_e2_features_extended({'TEST': df})
print('Columnas generadas:', len(feats['TEST'].columns))
print('Sample:')
print(feats['TEST'].tail(3))
"
```
**Esperado:** 28 columnas, sin NaN en las últimas 3 filas (warm-up de rolling máximo es 252 días de `dist_from_high_252`).

### Test 2: corrida completa del análisis
```bash
python scripts/evaluation/feature_ic_analysis.py
```
**Esperado:**
- 3 archivos en `reports/feature_analysis/` (CSV detallado, CSV summary, heatmap PNG).
- 1 archivo `e2_feature_ic_summary.md`.
- Tiempo de ejecución < 2 minutos.
- Logs muestran IC computado para 28 features × 10 tickers = 280 mediciones.

### Test 3: sanity checks sobre el reporte
Abrir `reports/feature_analysis/e2_feature_ic_summary.md` y verificar:
- **`rsi_14` debería tener IC negativo** (mean-reversion: RSI alto → retorno futuro bajo). Si sale positivo cerca de 0, indicaría problema en datos o cómputo.
- **`ret_20d` debería tener IC negativo** (mean-reversion 20d, consistente con la hipótesis de que el horizonte coincide con la ventana de momentum).
- **`vol_adj_ret_20d` debería tener |IC| > |IC de ret_20d|** si la normalización por volatilidad agrega información.
- **Features cross-sectional (rel_strength_20d, beta_60d_universe)** deberían mostrar mejor estabilidad temporal que las univariadas (menor `ic_stability_std`).

Si alguno de estos sanity checks falla → revisar implementación antes de confiar en el ranking.

### Test 4 (entregable final, manual)
Vos leés el `summary.md`, aplicás los 3 filtros documentados, y editás una constante `SELECTED_FEATURES` al inicio de `build_features_extended.py` con la lista final (estimado: 14-22 nombres). Esa lista es el output utilizable para el siguiente paso (re-entrenamiento del LSTM, fuera de scope de este plan).
