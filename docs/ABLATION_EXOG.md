# Ablación E2 — Features exógenas (Fase 1, Bloques A/B/C/D)

Evidencia detrás de `data.exog.enabled: false` en [`src/config/base.yaml`](../src/config/base.yaml).
Catálogo de las features: [`docs/FEATURES.md`](FEATURES.md).

## Conclusión vigente (Fase 1 cerrada, 2026-08-02)

**NO se activan las exógenas**, ni globalmente ni per-ticker.

- **In-sample sí mejoran.** Con hiperparámetros re-tuneados para el set exógeno (Optuna
  `--exog`, 21 features), la ablación justa v3 sobre los 30 tickers da **Δscore +0.036**, y
  Sharpe y Calmar pasan de negativos a positivos. El "resultado negativo" que se había
  concluido antes era un **confound**: se estaban usando los thresholds del baseline de 12
  features para evaluar el modelo de 21.
- **Out-of-sample no generaliza.** En held-out temporal multi-seed (train ≤2023, eval
  2024-2026) los 12 winners dan **Δ mediana −0.134** y solo 3/12 sobreviven; los tickers de
  control "mejoran" tan seguido como los winners, y los sobrevivientes cambian según la
  semilla. El beneficio per-ticker está dominado por el ruido de entrenamiento del LSTM
  (**std ~0.09 por ticker vs. un efecto de ~0.03–0.05**).
- **No hay allowlist defendible:** solo MA sobrevive en las dos corridas, y un nombre de 30
  no justifica el subsistema (riesgo de comparaciones múltiples).
- **Los `.BA` nunca se beneficiaron** en ninguna versión del experimento → su contexto propio
  llega en la Fase 3 (CCL, riesgo país, Merval).
- **Queda la infraestructura** (fetch macro + PIT + guardrail de cobertura + Optuna
  exog-aware + arnés de ablación/held-out multi-seed) y **un bug de datos real corregido**.

Las tres lecciones metodológicas, que valen más que el resultado:

1. **Verificar la cobertura histórica de cada serie externa** antes de confiar en un
   resultado — acá una serie con historia restringida recortó en silencio el train set.
2. **Re-tunear hiperparámetros por set de features** (sobre todo los thresholds), nunca
   reusar los del baseline: eso solo produjo un falso negativo.
3. **Validar con multi-seed y held-out.** El in-sample y el single-seed engañaron, en esta
   misma página, dos veces.

## Cómo leer lo que sigue

Es el registro cronológico del experimento, en cuatro etapas, y **cada una corrige a la
anterior**. Se conserva completo a propósito: el recorrido —incluidos los dos falsos
resultados y el bug que los causó— es el contenido, no la prolijidad del veredicto.

| Etapa | Qué concluyó | Estado |
|---|---|---|
| 1 · Ablación inicial (07-28) | 13 winners per-ticker | ❌ invalidada por el bug de `hy_oas_z` |
| 2 · Ablación corregida v2 (07-31) | efecto agregado ~nulo | ⚠️ superada (confound de hiperparámetros) |
| 3 · Held-out (07-31) | no activar, resultado negativo | ⚠️ superada (mismo confound) |
| 4 · Tuning justo v3 + held-out multi-seed (08-01) | in-sample sí, OOS no → **no activar** | ✅ vigente |

Los CSV crudos que se citan viven en `reports/ablation/` **fuera del control de versiones**
(`reports/**` está gitignoreado): son salidas regenerables con
`scripts/evaluation/ablation_e2_exog.py` y `holdout_e2_exog.py`.

---

# Etapa 1 — Ablación inicial (2026-07-28)

> ⚠️ **INVALIDADA (2026-07-31) por un bug de datos — los resultados de ESTA etapa no son
> confiables.**
> La feature `hy_oas_z` se bajaba de FRED `BAMLH0A0HYM2` (ICE BofA HY OAS), que tiene la
> **historia pública restringida** (solo ~3 años): quedaba NaN antes de 2023. Como
> `make_sequences` hace `dropna`, los modelos **exog** entrenaban con ~2023-2026 (~600
> muestras) y los **base** con 2016-2026 (~2400) → **comparación injusta**. La selección de
> "13 ganadores" de esta etapa se descarta.
> **Corrección:** `hy_oas_z` ahora es un proxy **HYG/LQD** (yfinance, historia completa) —
> ver `src/data/macro.py`. Champions exog contaminados (NVDA/AMZN) revertidos a base vía
> `restore_champion_from_retired`; exógenas en pausa (`data.exog.enabled: false`). La
> re-ablación con datos corregidos es la **Etapa 2**.

**Fecha:** 2026-07-28 · **Estrategia:** E2 (LSTM, 20d) · **Núcleo exógeno (9):** vix_ts, vix_pctl_252, hy_oas_z, curve_10y2y, spy_mom_20, rel_strength_sector, sector_rotation, breakeven_10y, nfci

## Método

Cada ticker se entrena en dos modos —**base** (12 features) y **exog** (12 + 9 núcleo)— y se compara el **score compuesto** de promoción (0.35·Sharpe + 0.25·IC + 0.20·DirAcc + 0.20·Calmar), sin registrar candidatos (`register_lifecycle=False`). Driver: `scripts/evaluation/ablation_e2_exog.py`.

Dos corridas: **single-seed** (1×) y **multi-seed robusta** (3 seeds promediados, para separar señal del ruido de entrenamiento del LSTM).

## Hallazgo metodológico crítico

El **std del score entre seeds (~0.09 por ticker) es MAYOR que el efecto agregado de las exógenas (~0.03–0.05)**. Es decir, el ruido de entrenamiento supera a la señal medida. Consecuencia: los Δ per-ticker de single-seed (±0.8) eran **casi todo ruido**. El multi-seed es imprescindible para cualquier conclusión per-ticker.

Prueba: la historia de los tickers argentinos **se dio vuelta** al de-ruidar:

| Grupo | Single-seed (mejoran) | Multi-seed 3× (mejoran) |
|---|---|---|
| .BA (5) | 4/5, Δ mediana **+0.116** | **1/5, Δ mediana −0.205** |

El "beneficio" de las exógenas en los .BA era ruido.

## Resultado robusto (3 seeds)

Δ = per-ticker (score_exog − score_base), promediado sobre seeds.

| Grupo | Δ media | Δ mediana | Mejoran |
|---|---|---|---|
| **Todos (30)** | **+0.051** | −0.025 | **14/30** |
| **US (25)** | **+0.097** | **+0.084** | 13/25 |
| **.BA (5)** | −0.175 | −0.205 | 1/5 |

Clasificación por robustez (Δ vs ±1 error estándar entre seeds):
- **Mejoran robustamente (13):** ABBV, GS, META, WMT, NFLX, MCD, CRM, AMZN, GE, SPGI, HD, AVGO, NVDA
- **Empeoran robustamente (12):** ISRG, QCOM, LOMA.BA, BBAR.BA, LLY, GOOGL, HON, COST, UNP, VRTX, TMUS, MA
- **Ambiguos (5):** el resto

## Conclusiones

1. **NO activar exógenas globalmente.** El efecto es net-positivo en media (por los ganadores grandes) pero la mediana per-ticker es ~neutral y solo 14/30 mejoran.
2. **Excluir los .BA del bloque exógeno** — robustamente negativo (1/5, mediana −0.205). Las features macro/sector de EE.UU. son un mal proxy para activos argentinos (mercado/moneda/drivers distintos). Su contexto propio llega en la **Fase 3 (Bloque G: CCL, riesgo país, Merval)**.
3. **En US hay señal real** (media +0.097, mediana +0.084) pero **repartida** (~13 ganan robustamente, ~12 pierden), sin una regla sectorial limpia (hay semis, farma, staples y comm en ambos lados) → parece idiosincrático por nombre.
4. La infraestructura funciona y hay evidencia de-ruidada; la decisión de activación debe ser **conservadora y per-ticker** (la promoción ya es por ticker), no un flip global.

## Datos crudos
`e2_exog_ablation.csv` (single-seed) · `e2_exog_ablation_multiseed.csv` (3 seeds), en
`reports/ablation/` (gitignoreado — regenerable con `scripts/evaluation/ablation_e2_exog.py`).

---

# Etapa 2 — Resultado corregido (v2, 2026-07-31): datos ya sin el bug

> ⚠️ **Superada por la Etapa 4.** El "efecto ~nulo" de abajo se midió con los
> hiperparámetros del baseline aplicados también al modo exógeno; con tuning justo el
> agregado deja de ser nulo.

Re-ablación multi-seed (3 seeds × 30) con `hy_oas_z` sourced de HYG/LQD (historia completa).
Datos: `e2_exog_ablation_multiseed_v2.csv`.

**Agregado (30):** score mediana base 0.283 → exog 0.301 (Δ **+0.018**); **media Δ ≈ +0.000** (¡nulo!); mejoran **17/30**. .BA: 2/5, mediana −0.008.

Robustez (Δ vs ±1 SE entre seeds):
- **Mejoran robustamente (8):** ORCL, NVDA, NFLX, ABBV, COST, ISRG, GE, UNP (todos US).
- **Empeoran robustamente (9):** TGSU2.BA, TMUS, GS, VRTX, CRM, WMT, QCOM, LLY, AVGO.
- **Ambiguos (13):** GOOGL, AMZN, META, BBAR.BA, BMA.BA, EDN.BA, LOMA.BA, NOW, MA, SPGI, HD, MCD, HON.

## Conclusiones corregidas (reemplazan a las de arriba)

1. **El efecto agregado es prácticamente NULO** (media Δ ≈ 0, mediana +0.018). Con datos correctos, las exógenas **no dan una ventaja amplia** a E2.
2. **La selección cambió por completo** vs la versión contaminada: ISRG pasó de peor (−0.84) a winner (+0.10); ORCL de −0.51 a +0.31; y ex-"ganadores" como GS, CRM, WMT, AVGO ahora son **perdedores robustos**. → la selección de 13 estaba mal; bien que se pausó.
3. Hay un subconjunto acotado (**8 winners robustos, todos US**) que mejora de forma estable, pero con 9 perdedores robustos y 13 ambiguos → **beneficio estrecho, no universal**.
4. Los `.BA` siguen sin beneficiarse (2/5, mediana negativa) → Fase 3.
5. **Decisión pendiente del held-out**: los 8 winners solo se activan si sobreviven la validación temporal (train ≤2023, eval 2024-2026, `holdout_e2_exog.py`). Si no sobreviven, NO activar y pasar a Fase 2/3.

---

# Etapa 3 — Held-out (2026-07-31): validación temporal

> ⚠️ **Superada por la Etapa 4**, y por el mismo motivo que la Etapa 2: este held-out
> también corrió con los hiperparámetros del baseline en modo exógeno. La conclusión "no
> activar" termina coincidiendo con la vigente, pero el razonamiento de acá no se sostiene.

Modelos entrenados con datos **≤2023**, evaluados en **2024-2026** (nunca vistos). Datos: `e2_exog_holdout.csv`.

| Grupo | Δ mediana | Δ media | Sobreviven (Δ>0) |
|---|---|---|---|
| **Winners (8)** | **−0.031** | −0.034 | **3/8** |
| Control/losers (3) | +0.141 | −0.198 | 2/3 |

Winners que sobreviven: **GE (+0.537), COST (+0.128), ORCL (+0.008)**. Fallan: NVDA (−0.307), ISRG (−0.341), ABBV (−0.237), NFLX, UNP. Y los "perdedores" de control **mejoran** out-of-sample tan seguido como los winners (GS +0.141, WMT +0.389).

## Conclusión final: NO activar exógenas en E2 (resultado negativo)

La selección per-ticker de la ablación **no generaliza**: solo 3/8 winners sobreviven (mediana negativa), y los controles "ganan" tanto como los winners → los Δ per-ticker eran esencialmente **ruido / sobreajuste a la ventana**. Sumado a que el **efecto agregado ya era ~nulo**, la evidencia dice que el bloque exógeno (tal como está diseñado) **no aporta una ventaja robusta y generalizable** a E2.

**Decisión:** dejar `data.exog.enabled: false` (resultado negativo documentado, como E3 intradía). La **infraestructura queda** (fetch macro + PIT + guardrail de cobertura + arnés de ablación/held-out) — reusable para Fase 2 (sentimiento) y Fase 3 (Argentina), que aportan señales distintas. Caveat: el held-out fue single-seed (ruidoso); GE es el único candidato que sobrevive con margen, pero un solo nombre no justifica activar el subsistema.

---

# Etapa 4 (vigente) — Re-validación con tuning justo (2026-08-01): el "negativo" era un CONFOUND de hiperparámetros

Todo lo anterior (ablación v2 Y held-out) usó los params Optuna tuneados para los modelos **base (12 feats)** también en el modo **exog (21 feats)** — incluidos los **thresholds `tau_buy`/`tau_sell`**. Como Sharpe/Calmar (55% del score) dependen del threshold, y el exog produce otra distribución de predicciones, la comparación estaba **sesgada contra el exog**. Diagnóstico: exog **mejora el IC** (+0.039, independiente del threshold) pero el score quedaba plano.

**Fix:** se agregó `--exog` a `optimize_e2_hyperparameters.py` (tunea sobre 21 feats) → yaml separado `reports/hyperparameter_optimization/exog/`. Los drivers de ablación/held-out ahora usan, en modo exog, esos params exog (comparación justa: base con params base, exog con params exog).

**Ablación JUSTA v3, subset de 8** (`e2_exog_ablation_v3_subset.csv` — NVDA, ORCL, GE, COST, ISRG, ABBV, CRM, GS):

| Métrica | Δ v2 (injusto) | Δ v3 (justo) |
|---|---|---|
| **Score** | +0.082 | **+0.144** (~×2) |
| IC (indep. threshold) | +0.072 | +0.072 (igual — control) |
| **Sharpe** (dep. threshold) | +0.117 | **+0.248** (~×2) |
| **Calmar** (dep. threshold) | +0.111 | **+0.198** (~×2) |

**7/8 tickers mejoran con exog** (solo GS empeora). Con thresholds bien calibrados, la mejora de IC **se traduce** en Sharpe/Calmar. → **el resultado "negativo" era un artefacto de tuning, no un nulo genuino.**

**Caveats:** subset de 8 elegido por prometedor (sesgo de selección) → hace falta el Optuna full en los 30 y un **held-out con tuning justo** (el held-out previo también estaba confounded). En curso: Optuna exog en los 22 restantes → ablación v3 completa → held-out justo → decisión (Paso 5 del plan).

## Ablación JUSTA v3 COMPLETA (30 tickers, sin sesgo de selección) — `e2_exog_ablation_v3.csv`

Optuna exog corrido en los 30 (params en `reports/hyperparameter_optimization/exog/`). Agregado v2 (injusto) vs v3 (justo):

| Métrica | Δ v2 | Δ v3 |
|---|---|---|
| **Score** | +0.001 | **+0.036** |
| IC | +0.038 | +0.040 (igual — control) |
| DirAcc | −0.010 | −0.011 |
| **Sharpe** | −0.005 | **+0.064** |
| **Calmar** | −0.026 | **+0.030** |

Con tuning justo el agregado pasa de ~nulo a **claramente positivo**: **Sharpe y Calmar se dan vuelta de negativo a positivo**, confirmando que la mejora de IC ahora se traduce al backtest. **Mejoran 19/30** (Δscore mediana +0.051). US 17/25 (med +0.056); **.BA 2/5 (med −0.079)** → los argentinos siguen sin beneficiarse (Fase 3).

- **Winners robustos (12):** COST, GE, NFLX, NVDA, GOOGL, LOMA.BA, AVGO, CRM, ORCL, AMZN, MA, ISRG.
- **Losers robustos (7):** VRTX, TGSU2.BA, HON, GS, WMT, TMUS, LLY.

→ **Fase 1 NO era un nulo genuino.** El held-out justo (con params exog) sobre los 12 winners decide la activación final.

## Held-out JUSTO multi-seed (3 seeds) — VEREDICTO FINAL (`e2_exog_holdout_v3_multiseed.csv`)

Train ≤2023, eval 2024-2026, params exog, 3 seeds promediados (el single-seed engañaba).

- **Winners (12): Δ mediana −0.134**, solo **3/12 sobreviven claro** (GOOGL, MA, ORCL). Controles (3): 2/3 "mejoran" → **sin separación** de los winners.
- **Inestabilidad entre single y multi-seed:** sobrevivientes single = {MA, LOMA.BA, GE}; multi = {GOOGL, MA, ORCL}. **Solo MA en ambos.** → el beneficio per-ticker OOS está dominado por el ruido de entrenamiento.

### Conclusión final de Fase 1

- **In-sample (ablación justa):** exog mejora E2 (+0.036 score; IC→Sharpe/Calmar). El "negativo" previo era un **confound de hiperparámetros** (thresholds base aplicados al exog) — corregido con Optuna sobre 21 features.
- **Out-of-sample (held-out multi-seed):** el beneficio per-ticker **NO generaliza de forma confiable** (mediana −0.134, sobrevivientes inestables entre seeds).
- **Decisión: NO activar exógenas per-ticker** (`data.exog.enabled: false`). No hay allowlist defendible (solo MA es consistente; un nombre de 30 no justifica el subsistema y hay riesgo de multiple-comparisons).
- **Lo que queda:** infra reusable (fetch macro + PIT + guardrail cobertura + Optuna exog-aware + arnés ablación/held-out multi-seed), un bug de datos real corregido, y la **lección metodológica**: comparar features SIEMPRE con hiperparámetros re-tuneados por set, y validar con multi-seed + held-out (el in-sample y el single-seed engañan). El aporte in-sample es real pero no cosechable per-ticker a 20d con este nivel de ruido → la señal orthogonal se busca en Fase 2 (eventos) / Fase 3 (Argentina).
