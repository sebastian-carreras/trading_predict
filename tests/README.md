# Tests

Suite de tests unitarios y de integración ligera para el proyecto de Trading Prediction (trabajo final de la Especialización en IA, FIUBA).

No son tests de "framework" genéricos: cada uno verifica una garantía metodológica concreta del pipeline — ausencia de data leakage, atomicidad de escritura del registry, consistencia de la lógica de promoción de modelos, correctud del cálculo de costos de backtest, etc. Son, en buena medida, la evidencia de que las prácticas descritas en `PORTFOLIO.md` (walk-forward con embargo, scaling sin leakage, promoción champion/candidate) están efectivamente implementadas y no solo documentadas.

---

## ▶️ Cómo correr

```bash
# Suite completa (lo que corre CI, ver .github/workflows/ci.yml)
python -m pytest -v

# Un archivo puntual
pytest tests/test_registry_atomicity.py -v

# Un test puntual dentro de un archivo
pytest tests/test_walkforward_splits.py::TestEmbargo::test_embargo_gap_e1_90_days -v
```

CI corre esta misma suite (`python -m pytest -v`) en Python 3.10 y 3.12 en cada push a `main` y en cada PR.

**Sin dependencias externas:** los tests no descargan datos ni llaman a APIs externas (yfinance, IOL, MLflow remoto). Usan datos sintéticos generados con `numpy.random.default_rng` (seeds fijas, reproducibles) y `tmp_path` de pytest para I/O (registry, CSVs, logs) aislado por test.

**Tests condicionales:** dos tests en `test_reevaluation.py` (`test_predict_latest_on_real_champion_matches_predict_series`, `test_predict_series_reproduces_stored_walkforward`) requieren `torch` instalado y al menos un modelo champion real en `models/registry.json` con sus artefactos (`*_model.pth`, `*_walkforward_predictions.csv`, CSV limpio en `data/clean/`). Si no se cumple alguna condición, se saltan (`pytest.skip`) en vez de fallar — no dependen de que el repo tenga un estado de entrenamiento particular.

---

## 📉 Prevención de data leakage

### `test_walkforward_splits.py`
Verifica las propiedades del split walk-forward (`sklearn.model_selection.TimeSeriesSplit` con `gap`) usado en `src/e1/train_pipeline.py` y análogos en E2/E3:
- Train y test son siempre disjuntos y train precede a test (orden temporal).
- El embargo (`gap`) entre el fin de train y el inicio de test es ≥ `horizon_days` de la estrategia (90 días en E1, 20 en E2) — evita que el horizonte de predicción se filtre entre folds.
- Cada fold sucesivo tiene un train set igual o mayor (walk-forward real, no k-fold aleatorio).
- Si `test_size <= embargo`, el pipeline ajusta `test_size = embargo + 1` (mismo guard que `train_pipeline.py:539-540`).

### `test_feature_scaling.py`
Replica la lógica de z-score scaling in-line de `src/e{1,2,3}/train_pipeline.py` y verifica el invariante central: **media y desvío se calculan solo con el fold de train**, nunca con val/test. Incluye casos con distribuciones de val/test deliberadamente desplazadas (para detectar leakage si lo hubiera), no-mutación de los arrays de entrada, y estabilidad numérica cuando una feature tiene std=0.

---

## 💰 Backtesting

### `test_backtest_costs.py`
Verifica la fórmula de costos de transacción compartida por `src/backtest/backtest_daily.py` (E1/E2) y `src/backtest/backtest_intraday.py` (E3):

```
costs = |Δposición| * (round_trip_bps / 10000) / 2
```

Chequea: costo cero con `round_trip_bps=0`, costos siempre no negativos, proporcionalidad exacta (20 bps = 2× el costo de 10 bps), `net_ret = gross_ret - costs`, y que el motor intraday use la misma fórmula.

---

## 🔄 Ciclo de vida del modelo (`src/lifecycle/`)

### `test_registry_atomicity.py`
Verifica que `ModelRegistry._save()` escribe de forma atómica (`tempfile` + `os.replace`, no un `write` directo sobre `registry.json`): si la escritura falla a mitad de camino, el archivo original queda intacto; no quedan `.tmp` huérfanos tras un guardado exitoso; y los datos persisten correctamente entre instancias sucesivas de `ModelRegistry`.

### `test_lifecycle_paths.py`
Verifica `resolve_lifecycle_paths()` (`src/utils.py`): rutas relativas del config se resuelven contra `root`, rutas absolutas se respetan tal cual, y a falta de config caen a los defaults (`models/registry.json`, `models/metrics_log.jsonl`).

### `test_guardrails.py`
Verifica los guardrails de Fase 1 (`src/lifecycle/guardrails.py::validate_candidate`) que bloquean la promoción de un candidato antes de siquiera comparar métricas: modelo faltante, NaN/Inf en predicciones, Sharpe por debajo del mínimo, e IC peor que el baseline.

### `test_promotion_per_strategy.py`
Verifica la resolución jerárquica de configuración de promoción (`get_strategy_promotion_config()` en `src/lifecycle/promotion.py`): `per_strategy.{strategy}` → `per_strategy.{prefix}` (p. ej. `e1_conservative` → `e1`) → fallback global. Cubre que cada estrategia puede tener sus propios `scoring_weights` (E3 pondera `profit_factor` y `win_rate` en vez de `calmar`), que el config original no se mockea/pierde, y que `compute_score()` produce scores distintos con pesos distintos.

### `test_promotion_common_window.py`
El test más "de negocio" de la suite: verifica `evaluate_and_promote()` / `compare_on_common_window()`, la comparación **justa** champion-vs-candidate sobre una ventana out-of-sample común (no cada uno con su propia ventana, que favorecería artificialmente al que tenga métricas más recientes). El inference con torch está mockeado (se inyectan predicciones directamente) para que corra sin GPU/modelo real. Cubre los cuatro modos de comparación: `fair_window` (ventana común válida), `identity_skip` (candidato idéntico al champion, se salta la comparación), `insufficient_evidence` (ventana OOS demasiado chica, se mantiene el champion) y `stored` (fallback a métricas ya guardadas cuando no hay `train_data_end` del champion).

### `test_reevaluation.py`
Tests de soporte para el re-backtest justo: `recompute_metrics_on_window()` devuelve las keys que consume el composite score (`bt_sharpe`, `bt_calmar`, `ml_ic`, `ml_directional_accuracy`) con valores finitos y coherentes; `load_walkforward_predictions()` hace round-trip correcto contra CSV. Los dos tests condicionales (ver arriba) verifican algo más fuerte: que `predict_series()` reconstruye, con el modelo congelado y los mismos scalers, prácticamente el mismo `y_pred` (`atol=1e-3`) que quedó guardado durante el entrenamiento original — es la prueba de que la inferencia post-hoc no introduce drift de reproducibilidad.

---

## 📊 Monitoreo de drift

### `test_drift.py`
Verifica `src/lifecycle/drift.py` (PSI + test de Kolmogorov-Smirnov para detectar drift de features entre el set de referencia y el actual): PSI ≈ 0 con distribuciones idénticas, PSI por encima del umbral con un shift grande, monotonicidad de PSI respecto a la magnitud del shift, manejo de casos borde (referencia constante, inputs vacíos, NaN/Inf ignorados sin propagar), que el reporte solo considere columnas numéricas comunes entre ambos frames, y que el logging a JSONL (`log_drift_report`) escriba una entrada por llamada y permita append.

---

## 🗂️ Convenciones de la suite

- Un archivo de test por módulo/concern de `src/`, nombrado `test_<módulo_o_concern>.py`.
- Datos sintéticos con `numpy.random.default_rng(seed)` — nunca datos reales de mercado ni llamadas de red.
- I/O de disco (registry, CSVs, logs) siempre bajo `tmp_path` (fixture de pytest), nunca contra `models/registry.json` real.
- Mocks solo donde el costo de levantar el objeto real es alto (inference con torch en `test_promotion_common_window.py`); el resto del código de negocio (guardrails, promotion, drift) se ejercita real, sin mockear.
