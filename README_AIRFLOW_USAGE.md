# Guía: Ejecutar DAGs para Tickers Específicos

## Índice de DAGs E1

- [E1 Conservative (GRU)](#e1-conservative-pipeline---ejecución-selectiva) - Modelo principal
- [E1 Simple (GRU simplificado)](#e1-simple-pipeline---ejecución-selectiva) - Pipeline rápido
- [E1 Baseline (Regresión Lineal)](#e1-baseline-linear-regression---comparación) - Baseline para comparación

**Ubicación de DAGs**: [dockerfiles/airflow/dags](dockerfiles/airflow/dags) con subcarpetas [E1](dockerfiles/airflow/dags/E1), [E2](dockerfiles/airflow/dags/E2), [E3](dockerfiles/airflow/dags/E3), [E4](dockerfiles/airflow/dags/E4). Airflow escanea subcarpetas automáticamente.

---

## E1 Conservative Pipeline - Ejecución Selectiva

### Desde Airflow UI (Recomendado)

1. **Acceder**: http://localhost:8080

2. **Trigger con Configuración**:
   - Click en `e1_conservative_pipeline`
   - Click en **"Play"** (▶) → **"Trigger DAG w/ config"**
   - En el JSON editor:
   
   ```json
   {
     "tickers": "AAPL,MSFT,GOOGL"
   }
   ```
   
   - Click **"Trigger"**

### Desde CLI

```bash
# Un ticker específico
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_conservative_pipeline \
  --conf '{"tickers": "AAPL"}'

# Múltiples tickers
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_conservative_pipeline \
  --conf '{"tickers": "AAPL,MSFT,NVDA"}'

# Todos los tickers (campo vacío)
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_conservative_pipeline \
  --conf '{"tickers": ""}'
```

## Ejemplos de Uso

### Debugging rápido (1 ticker)
```json
{"tickers": "AAPL"}
```
⏱ ~2-3 min | Útil para probar cambios

### Tech stocks comparison
```json
{"tickers": "AAPL,MSFT,GOOGL,META,NVDA"}
```
⏱ ~10-15 min | Compara ICs entre empresas

### Portfolio completo
```json
{"tickers": ""}
```
⏱ ~1-2 horas | Todos los tickers del config

## 🔍 Ver Resultados en MLflow

1. **MLflow UI**: http://localhost:5050
2. **Experimento**: "E1_Conservative_Strategy"
3. **Filtrar**: `params.ticker = "AAPL"`
4. **Comparar**: Seleccionar runs → "Compare"

## Notas

- SPY (benchmark) siempre se descarga automáticamente
- Solo entrena tickers válidos de E1 en `base.yaml`
- Cada ticker = 1 run en MLflow
- Run de resumen con métricas agregadas (IC mean, median, etc.)

---

## E1 Simple Pipeline - Ejecución Selectiva

Pipeline simplificado (GRU de 1 capa, sin walk-forward). Descarga, limpia y entrena automáticamente. Ideal para pruebas rápidas.

### Desde Airflow UI (Recomendado)

1. **Acceder**: http://localhost:8080

2. **Trigger con Configuración**:
   - Click en `e1_simple_pipeline`
   - Click en **"Play"** (▶) → **"Trigger DAG w/ config"**
   - En el JSON editor:
   
   ```json
   {
     "tickers": "AAPL,MSFT,GOOGL",
     "skip_download": "False",
     "skip_cleaning": "False"
   }
   ```
   
   - Click **"Trigger"**

### Desde CLI

```bash
# Un ticker específico
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_simple_pipeline \
  --conf '{"tickers": "AAPL", "skip_download": "False", "skip_cleaning": "False"}'

# Reutilizando datos descargados y limpios por E1 Conservative
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_simple_pipeline \
  --conf '{"tickers": "AAPL,MSFT", "skip_download": "True", "skip_cleaning": "True"}'

# Todos los tickers (según config)
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_simple_pipeline \
  --conf '{"tickers": "", "skip_download": "False", "skip_cleaning": "False"}'
```

### Parámetros del DAG E1 Simple

**`tickers`** (string)
- Lista separada por comas: "AAPL,MSFT,GOOGL"
- Vacío = todos los tickers E1 Simple del `base.yaml` (fallback a E1 Conservative si no están definidos)

**`skip_download`** (boolean)
- "True": No descarga, reutiliza `data/raw/daily/` existente
- "False": Descarga diaria con `skip_existing` y frescura mínima

**`skip_cleaning`** (boolean)
- "True": No limpia, reutiliza `data/clean/` existente
- "False": Ejecuta limpieza (forward fill, filtros, validación mínima de 252 días)

### 🔍 Ver Resultados en MLflow

1. **MLflow UI**: http://localhost:5050
2. **Experimento**: "E1_Simple"
3. **Filtrar**: `params.ticker = "AAPL"`
4. **Comparar**: Seleccionar runs → "Compare"

### Notas

- Descarga y limpieza pueden omitirse con `skip_*` para acelerar iteraciones
- SPY (benchmark) se carga desde `data/clean` y, si no existe, desde `data/raw/daily`
- Artifacts por ticker: `models/`, `predictions/`, `backtest/` bajo `runs/e1_simple/<timestamp>/<ticker>/`
- Run de resumen: `summary_all.csv` y métricas agregadas (IC, Decision Score)

---

## E1 Baseline Linear Regression - Comparación

### Desde Airflow UI

1. **Acceder**: http://localhost:8080
2. **Trigger con Configuración**:
   - Click en `e1_baseline_linear_regression`
   - Click en **"Play"** (▶) → **"Trigger DAG w/ config"**
   - En el JSON editor:
   
   ```json
   {
     "tickers": "AAPL,MSFT,GOOGL",
     "auto_compare_with_gru": "True"
   }
   ```
   
   - Click **"Trigger"**

### Desde CLI

```bash
# Baseline para tickers específicos con comparación
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_baseline_linear_regression \
  --conf '{"tickers": "AAPL,MSFT", "auto_compare_with_gru": "True"}'

# Sin comparación automática
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_baseline_linear_regression \
  --conf '{"tickers": "AAPL", "auto_compare_with_gru": "False"}'

# Todos los tickers E1 (campo vacío)
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_baseline_linear_regression \
  --conf '{"tickers": ""}'
```

### Parámetros del DAG Baseline

**`tickers`** (string)
- Lista separada por comas: `"AAPL,MSFT,GOOGL"`
- Vacío = todos los tickers E1
- Default: `"YPFD.BA, GGAL.BA, PAMP.BA, BYMA.BA, CEPU.BA, AAPL, MSFT, JNJ, PG, V"`

**`auto_compare_with_gru`** (boolean)
- `"True"`: Compara automáticamente con último run de GRU
- `"False"`: Solo entrena baseline (sin comparación)
- Default: `"True"`

### Flujo Recomendado: GRU + Baseline

```bash
# 1. Ejecutar GRU primero (Lunes 2 AM automático, o manual)
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_conservative_pipeline \
  --conf '{"tickers": "AAPL,MSFT"}'

# 2. Esperar que termine (~10-15 min)

# 3. Ejecutar Baseline con comparación (Lunes 3 AM automático, o manual)
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_baseline_linear_regression \
  --conf '{"tickers": "AAPL,MSFT", "auto_compare_with_gru": "True"}'
```

### Ver Resultados Baseline en MLflow

1. **MLflow UI**: http://localhost:5050
2. **Experimento**: "E1_Baseline_LinearRegression"
3. **Filtrar**: `params.ticker = "AAPL"`
4. **Comparar**: Seleccionar runs → "Compare"

### Comparación Baseline vs GRU

Resultado en: `runs/e1_baseline/<timestamp>/comparison_vs_gru.csv`

```bash
# Ver comparación del último run
ls -lht runs/e1_baseline/ | head -n 2
cat runs/e1_baseline/<timestamp>/comparison_vs_gru.csv
```

### Notas Baseline

- Usa mismos features que GRU (27 indicadores)
- Regresión Lineal simple (sklearn)
- Métricas: MAE, RMSE, IC (Spearman), Directional Accuracy
- Backtesting con mismos parámetros (tau_buy, tau_sell, costos)
- IC > 0.05 = significativo en finanzas

---

**Ver más**: 
- [README_E1_BASELINE.md](README_E1_BASELINE.md) - Documentación completa del baseline
- [QUICKSTART_E1_BASELINE.md](QUICKSTART_E1_BASELINE.md) - Guía rápida
- [docs/AIRFLOW_E1_BASELINE_DAG.md](docs/AIRFLOW_E1_BASELINE_DAG.md) - Documentación del DAG
- [README_DOCKER.md](README_DOCKER.md) - Setup Docker

---
