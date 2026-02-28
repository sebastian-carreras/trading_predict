# Dashboard de Monitoreo — MLflow UI

**Panel de supervisión** para tiempos de entrenamiento/predicción, métricas ML y de trading, con alertas por estrategia.

---

## Quick Start

### Opción A — Local (sin Docker)

```bash
# 1. Entrenar — MLflow guarda automáticamente en SQLite local
python -m src.train_e1_simple_pipeline --tickers AAPL

# 2. Levantar MLflow UI apuntando a la DB local
mlflow ui --backend-store-uri sqlite:///runs/mlflow_local/mlflow.db --port 5050

# 3. Ver dashboard en navegador
open http://localhost:5050

# 4. Generar reporte de alertas (vista summary)
python -m src.dashboard.checker

# 5. Vista per-ticker de una estrategia
python -m src.dashboard.checker --view ticker --strategy e1_conservative

# 6. Historial de un ticker específico
python -m src.dashboard.checker --view history --strategy e1_conservative --ticker AAPL
```

> **Nota:** No es necesario tener MLflow server corriendo para entrenar.
> Los pipelines guardan métricas en `runs/mlflow_local/mlflow.db` automáticamente.
> Al levantar MLflow UI después, se ven todos los runs anteriores.

### Opción B — Docker (Postgres + MinIO)

```bash
# 1. Levantar solo los servicios del dashboard
docker compose --profile dashboard up -d

# 2. Ver MLflow UI
open http://localhost:5050

# 3. Ver MinIO UI (artifacts)
open http://localhost:9001   # user: minio / pass: minio123

# 4. Generar reporte de alertas
python -m src.dashboard.checker
```

---

## Arquitectura

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Pipelines  │────▶│    MLflow     │────▶│  MLflow UI   │
│ E1/E2/E3/E4  │     │   Server     │     │  Dashboard   │
└──────┬───────┘     └──────┬───────┘     └──────────────┘
       │                    │
       │  timing JSONL      │  Backend Store
       ▼                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   reports/   │     │  PostgreSQL  │     │    MinIO      │
│   timing/    │     │  (metadata)  │     │ (artifacts)  │
└──────────────┘     └──────────────┘     └──────────────┘
       │
       ▼
┌──────────────┐
│   Checker    │──── reports/dashboard/
│   CLI        │     ├── dashboard_report_*.csv
│              │     ├── ticker_report_*.csv
│              │     └── history_report_*.csv
└──────────────┘
```

### Flujo de datos

1. **Pipelines** entrenan modelos y registran métricas + tiempos en MLflow
2. **MLflow Server** almacena metadata en Postgres y artifacts en MinIO
3. **MLflow UI** muestra experimentos por estrategia con filtros y gráficos
4. **Checker CLI** cruza datos de MLflow + timing log JSONL y genera reportes con alertas

---

## Métricas del Dashboard

### Por Estrategia

| Categoría | Métrica | Descripción |
|-----------|---------|-------------|
| ⏱ Timing | `train_seconds` | Tiempo de entrenamiento por ticker |
| ⏱ Timing | `predict_seconds` | Tiempo de predicción por ticker |
| 🤖 ML | `ml_mae` | Error absoluto medio |
| 🤖 ML | `ml_rmse` | Error cuadrático medio |
| 🤖 ML | `ml_ic` | Information Coefficient (Spearman) |
| 🤖 ML | `ml_directional_accuracy` | % aciertos de dirección |
| 📈 Trading | `bt_sharpe` | Sharpe ratio |
| 📈 Trading | `bt_sortino` | Sortino ratio |
| 📈 Trading | `bt_cagr` | Retorno anualizado compuesto |
| 📈 Trading | `bt_max_drawdown` | Máxima caída desde pico |
| 📈 Trading | `bt_calmar` | Calmar ratio (CAGR / Max DD) |
| 📈 Trading | `bt_profit_factor` | Profit factor (gross profit / gross loss) |

### Estadísticas Mostradas

Para cada métrica se muestra:
- **Promedio** (avg) — promedio de todos los runs
- **Último** (last) — valor del run más reciente
- **p50 / p95** — percentiles (solo para timing)
- **Alerta** — 🟢 verde / 🟡 amarillo / 🔴 rojo

---

## Umbrales de Alerta por Estrategia

### E1 Simple / E1 Conservative

Umbrales calibrados con distribución real de métricas (feb 2026).

| Métrica | 🟢 Verde | 🟡 Amarillo | 🔴 Rojo |
|---------|----------|-------------|---------|
| MAE | ≤ 0.15 | 0.15–0.35 | > 0.35 |
| RMSE | ≤ 0.20 | 0.20–0.45 | > 0.45 |
| IC | ≥ 0.10 | -0.05–0.10 | < -0.05 |
| Dir. Accuracy | ≥ 55% | 50–55% | < 50% |
| Sharpe | ≥ 1.0 | 0.3–1.0 | < 0.3 |
| Sortino | ≥ 1.2 | 0.3–1.2 | < 0.3 |
| CAGR (Simple) | ≥ 8% | 0–8% | < 0% |
| CAGR (Conservative) | ≥ 0% | -5%–0% | < -5% |
| Max Drawdown | ≤ 30% | 30–55% | > 55% |
| Calmar | ≥ 0.5 | 0.1–0.5 | < 0.1 |
| Profit Factor | ≥ 1.3 | 1.0–1.3 | < 1.0 |
| Train time (Simple) | ≤ 2 min | 2–5 min | > 5 min |
| Train time (Conservative) | ≤ 10 min | 10–20 min | > 20 min |

### E2 Simple / E2 Moderate

| Métrica | 🟢 Verde | 🟡 Amarillo | 🔴 Rojo |
|---------|----------|-------------|---------|
| MAE | ≤ 0.02 | 0.02–0.04 | > 0.04 |
| RMSE | ≤ 0.03 | 0.03–0.06 | > 0.06 |
| IC | ≥ 0.05 | 0.02–0.05 | < 0.02 |
| Dir. Accuracy | ≥ 55% | 50–55% | < 50% |
| Sharpe | ≥ 0.8 | 0.4–0.8 | < 0.4 |
| Max Drawdown | ≤ 15% | 15–20% | > 20% |

### E3 Intraday

| Métrica | 🟢 Verde | 🟡 Amarillo | 🔴 Rojo |
|---------|----------|-------------|---------|
| MAE | ≤ 0.001 | 0.001–0.003 | > 0.003 |
| RMSE | ≤ 0.002 | 0.002–0.005 | > 0.005 |
| IC | ≥ 0.03 | 0.01–0.03 | < 0.01 |
| Dir. Accuracy | ≥ 52% | 48–52% | < 48% |
| Profit Factor | ≥ 1.4 | 1.0–1.4 | < 1.0 |
| Max DD Intraday | ≤ 3% | 3–5% | > 5% |

### E4 Pairs

| Métrica | 🟢 Verde | 🟡 Amarillo | 🔴 Rojo |
|---------|----------|-------------|---------|
| Sharpe | ≥ 1.0 | 0.5–1.0 | < 0.5 |
| Max Drawdown | ≤ 10% | 10–20% | > 20% |
| Win Rate | ≥ 55% | 45–55% | < 45% |
| Total Return | ≥ 5% | 0–5% | < 0% |

> Los umbrales se configuran en [`src/config/dashboard_thresholds.yaml`](src/config/dashboard_thresholds.yaml)

---

## Checker CLI

El checker tiene 3 vistas: **summary**, **ticker** y **history**.

### Vista Summary (default)

Promedio y último valor por estrategia, con alertas.

```bash
# Todas las estrategias
python -m src.dashboard.checker

# Una estrategia específica
python -m src.dashboard.checker --strategy e1_conservative
```

**Output:**

```
======================================================================
  E1 Conservative
  GRU 2 capas, walk-forward 5 folds, horizon 90d
======================================================================

  ⏱ Timing
  ────────────────────────────────────────────────────────────────
  Metric                            Avg     Last  Alert
  ────────────────────────────────────────────────────────────────
  train_seconds                    45.2     42.1  🟢
  predict_seconds                   0.3      0.2  🟢

  🤖 ML
  ────────────────────────────────────────────────────────────────
  ml_mae                         0.0280   0.0250  🟢
  ml_rmse                        0.0420   0.0380  🟢
  ml_ic                          0.0650   0.0720  🟢
  ml_directional_accuracy        0.5600   0.5800  🟢

  📈 Trading
  ────────────────────────────────────────────────────────────────
  bt_sharpe                      0.8500   1.0200  🟢
  bt_cagr                        0.0900   0.1100  🟢
  bt_max_drawdown                0.1200   0.1400  🟢
```

### Vista Ticker — Per-ticker

Muestra métricas del último entrenamiento (o promedio de los últimos N) para cada ticker de una estrategia. Incluye **score compuesto** y alertas separadas por categoría (ML y Trading).

```bash
# Último entrenamiento por ticker
python -m src.dashboard.checker --view ticker --strategy e1_conservative

# Promedio de últimos 3 entrenamientos por ticker
python -m src.dashboard.checker --view ticker --strategy e1_conservative --last 3
```

**Output:**

```
================================================================================
  E1 Conservative — Per-Ticker (último entrenamiento)
  Score = 40% bt_sharpe + 30% bt_cagr + 30% bt_max_drawdown
================================================================================

  🤖 ML
  ──────────────────────────────────────────────────────────────────────────
  Ticker              MAE       RMSE         IC    Dir Acc
  ──────────────────────────────────────────────────────────────────────────
  AAPL             0.0820     0.1050     0.2100     0.5800  🟢
  GGAL.BA          0.1240     0.1680    -0.0300     0.4900  🟡

  📈 Trading
  ──────────────────────────────────────────────────────────────────────────
  Ticker           Sharpe    Sortino       CAGR     Max DD   Score
  ──────────────────────────────────────────────────────────────────────────
  AAPL             1.2300     1.8100     0.1500     0.1200  0.5640  🟢
  GGAL.BA          0.4500     0.5200     0.0200     0.3800  0.1820  🟡

  📊 Resumen ML:      1🟢 1🟡 0🔴 de 2 tickers
  📊 Resumen Trading: 1🟢 1🟡 0🔴 de 2 tickers
  📊 US avg Score: 0.5640 | AR avg Score: 0.1820
```

El **score** se calcula con los mismos pesos configurados en `src/config/base.yaml` (sección `lifecycle.promotion.scoring_weights`).

### Vista History — Historial de un ticker

Muestra los últimos N entrenamientos de un ticker con tendencia del score.

```bash
# Últimos 10 entrenamientos (default)
python -m src.dashboard.checker --view history --strategy e1_conservative --ticker AAPL

# Últimos 5 entrenamientos
python -m src.dashboard.checker --view history --strategy e1_conservative --ticker AAPL --last 5
```

**Output:**

```
================================================================================
  E1 Conservative — AAPL (últimos 5 entrenamientos)
  Score = 40% bt_sharpe + 30% bt_cagr + 30% bt_max_drawdown
================================================================================

  ────────────────────────────────────────────────────────────────────────────────
  Fecha            MAE     RMSE       IC  Dir Acc  Sharpe  Sortino     CAGR   Max DD    Score  ML  Trad
  ────────────────────────────────────────────────────────────────────────────────
  2026-02-25    0.0820   0.1050   0.2100   0.5800  1.2300   1.8100   0.1500   0.1200   0.5640  🟢 🟢
  2026-02-20    0.0900   0.1150   0.1800   0.5600  1.1000   1.6500   0.1200   0.1400   0.5120  🟢 🟢
  2026-02-15    0.1100   0.1400   0.0800   0.5300  0.8500   1.2000   0.0600   0.2200   0.3680  🟢 🟡
  2026-02-10    0.1250   0.1600   0.0500   0.5100  0.6200   0.8800   0.0300   0.2800   0.2560  🟡 🟡
  2026-02-05    0.1400   0.1800   0.0200   0.5000  0.4000   0.5500   0.0100   0.3500   0.1520  🟡 🟡

  📈 Tendencia score: 0.1520 → 0.5640 (+271.1%) ↑
```

### Referencia de argumentos CLI

| Argumento | Descripción | Default |
|-----------|-------------|---------|
| `--view` | Vista: `summary`, `ticker`, `history` | `summary` |
| `--strategy` | Clave de estrategia (ej: `e1_conservative`). Requerido para ticker/history. | Todas |
| `--ticker` | Ticker (ej: `AAPL`). Requerido para history. | — |
| `--last` | ticker: promedia últimos N. history: cantidad de runs. | ticker: 1, history: 10 |
| `--save` | Guardar CSV | `True` |
| `--quiet` | Suprimir output a consola | `False` |

### Output

- **stdout:** tabla formateada con alertas
- **CSV summary:** `reports/dashboard/dashboard_report_latest.csv`
- **CSV ticker:** `reports/dashboard/ticker_report_{strategy}_latest.csv`
- **CSV history:** `reports/dashboard/history_report_{strategy}_{ticker}_latest.csv`
- **CSV histórico:** `reports/dashboard/dashboard_report_YYYYMMDD_HHMMSS.csv`

---

## Uso de MLflow UI como Dashboard

### Vistas Recomendadas

#### 1. Vista por Estrategia
En MLflow UI, cada estrategia tiene su propio **Experiment**:

| Estrategia | Experiment Name |
|------------|-----------------|
| E1 Simple | `E1_Simple` |
| E1 Conservative | `E1_Conservative_Strategy` |
| E2 Simple | `E2_Simple` |
| E2 Moderate | `E2_Moderate` |
| E3 Intraday | `E3_Intraday` |
| E4 Pairs | `E4_Pairs` |

#### 2. Filtrar por Tags
Cada run incluye tags de dashboard para filtrado:
- `dashboard.strategy` — clave de la estrategia
- `dashboard.ticker` — ticker del activo
- `dashboard.run_type` — `train`, `predict`, `evaluate`, `aggregate`
- `dashboard.alert.overall` — 🟢/🟡/🔴
- `dashboard.alert.{metric}` — alerta por métrica individual

**Ejemplo de filtro en MLflow UI:**
```
tags.dashboard.strategy = "e1_simple" AND tags.dashboard.alert.overall = "🔴"
```
Nota: el tag es `dashboard.strategy` (con punto), no `dashboard_strategy`.

#### 3. Comparar Runs
1. Seleccionar múltiples runs en la tabla
2. Click "Compare" → ver métricas lado a lado
3. Usar "Chart" para graficar evolución temporal

#### 4. Gráficos Útiles
- **Scatter:** `ml_ic` vs `bt_sharpe` → correlación predicción-trading
- **Line:** `ml_mae` por run → evolución de calidad
- **Bar:** `dashboard.train_seconds` por ticker → identificar bottlenecks

---

## Tags de Dashboard

Cada pipeline registra automáticamente los siguientes tags y métricas:

### Tags (para filtrado)
```
dashboard.strategy         = "e1_simple"
dashboard.strategy_display = "E1 Simple"
dashboard.ticker           = "AAPL"
dashboard.run_type         = "train"
dashboard.alert.overall    = "🟢"
dashboard.alert.ml_mae     = "🟢"
dashboard.alert.bt_sharpe  = "🟡"
...
```

### Métricas adicionales
```
dashboard.train_seconds    = 45.2
dashboard.predict_seconds  = 0.3
timing_train_seconds       = 45.2   (también en namespace timing_*)
timing_predict_seconds     = 0.3
```

---

## Integración en Pipelines

### Context manager (recomendado)

Para integrar en cualquier pipeline, usar el context manager:

```python
from src.dashboard.integration import with_dashboard_logging

with mlflow.start_run(run_name=f"E2Simple_{ticker}_{ts}"):
    with with_dashboard_logging("e2_simple", ticker=ticker) as ctx:
        # ... entrenar y evaluar ...
        ctx["train_seconds"] = train_duration
        ctx["predict_seconds"] = pred_duration
        ctx["metrics"] = {"ml_mae": mae, "bt_sharpe": sharpe, ...}
```

El context manager registra automáticamente:
- Tags de estrategia y ticker
- Tiempos de entrenamiento/predicción (fallback a wall-clock si no se setean)
- Alertas por umbral para cada métrica

---

## Infraestructura

### Modo Local (desarrollo)

```bash
# MLflow UI con backend local SQLite
mlflow ui --backend-store-uri sqlite:///runs/mlflow_local/mlflow.db --port 5050
```

- **Backend store:** `runs/mlflow_local/mlflow.db` (SQLite)
- **Artifact store:** `runs/mlflow_local/artifacts/` (filesystem)
- **Adecuado para:** desarrollo, experimentación individual

### Modo Docker (producción local)

```bash
# Levantar Postgres + MinIO + MLflow
docker compose --profile dashboard up -d
```

- **Backend store:** PostgreSQL (`postgres:5432/mlflow_db`)
- **Artifact store:** MinIO S3 (`s3://mlflow/`)
- **Adecuado para:** producción local, múltiples usuarios

### Escalado Futuro — Tracking Server Remoto

Para escalar a un entorno compartido o cloud:

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Pipelines  │────▶│  MLflow Server   │────▶│  MLflow UI   │
│  (local/CI) │     │  (EC2/GKE/ECS)   │     │  (web)       │
└─────────────┘     └────────┬─────────┘     └──────────────┘
                             │
                    ┌────────┴─────────┐
                    │                  │
              ┌─────▼──────┐    ┌──────▼─────┐
              │ PostgreSQL │    │    S3       │
              │ (RDS/Cloud │    │ (AWS/GCS/  │
              │  SQL)      │    │  MinIO)    │
              └────────────┘    └────────────┘
```

#### Pasos para escalar:

1. **Postgres externo:**
   ```bash
   # Cambiar backend store URI
   MLFLOW_BACKEND_STORE_URI=postgresql://user:pass@rds-host:5432/mlflow_db
   ```

2. **S3 real (o MinIO remoto):**
   ```bash
   # Cambiar artifact root
   MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://my-mlflow-bucket/
   AWS_ACCESS_KEY_ID=...
   AWS_SECRET_ACCESS_KEY=...
   ```

3. **Autenticación (cuando sea necesario):**
   - MLflow con nginx reverse proxy + basic auth
   - O MLflow con OAuth (MLflow 2.10+)

4. **Docker Compose → Kubernetes:**
   ```yaml
   # Helm chart de MLflow o deployment custom
   # Postgres → CloudSQL/RDS
   # MinIO → S3/GCS
   ```

---

## Estructura de Archivos

```
src/
├── config/
│   ├── base.yaml                      # Config principal (incluye scoring_weights)
│   └── dashboard_thresholds.yaml      # Umbrales de alerta por estrategia
├── dashboard/
│   ├── __init__.py                    # Helpers: tags, timing, alerts
│   ├── checker.py                     # CLI: genera reporte de salud (3 vistas)
│   └── integration.py                 # Context manager para pipelines
├── lifecycle/
│   └── promotion.py                   # compute_score() — usado por vista ticker
runs/
└── mlflow_local/
    ├── mlflow.db                      # SQLite — backend store local (single source of truth)
    └── artifacts/                     # Artifacts de los runs (modelos, configs, etc.)
reports/
├── timing/
│   └── timing_log.jsonl               # Log de tiempos (JSONL)
└── dashboard/
    ├── dashboard_report_latest.csv    # Último reporte summary
    ├── ticker_report_*_latest.csv     # Último reporte per-ticker
    ├── history_report_*_latest.csv    # Último reporte historial
    └── dashboard_report_*.csv         # Histórico
```

---

## FAQ

### ¿Por qué MLflow UI y no Streamlit/Dash?

1. **Ya está integrado** — los pipelines ya registran métricas en MLflow
2. **Comparación nativa** — MLflow UI tiene comparación de runs built-in
3. **Filtros potentes** — por tags, métricas, parámetros
4. **Artifacts** — modelos, predicciones, backtests accesibles desde la UI
5. **Escalable** — de local a servidor remoto sin cambiar código

### ¿Cómo veo solo los runs con alertas rojas?

En MLflow UI, filtrar por:
```
tags.dashboard.alert.overall = "🔴"
```

### ¿Cómo comparo el promedio vs el último run?

Usar el checker CLI:
```bash
python -m src.dashboard.checker --strategy e1_conservative
```

### ¿Cómo veo métricas por ticker?

```bash
python -m src.dashboard.checker --view ticker --strategy e1_conservative
```

### ¿Cómo veo la evolución de un ticker?

```bash
python -m src.dashboard.checker --view history --strategy e1_conservative --ticker AAPL
```

### ¿Qué es el score compuesto?

Es un promedio ponderado de métricas de trading, configurado en `src/config/base.yaml` bajo `lifecycle.promotion.scoring_weights`. Se usa tanto para la vista ticker como para la lógica de promoción champion/challenger.

### ¿Los umbrales son editables?

Sí. Editar [`src/config/dashboard_thresholds.yaml`](src/config/dashboard_thresholds.yaml) y re-ejecutar el checker.

### ¿Dónde se guardan los datos de MLflow?

En **`runs/mlflow_local/mlflow.db`** (SQLite). Es la single source of truth para métricas.
Los pipelines y el checker usan la misma cascada de fallback:

1. Servidor remoto (`MLFLOW_TRACKING_URI`) si está configurado y accesible
2. SQLite local (`runs/mlflow_local/mlflow.db`)
3. File store por defecto (`mlruns/`) como último recurso

> **Nota:** La carpeta `mlruns/` ya no se usa. Fue eliminada porque contenía
> experimentos corruptos. Todo el tracking se hace via SQLite local o servidor remoto.

---

## Referencias

- [README_E1_SIMPLE.md](README_E1_SIMPLE.md) — E1 Simple pipeline
- [README_E1.md](README_E1.md) — E1 Conservative pipeline
- [README_E2.md](README_E2.md) — E2 Moderate pipeline
- [README_E3.md](README_E3.md) — E3 Intraday pipeline
- [README_E4.md](README_E4.md) — E4 Pairs pipeline
- [README_DOCKER.md](README_DOCKER.md) — Infraestructura Docker

---

**Última actualización:** Febrero 27, 2026
**Versión:** 2.0
