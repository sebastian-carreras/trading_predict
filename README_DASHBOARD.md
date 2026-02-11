# Dashboard de Monitoreo — MLflow UI

**Panel de supervisión** para tiempos de entrenamiento/predicción, métricas ML y de trading, con alertas por estrategia.

---

## Quick Start

### Opción A — Local (sin Docker)

```bash
# 1. Levantar MLflow UI local
mlflow ui --port 5050

# 2. Entrenar con tracking habilitado
export MLFLOW_TRACKING_URI=http://localhost:5050
python -m src.train_e1_simple_pipeline --tickers AAPL

# 3. Ver dashboard en navegador
open http://localhost:5050

# 4. Generar reporte de alertas
python -m src.dashboard.checker
```

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

**Output del checker:**

```
======================================================================
  E1 Simple
  GRU 1 capa, time split, horizon 90d
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

✓ Report saved: reports/dashboard/dashboard_report_20260207_120000.csv
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
│   Checker    │──── reports/dashboard/dashboard_report_*.csv
│   CLI        │
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
| 📈 Trading | `bt_cagr` | Retorno anualizado compuesto |
| 📈 Trading | `bt_max_drawdown` | Máxima caída desde pico |

### Estadísticas Mostradas

Para cada métrica se muestra:
- **Promedio** (avg) — promedio de todos los runs
- **Último** (last) — valor del run más reciente
- **p50 / p95** — percentiles (solo para timing)
- **Alerta** — 🟢 verde / 🟡 amarillo / 🔴 rojo

---

## Umbrales de Alerta por Estrategia

### E1 Simple

| Métrica | 🟢 Verde | 🟡 Amarillo | 🔴 Rojo |
|---------|----------|-------------|---------|
| MAE | ≤ 0.03 | 0.03–0.06 | > 0.06 |
| RMSE | ≤ 0.05 | 0.05–0.08 | > 0.08 |
| IC | ≥ 0.05 | 0.02–0.05 | < 0.02 |
| Dir. Accuracy | ≥ 55% | 50–55% | < 50% |
| Sharpe | ≥ 1.0 | 0.5–1.0 | < 0.5 |
| CAGR | ≥ 8% | 0–8% | < 0% |
| Max Drawdown | ≤ 15% | 15–25% | > 25% |
| Train time | ≤ 2 min | 2–5 min | > 5 min |

### E1 Conservative

| Métrica | 🟢 Verde | 🟡 Amarillo | 🔴 Rojo |
|---------|----------|-------------|---------|
| Sharpe | ≥ 0.6 | 0.3–0.6 | < 0.3 |
| Max Drawdown | ≤ 15% | 15–25% | > 25% |
| Train time | ≤ 10 min | 10–20 min | > 20 min |

### E2 Simple / Moderate

| Métrica | 🟢 Verde | 🟡 Amarillo | 🔴 Rojo |
|---------|----------|-------------|---------|
| MAE | ≤ 0.02 | 0.02–0.04 | > 0.04 |
| Sharpe | ≥ 0.8 | 0.4–0.8 | < 0.4 |
| Max Drawdown | ≤ 15% | 15–20% | > 20% |

### E3 Intraday

| Métrica | 🟢 Verde | 🟡 Amarillo | 🔴 Rojo |
|---------|----------|-------------|---------|
| Profit Factor | ≥ 1.4 | 1.0–1.4 | < 1.0 |
| Max DD Intraday | ≤ 3% | 3–5% | > 5% |

### E4 Pairs

| Métrica | 🟢 Verde | 🟡 Amarillo | 🔴 Rojo |
|---------|----------|-------------|---------|
| Sharpe | ≥ 1.0 | 0.5–1.0 | < 0.5 |
| Max Drawdown | ≤ 10% | 10–20% | > 20% |
| Win Rate | ≥ 55% | 45–55% | < 45% |

> Los umbrales se configuran en [`src/config/dashboard_thresholds.yaml`](src/config/dashboard_thresholds.yaml)

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

## Checker CLI

### Todas las estrategias
```bash
python -m src.dashboard.checker
```

### Una estrategia específica
```bash
python -m src.dashboard.checker --strategy e1_simple
```

### Solo guardar CSV (sin output a consola)
```bash
python -m src.dashboard.checker --quiet
```

### Output
- **stdout:** tabla formateada con alertas
- **CSV:** `reports/dashboard/dashboard_report_latest.csv`
- **CSV histórico:** `reports/dashboard/dashboard_report_YYYYMMDD_HHMMSS.csv`

---

## Integración en Pipelines

### E1 Simple (ya integrado)

El pipeline `train_e1_simple_pipeline.py` ya registra:
1. Tags de dashboard al inicio del run
2. Tiempos de entrenamiento y predicción
3. Alertas por métrica al finalizar

### Otros Pipelines — Integración rápida

Para integrar en E2/E3/E4, usar el context manager:

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
- Tiempos de entrenamiento/predicción
- Alertas por umbral para cada métrica

---

## Infraestructura

### Modo Local (desarrollo)

```bash
# MLflow UI con backend local (SQLite + filesystem)
mlflow ui --port 5050
```

- **Backend store:** `mlruns/` (SQLite)
- **Artifact store:** `mlruns/` (filesystem)
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
│   ├── base.yaml                      # Config principal
│   └── dashboard_thresholds.yaml      # Umbrales de alerta por estrategia
├── dashboard/
│   ├── __init__.py                    # Helpers: tags, timing, alerts
│   ├── checker.py                     # CLI: genera reporte de salud
│   └── integration.py                 # Context manager para pipelines
reports/
├── timing/
│   └── timing_log.jsonl               # Log de tiempos (JSONL)
└── dashboard/
    ├── dashboard_report_latest.csv    # Último reporte
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
python -m src.dashboard.checker --strategy e1_simple
```

O en MLflow UI: seleccionar runs y usar "Compare".

### ¿Los umbrales son editables?

Sí. Editar [`src/config/dashboard_thresholds.yaml`](src/config/dashboard_thresholds.yaml) y re-ejecutar el checker.

---

## Referencias

- [README_E1_SIMPLE.md](README_E1_SIMPLE.md) — E1 Simple pipeline
- [README_E1.md](README_E1.md) — E1 Conservative pipeline
- [README_E2.md](README_E2.md) — E2 Moderate pipeline
- [README_E3.md](README_E3.md) — E3 Intraday pipeline
- [README_E4.md](README_E4.md) — E4 Pairs pipeline
- [README_DOCKER.md](README_DOCKER.md) — Infraestructura Docker
- [README_CONTINUOUS_EVALUATION.md](README_CONTINUOUS_EVALUATION.md) — Evaluación continua

---

**Última actualización:** Febrero 7, 2026
**Versión:** 1.0
