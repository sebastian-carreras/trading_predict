# Scripts

Scripts organizados por funcionalidad para el proyecto de Trading Prediction.

## 📁 Estructura

```
scripts/
├── data/                   # Gestión de datos
├── optimization/           # Optimización de hiperparámetros
├── evaluation/             # Evaluación y validación de modelos
├── trading/                # Trading en vivo (IOL API)
├── airflow/                # Validación de DAGs Airflow
└── mlflow/                 # Utilidades de MLflow
```

---

## 📂 data/ - Gestión de Datos

### `run_data_cleaning.py`
Descarga y limpia datos históricos de yfinance.

**Uso:**
```bash
python scripts/data/run_data_cleaning.py
```

**Output:**
- `data/raw/` - Datos crudos de yfinance
- `data/clean/` - Datos limpios en formato CSV

---

## 🔧 optimization/ - Optimización de Hiperparámetros

### `optimize_e1_hyperparameters.py`
Optimización con Optuna para E1 Conservative (GRU) con logging conciso por trial.

**Parámetros optimizados (actual):**
- `tau_buy`, `tau_sell`
- `gru_units_1`, `gru_units_2`, `dropout`
- `learning_rate`, `weight_decay`, `batch_size`

**Importante:**
- En E1 ya **no** se optimizan parámetros de split (`n_folds`, `internal_val_fraction`).
- Esos parámetros quedan fijos en la configuración del pipeline.

**Uso típico:**
```bash
# 1 ticker (smoke test)
python scripts/optimization/optimize_e1_hyperparameters.py --ticker YPFD.BA --n_trials 1

# modo rápido (subset)
python scripts/optimization/optimize_e1_hyperparameters.py --quick --n_trials 10

# optimización por ticker (recomendado para generar YAML consumible)
python scripts/optimization/optimize_e1_hyperparameters.py --per_ticker --n_trials 50
```

**Salida por trial (modo conciso):**
```text
[Trial 0007] obj=+0.6042 ic=+0.3121 sharpe=+0.8963 trades=8/10 time=42.5s | tau=(0.050,0.000) gru=[112,48] do=0.25 lr=0.000200 wd=0.000120 bs=64
```

`weight_decay` se optimiza en log scale con rango default `1e-6 → 1e-2`.
Rango recomendado para pruebas iniciales: `1e-5 → 1e-3`.

**Artifacts principales:**
- `reports/hyperparameter_optimization/best_params_e1.yaml`
- `reports/hyperparameter_optimization/e1_all_trials.csv`
- `reports/hyperparameter_optimization/e1_tuned_params_by_ticker.yaml` (consumible)
- `reports/hyperparameter_optimization/e1_tuned_params_by_ticker.meta.yaml` (con metadata)

**Integración con entrenamiento E1:**
```bash
export E1_TUNED_PARAMS_PATH="reports/hyperparameter_optimization/e1_tuned_params_by_ticker.yaml"
```

El entrenamiento E1 aplica overrides de `thresholds` y `model` por ticker.

### `optimize_e2_hyperparameters.py`
Optimización con Optuna para E2 Moderate.

**Uso:**
```bash
python scripts/optimization/optimize_e2_hyperparameters.py \
    --ticker MSFT \
    --n-trials 100
```

### `analyze_optuna_db.py`
Analiza resultados almacenados en Optuna DB.

**Uso:**
```bash
python scripts/optimization/analyze_optuna_db.py \
    --study-name e1_conservative_optimization
```

---

## 📊 evaluation/ - Evaluación y Validación

### `e1_retrospective_validation.py`
Validación out-of-time con modelo GRU (E1 Simple).

**Uso:**
```bash
python scripts/evaluation/e1_retrospective_validation.py \
    --ticker AAPL \
    --train-days-ago 360 \
    --horizon 90
```

**Características:**
- Usa arquitectura GRU real (2 capas: 64, 32)
- Verifica `data/clean/` antes de descargar
- Guarda resultados en MLflow

### `compare_e1_models.py`
Compara modelos E1 entrenados.

**Uso:**
```bash
python scripts/evaluation/compare_e1_models.py \
    --run-ids run1 run2 run3
```

### `compare_e1_versions.py`
Compara resultados agregados de las 3 versiones E1: baseline, simple y conservative.

**Uso:**
```bash
# Último run de cada versión
python scripts/evaluation/compare_e1_versions.py

# Filtrado por tickers
python scripts/evaluation/compare_e1_versions.py --tickers AAPL,MSFT

# Comparación ticker-vs-ticker entre versiones
python scripts/evaluation/compare_e1_versions.py --tickers AAPL,MSFT --per-ticker
```

**Incluye:**
- Tiempo de entrenamiento (`train_time_seconds_avg`, `train_time_seconds_total`)
- Métricas ML (`mae`, `rmse`, `ic`, `directional_accuracy`)
- Métricas de trading (`bt_sharpe`, `bt_cagr`, `bt_max_drawdown`, `bt_num_trades`)

**Salida:**
- Carpeta: `reports/tables/e1_versions_comparison/`
- Archivos con timestamp: `e1_versions_comparison_<YYYYMMDD_HHMMSS>*.csv/.md`

---

## 💰 trading/ - Trading en Vivo

### `e1_simple_iol_live_trade.py`
Trading en vivo con IOL API usando modelos E1 Simple.

**Uso:**
```bash
# Dry-run (sin ejecutar órdenes)
python scripts/trading/e1_simple_iol_live_trade.py \
    --ticker GGAL.BA \
    --action BUY \
    --market bCBA

# Ejecución real en producción
python scripts/trading/e1_simple_iol_live_trade.py \
    --ticker AAPL \
    --action BUY \
    --market aNYS \
    --production \
    --quantity 10
```

**Features:**
- Predicción con modelos E1 Simple GRU
- Integración con IOL API (producción/sandbox)
- Logging automático en MLflow
- Decision score para señales BUY/HOLD/SELL

**Ver:** Documentación IOL API en `README_IOL_FALLBACK.md`

### `iol_list_e1_prices.py`
Lista precios actuales de IOL para tickers E1 Simple.

**Uso:**
```bash
python scripts/trading/iol_list_e1_prices.py
```

**Output:**
- Tabla con precios en consola
- JSON en `reports/iol_prices_YYYYMMDD_HHMMSS.json`

---

## 🌊 airflow/ - Validación Airflow

### `validate_dags.py`
Valida sintaxis de DAGs de Airflow.

**Uso:**
```bash
python scripts/airflow/validate_dags.py
```

---

## 🧪 mlflow/ - Utilidades MLflow

### `up_transparent_mlflow.sh`
Levanta MLflow en modo transparente offline→online:
- Detecta tu versión local de `mlflow`
- Reconstruye el contenedor MLflow con la misma versión
- Inicia MLflow con backend/artifacts compartidos en `runs/mlflow_local`

**Uso:**
```bash
bash scripts/mlflow/up_transparent_mlflow.sh
```

**Override opcional de versión:**
```bash
MLFLOW_VERSION=3.8.1 bash scripts/mlflow/up_transparent_mlflow.sh
```

---

## 🔗 Referencias

- **Data Cleaning:** `README_DATA_CLEANING.md`
- **Hyperparameter Tuning:** `README_HYPERPARAMETER_TUNING.md`
- **IOL Trading:** `README_IOL_FALLBACK.md`

---

## 📝 Notas

### Cambios recientes
- **2026-01-21:** Reorganización de scripts en subcarpetas por funcionalidad
- Scripts temporales eliminados (`check_iol_msft.py`)

### Paths importantes
- Datos: `data/clean/*.csv`
- Modelos: `runs/e1_simple/`, `runs/e2_moderate/`
- MLflow: `runs/mlflow_local/mlflow.db`
- Optuna: `runs/optuna_trials/optuna.db`
