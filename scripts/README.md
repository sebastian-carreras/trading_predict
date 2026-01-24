# Scripts

Scripts organizados por funcionalidad para el proyecto de Trading Prediction.

## 📁 Estructura

```
scripts/
├── data/                   # Gestión de datos
├── optimization/           # Optimización de hiperparámetros
├── evaluation/             # Evaluación y validación de modelos
├── trading/                # Trading en vivo (IOL API)
└── airflow/                # Validación de DAGs Airflow
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
Optimización con Optuna para E1 Conservative.

**Uso:**
```bash
python scripts/optimization/optimize_e1_hyperparameters.py \
    --ticker AAPL \
    --n-trials 50
```

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

### `e1_continuous_evaluation.py`
Sistema de evaluación continua con MLflow.

**Uso:**
```bash
# Guardar predicción
python scripts/evaluation/e1_continuous_evaluation.py \
    --mode save \
    --ticker AAPL \
    --prediction 0.15

# Evaluar predicciones pendientes
python scripts/evaluation/e1_continuous_evaluation.py \
    --mode evaluate
```

**Ver:** `README_CONTINUOUS_EVALUATION.md`

### `e1_retrospective_validation.py`
Validación out-of-time con modelo GRU (E1 Simple).

**Uso:**
```bash
python scripts/evaluation/e1_retrospective_validation.py \
    --ticker AAPL \
    --train-days-ago 180 \
    --horizon 90
```

**Características:**
- Usa arquitectura GRU real (2 capas: 128, 64)
- Verifica `data/clean/` antes de descargar
- Guarda resultados en MLflow

### `e1_baseline_retrospective_validation.py`
Validación out-of-time con LinearRegression (baseline).

**Uso:**
```bash
python scripts/evaluation/e1_baseline_retrospective_validation.py \
    --ticker AAPL \
    --train-days-ago 180 \
    --horizon 90
```

**Características:**
- Modelo baseline simple (LinearRegression)
- Más rápido que GRU
- Útil para comparación

### `compare_e1_models.py`
Compara modelos E1 entrenados.

**Uso:**
```bash
python scripts/evaluation/compare_e1_models.py \
    --run-ids run1 run2 run3
```

### `validate_e2_optimization.py`
Valida resultados de optimización E2.

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

## 🔗 Referencias

- **Data Cleaning:** `README_DATA_CLEANING.md`
- **Hyperparameter Tuning:** `README_HYPERPARAMETER_TUNING.md`
- **Continuous Evaluation:** `README_CONTINUOUS_EVALUATION.md`
- **IOL Trading:** `README_IOL_FALLBACK.md`
- **Backtesting:** `README_BACKTESTING.md`

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
