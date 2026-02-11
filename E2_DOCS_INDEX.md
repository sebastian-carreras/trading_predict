# Índice de Documentación E2

## 📖 Documentación Principal

### E2 Moderate (Producción)
- **[README_E2.md](README_E2.md)** - Documentación completa de E2 Moderate
  - Arquitectura LSTM 2 capas, walk-forward validation
  - Features momentum-centric (RSI, MACD, Stochastic, OBV)
  - Filtros complejos y decision score perfil "moderate"
  - Uso CLI y Airflow

### E2 Simple (Desarrollo)
- **[README_E2_SIMPLE.md](README_E2_SIMPLE.md)** - Documentación completa de E2 Simple
  - Pipeline simplificado, ~40% más rápido que E2 Moderate
  - LSTM 2 capas, decision score 5 métricas, time split simple
  - Quick start, uso CLI y Airflow
  - Ideal para prototipado rápido

---

## 📊 Optimización

### Hyperparameter Tuning
- **[README_E2_OPTIMIZATION.md](README_E2_OPTIMIZATION.md)** - Optimización de hiperparámetros con Optuna
  - Búsqueda de tau_buy, tau_sell, lstm_units, dropout
  - Filtros RSI, learning_rate, batch_size
  - Walk-forward params (n_folds, val_fraction)

---

## 📈 MLflow & Tracking

### Target Metrics Enhancement
- **[docs/E2_TARGET_METRICS_ENHANCEMENT.md](docs/E2_TARGET_METRICS_ENHANCEMENT.md)** - Target metrics en MLflow Summary
  - Logging de valores objetivo (ic_min, sharpe_min)
  - Métricas agregadas de Sharpe
  - Comparación rápida: métricas reales vs targets
  - Trazabilidad completa de objetivos en MLflow

---

## 🚀 Quick Start por Perfil

**Nuevo usuario → desarrollo rápido:**
1. [README_E2_SIMPLE.md](README_E2_SIMPLE.md) - Quick start con E2 Simple
   - Entrenar en minutos
   - Experimentar con hyperparams
   - Validar ideas rápidamente

**Investigador → evaluación rigurosa:**
1. [README_E2_SIMPLE.md](README_E2_SIMPLE.md) - Prototipar rápido
2. [README_E2.md](README_E2.md) - Validar con walk-forward
3. [README_E2_OPTIMIZATION.md](README_E2_OPTIMIZATION.md) - Tunear parámetros

**DevOps → deploy producción:**
1. [README_E2.md](README_E2.md) - E2 Moderate completo
2. [README_AIRFLOW_USAGE.md](README_AIRFLOW_USAGE.md) - DAGs Airflow
3. [README_E2_OPTIMIZATION.md](README_E2_OPTIMIZATION.md) - Tuning Optuna

---

## 🔗 Comparación con E1

- **E2 vs E1 (Diferencias clave)**:
  - **Horizonte**: E2 = 20 días, E1 = 90 días
  - **Features**: E2 = momentum (RSI, MACD), E1 = tendencia (SMA, ADX)
  - **Modelo**: E2 = LSTM, E1 = GRU
  - **Rebalanceo**: E2 = semanal, E1 = mensual
  - **Lookback**: E2 = 60d, E1 = 360d

- **Ver también**:
  - [README_E1.md](README_E1.md) - E1 Conservative
  - [README_E1_SIMPLE.md](README_E1_SIMPLE.md) - E1 Simple
  - [E1_DOCS_INDEX.md](E1_DOCS_INDEX.md) - Índice E1

---

## 📁 Archivos Implementados

### Core Pipeline
- `src/train_e2_pipeline.py` - Pipeline E2 Moderate (walk-forward)
- `src/train_e2_simple_pipeline.py` - Pipeline E2 Simple (time split)

### Airflow DAGs
- `dockerfiles/airflow/dags/E2/e2_moderate_pipeline.py` - DAG E2 Moderate
- `dockerfiles/airflow/dags/E2/e2_simple_pipeline.py` - DAG E2 Simple

### Features & Model
- `src/features/build_features_e2.py` - Features E2 (compartidas por Moderate & Simple)
- `src/models/e2_lstm.py` - Modelo LSTM PyTorch

### Config
- `src/config/base.yaml` - Configuración centralizada
  - `strategies.e2_moderate` - Parámetros E2 Moderate
  - `strategies.e2_simple` - Parámetros E2 Simple
  - `universe.tickers_by_strategy.e2_moderate` - Tickers E2 Moderate
  - `universe.tickers_by_strategy.e2_simple` - Tickers E2 Simple
  - `splits.strategy_profile.e2_moderate: moderate`
  - `splits.strategy_profile.e2_simple: moderate`

---

**Última actualización**: 2026-01-xx
