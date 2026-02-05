# Cheat Sheet - Optimización de Hiperparámetros E1

## Comandos Más Usados

```bash
# 🚀 INICIO RÁPIDO (10 min)
python scripts/optimize_e1_hyperparameters.py --n_trials 10 --quick

# 📊 OPTIMIZACIÓN COMPLETA (3 horas)
python scripts/optimize_e1_hyperparameters.py --n_trials 50

# 🎯 TICKER ESPECÍFICO (30 min)
python scripts/optimize_e1_hyperparameters.py --ticker AAPL --n_trials 30

# ⏱️ CON LÍMITE DE TIEMPO (2 horas)
python scripts/optimize_e1_hyperparameters.py --n_trials 100 --timeout 7200

# 🔄 CONTINUAR OPTIMIZACIÓN
python scripts/optimize_e1_hyperparameters.py --n_trials 20  # Suma a los existentes

# 🌐 USAR SERVIDOR REMOTO (requiere Docker)
python scripts/optimize_e1_hyperparameters.py --n_trials 50 --mlflow_uri http://localhost:5050

# 🎚️ ACOTAR RANGOS / TICKERS
python scripts/optimize_e1_hyperparameters.py --n_trials 40 --tau_buy_min 0.03 --tau_buy_max 0.07 --dropout_max 0.4 --batch_sizes 32,48,64 --tickers AAPL,MSFT,GOOGL
```

## Ver Resultados

```bash
# Ver mejores parámetros
cat reports/hyperparameter_optimization/best_params_e1.yaml

# Ver todos los trials
cat reports/hyperparameter_optimization/all_trials.csv

# Abrir gráficos
open reports/hyperparameter_optimization/figures/

# Iniciar MLflow UI (local)
mlflow ui --backend-store-uri sqlite:///runs/mlflow_local/mlflow.db --port 5001
# → http://localhost:5001

# MLflow UI (remoto - si usas Docker)
open http://localhost:5050
```

## MLflow CLI

```bash
# Exportar variable de entorno
export MLFLOW_TRACKING_URI=sqlite:///runs/mlflow_local/mlflow.db

# Buscar experimentos
mlflow experiments search

# Buscar runs
mlflow runs search --experiment-name E1_Hyperparameter_Optimization
```

## Análisis en Python

```python
import pandas as pd
import yaml

# Cargar mejores parámetros
with open('reports/hyperparameter_optimization/best_params_e1.yaml') as f:
    best = yaml.safe_load(f)
print(best['best_params'])

# Analizar trials
df = pd.read_csv('reports/hyperparameter_optimization/all_trials.csv')

# Top 10
print(df.nlargest(10, 'value'))

# Correlaciones
params = ['params_tau_buy', 'params_dropout', 'params_learning_rate']
print(df[params + ['value']].corr()['value'])

# Rango de parámetros en top 10%
top = df.nlargest(int(len(df)*0.1), 'value')
for p in params:
    print(f"{p}: {top[p].min():.4f} - {top[p].max():.4f}")
```

## Aplicar Resultados

```bash
# 1. Ver mejores parámetros
cat reports/hyperparameter_optimization/best_params_e1.yaml

# 2. Editar configuración
vim src/config/base.yaml
# Copiar valores de best_params

# 3. Re-entrenar con nuevos parámetros
python src/train_e1_pipeline.py

# 4. Comparar en MLflow
mlflow ui --backend-store-uri sqlite:///runs/mlflow_local/mlflow.db --port 5001
```

## Troubleshooting Rápido

```bash
# Limpiar MLflow local (si hay errores)
rm -rf runs/mlflow_local/*

# Limpiar database Optuna (empezar desde cero)
rm optuna_studies.db

# Ver progreso en tiempo real
# Terminal 1: Ejecutar optimización
python scripts/optimize_e1_hyperparameters.py --n_trials 50

# Terminal 2: Ver en Optuna Dashboard (opcional)
pip install optuna-dashboard
optuna-dashboard sqlite:///optuna_studies.db
# → http://localhost:8080

# Terminal 3: Ver logs
tail -f reports/hyperparameter_optimization/optimization_summary.txt
```

## Rangos de Hiperparámetros

| Parámetro | Min | Max | Step | Escala |
|-----------|-----|-----|------|--------|
| `tau_buy` | 0.02 | 0.10 | 0.01 | Linear |
| `tau_sell` | -0.02 | 0.02 | 0.01 | Linear |
| `gru_units_1` | 32 | 128 | 16 | Linear |
| `gru_units_2` | 16 | 64 | 16 | Linear |
| `dropout` | 0.2 | 0.5 | 0.05 | Linear |
| `learning_rate` | 1e-4 | 1e-2 | - | Log |
| `batch_size` | - | - | - | [32, 64, 128] |

## Métricas Clave

```python
# En MLflow UI, filtrar por:
metrics.sharpe_mean > 1.0                     # Sharpe bueno
metrics.ic_mean > 0.15                        # IC excelente
metrics.ic_mean > 0.10 AND metrics.sharpe_mean > 0.5  # Balance
metrics.tickers_with_trades >= 5              # Suficientes señales
```

## Outputs Esperados

```
reports/hyperparameter_optimization/
├── best_params_e1.yaml          # ← Copiar a src/config/base.yaml
├── all_trials.csv               # ← Analizar en pandas
├── optimization_summary.txt     # ← Leer resumen
└── figures/
    ├── optimization_history.png # ← ¿Converge?
    ├── param_importances.png    # ← ¿Qué parámetros importan?
    └── parallel_coordinate.png  # ← Patrones en top trials
```

## Airflow

- DAG: [dockerfiles/airflow/dags/E1/e1_optuna_tuning.py](dockerfiles/airflow/dags/E1/e1_optuna_tuning.py)
- Configura tickers, rangos y conexión MLflow desde `Params` antes de ejecutar
- Revisa XCom `cli_command` para copiar el comando exacto en terminal

## Tiempos Estimados

| Comando | Trials | Tickers | Tiempo |
|---------|--------|---------|--------|
| `--quick` | 10 | 3 | ~20 min |
| Ticker único | 30 | 1 | ~30 min |
| Estándar | 50 | ~10 | ~3 horas |
| Completo | 100 | ~10 | ~6 horas |

## Valores Típicos (Referencia)

Basado en optimizaciones previas de E1:

```yaml
# Conservador (bajo riesgo, menos trades)
tau_buy: 0.06 - 0.08
tau_sell: -0.01 - 0.01

# Moderado (balance)
tau_buy: 0.04 - 0.06
tau_sell: -0.02 - 0.00

# Agresivo (más trades, mayor riesgo)
tau_buy: 0.02 - 0.04
tau_sell: -0.02 - -0.01

# Arquitectura típica
gru_units: [64-96, 32-48]
dropout: 0.30 - 0.40
learning_rate: 0.0005 - 0.002
batch_size: 64 (más común)
```

## Checklist Pre-Optimización

- [ ] Datos limpios en `data/clean/`
- [ ] Configuración base en `src/config/base.yaml`
- [ ] Suficiente espacio en disco (~1 GB para 50 trials)
- [ ] Tiempo disponible (3+ horas para optimización completa)
- [ ] Ambiente Python activo (`conda activate ia_ceia_18co`)

## Checklist Post-Optimización

- [ ] Revisar gráfico de convergencia (línea roja estabiliza)
- [ ] Verificar que mejores trials tienen valor objetivo > 0.3
- [ ] Verificar que top trials generan trades (tickers_with_trades > 50%)
- [ ] Comparar top 10 trials - parámetros similares = robusto
- [ ] Aplicar mejores parámetros en `base.yaml`
- [ ] Re-entrenar y validar mejora

---

📖 **Documentación completa**: [README_HYPERPARAMETER_TUNING.md](../README_HYPERPARAMETER_TUNING.md)
