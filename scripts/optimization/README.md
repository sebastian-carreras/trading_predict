# Optimización de Hiperparámetros con Optuna

Scripts de búsqueda automática de hiperparámetros para las estrategias E1, E2 y E3 usando Optuna como framework de optimización bayesiana y MLflow para tracking de experimentos.

---

## Visión general

Cada script sigue el mismo flujo:

```
Optuna propone hiperparámetros (TPE sampler)
    → Pipeline de entrenamiento corre sobre todos los tickers
    → Se calculan métricas agregadas (Sharpe, IC, Calmar, DirAcc)
    → Se devuelve una métrica objetivo compuesta
    → Optuna aprende del resultado y propone el siguiente trial
```

Los estudios son persistentes en SQLite (`runs/optuna_trials/optuna_studies.db`), lo que permite interrumpir y retomar corridas sin perder progreso.

---

## Comparativa de estrategias

| Aspecto | E1 Conservative | E2 Moderate | E3 Intraday |
|---|---|---|---|
| Modelo | GRU 2 capas | LSTM 2 capas | LSTM Ensemble |
| Horizonte | 90 días | 20 días | 30 min (5-min bars) |
| Script | `optimize_e1_hyperparameters.py` | `optimize_e2_hyperparameters.py` | `optimize_e3_hyperparameters.py` |
| Study name default | `e1_hyperparameter_optimization` | `e2_hyperparameter_optimization_timesplit` | `e3_hyperparameter_optimization` |

---

## Espacio de búsqueda

### E1 — GRU Conservative

| Parámetro | Rango | Tipo |
|---|---|---|
| `tau_buy` | [0.02, 0.10] step=0.01 | Umbral señal de compra |
| `tau_sell` | [-0.02, 0.02] step=0.01 | Umbral señal de venta |
| `gru_units_1` | [32, 128] step=16 | Unidades capa GRU 1 |
| `gru_units_2` | [16, 64] step=16 | Unidades capa GRU 2 |
| `dropout` | [0.20, 0.50] step=0.05 | Regularización |
| `learning_rate` | [1e-4, 1e-2] | Log-uniforme |
| `weight_decay` | [1e-6, 1e-2] | Log-uniforme |
| `batch_size` | {32, 64, 128} | Categórico |

**Función objetivo:** `0.35×Sharpe + 0.25×IC + 0.20×DirAcc + 0.20×Calmar + trade_penalty`

### E2 — LSTM Moderate

| Parámetro | Rango | Tipo |
|---|---|---|
| `tau_buy` | [0.015, 0.05] step=0.005 | Umbral señal de compra |
| `tau_sell` | [-0.01, 0.01] step=0.005 | Umbral señal de venta |
| `lstm_units_1` | [64, 256] step=32 | Unidades capa LSTM 1 |
| `lstm_units_2` | [32, 128] step=16 | Unidades capa LSTM 2 |
| `dropout` | [0.10, 0.40] step=0.05 | Regularización |
| `learning_rate` | [5e-5, 5e-3] | Log-uniforme |
| `batch_size` | {32, 64, 128} | Categórico |

**Función objetivo:** `0.30×Sharpe + 0.20×IC + 0.30×DirAcc + 0.20×Calmar + trade_penalty`

### E3 — LSTM Ensemble Intraday

| Parámetro | Rango | Tipo |
|---|---|---|
| `tau_buy` | [0.001, 0.005] step=0.0005 | Umbral señal de compra (en bps) |
| `tau_sell` | [0.001, 0.005] step=0.0005 | Umbral señal de venta |
| `lstm_hidden_size` | [64, 256] step=32 | Tamaño capa LSTM oculta |
| `dense_units` | [16, 64] step=16 | Unidades capa densa |
| `dropout` | [0.10, 0.40] step=0.05 | Regularización |
| `learning_rate` | [1e-4, 5e-3] | Log-uniforme |
| `weight_decay` | [1e-6, 1e-2] | Log-uniforme |
| `batch_size` | {64, 128, 256, 512} | Categórico |
| `ensemble_members` | [2, 5] step=1 | Cantidad de miembros del ensemble |
| `consensus_tol` | [0.0001, 0.002] step=0.0001 | Filtro de consenso entre miembros |

**Función objetivo:** `0.30×Sharpe + 0.20×IC + 0.25×DirAcc + 0.25×Calmar + trade_penalty`

> `trade_penalty = -5` si menos del 50% de tickers generan trades (penaliza thresholds demasiado altos que no emiten señales).

---

## Uso

### Comandos básicos

```bash
# E1 — optimización sobre universo completo (50 trials por defecto)
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 50

# E2
python scripts/optimization/optimize_e2_hyperparameters.py --n_trials 50

# E3
python -m scripts.optimization.optimize_e3_hyperparameters --n_trials 50
```

### Opciones comunes (disponibles en los tres scripts)

```bash
# Ticker único (más rápido, útil para pruebas)
--ticker AAPL

# Lista explícita de tickers
--tickers AAPL,MSFT,GOOG

# Modo rápido: 3 tickers aleatorios del universo (2 para E3)
--quick

# Optimización independiente por ticker (genera YAML por ticker)
--per_ticker

# Continuar un estudio existente
--study_name nombre_del_estudio

# Timeout total en segundos
--timeout 3600

# MLflow remoto (Docker)
--mlflow_uri http://localhost:5050

# Usar datos hasta la fecha actual (por defecto usa ventana fija del config)
--use-latest-data
```

### Ajustar rangos de búsqueda en tiempo de ejecución

```bash
# Ejemplo E1: acotar rango de tau_buy y learning_rate
python scripts/optimization/optimize_e1_hyperparameters.py \
  --tau_buy_min 0.03 --tau_buy_max 0.07 \
  --learning_rate_min 1e-4 --learning_rate_max 1e-3 \
  --n_trials 30

# Ejemplo E3: limitar ensemble_members y ajustar batch sizes
python -m scripts.optimization.optimize_e3_hyperparameters \
  --ensemble_members_min 2 --ensemble_members_max 3 \
  --batch_sizes 64,128 \
  --n_trials 20
```

### Método de validación

Por defecto se usa `walk_forward` con 3 folds (configurable en `src/config/base.yaml` bajo `optuna.<estrategia>.validation`). Se puede cambiar por CLI:

```bash
--validation_method time_split   # split único más rápido
--validation_method walk_forward --optuna_folds 5
```

---

## Outputs

Todos los resultados se guardan en `reports/hyperparameter_optimization/`:

| Archivo | Contenido |
|---|---|
| `best_params_e{1,2,3}.yaml` | Mejores hiperparámetros del estudio |
| `e{1,2,3}_all_trials.csv` | Todos los trials del estudio |
| `e{1,2,3}_run_trials.csv` | Solo los trials de la corrida actual |
| `e{1,2,3}_tuned_params_by_ticker.yaml` | Overrides por ticker (formato consumible por training) |
| `e{1,2,3}_tuned_params_by_ticker.meta.yaml` | Idem con metadata del estudio |
| `e{1,2,3}_optimization_summary.txt` | Resumen textual con top 10 trials |
| `figures/` | Gráficos: historial, importancias, coordenadas paralelas |

Con `--per_ticker`, los resultados de cada ticker se guardan además en `by_ticker/<TICKER>/`.

---

## Persistencia y continuación de estudios

Los trials se acumulan en `runs/optuna_trials/optuna_studies.db`. Si se interrumpe una corrida, se puede retomar con el mismo `--study_name` y los trials anteriores se preservan:

```bash
# Retomar un estudio interrumpido
python scripts/optimization/optimize_e1_hyperparameters.py \
  --study_name e1_hyperparameter_optimization \
  --n_trials 20
```

---

## Tracking con MLflow

El tracking usa cascade fallback automático:

1. Servidor remoto (`MLFLOW_TRACKING_URI` en `.env` o `--mlflow_uri`)
2. SQLite local (`runs/mlflow_local/mlflow.db`)
3. File store (`mlruns/`)

Para visualizar localmente sin Docker:

```bash
mlflow ui --backend-store-uri runs/mlflow_local/mlflow.db
# UI disponible en http://localhost:5000
```

Para visualizar los estudios de Optuna:

```bash
optuna-dashboard sqlite:///runs/optuna_trials/optuna_studies.db
```

---

## Integración con el pipeline de entrenamiento

Los parámetros optimizados se inyectan en el pipeline de entrenamiento a través del YAML generado. El path está registrado en `src/config/base.yaml` bajo `optuna.<estrategia>.tuned_params_path`. El pipeline lo lee automáticamente si el archivo existe, sobreescribiendo los valores del config base.
