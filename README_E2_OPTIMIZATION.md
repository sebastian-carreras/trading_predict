# Optimización de Hiperparámetros - Estrategia E2 (LSTM Moderada)

Este documento describe cómo optimizar los hiperparámetros de la estrategia E2 usando **Optuna** para búsqueda automática y **MLflow** para tracking de experimentos.

## 📋 Hiperparámetros Optimizados

### Trading Thresholds
- **tau_buy**: Umbral de señal de compra (0.015 - 0.05)
- **tau_sell**: Umbral de señal de venta (-0.01 - 0.01)

### Arquitectura LSTM
- **lstm_units_1**: Neuronas en primera capa LSTM (64 - 256)
- **lstm_units_2**: Neuronas en segunda capa LSTM (32 - 128)
- **dropout**: Tasa de dropout para regularización (0.1 - 0.4)

### Entrenamiento
- **learning_rate**: Tasa de aprendizaje (5e-5 - 5e-3, log scale)
- **batch_size**: Tamaño de batch (32, 64, 128)

### Filtros de Entrada
- **rsi14_min**: RSI mínimo para entrar (25 - 40)
- **rsi14_max**: RSI máximo para entrar (60 - 75)

### Walk-Forward Validation (Nuevo)
- **n_folds**: Número de ventanas temporales de validación (3 - 7)
  - Más folds = mayor robustez temporal pero más costo computacional
  - Recomendado: 5 para balance óptimo
- **internal_val_fraction**: Fracción de validación dentro de cada fold (0.10 - 0.25)
  - Determina el split train/val interno en cada ventana
  - Recomendado: 0.15 (15% validación, 85% entrenamiento)

## 🚀 Uso Básico

### 1. Optimización Rápida (Prueba)

```bash
# 10 trials, 3 tickers aleatorios (~10-15 minutos)
python scripts/optimize_e2_hyperparameters.py --n_trials 10 --quick
```

### 2. Optimización Completa

```bash
# 50 trials, todos los tickers E2 (~2-3 horas)
python scripts/optimize_e2_hyperparameters.py --n_trials 50
```

### 3. Optimizar Un Solo Ticker

```bash
# Útil para entender comportamiento de ticker específico
python scripts/optimize_e2_hyperparameters.py --ticker NVDA --n_trials 30
```

### 4. Optimización Por Ticker (Recomendado)

```bash
# Genera YAML con mejores params para cada ticker
python scripts/optimize_e2_hyperparameters.py --per_ticker --n_trials 30
```

Esto genera `reports/hyperparameter_optimization/e2_tuned_params_by_ticker.yaml`:

```yaml
NVDA:
  tau_buy: 0.025
  tau_sell: 0.005
  lstm_units_1: 128
  lstm_units_2: 64
  dropout: 0.2
  learning_rate: 0.0005
  batch_size: 64
  rsi14_min: 30
  rsi14_max: 70

GOOGL:
  tau_buy: 0.030
  tau_sell: 0.000
  # ... etc
```

## 📊 Outputs

### 1. Mejores Parámetros (YAML)

**Global**: `reports/hyperparameter_optimization/best_params_e2.yaml`
```yaml
optimization:
  study_name: e2_hyperparameter_optimization
  n_trials: 50
  best_value: 1.2345
  best_trial: 23

best_params:
  tau_buy: 0.025
  tau_sell: 0.005
  lstm_units_1: 128
  lstm_units_2: 64
  dropout: 0.25
  learning_rate: 0.0008
  batch_size: 64
  rsi14_min: 30
  rsi14_max: 70
```

**Por ticker**: `reports/hyperparameter_optimization/e2_tuned_params_by_ticker.yaml`
- Archivos individuales: `reports/hyperparameter_optimization/by_ticker/<TICKER>/best_params_e2.yaml`

### 2. Trials Completos (CSV)

`reports/hyperparameter_optimization/e2_all_trials.csv`
- Todos los trials con parámetros y métricas
- Útil para análisis posterior en Excel/Pandas

### 3. Visualizaciones

Carpeta: `reports/hyperparameter_optimization/figures/`

- **e2_optimization_history.png**: Evolución del valor objetivo por trial
- **e2_param_importances.png**: Importancia relativa de cada parámetro
- **e2_parallel_coordinate.png**: Visualización multi-dimensional de parámetros

### 4. MLflow Tracking

**Local**: `runs/mlflow_local/mlflow.db`
```bash
# Ver experimentos
mlflow ui --backend-store-uri sqlite:///runs/mlflow_local/mlflow.db
```

**Docker** (si usas `--mlflow_uri http://localhost:5050`):
```bash
# MLflow ya disponible en http://localhost:5050
```

Experimento: `E2_Hyperparameter_Optimization`

Métricas trackeadas:
- `ic_mean`, `ic_median`, `ic_std`
- `sharpe_mean`, `sharpe_median`, `sharpe_std`
- `profit_factor_mean`, `cagr_mean`
- `ic_positive_pct`, `sharpe_positive_pct`
- `tickers_with_trades`, `avg_num_trades`
- `objective_value` (métrica a maximizar)

## 🎯 Función Objetivo

La métrica objetivo es una **combinación ponderada**:

```python
objective = (
    0.4 * sharpe_mean +           # 40% Sharpe ratio
    0.3 * min(pf_mean, 3.0) +     # 30% Profit Factor (capped en 3)
    0.3 * cagr_mean * 10 +        # 30% CAGR (escalado)
    trade_penalty                  # -5 si <50% tickers tienen trades
)
```

**Justificación**:
- **Sharpe (40%)**: Balance riesgo/retorno (métrica principal)
- **Profit Factor (30%)**: Eficiencia de trades (ganancia/pérdida)
- **CAGR (30%)**: Retorno anualizado
- **Trade Penalty**: Evita thresholds muy altos que no generan señales

### Por Qué Optimizar Walk-Forward Parameters

**`n_folds` (Número de Ventanas):**
- **Menos folds (3)**: Más datos por ventana → métricas más estables, pero menor robustez temporal
- **Más folds (7)**: Mayor robustez temporal → detecta mejor concept drift, pero más varianza por ventana
- **Impacto**: Un modelo puede funcionar bien con 5 folds pero mal con 3 (overfitting a períodos largos)

**`internal_val_fraction` (Validación Interna):**
- **Menor (0.10)**: Más datos para entrenamiento → menor underfitting
- **Mayor (0.25)**: Mejor early stopping → menor overfitting
- **Impacto**: Afecta cuándo se detiene el entrenamiento en cada fold

**Ejemplo de Impacto:**
```
Trial A: n_folds=5, val_frac=0.15 → Sharpe=1.2 (balance óptimo)
Trial B: n_folds=3, val_frac=0.25 → Sharpe=0.8 (pocas ventanas, overfitting a val)
Trial C: n_folds=7, val_frac=0.10 → Sharpe=1.5 (robusto temporalmente)
```

Optuna encontrará la combinación óptima para tus datos específicos.

## 🔄 Aplicar Parámetros Optimizados

### Opción 1: Actualizar base.yaml Manualmente

Copia los mejores valores de `best_params_e2.yaml` a `src/config/base.yaml`:

```yaml
strategies:
  e2_moderate:
    thresholds:
      tau_buy: 0.025     # ← Desde best_params
      tau_sell: 0.005
    
    filters:
      rsi14_min: 30      # ← Desde best_params
      rsi14_max: 70
    
    model:
      lstm_units: [128, 64]  # ← Desde best_params
      dropout: 0.25
      learning_rate: 0.0008
      batch_size: 64
```

### Opción 2: Usar Parámetros Por Ticker en Airflow

Edita el DAG `e2_moderate_pipeline.py` para usar el YAML optimizado:

```python
# En la UI de Airflow, al ejecutar el DAG:
use_tuned_params = True
tuned_params_path = "reports/hyperparameter_optimization/e2_tuned_params_by_ticker.yaml"
```

Esto aplica automáticamente los mejores parámetros para cada ticker.

## 📈 Análisis de Resultados

### Ver en MLflow UI

```bash
# Local
mlflow ui --backend-store-uri sqlite:///runs/mlflow_local/mlflow.db

# Navegar a http://localhost:5000
# Experimento: "E2_Hyperparameter_Optimization"
```

Comparar trials:
1. Seleccionar múltiples runs
2. Click "Compare"
3. Ver gráficos de métricas vs parámetros

### Analizar CSV en Python

```python
import pandas as pd

# Cargar trials
df = pd.read_csv("reports/hyperparameter_optimization/e2_all_trials.csv")

# Top 10 trials
top10 = df.nlargest(10, "value")

# Correlación parámetros vs métrica objetivo
correlations = df.corr()["value"].sort_values(ascending=False)
print(correlations)

# Distribución de mejores valores
import matplotlib.pyplot as plt
df.nlargest(20, "value")["params_tau_buy"].hist(bins=10)
plt.xlabel("tau_buy")
plt.show()
```

## ⚙️ Opciones Avanzadas

### Continuar Optimización Existente

```bash
# Agrega 20 trials más al estudio existente
python scripts/optimize_e2_hyperparameters.py \
    --study_name e2_hyperparameter_optimization \
    --n_trials 20
```

### Usar MLflow Remoto (Docker)

```bash
python scripts/optimize_e2_hyperparameters.py \
    --n_trials 50 \
    --mlflow_uri http://localhost:5050
```

### Optimizar Subset de Tickers

```bash
# Tech stocks
python scripts/optimize_e2_hyperparameters.py \
    --tickers "NVDA,GOOGL,META,AMZN" \
    --n_trials 40
```

### Ajustar Rangos de Búsqueda

Edita el script `optimize_e2_hyperparameters.py` para cambiar rangos:

```python
# Líneas 142-149
self.tau_buy_bounds = {"min": 0.01, "max": 0.08, "step": 0.005}  # Ampliar rango
self.lstm_units_1_bounds = {"min": 96, "max": 192, "step": 32}   # Reducir rango
```

### Timeout (Límite de Tiempo)

```bash
# Máximo 2 horas (7200 segundos)
python scripts/optimize_e2_hyperparameters.py \
    --n_trials 100 \
    --timeout 7200
```

## 🐛 Troubleshooting

### Error: "No module named 'optuna'"

```bash
pip install optuna kaleido plotly
```

### Error: "Permission denied" en optuna_studies.db

```bash
# Eliminar DB corrupta
rm optuna_studies.db

# Re-ejecutar optimización
python scripts/optimize_e2_hyperparameters.py --n_trials 10
```

### Trials muy lentos

**Causas**:
- Muchos tickers (10 tickers × 50 trials = 500 entrenamientos)
- Epochs altos en base.yaml

**Soluciones**:
```bash
# Modo rápido (3 tickers)
python scripts/optimize_e2_hyperparameters.py --quick --n_trials 20

# Reducir max_epochs temporalmente en base.yaml
# strategies.e2_moderate.model.max_epochs: 150 → 50
```

### Visualizaciones no generadas

Requiere `kaleido` para exportar a PNG:

```bash
pip install kaleido

# Si falla, usar formato HTML interactivo
# Editar script y cambiar:
# fig.write_image(...) → fig.write_html(...)
```

## 📊 Comparación de Parámetros Optimizables

| Parámetro | Tipo | Rango | Impacto | Prioridad |
|-----------|------|-------|---------|-----------|
| **tau_buy** | Trading | 0.015-0.05 | ⭐⭐⭐⭐⭐ Crítico para señales | Alta |
| **tau_sell** | Trading | -0.01-0.01 | ⭐⭐⭐⭐ Gestión de salidas | Alta |
| **lstm_units_1** | Arquitectura | 64-256 | ⭐⭐⭐⭐⭐ Capacidad del modelo | Alta |
| **lstm_units_2** | Arquitectura | 32-128 | ⭐⭐⭐⭐ Refinamiento | Alta |
| **dropout** | Regularización | 0.1-0.4 | ⭐⭐⭐⭐ Previene overfitting | Alta |
| **learning_rate** | Entrenamiento | 5e-5 - 5e-3 | ⭐⭐⭐⭐⭐ Convergencia | Alta |
| **batch_size** | Entrenamiento | 32,64,128 | ⭐⭐ Estabilidad gradientes | Media |
| **rsi14_min** | Filtros | 25-40 | ⭐⭐⭐ Filtrado entrada | Media |
| **rsi14_max** | Filtros | 60-75 | ⭐⭐⭐ Filtrado entrada | Media |
| **n_folds** | Walk-Forward | 3-7 | ⭐⭐⭐⭐ Robustez temporal | Alta |
| **internal_val_fraction** | Walk-Forward | 0.10-0.25 | ⭐⭐⭐ Early stopping | Media |

**Total parámetros optimizables:** 11 (vs 9 en E1)

**Espacio de búsqueda:** ~10^11 combinaciones → Optuna esencial

**Diferencia clave vs E1:**
- E2 optimiza filtros RSI (2 params adicionales)
- E2 usa LSTM (más complejo que GRU de E1)
- E2 prefiere menos folds (3-5) por horizonte corto (20 días)

## � Ejecución desde Airflow

El DAG `e2_optuna_hyperparameter_tuning` permite ejecutar la optimización desde la UI de Airflow con parámetros configurables.

### 1. Iniciar Airflow

```bash
# Desde docker-compose
docker-compose up -d

# Acceder a la UI
# http://localhost:8080
# Usuario: airflow
# Password: airflow
```

### 2. Localizar el DAG

En la UI de Airflow:
- **DAG ID**: `e2_optuna_hyperparameter_tuning`
- **Tags**: `optuna`, `mlflow`, `e2`, `lstm`, `hyperparameter_tuning`
- **Descripción**: Ejecuta Optuna para E2 (LSTM) con parámetros configurables

### 3. Trigger con Configuración Personalizada

Click en "Trigger DAG w/ config" y personalizar parámetros en JSON:

#### Ejemplo 1: Optimización Rápida

```json
{
  "n_trials": "10",
  "quick_mode": "True",
  "mlflow_uri": "local"
}
```

#### Ejemplo 2: Optimización de Tickers Específicos

```json
{
  "n_trials": "30",
  "tickers": "NVDA, AMD, TSLA, GOOGL, META",
  "tau_buy_min": "0.02",
  "tau_buy_max": "0.06",
  "lstm_units_1_min": "128",
  "lstm_units_1_max": "384",
  "dropout_min": "0.15",
  "dropout_max": "0.35",
  "batch_sizes": "64,128"
}
```

#### Ejemplo 3: Optimización por Ticker

```json
{
  "per_ticker": "True",
  "n_trials": "30",
  "tickers": "NVDA, AMD, MSFT",
  "mlflow_uri": "local"
}
```

#### Ejemplo 4: Un Solo Ticker (Prueba)

```json
{
  "single_ticker": "NVDA",
  "n_trials": "20",
  "lstm_units_1_min": "128",
  "lstm_units_1_max": "256"
}
```

### 4. Parámetros Configurables del DAG

#### Parámetros Generales

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `config_path` | str | `src/config/base.yaml` | Path al archivo de configuración |
| `n_trials` | int | `20` | Número de trials de Optuna |
| `study_name` | str | `e2_hyperparameter_optimization` | Nombre del estudio |
| `mlflow_uri` | str | `local` | URI de MLflow (local o http://...) |
| `output_dir` | str | `reports/hyperparameter_optimization` | Directorio de outputs |

#### Selección de Tickers

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `tickers` | str | `NVDA, AMD, MSFT, ...` | Lista separada por comas |
| `single_ticker` | str | `""` | Un único ticker (para pruebas) |
| `quick_mode` | bool | `False` | Modo rápido (3 tickers aleatorios) |
| `per_ticker` | bool | `False` | Optimizar cada ticker independientemente |

#### Overrides del Espacio de Búsqueda

Todos los parámetros del espacio de búsqueda son configurables:

**Thresholds de Trading:**
- `tau_buy_min`, `tau_buy_max`
- `tau_sell_min`, `tau_sell_max`

**Arquitectura LSTM:**
- `lstm_units_1_min`, `lstm_units_1_max`
- `lstm_units_2_min`, `lstm_units_2_max`

**Regularización:**
- `dropout_min`, `dropout_max`

**Entrenamiento:**
- `learning_rate_min`, `learning_rate_max`
- `batch_sizes` (string: "32,64,128")

**Filtros RSI:**
- `rsi14_min_min`, `rsi14_min_max`
- `rsi14_max_min`, `rsi14_max_max`

**Walk-forward:**
- `n_folds_min`, `n_folds_max`
- `internal_val_fraction_min`, `internal_val_fraction_max`

### 5. Monitorear Progreso

- **Logs**: Click en task `run_optuna_tuning` → Logs
- **MLflow**: `http://localhost:5000` (si usas mlflow ui local)
- **Outputs**: `reports/hyperparameter_optimization/`

### 6. Acceder a Resultados

Los resultados se guardan automáticamente en:

```
reports/hyperparameter_optimization/
├── best_params_e2.yaml                    # Mejores parámetros
├── e2_tuned_params_by_ticker.yaml         # Params por ticker
├── e2_all_trials.csv                      # Todos los trials
└── figures/
    ├── e2_optimization_history.png
    ├── e2_param_importances.png
    └── e2_parallel_coordinate.png
```

### 7. Recuperar Comando CLI Ejecutado

El DAG guarda el comando CLI exacto en XCom:

```python
# En Airflow UI → XCom
{
  "key": "cli_command",
  "value": "python scripts/optimize_e2_hyperparameters.py --config ..."
}
```

Útil para replicar la ejecución desde terminal.

### Ventajas del DAG vs CLI

✅ **Interfaz gráfica**: No requiere terminal  
✅ **Configuración JSON**: Parámetros legibles y versionables  
✅ **Historial**: Todas las ejecuciones quedan registradas  
✅ **Logs centralizados**: Fácil debugging  
✅ **Scheduling**: Posibilidad de automatizar (ej: mensual)  
✅ **Notificaciones**: Integración con Slack/email en failures  

## �📊 Benchmarks de Rendimiento

**Hardware**: MacBook Pro M1, 16GB RAM

| Configuración | Tiempo Estimado |
|--------------|-----------------|
| `--quick --n_trials 10` | 10-15 min |
| `--ticker NVDA --n_trials 30` | 20-30 min |
| `--n_trials 50` (10 tickers) | 2-3 horas |
| `--per_ticker --n_trials 30` (10 tickers) | 3-4 horas |

**Paralelización**: Optuna usa TPESampler (secuencial). Para paralelizar, ejecutar múltiples estudios independientes por ticker (`--per_ticker`).

## 🎓 Mejores Prácticas

1. **Empezar con `--quick`**: Validar que el script funciona (10 min)
2. **Optimizar por ticker**: Mejores resultados que optimización global
3. **Monitorear MLflow**: Revisar métricas en tiempo real
4. **Guardar estudios**: Usar `--study_name` único para cada experimento
5. **Iterar rangos**: Si todos los trials convergen en un límite, expandir rango
6. **Validar OOS**: Después de optimización, validar en datos no vistos (walk-forward)

## 📚 Referencias

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [Hyperparameter Tuning Best Practices](https://www.youtube.com/watch?v=ttE0F7fghfk)

## 🔗 Archivos Relacionados

- Script: [scripts/optimize_e2_hyperparameters.py](scripts/optimize_e2_hyperparameters.py)
- Pipeline: [src/train_e2_pipeline.py](src/train_e2_pipeline.py)
- Config: [src/config/base.yaml](src/config/base.yaml)
- DAG Training: [dockerfiles/airflow/dags/e2_moderate_pipeline.py](dockerfiles/airflow/dags/e2_moderate_pipeline.py)
- **DAG Optuna**: [dockerfiles/airflow/dags/e2_optuna_tuning.py](dockerfiles/airflow/dags/e2_optuna_tuning.py) ← **Nuevo**
