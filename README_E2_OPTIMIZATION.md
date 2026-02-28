# Optimizacion de Hiperparametros - Estrategia E2 (LSTM Moderada)

Este documento describe como optimizar los hiperparametros de la estrategia E2 usando **Optuna** para busqueda automatica y **MLflow** para tracking de experimentos.

## Hiperparametros Optimizados

### Trading Thresholds
- **tau_buy**: Umbral de senal de compra (0.015 - 0.05, step=0.005)
- **tau_sell**: Umbral de senal de venta (-0.01 - 0.01, step=0.005)

### Arquitectura LSTM
- **lstm_units_1**: Neuronas en primera capa LSTM (64 - 256, step=32)
- **lstm_units_2**: Neuronas en segunda capa LSTM (32 - 128, step=16)
- **dropout**: Tasa de dropout para regularizacion (0.1 - 0.4, step=0.05)

### Entrenamiento
- **learning_rate**: Tasa de aprendizaje (5e-5 - 5e-3, log scale)
- **batch_size**: Tamano de batch (32, 64, 128)

**Nota sobre weight_decay:** El modelo LSTM usa AdamW que soporta weight_decay, pero actualmente no se expone como parametro optimizable en `lstm.py`. Se usa el default de PyTorch (0.01).

**Total parametros optimizables:** 7

## Uso Basico

### 1. Optimizacion Rapida (Prueba)

```bash
# 10 trials, 3 tickers aleatorios (~10-15 minutos)
python scripts/optimization/optimize_e2_hyperparameters.py --n_trials 10 --quick
```

### 2. Optimizacion Completa

```bash
# 50 trials, todos los tickers E2 (~2-3 horas)
python scripts/optimization/optimize_e2_hyperparameters.py --n_trials 50
```

### 3. Optimizar Un Solo Ticker

```bash
# Util para entender comportamiento de ticker especifico
# El study se aisla automaticamente con sufijo __TICKER
python scripts/optimization/optimize_e2_hyperparameters.py --ticker NVDA --n_trials 30
```

### 4. Optimizacion Por Ticker (Recomendado)

```bash
# Genera YAML con mejores params para cada ticker + meta YAML con metadata
python scripts/optimization/optimize_e2_hyperparameters.py --per_ticker --n_trials 30
```

Esto genera:
- `reports/hyperparameter_optimization/e2_tuned_params_by_ticker.yaml` (consumible por training)
- `reports/hyperparameter_optimization/e2_tuned_params_by_ticker.meta.yaml` (con metadata de estudios)

Ejemplo del YAML compacto:

```yaml
NVDA:
  thresholds:
    tau_buy: 0.025
    tau_sell: 0.005
  model:
    lstm_units: [128, 64]
    dropout: 0.2
    learning_rate: 0.0005
    batch_size: 64

GOOGL:
  thresholds:
    tau_buy: 0.030
    tau_sell: 0.000
  # ... etc
```

### 5. Overrides de Rangos via CLI

```bash
# Ajustar rangos de busqueda sin editar el script
python scripts/optimization/optimize_e2_hyperparameters.py \
    --n_trials 30 \
    --tau_buy_min 0.02 --tau_buy_max 0.06 \
    --lstm_units_1_min 128 --lstm_units_1_max 384 \
    --dropout_min 0.15 --dropout_max 0.35 \
    --batch_sizes "64,128"
```

## Outputs

### 1. Mejores Parametros (YAML)

**Global**: `reports/hyperparameter_optimization/best_params_e2.yaml`
```yaml
optimization:
  study_name: e2_hyperparameter_optimization_timesplit
  n_trials: 50
  n_completed_trials: 48
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
```

**Por ticker**:
- Compacto: `reports/hyperparameter_optimization/e2_tuned_params_by_ticker.yaml`
- Con metadata: `reports/hyperparameter_optimization/e2_tuned_params_by_ticker.meta.yaml`
- Individuales: `reports/hyperparameter_optimization/by_ticker/<TICKER>/best_params_e2.yaml`

### 2. Trials (CSV)

- `reports/hyperparameter_optimization/e2_all_trials.csv` - Todos los trials (historial completo)
- `reports/hyperparameter_optimization/e2_run_trials.csv` - Solo trials de la corrida actual

### 3. Visualizaciones

Carpeta: `reports/hyperparameter_optimization/figures/`

- **e2_optimization_history.png**: Evolucion del objetivo penalizado (historial completo)
- **e2_optimization_history_current_run.png**: Idem, solo corrida actual
- **e2_optimization_history_raw.png**: Objetivo sin penalizacion (historial completo)
- **e2_optimization_history_raw_current_run.png**: Idem, solo corrida actual
- **e2_param_importances.png**: Importancia relativa de cada parametro (con fallback de correlacion)
- **e2_parallel_coordinate.png**: Visualizacion multi-dimensional de parametros

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

Metricas trackeadas:
- `ic_mean`, `ic_median`, `ic_std`
- `sharpe_mean`, `sharpe_median`, `sharpe_std`
- `profit_factor_mean`, `cagr_mean`
- `ic_positive_pct`, `sharpe_positive_pct`, `profit_factor_above_1_pct`
- `tickers_with_trades`, `avg_num_trades`
- `objective_value` (metrica a maximizar, con penalizacion)
- `objective_raw` (sin penalizacion)
- `trade_penalty`
- `trial_duration_seconds`

**Nota**: Si MLflow no esta disponible (ej: servidor caido), el script continua sin tracking gracias al fallback automatico (`self.mlflow_enabled`).

## Funcion Objetivo

La metrica objetivo es una **combinacion ponderada**:

```python
objective_raw = (
    0.4 * sharpe_mean +           # 40% Sharpe ratio
    0.3 * min(pf_mean, 3.0) +     # 30% Profit Factor (capped en 3)
    0.3 * cagr_mean * 10           # 30% CAGR (escalado)
)
objective_value = objective_raw + trade_penalty  # -5 si <50% tickers tienen trades
```

**Justificacion**:
- **Sharpe (40%)**: Balance riesgo/retorno (metrica principal)
- **Profit Factor (30%)**: Eficiencia de trades (ganancia/perdida), capped en 3 para evitar outliers
- **CAGR (30%)**: Retorno anualizado, escalado x10 porque tipicamente es 0.1-0.3
- **Trade Penalty**: Evita thresholds muy altos que no generan senales

**Nota metodologica**: La validacion walk-forward se reserva para la fase de entrenamiento/evaluacion final, fuera del loop de Optuna. Durante la optimizacion se usa `time_split` (split temporal simple) para reducir costo computacional.

## Aplicar Parametros Optimizados

### Opcion 1: Variable de Entorno

```bash
export E2_TUNED_PARAMS_PATH=reports/hyperparameter_optimization/e2_tuned_params_by_ticker.yaml
python scripts/run_e2.py
```

### Opcion 2: Actualizar base.yaml Manualmente

Copia los mejores valores de `best_params_e2.yaml` a `src/config/base.yaml`:

```yaml
strategies:
  e2_moderate:
    thresholds:
      tau_buy: 0.025
      tau_sell: 0.005

    model:
      lstm_units: [128, 64]
      dropout: 0.25
      learning_rate: 0.0008
      batch_size: 64
```

### Opcion 3: Usar Parametros Por Ticker en Airflow

Edita el DAG `e2_moderate_pipeline.py` para usar el YAML optimizado:

```python
# En la UI de Airflow, al ejecutar el DAG:
use_tuned_params = True
tuned_params_path = "reports/hyperparameter_optimization/e2_tuned_params_by_ticker.yaml"
```

## Analisis de Resultados

### Ver en MLflow UI

```bash
# Local
mlflow ui --backend-store-uri sqlite:///runs/mlflow_local/mlflow.db

# Navegar a http://localhost:5000
# Experimento: "E2_Hyperparameter_Optimization"
```

Comparar trials:
1. Seleccionar multiples runs
2. Click "Compare"
3. Ver graficos de metricas vs parametros

### Optuna Dashboard

```bash
optuna-dashboard sqlite:///runs/optuna_trials/optuna_studies.db
```

### Analizar CSV en Python

```python
import pandas as pd

# Cargar trials
df = pd.read_csv("reports/hyperparameter_optimization/e2_all_trials.csv")

# Top 10 trials
top10 = df.nlargest(10, "value")

# Correlacion parametros vs metrica objetivo
param_cols = [c for c in df.columns if c.startswith("params_")]
correlations = df[param_cols + ["value"]].corr()["value"].sort_values(ascending=False)
print(correlations)
```

## Opciones Avanzadas

### Continuar Optimizacion Existente

```bash
# Agrega 20 trials mas al estudio existente
python scripts/optimization/optimize_e2_hyperparameters.py \
    --study_name e2_hyperparameter_optimization_timesplit \
    --n_trials 20
```

### Usar MLflow Remoto (Docker)

```bash
python scripts/optimization/optimize_e2_hyperparameters.py \
    --n_trials 50 \
    --mlflow_uri http://localhost:5050
```

### Optimizar Subset de Tickers

```bash
# Tech stocks
python scripts/optimization/optimize_e2_hyperparameters.py \
    --tickers "NVDA,GOOGL,META,AMZN" \
    --n_trials 40
```

### Timeout (Limite de Tiempo)

```bash
# Maximo 2 horas (7200 segundos)
python scripts/optimization/optimize_e2_hyperparameters.py \
    --n_trials 100 \
    --timeout 7200
```

### Interrupcion Segura

El script maneja `Ctrl+C` guardando resultados parciales automaticamente. Los trials completos se persisten en la base de datos de Optuna y pueden continuarse.

## Protecciones y Robustez

- **Env vars**: Se limpian `E2_TUNED_PARAMS_PATH` y `TUNED_PARAMS_PATH` antes de cada trial para evitar contaminacion
- **MLflow fallback**: Si el servidor remoto falla, se usa SQLite local automaticamente. Si todo falla, continua sin tracking
- **Escritura atomica**: Los archivos de salida se escriben via temp file + rename con retries (protege contra timeouts de iCloud/FS remotos)
- **Captura de logs**: stdout/stderr del training se captura para mantener la salida de Optuna limpia
- **Study isolation**: En modo single-ticker, el study name se aisla con sufijo `__TICKER` para evitar mezclar trials

## Troubleshooting

### Error: "No module named 'optuna'"

```bash
pip install optuna kaleido plotly
```

### Error: "Permission denied" en runs/optuna_trials/optuna_studies.db

```bash
# Eliminar DB corrupta
rm runs/optuna_trials/optuna_studies.db

# Re-ejecutar optimizacion
python scripts/optimization/optimize_e2_hyperparameters.py --n_trials 10
```

### Trials muy lentos

**Causas**:
- Muchos tickers (10 tickers x 50 trials = 500 entrenamientos)
- Epochs altos en base.yaml

**Soluciones**:
```bash
# Modo rapido (3 tickers)
python scripts/optimization/optimize_e2_hyperparameters.py --quick --n_trials 20

# Reducir max_epochs temporalmente en base.yaml
# strategies.e2_moderate.model.max_epochs: 150 -> 50
```

### Visualizaciones no generadas

Requiere `kaleido` para exportar a PNG:

```bash
pip install kaleido

# Si kaleido falla, los graficos se generan en HTML interactivo automaticamente
```

## Comparacion de Parametros Optimizables

| Parametro | Tipo | Rango | Impacto | Prioridad |
|-----------|------|-------|---------|-----------|
| **tau_buy** | Trading | 0.015-0.05 | Critico para senales | Alta |
| **tau_sell** | Trading | -0.01-0.01 | Gestion de salidas | Alta |
| **lstm_units_1** | Arquitectura | 64-256 | Capacidad del modelo | Alta |
| **lstm_units_2** | Arquitectura | 32-128 | Refinamiento | Alta |
| **dropout** | Regularizacion | 0.1-0.4 | Previene overfitting | Alta |
| **learning_rate** | Entrenamiento | 5e-5 - 5e-3 | Convergencia | Alta |
| **batch_size** | Entrenamiento | 32,64,128 | Estabilidad gradientes | Media |

**Diferencia clave vs E1:**
- E1 usa GRU, E2 usa LSTM (mas parametros internos)
- E1 optimiza `weight_decay`, E2 no (pendiente de exponer en lstm.py)
- E1 objetivo: 0.5*IC + 0.5*Sharpe. E2 objetivo: 0.4*Sharpe + 0.3*PF + 0.3*CAGR (perfil moderado)
- E1 early_stopping_patience=10, E2=12

## Referencias de Rendimiento

**Hardware**: MacBook Pro M1, 16GB RAM

| Configuracion | Tiempo Estimado |
|--------------|-----------------|
| `--quick --n_trials 10` | 10-15 min |
| `--ticker NVDA --n_trials 30` | 20-30 min |
| `--n_trials 50` (10 tickers) | 2-3 horas |
| `--per_ticker --n_trials 30` (10 tickers) | 3-4 horas |

**Paralelizacion**: Optuna usa TPESampler (secuencial). Para paralelizar, ejecutar multiples estudios independientes por ticker (`--per_ticker`).

## Mejores Practicas

1. **Empezar con `--quick`**: Validar que el script funciona (10 min)
2. **Optimizar por ticker**: Mejores resultados que optimizacion global
3. **Monitorear MLflow**: Revisar metricas en tiempo real
4. **Guardar estudios**: Usar `--study_name` unico para cada experimento
5. **Iterar rangos**: Si todos los trials convergen en un limite, expandir rango con `--tau_buy_max`, etc.
6. **Validar OOS**: Despues de optimizacion, validar en datos no vistos (walk-forward)

## Ejecucion desde Airflow

El DAG `e2_optuna_hyperparameter_tuning` permite ejecutar la optimizacion desde la UI de Airflow con parametros configurables.

### Trigger con Configuracion Personalizada

Click en "Trigger DAG w/ config" y personalizar parametros en JSON:

```json
{
  "n_trials": "10",
  "quick_mode": "True",
  "mlflow_uri": "local"
}
```

```json
{
  "per_ticker": "True",
  "n_trials": "30",
  "tickers": "NVDA, AMD, MSFT",
  "mlflow_uri": "local"
}
```

Los resultados se guardan automaticamente y el comando CLI exacto queda en XCom para replicar desde terminal.

## Referencias

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)

## Archivos Relacionados

- Script: [scripts/optimization/optimize_e2_hyperparameters.py](scripts/optimization/optimize_e2_hyperparameters.py)
- Pipeline: [src/e2/train_pipeline.py](src/e2/train_pipeline.py)
- Modelo LSTM: [src/e2/lstm.py](src/e2/lstm.py)
- Config: [src/config/base.yaml](src/config/base.yaml)
- DAG Training: [dockerfiles/airflow/dags/E2/e2_moderate_pipeline.py](dockerfiles/airflow/dags/E2/e2_moderate_pipeline.py)
- DAG Optuna: [dockerfiles/airflow/dags/E2/e2_optuna_tuning.py](dockerfiles/airflow/dags/E2/e2_optuna_tuning.py)
