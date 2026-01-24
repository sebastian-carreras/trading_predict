# Optimización de Hiperparámetros - Estrategia E1 (GRU Conservadora)

Este documento describe cómo optimizar los hiperparámetros de la estrategia E1 usando **Optuna** para búsqueda automática y **MLflow** para tracking de experimentos.

## 📋 Hiperparámetros Optimizados

### Trading Thresholds
- **tau_buy**: Umbral de señal de compra (0.015 - 0.05)
- **tau_sell**: Umbral de señal de venta (-0.01 - 0.01)

### Arquitectura GRU
- **gru_units_1**: Neuronas en primera capa GRU (64 - 256)
- **gru_units_2**: Neuronas en segunda capa GRU (32 - 128)
- **dropout**: Tasa de dropout para regularización (0.1 - 0.4)

### Entrenamiento
- **learning_rate**: Tasa de aprendizaje (5e-5 - 5e-3, log scale)
- **batch_size**: Tamaño de batch (32, 64, 128)

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
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 10 --quick
```

### 2. Optimización Completa

```bash
# 50 trials, todos los tickers E1 (~2-3 horas)
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 50
```

### 3. Optimizar Un Solo Ticker

```bash
# Útil para entender comportamiento de ticker específico
python scripts/optimization/optimize_e1_hyperparameters.py --ticker GGAL.BA --n_trials 30
```

### 4. Optimización Por Ticker (Recomendado)

```bash
# Genera YAML con mejores params para cada ticker
python scripts/optimization/optimize_e1_hyperparameters.py --per_ticker --n_trials 30
```

Esto genera `reports/hyperparameter_optimization/e1_tuned_params_by_ticker.yaml`:

```yaml
GGAL.BA:
  thresholds:
    tau_buy: 0.05
    tau_sell: 0.02
  model:
    gru_units: [112, 48]
    dropout: 0.25
    learning_rate: 0.0002
    batch_size: 64

YPFD.BA:
  thresholds:
    tau_buy: 0.045
    tau_sell: 0.015
  model:
    gru_units: [128, 64]
    # ... etc
```

## 📊 Outputs

### 1. Mejores Parámetros (YAML)

**Global**: `reports/hyperparameter_optimization/best_params_e1.yaml`
```yaml
optimization:
  study_name: e1_hyperparameter_optimization
  n_trials: 50
  best_value: 1.1234
  best_trial: 18

best_params:
  tau_buy: 0.025
  tau_sell: 0.005
  gru_units_1: 128
  gru_units_2: 64
  dropout: 0.25
  learning_rate: 0.0005
  batch_size: 64
```

**Por ticker**: `reports/hyperparameter_optimization/e1_tuned_params_by_ticker.yaml`

### 2. Trials Completos (CSV)

`reports/hyperparameter_optimization/e1_all_trials.csv`
- Todos los trials con parámetros y métricas
- Útil para análisis posterior en Excel/Pandas

### 3. Visualizaciones

Carpeta: `reports/hyperparameter_optimization/figures/`

- **e1_optimization_history.png**: Evolución del valor objetivo por trial
- **e1_param_importances.png**: Importancia relativa de cada parámetro
- **e1_parallel_coordinate.png**: Visualización multi-dimensional de parámetros

### 4. MLflow Tracking

**Local**: `runs/mlflow_local/mlflow.db`
```bash
# Ver experimentos
mlflow ui --backend-store-uri sqlite:///runs/mlflow_local/mlflow.db
```

Experimento: `E1_Hyperparameter_Optimization`

Métricas trackeadas:
- `ic_mean`, `ic_median`, `ic_std`
- `sharpe_mean`, `sharpe_median`, `sharpe_std`
- `ic_positive_pct`, `sharpe_positive_pct`
- `tickers_with_trades`, `avg_num_trades`
- `objective_value` (métrica a maximizar)

## 🎯 Función Objetivo

La métrica objetivo es una **combinación ponderada**:

```python
objective = (
    0.5 * ic_mean +               # 50% Information Coefficient
    0.5 * sharpe_mean +           # 50% Sharpe ratio
    trade_penalty                  # -5 si <50% tickers tienen trades
)
```

**Justificación**:
- **Information Coefficient**: Prioridad en estrategia conservadora - correlación entre predicciones y retornos
- **Sharpe Ratio**: Balance entre retorno y riesgo
- **Trade Penalty**: Asegura que los parámetros generen señales de trading, no solo predicciones

**Diferencias con E2**:
- E1 usa IC como métrica principal (más conservadora)
- E2 usa Profit Factor y CAGR (más agresiva)
- E1 GRU es computacionalmente más eficiente que E2 LSTM

### Por Qué Optimizar Walk-Forward Parameters

**`n_folds` (Número de Ventanas):**
- **Menos folds (3)**: Más datos por ventana → métricas más estables, pero menor robustez temporal
- **Más folds (7)**: Mayor robustez temporal → detecta mejor concept drift, pero más varianza por ventana
- **Impacto**: Un modelo puede funcionar bien con 5 folds pero mal con 3 (overfitting a períodos largos)

**`internal_val_fraction` (Validación Interna):**
- **Menor (0.10)**: Más datos para entrenamiento → menor underfitting
- **Mayor (0.25)**: Mejor early stopping → menor overfitting
- **Impacto**: Afecta cuándo se detiene el entrenamiento en cada fold

**Ejemplo de Impacto en E1:**
```
Trial A: n_folds=5, val_frac=0.15 → IC=0.35, Sharpe=0.8 (balance óptimo)
Trial B: n_folds=3, val_frac=0.25 → IC=0.28, Sharpe=0.6 (pocas ventanas)
Trial C: n_folds=7, val_frac=0.10 → IC=0.42, Sharpe=0.9 (robusto temporalmente)
```

Optuna encontrará la combinación óptima para tus datos específicos.

**Nota para E1 (Horizonte 90 días):**
- Ventanas más largas que E2 (horizon=90 vs 20)
- Puede beneficiarse de MENOS folds (3-5) para tener suficientes datos por ventana
- Optuna balanceará automáticamente este trade-off

## 🔄 Integración con Pipeline de Entrenamiento

### Opción 1: Airflow DAG

1. **Ejecutar optimización** (fuera de Airflow):
```bash
python scripts/optimization/optimize_e1_hyperparameters.py --per_ticker --n_trials 50
```

2. **Copiar YAML** al contenedor Airflow:
```bash
# Si usas Docker Compose
docker cp reports/hyperparameter_optimization/e1_tuned_params_by_ticker.yaml \
  airflow-webserver:/opt/airflow/reports/hyperparameter_optimization/
```

3. **Activar en Airflow UI**:
   - Ir a DAG `e1_conservative_pipeline`
   - Click en "Trigger DAG w/ config"
   - Cambiar `use_tuned_params: True`
   - Run

4. **Verificar logs**:
```
✓ Aplicando parámetros tuneados desde: e1_tuned_params_by_ticker.yaml
  Ticker: GGAL.BA
  tau_buy: 0.05, tau_sell: 0.02
  gru_units: [112, 48], dropout: 0.25
```

### Opción 2: CLI Local

```bash
# Exportar variable de entorno
export E1_TUNED_PARAMS_PATH="reports/hyperparameter_optimization/e1_tuned_params_by_ticker.yaml"

# Entrenar con params tuneados
python src/train_e1_pipeline.py
```

## ⚙️ Parámetros Avanzados

### Impacto de Walk-Forward Parameters

**Ejemplo Real (GGAL.BA):**

```
Configuración Fija (base.yaml):
  n_folds: 5
  internal_val_fraction: 0.15
  
  Resultado: IC=0.25, Sharpe=0.65
  → Funciona, pero podría mejorar

Optimización con Optuna:
  Trial 0: n_folds=3, val_frac=0.20 → IC=0.18, Sharpe=0.45 (peor)
  Trial 5: n_folds=5, val_frac=0.10 → IC=0.31, Sharpe=0.72 (mejor)
  Trial 12: n_folds=6, val_frac=0.10 → IC=0.35, Sharpe=0.78 (óptimo!)
  
  Mejora: +40% IC, +20% Sharpe
  → 6 ventanas + 10% validación = balance perfecto para este ticker
```

**Lecciones:**
- E1 con horizonte largo (90 días) prefiere MÁS folds (6-7) que E2
- Validación interna baja (0.10) permite más datos para entrenamiento
- La configuración óptima varía por ticker y régimen de mercado

### Persistencia de Estudios

Por defecto, Optuna guarda estudios en **SQLite**:
```
optuna_studies.db
```

**Continuar estudio previo**:
```bash
# Usar mismo study_name para continuar
python scripts/optimization/optimize_e1_hyperparameters.py \
  --n_trials 50 \
  --study_name e1_hyperparameter_optimization
```

**Ver estudios guardados**:
```bash
optuna studies --storage sqlite:///optuna_studies.db
```

### Cambiar Sampler

Por defecto: **TPESampler** (Bayesian Optimization)

```python
# En optimize_e1_hyperparameters.py, línea ~115
study = optuna.create_study(
    study_name=study_name,
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),  # ← TPE
    storage=f"sqlite:///{self.root / 'optuna_studies.db'}",
    load_if_exists=True,
)
```

Alternativas:
- `RandomSampler()`: Búsqueda aleatoria (baseline)
- `GridSampler()`: Búsqueda exhaustiva (lento)
- `CmaEsSampler()`: Estrategia de evolución

### Paralelización

**Múltiples procesos** (cuidado con MLflow):
```bash
# Terminal 1
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 25 --study_name e1_opt

# Terminal 2 (mismo study_name → comparte DB)
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 25 --study_name e1_opt
```

Optuna maneja concurrencia en SQLite, pero MLflow puede tener conflictos.

### Pruning (Early Stopping de Trials)

```python
# En objective(), agregar intermediate values
mlflow.log_metric("ic_mean", ic_mean, step=fold)
trial.report(ic_mean, fold)

# Optuna puede detener trial si va mal
if trial.should_prune():
    raise optuna.TrialPruned()
```

## 📈 Interpretación de Resultados

### Revisión de Trials

```bash
# Ver todos los trials ordenados
python -c "
import pandas as pd
df = pd.read_csv('reports/hyperparameter_optimization/e1_all_trials.csv')
print(df.sort_values('objective_value', ascending=False).head(10))
"
```

### Análisis de Importancia

`e1_param_importances.png` muestra:
- **Alta importancia**: Parámetro tiene gran impacto en objetivo
- **Baja importancia**: Parámetro no afecta mucho (puede fijarse)

**Ejemplo**:
```
Importances:
  learning_rate: 0.45  ← Muy crítico
  gru_units_1: 0.25
  tau_buy: 0.15
  dropout: 0.10
  batch_size: 0.05     ← Poco impacto
```

→ **Decisión**: Fijar `batch_size=64`, enfocar trials en `learning_rate` y `gru_units_1`

### Validación Cruzada

El script usa **walk-forward** interno (5 folds) para evaluar cada trial:
- Evita overfitting a un período específico
- Métricas promediadas entre folds
- Más robusto que single train/test split

## 🐛 Troubleshooting

### Error: Permission Denied `/opt/airflow`

**Causa**: MLflow experiment creado en Docker con artifact_location incorrecto.

**Solución**:
```bash
# Borrar DB de MLflow y recrear
rm -f runs/mlflow_local/mlflow.db

# Volver a ejecutar
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 10 --quick
```

El script ahora fuerza `artifact_location` local.

### Error: `kaleido` Package Not Found

**Causa**: Problemas con Plotly exportación a PNG en macOS.

**Solución**: El script automáticamente usa **HTML fallback**:
```
⚠️  Error saving PNG: kaleido not found
✓  Saved as HTML: e1_optimization_history.html
```

Abrir HTML en navegador para visualizar.

### Optimización Muy Lenta

**Diagnóstico**:
```bash
# Ver tiempo por trial
tail -f nohup.out
```

**Causas comunes**:
1. **Demasiados tickers**: Usar `--quick` (3 tickers) o `--ticker TICKER`
2. **Walk-forward muy largo**: Reducir folds en código (5 → 3)
3. **Modelo muy grande**: `gru_units` elevado aumenta tiempo

**Optimización**:
```bash
# Solo 3 tickers más rápidos
python scripts/optimization/optimize_e1_hyperparameters.py \
  --ticker AAPL --ticker GOOGL --ticker MSFT \
  --n_trials 20
```

### MLflow No Muestra Métricas

**Verificar**:
```bash
sqlite3 runs/mlflow_local/mlflow.db "SELECT name FROM experiments;"
```

Debería mostrar: `E1_Hyperparameter_Optimization`

**Ver runs**:
```bash
mlflow ui --backend-store-uri sqlite:///$(pwd)/runs/mlflow_local/mlflow.db --port 5001
```

Abrir: http://localhost:5001

## 📚 Ejemplos de Flujos Completos

### Flujo 1: Optimización Rápida → Producción

```bash
# 1. Prueba rápida (3 tickers, 10 trials)
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 10 --quick

# 2. Ver resultados
cat reports/hyperparameter_optimization/best_params_e1.yaml

# 3. Si resultados buenos → optimización completa
python scripts/optimization/optimize_e1_hyperparameters.py --per_ticker --n_trials 50

# 4. Entrenar con params optimizados
export E1_TUNED_PARAMS_PATH="reports/hyperparameter_optimization/e1_tuned_params_by_ticker.yaml"
python src/train_e1_pipeline.py
```

### Flujo 2: Optimización Ticker Específico

```bash
# 1. Optimizar ticker problemático
python scripts/optimization/optimize_e1_hyperparameters.py \
  --ticker GGAL.BA \
  --n_trials 50

# 2. Ver resultados individuales
cat reports/hyperparameter_optimization/by_ticker/GGAL.BA/best_params_e1.yaml

# 3. Integrar en YAML global
# Editar manualmente e1_tuned_params_by_ticker.yaml
```

### Flujo 3: Comparación con Baseline

```bash
# 1. Entrenar con params default (baseline)
python src/train_e1_pipeline.py > baseline_results.log

# 2. Optimizar
python scripts/optimization/optimize_e1_hyperparameters.py --per_ticker --n_trials 50

# 3. Entrenar con params optimizados
export E1_TUNED_PARAMS_PATH="reports/hyperparameter_optimization/e1_tuned_params_by_ticker.yaml"
python src/train_e1_pipeline.py > optimized_results.log

# 4. Comparar métricas en MLflow
mlflow ui --backend-store-uri sqlite:///runs/mlflow_local/mlflow.db
```

## 🔗 Referencias

- **Optuna Docs**: https://optuna.readthedocs.io/
- **MLflow Tracking**: https://www.mlflow.org/docs/latest/tracking.html
- **TPE Sampler Paper**: Bergstra et al., "Algorithms for Hyper-Parameter Optimization" (2011)
- **Walk-Forward Validation**: Prado, "Advances in Financial Machine Learning" (2018)

## ⚡ Quick Reference

```bash
# Optimización rápida
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 10 --quick

# Optimización por ticker (producción)
python scripts/optimization/optimize_e1_hyperparameters.py --per_ticker --n_trials 50

# Ver resultados MLflow
mlflow ui --backend-store-uri sqlite:///runs/mlflow_local/mlflow.db

# Usar params optimizados
export E1_TUNED_PARAMS_PATH="reports/hyperparameter_optimization/e1_tuned_params_by_ticker.yaml"

# Ver estudios Optuna
optuna studies --storage sqlite:///optuna_studies.db
```

## 📊 Comparación de Parámetros Optimizables

| Parámetro | Tipo | Rango | Impacto | Prioridad |
|-----------|------|-------|---------|-----------|
| **tau_buy** | Trading | 0.015-0.05 | ⭐⭐⭐⭐⭐ Crítico para señales | Alta |
| **tau_sell** | Trading | -0.01-0.01 | ⭐⭐⭐⭐ Gestión de salidas | Alta |
| **gru_units_1** | Arquitectura | 64-256 | ⭐⭐⭐⭐ Capacidad del modelo | Alta |
| **gru_units_2** | Arquitectura | 32-128 | ⭐⭐⭐ Refinamiento | Media |
| **dropout** | Regularización | 0.1-0.4 | ⭐⭐⭐⭐ Previene overfitting | Alta |
| **learning_rate** | Entrenamiento | 5e-5 - 5e-3 | ⭐⭐⭐⭐⭐ Convergencia | Alta |
| **batch_size** | Entrenamiento | 32,64,128 | ⭐⭐ Estabilidad gradientes | Media |
| **n_folds** | Walk-Forward | 3-7 | ⭐⭐⭐⭐ Robustez temporal | Alta |
| **internal_val_fraction** | Walk-Forward | 0.10-0.25 | ⭐⭐⭐ Early stopping | Media |

**Total parámetros optimizables:** 9 (11 con lstm_units_1/2 separados)

**Espacio de búsqueda:** ~10^9 combinaciones → Optuna esencial
