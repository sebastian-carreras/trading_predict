# Optimización de Hiperparámetros E1 con Optuna + MLflow

Sistema completo de búsqueda automática de hiperparámetros óptimos para la estrategia E1 Conservative.

## 📋 Tabla de Contenidos

- [Descripción](#descripción)
- [Hiperparámetros Optimizados](#hiperparámetros-optimizados)
- [Métrica Objetivo](#métrica-objetivo)
- [Uso](#uso)
- [Interpretación de Resultados](#interpretación-de-resultados)
- [Visualizaciones](#visualizaciones)

---

## Descripción

El script `optimize_e1_hyperparameters.py` implementa **búsqueda bayesiana** de hiperparámetros usando:

- **Optuna**: Framework de optimización automática con Tree-structured Parzen Estimator (TPE)
- **MLflow**: Tracking de experimentos, parámetros y métricas
- **Validación robusta**: Entrenamiento en todos los tickers de E1 para asegurar generalización

### Ventajas sobre Grid Search

| Método | Grid Search | Optuna (Bayesian) |
|--------|-------------|-------------------|
| **Eficiencia** | Prueba todas las combinaciones | Aprende de trials previos |
| **Trials necesarios** | ~1000 para 5 parámetros | ~50 para convergencia |
| **Adaptabilidad** | Fija | Explora zonas prometedoras |
| **Early stopping** | No | Sí (pruning) |

**Ejemplo**: Con 5 hiperparámetros y 10 valores cada uno:
- Grid Search: 10^5 = **100,000 trials** ❌
- Optuna: ~**50 trials** para encontrar óptimo ✅

---

## Hiperparámetros Optimizados

### 1. Trading Thresholds (Críticos para P&L)

```python
tau_buy:  0.02 → 0.10  # Umbral para señal de COMPRA
tau_sell: -0.02 → 0.02  # Umbral para señal de VENTA
```

**Impacto**:
- `tau_buy` muy bajo → muchos trades, comisiones altas
- `tau_buy` muy alto → pocas señales, oportunidades perdidas
- `tau_sell` negativo → vende en predicción de caída

**Ejemplo**:
```
tau_buy = 0.04 (4%)
→ Compra solo si predice ganancia > 4% en 90 días
→ Conservador pero con alta convicción

tau_sell = -0.01 (-1%)
→ Vende si predice caída > 1%
→ Salida rápida ante señales bajistas
```

### 2. Arquitectura GRU

```python
gru_units_1: 32 → 128   # Neuronas en primera capa
gru_units_2: 16 → 64    # Neuronas en segunda capa
```

**Impacto**:
- Más neuronas → mayor capacidad, riesgo de overfitting
- Menos neuronas → más rápido, riesgo de underfitting

**Rango típico**:
- Modelos simples: [32, 16]
- Modelos complejos: [128, 64]

### 3. Regularización

```python
dropout: 0.2 → 0.5  # Proporción de neuronas desactivadas
```

**Impacto**:
- Dropout bajo → memoriza entrenamiento (overfitting)
- Dropout alto → generaliza mejor pero aprende más lento

### 4. Entrenamiento

```python
learning_rate: 1e-4 → 1e-2  # Tasa de aprendizaje (log scale)
batch_size: [32, 64, 128]   # Tamaño de batch
```

**Impacto learning_rate**:
- Alto (1e-2) → aprende rápido pero inestable
- Bajo (1e-4) → aprende lento pero estable

**Impacto batch_size**:
- 32 → más actualizaciones, más ruidoso
- 128 → menos actualizaciones, más estable

---

## Métrica Objetivo

El optimizador maximiza una **métrica compuesta**:

```python
objective = 0.5 * IC_mean + 0.5 * Sharpe_mean + trade_penalty
```

### Componentes

**1. IC (Information Coefficient)** - Calidad de predicción
- Rango: -1 a +1
- IC > 0.05 = predicción útil
- IC > 0.20 = predicción excelente

**2. Sharpe Ratio** - Performance de trading
- Rango: -∞ a +∞
- Sharpe > 1.0 = estrategia buena
- Sharpe > 2.0 = estrategia excelente

**3. Trade Penalty** - Evitar thresholds irreales
- Penaliza si <50% de tickers generan trades
- Evita tau_buy muy alto (ej: 10%) que no opera nunca

### ¿Por qué esta métrica?

| Solo IC | Solo Sharpe | IC + Sharpe (USADO) |
|---------|-------------|---------------------|
| Buena predicción | Buen trading | Balance óptimo |
| Pero puede no tradear | Pero predicción pobre | Predice bien Y tradea bien |
| ❌ | ❌ | ✅ |

---

## Uso

### 1. Optimización Estándar (50 trials)

```bash
python scripts/optimize_e1_hyperparameters.py --n_trials 50
```

**Tiempo estimado**: ~3 horas (depende de hardware y tickers)

**Output**:
- `reports/hyperparameter_optimization/best_params_e1.yaml`
- `reports/hyperparameter_optimization/all_trials.csv`
- `reports/hyperparameter_optimization/figures/*.png`
- MLflow tracking: http://localhost:5050

### 2. Modo Rápido (Prueba)

```bash
# Solo 3 tickers, 10 trials
python scripts/optimize_e1_hyperparameters.py --n_trials 10 --quick
```

**Tiempo estimado**: ~20 minutos

**Uso**: Validar que funciona antes de optimización completa

### 3. Optimizar Ticker Específico

```bash
# Optimizar solo para AAPL (útil para análisis individual)
python scripts/optimize_e1_hyperparameters.py --ticker AAPL --n_trials 30
```

**Tiempo estimado**: ~45 minutos

**Uso**: Tunear estrategia para un activo específico

### 4. Continuar Optimización Existente

```bash
# Agregar 20 trials más a estudio existente
python scripts/optimize_e1_hyperparameters.py \
  --study_name e1_hyperparameter_optimization \
  --n_trials 20
```

**Uso**: Si optimización se interrumpió o quieres más trials

### 5. Con Timeout (Limitar Tiempo)

```bash
# Detener después de 2 horas (7200 segundos)
python scripts/optimize_e1_hyperparameters.py \
  --n_trials 100 \
  --timeout 7200
```

---

## Interpretación de Resultados

### 1. Archivo `best_params_e1.yaml`

```yaml
optimization:
  study_name: e1_hyperparameter_optimization
  n_trials: 50
  best_value: 0.4523
  best_trial: 37

best_params:
  tau_buy: 0.04
  tau_sell: -0.01
  gru_units_1: 64
  gru_units_2: 32
  dropout: 0.35
  learning_rate: 0.0015
  batch_size: 64
```

**Interpretación**:
- Trial #37 fue el mejor de 50
- Objective value = 0.4523 (IC + Sharpe combinados)
- `tau_buy = 0.04` → estrategia conservadora (4% threshold)
- `dropout = 0.35` → regularización moderada
- `learning_rate = 0.0015` → aprendizaje medio

**Acción**: Actualizar `src/config/base.yaml` con estos parámetros

### 2. CSV `all_trials.csv`

Columnas importantes:
- `number`: ID del trial
- `value`: Métrica objetivo
- `params_tau_buy`, `params_tau_sell`, etc.: Parámetros probados
- `state`: COMPLETE, PRUNED, FAIL

**Análisis útil**:

```python
import pandas as pd

df = pd.read_csv('reports/hyperparameter_optimization/all_trials.csv')

# Top 10 trials
print(df.nlargest(10, 'value'))

# Correlación entre parámetros y objetivo
params = ['params_tau_buy', 'params_dropout', 'params_learning_rate']
print(df[params + ['value']].corr()['value'])

# Rango de parámetros en top 10%
top_10pct = df.nlargest(int(len(df)*0.1), 'value')
for param in params:
    print(f"{param}: {top_10pct[param].min():.4f} - {top_10pct[param].max():.4f}")
```

### 3. MLflow UI

Acceder a http://localhost:5050 y filtrar experimento "E1_Hyperparameter_Optimization"

**Visualizaciones útiles**:
- **Parallel Coordinates**: Ver relación entre parámetros
- **Scatter Matrix**: Correlaciones
- **Compare Runs**: Comparar top trials

**Métricas agregadas por trial**:
- `ic_mean`: IC promedio en todos los tickers
- `sharpe_mean`: Sharpe promedio
- `ic_positive_pct`: % de tickers con IC > 0.05
- `tickers_with_trades`: Cuántos tickers generan señales

---

## Visualizaciones

El script genera 3 gráficos automáticamente en `reports/hyperparameter_optimization/figures/`:

### 1. `optimization_history.png`

![Optimization History Example](https://via.placeholder.com/800x400?text=Optimization+History)

**Interpretación**:
- **Eje Y**: Valor objetivo
- **Eje X**: Número de trial
- **Azul**: Valor de cada trial
- **Rojo**: Mejor valor acumulado

**Qué buscar**:
- ✅ Línea roja estabiliza → convergencia
- ❌ Línea roja sigue subiendo → necesita más trials

### 2. `param_importances.png`

![Parameter Importances Example](https://via.placeholder.com/800x400?text=Parameter+Importances)

**Interpretación**:
- Muestra **qué parámetros impactan más** el objetivo
- Calculado vía mutual information

**Ejemplo**:
```
tau_buy: 45% importance      ← MUY importante
dropout: 25% importance      ← Moderadamente importante
batch_size: 5% importance    ← Poco importante
```

**Acción**: Focalizar tuning manual en parámetros importantes

### 3. `parallel_coordinate.png`

![Parallel Coordinate Example](https://via.placeholder.com/800x600?text=Parallel+Coordinate)

**Interpretación**:
- Cada línea = 1 trial
- Color = valor objetivo (rojo = mejor, azul = peor)
- Permite ver **patrones** en combinaciones de parámetros

**Qué buscar**:
- Líneas rojas agrupadas en ciertos valores → rango óptimo
- Ejemplo: "Mejores trials tienen tau_buy entre 0.03-0.05"

---

## Ejemplo de Workflow Completo

### Paso 1: Optimización Rápida (Exploración)

```bash
# 10 trials con 3 tickers para validar
python scripts/optimize_e1_hyperparameters.py --n_trials 10 --quick
```

**Resultado**: Valores aproximados en ~20 minutos

### Paso 2: Optimización Completa

```bash
# 50 trials con todos los tickers
python scripts/optimize_e1_hyperparameters.py --n_trials 50
```

**Resultado**: Parámetros óptimos en ~3 horas

### Paso 3: Analizar Resultados

```bash
# Ver gráficos
open reports/hyperparameter_optimization/figures/

# Ver mejores parámetros
cat reports/hyperparameter_optimization/best_params_e1.yaml

# Explorar en MLflow
open http://localhost:5050
```

### Paso 4: Aplicar Mejores Parámetros

Editar `src/config/base.yaml`:

```yaml
strategies:
  e1_conservative:
    thresholds:
      tau_buy: 0.04  # Del optimization
      tau_sell: -0.01
    
    model:
      gru_units: [64, 32]  # Del optimization
      dropout: 0.35
      learning_rate: 0.0015
      batch_size: 64
```

### Paso 5: Validar Mejora

```bash
# Re-entrenar con nuevos parámetros
docker compose exec airflow-scheduler airflow dags trigger e1_conservative_pipeline

# Comparar métricas en MLflow
# Experimento anterior vs nuevo
```

---

## Troubleshooting

### Error: "No module named 'optuna'"

```bash
pip install optuna optuna-dashboard kaleido
```

### Error: MLflow connection refused

```bash
# Verificar que MLflow está corriendo
docker compose ps mlflow

# Si no está, iniciar
docker compose up -d mlflow
```

### Optimización muy lenta

**Opciones**:
1. Usar `--quick` para menos tickers
2. Reducir `--n_trials`
3. Usar `--ticker AAPL` para un solo ticker
4. Usar `--timeout` para limitar tiempo

### No converge (línea roja sigue subiendo)

**Solución**: Aumentar `--n_trials`

Típicamente:
- 20 trials: exploración inicial
- 50 trials: suficiente para mayoría
- 100 trials: si espacio de búsqueda muy grande

---

## Referencias

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html)
- [Hyperparameter Optimization in ML](https://arxiv.org/abs/1502.02127)

---

## Próximos Pasos

Después de optimización:

1. ✅ **Aplicar parámetros** en `base.yaml`
2. 📊 **Validar con Walk-Forward** (implementar siguiente)
3. 🔍 **Feature Importance** (analizar qué features usar)
4. 💼 **Portfolio Optimization** (multi-ticker allocation)
