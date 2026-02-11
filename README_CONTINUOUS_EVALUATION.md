# Sistema de Evaluación Continua - E1 Simple

## 🎯 Objetivo

Monitorear la performance de modelos E1 Simple en producción de forma continua, comparando predicciones diarias con retornos reales cuando se cumple el horizonte (90 días).

## 📊 Flujo de Trabajo

```
Día 0: Predicción          Día 90: Evaluación        Día 120: Métricas Rolling
  ├─ Hacer predicción        ├─ Descargar precio      ├─ IC últimos 30 días
  ├─ Guardar en MLflow       ├─ Calcular retorno      ├─ MAE rolling
  └─ Status: pending         ├─ Evaluar error         ├─ Comparar con training
                             └─ Status: evaluated     └─ Alertar degradación
```

## 🔧 Integración con MLflow

### Experimentos

1. **E1_Simple** (original)
   - Runs de entrenamiento
   - Métricas: IC, MAE, Sharpe, etc.
   - Artifacts: modelos, backtests

2. **E1_Simple_Production_Tracking** (nuevo)
   - Predicciones diarias
   - Evaluaciones periódicas
   - Comparaciones con training

### Estructura de Runs

**Run de Predicción** (se crea cada vez que ejecutas trading)
```python
params:
  ticker: "AAPL"
  prediction_date: "2026-01-19"
  evaluation_date: "2026-04-19"  # +90 días
  horizon_days: 90
  model_path: "runs/e1_simple/.../AAPL_model.pth"
  status: "pending"  # → "evaluated" después

metrics:
  predicted_return: 0.0521  # Predicción: +5.21%

artifacts:
  features/AAPL_features_20260119.csv
```

**Run de Evaluación** (se crea cuando pasaron 90 días)
```python
params:
  ticker: "AAPL"
  evaluation_type: "production_tracking"
  n_predictions: 30  # Últimas 30 predicciones evaluadas

metrics:
  mae: 0.0342
  rmse: 0.0487
  ic: 0.0821
  directional_accuracy: 0.6333
  n_evaluated: 30

artifacts:
  evaluations/AAPL_evaluation_20260419.csv
```

## 🚀 Uso

### 1. Trading Diario (Automático)

Cuando ejecutas el script de trading con IOL, automáticamente se guarda la predicción:

```bash
python scripts/trading/e1_simple_iol_live_trade.py \
  --ticker AAPL \
  --model runs/e1_simple/20260116_210239/AAPL/AAPL_model.pth \
  --quantity 10
```

Esto crea un run en MLflow con `status=pending`.

### 2. Evaluación Periódica (Manual o Cron)

Cada semana o mes, ejecuta:

```bash
# Evaluar todas las predicciones pendientes de AAPL
python scripts/evaluation/e1_continuous_evaluation.py \
  --ticker AAPL \
  --evaluate-pending

# Evaluar múltiples tickers
for ticker in AAPL MSFT GGAL.BA YPFD.BA; do
  python scripts/evaluation/e1_continuous_evaluation.py \
    --ticker $ticker \
    --evaluate-pending
done
```

Esto:
- Busca predicciones con `status=pending` y fecha de evaluación pasada
- Descarga datos reales de yfinance
- Calcula retorno real vs. predicho
- Guarda métricas (MAE, IC, Dir Acc) en MLflow
- Actualiza `status=evaluated`
- Compara con métricas del entrenamiento original
- Genera alertas si hay degradación >15%

### 3. Automatizar con Cron

```bash
# Crontab: Evaluar todos los domingos a las 8am
0 8 * * 0 cd /path/to/trading_predict && python scripts/evaluation/e1_continuous_evaluation.py --ticker AAPL --evaluate-pending
```

## 📈 Visualización en MLflow

### 1. Ver Predicciones

```bash
mlflow ui --backend-store-uri sqlite:///runs/mlflow_local/mlflow.db
```

- Experiment: "E1_Simple_Production_Tracking"
- Filter: `params.ticker = 'AAPL' AND params.status = 'pending'`
- Ver: `metrics.predicted_return`

### 2. Ver Evaluaciones

- Filter: `params.evaluation_type = 'production_tracking'`
- Comparar métricas: `ic`, `mae`, `directional_accuracy`
- Download artifact: `evaluations/AAPL_evaluation_*.csv`

### 3. Comparar con Training

```python
# En MLflow UI
# 1. Buscar run original: Experiment "E1_Simple", ticker="AAPL"
# 2. Buscar evaluación: Experiment "E1_Simple_Production_Tracking"
# 3. Comparar métricas:
#    - Training IC: 0.12
#    - Production IC: 0.08  → -33% degradación ⚠️
```

## 🔔 Alertas de Degradación

El script automáticamente detecta si las métricas empeoraron:

```
🚨 ALERTAS DE DEGRADACIÓN
================================================================================
⚠️  IC degradó 25.3% (threshold: 15%)
⚠️  MAE degradó 18.7% (threshold: 20%)
```

**Acciones recomendadas:**
1. Verificar cambios en el mercado (régimen diferente)
2. Re-entrenar modelo con datos recientes
3. Ajustar thresholds de decisión
4. Considerar ensemble con modelos nuevos

## 📊 Métricas Rolling

Para calcular IC/Sharpe rolling de últimos N días:

```python
from scripts.e1_continuous_evaluation import get_pending_predictions, evaluate_predictions

# Obtener últimas 30 evaluaciones
pending = get_pending_predictions("AAPL", mlflow_uri)
metrics = evaluate_predictions("AAPL", pending[-30:], mlflow_uri)

print(f"IC últimos 30 días: {metrics['ic']:.4f}")
```

## 🎯 Ejemplo Completo

### Día 1 - Enero 19, 2026

```bash
# Hacer predicción y ejecutar orden
python scripts/trading/e1_simple_iol_live_trade.py \
  --ticker AAPL \
  --model runs/e1_simple/20260116_210239/AAPL/AAPL_model.pth \
  --quantity 10

# Output:
# ✓ Predicción: retorno esperado = +5.21%
# ✓ Predicción guardada en MLflow para evaluación futura
# 🟢 Señal: COMPRAR 10 AAPL
```

**MLflow Run creado:**
- `prediction_date`: 2026-01-19
- `evaluation_date`: 2026-04-19
- `predicted_return`: 0.0521
- `status`: pending

### Día 90 - Abril 19, 2026

```bash
# Evaluar predicción
python scripts/evaluation/e1_continuous_evaluation.py \
  --ticker AAPL \
  --evaluate-pending

# Output:
# ✓ Encontradas 1 predicciones para evaluar
# ✓ Evaluación guardada en MLflow
#   MAE: 0.0234
#   IC: 0.0891
#   Dir Acc: 66.67%
```

**Retorno real:** +3.12% (vs. predicho +5.21%)
**Error:** 2.09% (MAE)
**Dirección:** ✓ Correcta (ambos positivos)

### Día 120 - Mayo 19, 2026

Si tienes 30 predicciones evaluadas:

```bash
# Métricas rolling
python scripts/evaluation/e1_continuous_evaluation.py --ticker AAPL --evaluate-pending

# Output con comparación:
# ✓ IC actual: 0.0821 vs. Training IC: 0.1200 → -31.6% degradación
# 🚨 ALERTAS: IC degradó 31.6% (threshold: 15%)
# 💡 Recomendación: Re-entrenar modelo con datos recientes
```

## 🗂️ Archivos Generados

```
runs/
├── mlflow_local/
│   └── mlflow.db                    # Base de datos MLflow
└── e1_simple/
    └── 20260116_210239/
        └── AAPL/
            └── AAPL_model.pth       # Modelo usado

/tmp/ (artifacts temporales):
├── AAPL_features_20260119.csv       # Features del día
└── AAPL_evaluation_20260419.csv     # Evaluación detallada
```

## ⚙️ Configuración

### Variables de Entorno (.env)

```bash
IOL_USERNAME=tu_usuario
IOL_PASSWORD=tu_password
```

### Thresholds de Degradación

En `e1_continuous_evaluation.py`:

```python
# MAE: si aumenta >20% → alerta
threshold_mae = 0.20

# IC, Dir Acc: si disminuye >15% → alerta
threshold_performance = 0.15
```

## 🔍 Queries Útiles en MLflow

### 1. Predicciones del último mes

```python
filter_string = """
  params.ticker = 'AAPL' AND 
  params.prediction_date >= '2026-04-19'
"""
```

### 2. Predicciones con señal BUY

```python
filter_string = "metrics.predicted_return >= 0.05"
```

### 3. Evaluaciones con IC bajo

```python
filter_string = """
  params.evaluation_type = 'production_tracking' AND
  metrics.ic < 0.05
"""
```

## 📝 Notas Importantes

1. **Horizonte fijo**: El sistema asume horizonte de 90 días (configurable)
2. **Datos de yfinance**: Para evaluar necesitas que yfinance tenga datos del ticker
3. **Timezone**: Todas las fechas en UTC/local según sistema
4. **Storage**: MLflow crece con el tiempo, considera limpieza periódica
5. **Comparación**: Solo compara si el modelo original está en MLflow

---

## 6. Validación Retrospectiva (Out-of-Time)

### 6.1 ¿Qué es?

**Diferencia con backtesting tradicional:**
- **Backtesting**: Entrena con 2020-2023, evalúa con 2024-2025 (walk-forward histórico)
- **Out-of-Time**: Entrena con datos hasta hace N días, evalúa con datos de hoy

**Ventaja**: Simula exactamente producción con mercados actuales.

### 6.2 Uso

**Modelo GRU (E1 Simple completo):**
```bash
# Entrenar con datos hasta hace 360 días, evaluar con datos de hoy
python scripts/evaluation/e1_retrospective_validation.py \
    --ticker AAPL \
    --train-days-ago 360 \
    --horizon 90

# Usar MLflow en Docker
python scripts/evaluation/e1_retrospective_validation.py \
    --ticker GGAL.BA \
    --train-days-ago 360 \
    --mlflow-docker
```

**Baseline Linear (más rápido, para comparación):**
```bash
# Entrenar modelo baseline (Ridge Regression)
python scripts/evaluation/e1_baseline_retrospective_validation.py \
    --ticker AAPL \
    --train-days-ago 360 \
    --horizon 90
```

**Ventaja del script GRU:**
- Usa arquitectura real de E1 Simple (2 capas GRU)
- Verifica `data/clean/` antes de descargar (más rápido)
- Resultados más realistas para producción

### 6.3 Interpretación

El script compara métricas in-sample test vs. out-of-time:

```
📊 COMPARACIÓN: In-Sample Test vs. Out-of-Time
================================================================================
           In-Sample Test  Out-of-Time  Degradación %
MAE                0.0842       0.0903           +7.2
IC                 0.1234       0.1156           -6.3
Dir Acc            0.5789       0.5621           -2.9
```

**Criterios de evaluación:**
- ✅ **Degradación < 15%**: Modelo robusto, listo para producción
- ⚠️  **Degradación 15-30%**: Revisar features o reentrenar
- 🚨 **Degradación > 30%**: No usar en producción, modelo obsoleto

### 6.4 Cuándo usar

**Antes de deployar a producción:**
```bash
# Validar múltiples tickers
for ticker in AAPL MSFT GGAL.BA; do
    python scripts/evaluation/e1_retrospective_validation.py \
        --ticker $ticker \
        --train-days-ago 90
done
```

**Periódicamente (mensual):**
```bash
# Verificar si modelos siguen válidos con mercados recientes
python scripts/evaluation/e1_retrospective_validation.py \
    --ticker AAPL \
    --train-days-ago 30
```

### 6.5 MLflow Integration

Resultados se guardan en experimento `E1_Simple_Retrospective_Validation`:

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///runs/mlflow_local/mlflow.db")
experiment = mlflow.get_experiment_by_name("E1_Simple_Retrospective_Validation")

runs = mlflow.search_runs(experiment.experiment_id)
print(runs[['params.ticker', 'metrics.oot_ic', 'metrics.ic_degradation_pct']])
```

### 6.6 Comparación: Backtesting vs. Out-of-Time

| Aspecto | Backtesting (Walk-Forward) | Out-of-Time |
|---------|----------------------------|-------------|
| **Período** | Histórico (2020-2025) | Reciente (90 días atrás → hoy) |
| **Propósito** | Validar estrategia general | Validar modelo en mercado actual |
| **Frecuencia** | Una vez durante desarrollo | Mensual/antes de deploy |
| **Detecta** | Overfitting, robustez histórica | Regime changes recientes |
| **Ventaja** | Cobertura temporal amplia | Realismo de producción |

**Recomendación**: Usar ambos. Backtesting para confianza general, out-of-time para validación reciente.

---

## 🚧 Próximas Mejoras

- [ ] Dashboard automático con métricas rolling
- [ ] Alertas por email/Slack cuando degradación >threshold
- [ ] Re-entrenamiento automático si performance cae
- [ ] Soporte para múltiples horizontes (30d, 60d, 90d)
- [ ] Integración con Airflow para evaluación automática
- [ ] Análisis de features drift (distribución actual vs. training)
- [ ] A/B testing: comparar modelo nuevo vs. modelo en producción

---

## 📚 Referencias

- MLflow Tracking: https://mlflow.org/docs/latest/tracking.html
- Backtesting: `README_BACKTESTING.md`
- E1 Simple: `README_E1_BASELINE.md`
- Out-of-Time Validation: `scripts/evaluation/e1_retrospective_validation.py`
