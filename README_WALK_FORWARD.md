# Walk-Forward Validation - E1 Estrategia Conservadora

## 🎯 Objetivo

Validar la **robustez temporal** de la estrategia E1 mediante validación walk-forward, demostrando que el modelo mantiene capacidad predictiva en múltiples ventanas temporales no vistas durante el entrenamiento.

## 📊 ¿Qué es Walk-Forward Validation?

Walk-forward es una técnica de validación específica para series temporales que:

1. **Divide los datos temporalmente** en N folds consecutivos
2. **Entrena progresivamente**: cada fold usa todos los datos anteriores para entrenamiento
3. **Valida hacia adelante**: cada fold evalúa en una ventana de test futura
4. **Respeta causalidad**: nunca usa información del futuro

### Ventajas vs Split Simple (70/15/15)

| Aspecto | Split Simple | Walk-Forward |
|---------|-------------|--------------|
| **Ventanas de validación** | 1 única ventana (15%) | N ventanas independientes |
| **Robustez temporal** | No detecta concept drift | Detecta degradación en el tiempo |
| **Uso de datos** | Descarta 85% en test | Usa 100% de datos de forma eficiente |
| **Rigor académico** | Básico | Gold standard para series temporales |

### Esquema Visual

```
Walk-Forward con 5 folds (expanding window):

Fold 1: [train.............] -> [test1]
Fold 2: [train..................] -> [test2]
Fold 3: [train.......................] -> [test3]
Fold 4: [train................................] -> [test4]
Fold 5: [train.......................................] -> [test5]

Ventana completa OOS: [test1][test2][test3][test4][test5]
```

## ⚙️ Configuración

### En `src/config/base.yaml`

```yaml
splits:
  method: "walk_forward"  # Activa walk-forward (vs "time_split")
  folds: 5                # Número de ventanas de validación
  
  # Embargo para evitar leakage temporal
  embargo_days:
    e1: 90   # Gap entre train y test (= horizon_days)
    e2: 20
    e3: 0
    e4: 0
```

### Parámetros Clave

- **`folds`**: Número de ventanas de test (5 por defecto)
  - Más folds = más granularidad temporal, pero menor tamaño de test por fold
  - Recomendado: 5-10 para series largas (>1000 muestras)

- **`embargo_days.e1`**: Gap temporal entre train y test
  - **CRÍTICO**: debe ser ≥ `horizon_days` (90 para E1)
  - Evita contaminar train con información que "mira hacia el futuro"
  - Sin embargo: al predecir retorno a 90 días en t, ya conoceríamos el target en train+validation

- **`test_size`**: Tamaño de cada ventana de test
  - Por defecto: `n_samples / (folds + 1)` = distribución uniforme
  - Puede ajustarse manualmente en config con clave `splits.test_size`

## 🚀 Ejecución

### Opción 1: Ejecutar con Walk-Forward (Modo Actual)

```bash
# Asegurar que splits.method = "walk_forward" en base.yaml
python -m src.train_e1_pipeline --tickers AAPL
```

**Output esperado:**
```
✓ Usando datos limpios: AAPL_daily.csv
▶ Ejecutando walk-forward (5 folds, test=auto)
  Fold 1: 2018-11-19 -> 2020-03-27 | MAE=0.1372 IC=0.319 Sharpe=0.61
  Fold 2: 2020-03-30 -> 2021-08-03 | MAE=0.2355 IC=0.766 Sharpe=2.49
  Fold 3: 2021-08-04 -> 2022-12-07 | MAE=0.2646 IC=0.108 Sharpe=-0.08
  Fold 4: 2022-12-08 -> 2024-04-17 | MAE=0.1014 IC=0.692 Sharpe=0.59
  Fold 5: 2024-04-18 -> 2025-08-26 | MAE=0.1249 IC=0.274 Sharpe=0.79
✓ AAPL: MAE=0.1727 IC=-0.254
```

### Opción 2: Volver a Split Simple

```yaml
# En src/config/base.yaml
splits:
  method: "time_split"  # Desactiva walk-forward
```

Luego ejecutar normalmente:
```bash
python -m src.train_e1_pipeline --tickers AAPL
```

## 📁 Archivos Generados

Cada ejecución walk-forward genera (en `runs/e1_conservative/<timestamp>/<ticker>/`):

### 1. `<ticker>_walkforward_folds.csv`

Métricas por fold individual:

| fold | window | ml_mae | ml_ic | bt_sharpe | bt_calmar | ... |
|------|--------|--------|-------|-----------|-----------|-----|
| 1 | 2018-11-19 -> 2020-03-27 | 0.137 | 0.319 | 0.61 | 1.07 | ... |
| 2 | 2020-03-30 -> 2021-08-03 | 0.236 | 0.766 | 2.49 | 9.16 | ... |
| ... | ... | ... | ... | ... | ... | ... |

**Columnas clave:**
- `test_start`, `test_end`: ventana temporal de cada fold
- `n_train`, `n_val`, `n_test`: tamaños de splits
- `ml_*`: métricas de machine learning (MAE, IC, directional accuracy)
- `bt_*`: métricas de trading (Sharpe, Calmar, max drawdown, profit factor)

### 2. `<ticker>_walkforward_predictions.csv`

Predicciones combinadas de todos los folds:

| timestamp | fold | y_true | y_pred |
|-----------|------|--------|--------|
| 2018-11-19 | 1 | 0.123 | 0.087 |
| 2020-03-30 | 2 | -0.045 | 0.012 |
| ... | ... | ... | ... |

### 3. `<ticker>_walkforward_backtest.csv`

Backtest completo (concatenación de todos los folds):

| timestamp | pos | signal | gross_ret | costs | net_ret | equity | turnover |
|-----------|-----|--------|-----------|-------|---------|--------|----------|
| 2018-11-19 | 1.0 | 1.0 | 0.0023 | 0.0005 | 0.0018 | 100180 | 1.0 |
| ... | ... | ... | ... | ... | ... | ... | ... |

### 4. `<ticker>_walkforward_metrics.png`

Gráfico de IC y Sharpe por fold:

![Walk-Forward Metrics Example](docs/walkforward_example.png)

- **Panel superior**: IC (Information Coefficient) por ventana
- **Panel inferior**: Sharpe ratio por ventana
- **Eje X**: etiquetas de ventanas temporales

### 5. `<ticker>_fold<N>_backtest.csv`

Backtest detallado de cada fold individual (5 archivos):
- `AAPL_fold1_backtest.csv`
- `AAPL_fold2_backtest.csv`
- ...

Útil para debugging de performance en ventanas específicas.

### 6. `summary_all.csv`

Resumen agregado con métricas combinadas:

```csv
ticker,n_samples,n_test,folds,ml_mae,ml_ic,bt_sharpe,bt_calmar,...
AAPL,2045,1700,5,0.173,-0.254,0.635,0.581,...
```

**Métricas agregadas:**
- `ml_ic`: IC calculado sobre predicciones concatenadas de todos los folds
- `bt_sharpe`: Sharpe del backtest completo (all folds combined)

## 📊 Interpretación de Resultados

### Métricas ML

**Information Coefficient (IC):**
- **IC > 0.05**: Capacidad predictiva positiva (benchmark mínimo)
- **IC > 0.10**: Buena capacidad predictiva
- **IC < 0**: Señal de overfitting o modelo sin capacidad predictiva

**Directional Accuracy:**
- **> 50%**: Modelo predice correctamente la dirección del movimiento
- **< 50%**: Modelo peor que random guess (preocupante)

### Métricas de Trading

**Sharpe Ratio:**
- **> 1.0**: Excelente estrategia risk-adjusted
- **0.5 - 1.0**: Buena estrategia
- **< 0.0**: Estrategia no rentable

**Calmar Ratio:**
- Retorno anualizado / Max Drawdown
- **> 1.0**: Estrategia con buen control de riesgo

**Profit Factor:**
- Ganancias brutas / Pérdidas brutas
- **> 1.5**: Estrategia robusta
- **1.0 - 1.5**: Aceptable
- **< 1.0**: Estrategia perdedora

### Ejemplo de Análisis

```
Fold 1: IC=0.319, Sharpe=0.61  → Modelo funciona, pero Sharpe moderado
Fold 2: IC=0.766, Sharpe=2.49  → Excelente performance (bull market 2020)
Fold 3: IC=0.108, Sharpe=-0.08 → Concept drift (bear market 2022)
Fold 4: IC=0.692, Sharpe=0.59  → Modelo recupera capacidad predictiva
Fold 5: IC=0.274, Sharpe=0.79  → Performance estable

Agregado: IC=-0.254, Sharpe=0.635
```

**Interpretación:**
- ✅ Modelo muestra capacidad predictiva en 4/5 folds (IC > 0.1)
- ⚠️  Fold 3 (2021-2022): degradación severa (bear market)
- ⚠️  IC agregado negativo sugiere que el modelo podría beneficiarse de:
  - Retraining periódico (drift detection)
  - Features adicionales para capturar regímenes de mercado
  - Ensemble de modelos para mayor robustez

## 🔍 Diferencias Técnicas Walk-Forward vs Split Simple

### Split Simple (`time_split`)

```python
# En train_e1_pipeline.py (sin walk-forward)
idx_train, idx_val, idx_test = time_split(len(X))  # 70/15/15
X_train, y_train = X[idx_train], y[idx_train]
X_test, y_test = X[idx_test], y[idx_test]

# Entrenar 1 modelo
model.fit(X_train, y_train, X_val, y_val)
y_pred = model.predict(X_test)

# 1 métrica IC en test único
ic = np.corrcoef(y_test, y_pred)[0, 1]
```

**Output:** 1 valor de IC, 1 backtest

### Walk-Forward (`walk_forward`)

```python
# En train_e1_pipeline.py (con walk-forward)
from sklearn.model_selection import TimeSeriesSplit

splitter = TimeSeriesSplit(n_splits=5, test_size=340, gap=90)

for fold, (train_idx, test_idx) in enumerate(splitter.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    
    # Entrenar modelo específico para este fold
    model = GRURegressor(...)
    model.fit(X_train, y_train, X_val, y_val)
    
    # Predecir en test fold
    y_pred = model.predict(X_test)
    
    # Métricas por fold
    ic_fold = np.corrcoef(y_test, y_pred)[0, 1]
    sharpe_fold = compute_sharpe(backtest(y_pred))
    
    print(f"Fold {fold}: IC={ic_fold:.3f} Sharpe={sharpe_fold:.2f}")

# Métricas agregadas (concatenando todos los folds)
ic_combined = compute_ic(all_y_true, all_y_pred)
```

**Output:** N valores de IC (1 por fold) + 1 IC agregado

## 🎓 Justificación Académica

Walk-forward validation es **gold standard** en investigación de trading porque:

1. **Replica producción real**: modelos se reentrenan con datos históricos crecientes
2. **Detecta concept drift**: identifica cuándo el modelo deja de funcionar
3. **Evita cherry-picking**: no permite seleccionar "la mejor ventana de test"
4. **Maximiza uso de datos**: valida en 100% de muestras (vs 15% en split simple)

### Referencias

- **Prado, M. L. (2018).** *Advances in Financial Machine Learning*. Wiley. Cap. 7: "Cross-Validation in Finance"
- **Aronson, D. (2006).** *Evidence-Based Technical Analysis*. Wiley. Cap. 9: "Walk-Forward Analysis"

## 🛠️ Troubleshooting

### Error: "No hay muestras disponibles para walk-forward"

**Causa:** Dataset muy pequeño después de eliminar NaNs y aplicar lookback.

**Solución:**
```yaml
# Reducir tamaño de test por fold
splits:
  test_size: 100  # En lugar de test_size automático
```

### Warning: "IC=nan en fold X"

**Causa:** Todas las predicciones o targets son constantes en ese fold (std = 0).

**Solución:**
- Verificar que el fold tenga suficiente variabilidad en retornos
- Puede ser señal de datos de mala calidad en esa ventana

### Error: "test_size <= gap_samples"

**Causa:** El tamaño de test es menor o igual que el embargo (gap).

**Solución:**
```yaml
splits:
  test_size: 200  # Aumentar manualmente
  embargo_days:
    e1: 60  # O reducir embargo (con precaución!)
```

## 📈 Próximos Pasos

1. **Optimizar hiperparámetros por fold** (opcional):
   - Buscar `tau_buy`, `tau_sell` óptimos para cada régimen
   - Ver [README_HYPERPARAMETER_TUNING.md](README_HYPERPARAMETER_TUNING.md)

2. **Detección de drift**:
   - Implementar monitoreo de IC por ventana deslizante
   - Retraining automático cuando IC < threshold

3. **Ensemble de modelos**:
   - Combinar predicciones de todos los folds con pesos adaptativos
   - Mejorar robustez ante concept drift

4. **Portfolio optimization**:
   - Aplicar walk-forward a múltiples tickers simultáneamente
   - Optimizar pesos de portfolio maximizando Sharpe global

---

**Documentado:** 2026-01-07  
**Autor:** Sebastian Carreras  
**Proyecto:** Trading Predict - FIUBA IA CEIA 18co
