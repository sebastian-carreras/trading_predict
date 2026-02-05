# E2 MLflow Target Metrics Enhancement

## Resumen de Cambios

Se agregó logging de **targets/valores mínimos** como parámetros en el Summary Run de MLflow para facilitar la comparación entre las métricas reales y los objetivos esperados.

## Motivación

Cuando se ejecuta el pipeline de E2 (Simple o Moderate) y se guardan las métricas agregadas en el Summary Run, es útil tener una referencia de los valores mínimos/objetivo esperados para cada métrica. Esto permite:

1. **Comparación rápida**: Ver si las métricas agregadas cumplen con los objetivos
2. **Trazabilidad**: Los targets quedan registrados en MLflow junto con las métricas
3. **Análisis histórico**: Comparar runs pasados contra los mismos targets

## Implementación

### Targets Definidos

```python
default_targets = {
    'ic_min': 0.05,                    # IC mínimo esperado
    'directional_accuracy_min': 0.55,   # Precisión direccional mínima
    'sharpe_min': 1.0,                  # Sharpe Ratio mínimo
    'mae_max': 0.03,                    # MAE máximo permitido
    'rmse_max': 0.05,                   # RMSE máximo permitido
}
```

Estos valores se combinan con los definidos en `config["decision"]["targets"]` si existen.

### Métricas Logueadas como Params

En el **Summary Run** de MLflow, ahora se loguean los siguientes targets como **parámetros**:

1. **`ic_target_min`**: Valor mínimo esperado de IC (0.05)
2. **`sharpe_target_min`**: Valor mínimo esperado de Sharpe Ratio (1.0)

### Ejemplo de Uso en MLflow

Cuando revisas el Summary Run en la UI de MLflow, ahora verás:

**Metrics**:
- `ic_mean`: 0.191
- `ic_median`: 0.234
- `ic_min`: 0.092
- `ic_max`: 0.283
- `ic_above_threshold`: 4

**Params**:
- `ic_target_min`: 0.05 ⭐

Esto te permite saber inmediatamente que el IC promedio (0.191) está **muy por encima** del mínimo esperado (0.05).

### Métricas Agregadas Adicionales

También se agregaron métricas agregadas de **Sharpe**:

**Sharpe Metrics**:
- `sharpe_mean`, `sharpe_median`, `sharpe_min`, `sharpe_max`
- `sharpe_above_threshold`: Cantidad de tickers con Sharpe > 1.0
- `sharpe_target_min` (param): 1.0

## Archivos Modificados

1. **`dockerfiles/airflow/dags/E2/e2_simple_pipeline.py`**:
  - Agregado cálculo de `sharpe_values`
  - Agregado logging de targets como params
  - Agregadas métricas agregadas de Sharpe

2. **`dockerfiles/airflow/dags/E2/e2_moderate_pipeline.py`**:
   - Mismos cambios que E2 Simple para mantener consistencia

3. **`README_E2_SIMPLE.md`**:
   - Nueva sección "MLflow Tracking" explicando targets

## Validación

```bash
# Validar sintaxis de ambos DAGs
python -m py_compile dockerfiles/airflow/dags/E2/e2_simple_pipeline.py
python -m py_compile dockerfiles/airflow/dags/E2/e2_moderate_pipeline.py

```

**Output esperado**:
```
✓ IC Metrics:
  ic_mean: 0.191
  → ic_target_min (param): 0.05 ⭐

✓ Sharpe Metrics:
  sharpe_mean: 1.400
  → sharpe_target_min (param): 1.0 ⭐

```

## Beneficios

### Para el Usuario
- **Claridad**: Sabe inmediatamente si las métricas cumplen con los objetivos
- **Decisiones**: Puede decidir si un run es "bueno" basándose en targets objetivos
- **Comparación**: Puede comparar runs contra el mismo baseline

### Para el Sistema
- **Trazabilidad**: Los targets quedan registrados en MLflow
- **Reproducibilidad**: Los targets usados para evaluar cada run están documentados
- **Consistencia**: Mismo patrón de logging en E2 Simple y E2 Moderate

## Próximos Pasos (Opcional)

1. **Alertas**: Configurar alertas si métricas agregadas caen bajo targets
2. **Dashboard**: Crear dashboard Grafana mostrando métricas vs targets
3. **Auto-tuning**: Usar targets para early stopping en optimización de hiperparámetros
4. **Targets dinámicos**: Calcular targets basados en performance histórica

## Referencias

- [E2 Simple README](README_E2_SIMPLE.md)
- [E2 Moderate README](README_E2.md)
- [MLflow Tracking Guide](https://mlflow.org/docs/latest/tracking.html)
