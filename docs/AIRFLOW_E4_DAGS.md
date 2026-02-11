# DAGs de Airflow para Estrategia E4 - Pairs Trading

## Resumen

Se han implementado **2 DAGs** para la estrategia E4 de pairs trading:

1. **`e4_pairs_trading_pipeline`**: Pipeline principal semanal
2. **`e4_monthly_recalibration`**: Re-calibración mensual de parámetros

---

## DAG 1: e4_pairs_trading_pipeline

### Descripción
Pipeline completo de pairs trading con cointegración, modelo Ornstein-Uhlenbeck y k-NN opcional.

### Schedule
- **Frecuencia**: Semanal (Lunes 4:00 AM)
- **Timezone**: UTC
- **Catchup**: Deshabilitado

### Flujo de Tareas

```
download_pairs_data
    ↓
clean_pairs_data
    ↓
validate_cointegration
    ↓
process_pairs_with_mlflow
    ↓
generate_report
```

### Tareas

#### 1. `download_pairs_data`
- **Función**: Descargar datos OHLCV para todos los activos en los pares
- **Período**: 10 años (histórico largo para cointegración)
- **Skip existing**: Sí (solo actualiza archivos obsoletos)

#### 2. `clean_pairs_data`
- **Función**: Limpiar datos (forward fill, remove zero volume)
- **Estrategia**: Forward fill (conservadora)
- **Mínimo**: 252 días (1 año)
- **Output**: `data/clean/`

#### 3. `validate_cointegration`
- **Función**: Validar cointegración de cada par candidato
- **Test**: Engle-Granger
- **Umbral**: p-value < 0.05
- **Output**: CSV con resultados de validación
- **XCom**: Lista de pares válidos → tarea siguiente

#### 4. `process_pairs_with_mlflow`
- **Función**: Procesar cada par válido y registrar en MLflow
- **Incluye**:
  - Construcción de spread con hedge ratio rolling
  - Estimación de parámetros OU (θ, μ, σ, half-life)
  - Entrenamiento k-NN (opcional)
  - Generación de señales basadas en z-score
  - Backtest con gestión dollar-neutral
  - Registro de métricas en MLflow
- **Output**: 
  - `runs/e4_pairs/<timestamp>/`
  - Por cada par: spread, OU params, trades, summary

#### 5. `generate_report`
- **Función**: Consolidar resultados y generar reporte
- **Output**: `reports/e4_pairs/pipeline_summary_<timestamp>.txt`

### Parámetros del DAG (Configurables desde UI)

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `pairs` | string | '' | Pares específicos (formato: "GGAL.BA,BMA.BA;KO,PEP"). Vacío = usar config |
| `use_knn` | bool | True | Activar confirmación k-NN |
| `entry_z` | float | 2.0 | Umbral de z-score para entrada |
| `exit_z` | float | 0.25 | Umbral de z-score para salida |
| `stop_z` | float | 3.0 | Umbral de z-score para stop-loss |
| `output_tag` | string | '' | Tag custom para output (vacío = timestamp) |

### Métricas en MLflow

**Por par**:
- `total_return`: Retorno total
- `cagr`: CAGR anualizado
- `sharpe_ratio`: Sharpe ratio
- `max_drawdown`: Máximo drawdown
- `num_trades`: Número de operaciones
- `win_rate`: Tasa de acierto
- `avg_holding_days`: Días promedio por trade
- `net_exposure_mean`: Exposición neta promedio (debe ser ~0)
- `cointegration_pvalue`: P-value de cointegración
- `ou_half_life`: Half-life del spread (días)

**Agregado**:
- `pairs_processed`: Número de pares procesados
- `avg_sharpe`: Sharpe promedio
- `avg_return`: Retorno promedio
- `avg_win_rate`: Win rate promedio
- `total_trades`: Trades totales

### Artifacts en MLflow

- `pairs/<pair_name>/spread_timeseries.csv`
- `pairs/<pair_name>/ou_params.json`
- `pairs/<pair_name>/trades.csv`
- `pairs/<pair_name>/backtest_summary.json`
- `summary_all_pairs.csv` (agregado)

### Ejemplo de Uso

**Desde UI de Airflow**:
1. Ir a DAG `e4_pairs_trading_pipeline`
2. Click en "Trigger DAG w/ config"
3. Configurar parámetros:
   ```json
   {
     "pairs": "GGAL.BA,BMA.BA;YPFD.BA,PAMP.BA",
     "use_knn": "True",
     "entry_z": "2.5",
     "exit_z": "0.2",
     "stop_z": "3.0"
   }
   ```
4. Click "Trigger"

**Desde CLI**:
```bash
airflow dags trigger e4_pairs_trading_pipeline \
  --conf '{"pairs": "GGAL.BA,BMA.BA", "use_knn": "True"}'
```

---

## DAG 2: e4_monthly_recalibration

### Descripción
Re-calibración mensual de parámetros OU y validación de cointegración para detectar breakdown de relaciones.

### Schedule
- **Frecuencia**: Mensual (Día 15 a las 5:00 AM)
- **Timezone**: UTC
- **Catchup**: Deshabilitado

### Flujo de Tareas

```
download_latest_data
    ↓
clean_latest_data
    ↓
recalibrate_ou_parameters
    ↓
send_recalibration_alert
```

### Tareas

#### 1. `download_latest_data`
- **Función**: Descargar últimos 5 años de datos
- **Skip existing**: No (forzar actualización)

#### 2. `clean_latest_data`
- **Función**: Limpiar datos actualizados

#### 3. `recalibrate_ou_parameters`
- **Función**: Re-validar cointegración y re-estimar parámetros OU
- **Detecta**:
  - Breakdown de cointegración (p-value > 0.05)
  - Half-life demasiado largo (> 30 días)
  - No-estacionariedad del spread
- **Output**:
  - `runs/e4_pairs/recalibration/recalibration_<date>.csv`
  - `runs/e4_pairs/recalibration/pairs_status_current.json`
- **Clasificación**:
  - **ACTIVE**: Par sigue cointegrado y estable
  - **PAUSED**: Breakdown detectado, pausar trading
  - **ERROR**: Error en procesamiento

#### 4. `send_recalibration_alert`
- **Función**: Alertar si > 30% de pares tienen breakdown
- **Output**: Log de alerta

### Métricas en MLflow

- `active_pairs`: Número de pares activos
- `paused_pairs`: Número de pares pausados
- `breakdown_rate`: Tasa de breakdown
- `avg_half_life`: Half-life promedio (pares activos)
- `avg_coint_pvalue`: P-value promedio (pares activos)

### Uso del Status de Pares

El archivo `pairs_status_current.json` contiene:
```json
{
  "last_update": "2026-01-15 05:30:00",
  "active_pairs": ["GGAL.BA-BMA.BA", "KO-PEP"],
  "paused_pairs": ["YPFD.BA-PAMP.BA"],
  "active_count": 2,
  "paused_count": 1
}
```

**Integración con trading en vivo**:
- Leer este archivo antes de generar señales
- Solo operar pares en `active_pairs`
- Ignorar pares en `paused_pairs`

---

## Integración MLflow

### Experiments

- **E4_Pairs_Trading_Strategy**: Pipeline principal
- **E4_Monthly_Recalibration**: Re-calibración mensual

### Viewing Results

**MLflow UI**:
```bash
# Acceder a http://localhost:5050
# Navegar a experiment "E4_Pairs_Trading_Strategy"
# Filtrar por métricas (e.g., sharpe_ratio > 1.0)
```

**CLI**:
```bash
# Listar runs del experiment E4
mlflow runs list --experiment-name "E4_Pairs_Trading_Strategy"

# Descargar artifacts de un run
mlflow artifacts download --run-id <run_id> --dst-path ./e4_results
```

---

## Comparación con E1/E2

| Aspecto | E1 (GRU) | E2 (LSTM) | E4 (Pairs) |
|---------|----------|-----------|------------|
| **Schedule** | Lunes 2 AM | Lunes 3 AM | Lunes 4 AM |
| **Frecuencia** | Semanal | Semanal | Semanal + Mensual |
| **Datos** | 10 años | 10 años | 10 años |
| **Modelo** | Deep Learning | Deep Learning | Estadístico |
| **Entrenamiento** | Epochs + backprop | Epochs + backprop | MLE + k-NN |
| **Validación** | Cointegración ❌ | Cointegración ❌ | Cointegración ✅ |
| **Re-calibración** | No | No | Sí (mensual) |
| **Artifacts** | Modelo .keras | Modelo .keras | Spreads + OU params |

---

## Monitoreo y Alertas

### Métricas Clave a Monitorear

1. **Breakdown Rate** (mensual):
   - Normal: < 20%
   - Alerta: 20-30%
   - Crítico: > 30%

2. **Sharpe Promedio** (semanal):
   - Objetivo: > 1.0
   - Mínimo aceptable: > 0.5

3. **Cointegración**:
   - P-value promedio: < 0.03 (ideal)
   - Half-life promedio: 10-15 días (ideal)

### Alertas Automáticas

- Alta tasa de breakdown (> 30%) → Email/Slack
- Sharpe agregado < 0.5 → Revisar estrategia
- Pares con half-life > 30 días → Pausar automáticamente

---

## Troubleshooting

### Error: "No cointegrated pairs found"

**Causa**: Ningún par pasó el test de cointegración (p-value > 0.05)

**Solución**:
1. Verificar calidad de datos (`data/clean/`)
2. Revisar período de datos (mínimo 1 año)
3. Considerar pares alternativos
4. Revisar correlaciones preliminares

### Error: "Insufficient data points"

**Causa**: < 252 días de datos limpios

**Solución**:
1. Extender período de descarga
2. Revisar limpieza de datos (muchos NaNs removidos?)
3. Ajustar `min_days` en config si es necesario

### Warning: "Half-life too long"

**Causa**: Half-life > 20-30 días (reversión muy lenta)

**Solución**:
1. Es solo warning, par se procesa igual
2. Considerar aumentar `time_stop_days` para ese par
3. O pausar el par si half-life > 30 días

---

## Best Practices

1. **Run inicial**:
   - Usar `output_tag` descriptivo (e.g., "production_2026_01")
   - Revisar resultados manualmente antes de automatizar

2. **Parámetros**:
   - Empezar con defaults (entry_z=2.0, exit_z=0.25)
   - Ajustar solo si backtests muestran necesidad
   - No over-optimize (riesgo de overfitting)

3. **Monitoreo**:
   - Revisar reporte mensual de re-calibración
   - Pausar pares con breakdown persistente
   - Comparar Sharpe real vs backtested (slippage)

4. **MLflow**:
   - Usar tags para experimentos (e.g., "production", "test")
   - Registrar modelos k-NN con mejores resultados
   - Comparar runs con diferentes umbrales

---

## Próximos Pasos

1. **Ejecutar DAG** `e4_pairs_trading_pipeline` con pares de config
2. **Revisar resultados** en MLflow y `summary_all_pairs.csv`
3. **Configurar alertas** en Airflow para breakdown > 30%
4. **Integrar con FastAPI** para servir señales en tiempo real
5. **Comparar E1 vs E2 vs E4** en términos de Sharpe y drawdown

---

## Referencias

- [README_E4.md](../../../README_E4.md) - Documentación completa de la estrategia
- [train_e4_pipeline.py](../../../src/train_e4_pipeline.py) - Pipeline principal
- [base.yaml](../../../src/config/base.yaml) - Configuración de pares
