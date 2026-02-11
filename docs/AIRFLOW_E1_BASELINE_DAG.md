# DAG E1 Baseline - Regresión Lineal

DAG de Airflow para ejecutar automáticamente el baseline de Regresión Lineal del modelo E1 y compararlo con el GRU.

## 📋 Descripción

Este DAG ejecuta el pipeline completo del baseline E1:
1. Descarga datos diarios (compartido con GRU)
2. Limpia y valida datos
3. Entrena modelo de Regresión Lineal
4. Realiza backtesting con mismos parámetros que GRU
5. Compara automáticamente con último run de GRU (opcional)

## ⏰ Schedule

- **Frecuencia**: Lunes a las 3:00 AM (1 hora después del GRU)
- **Catchup**: False (no ejecuta runs históricos)
- **Tags**: `trading`, `e1`, `baseline`, `linear_regression`, `comparison`

## 🎯 Parámetros del DAG

### `tickers` (string)
Lista de tickers separados por comas para entrenar.

**Ejemplo**:
```
YPFD.BA, GGAL.BA, PAMP.BA, BYMA.BA, CEPU.BA, AAPL, MSFT
```

**Default**: Vacío (usa todos los tickers E1 del config)

### `auto_compare_with_gru` (boolean)
Habilita comparación automática con el último run del GRU.

**Valores**: `True` / `False`  
**Default**: `True`

## 🔄 Flujo de Tareas

```
download_daily_data
        ↓
clean_daily_data
        ↓
train_baseline_models
        ↓
compare_with_gru
```

### Task 1: `download_daily_data`
- Descarga datos OHLCV diarios desde Yahoo Finance
- Skip si datos frescos (< 1 día)
- Output: `data/raw/daily/`

### Task 2: `clean_daily_data`
- Valida y limpia datos descargados
- Forward fill para valores faltantes
- Elimina días con volumen = 0
- Output: `data/clean/`

### Task 3: `train_baseline_models`
- Entrena Regresión Lineal para cada ticker
- Walk-forward validation (5 folds)
- Calcula métricas ML: MAE, RMSE, IC, Directional Accuracy
- Ejecuta backtesting con costos
- Output: `runs/e1_baseline/<timestamp>/`

### Task 4: `compare_with_gru`
- Carga último run de GRU
- Compara métricas ML y Trading
- Genera reporte comparativo
- Output: `runs/e1_baseline/<timestamp>/comparison_vs_gru.csv`

## 📊 Métricas Logueadas en MLflow

### Por Ticker
- **ML Metrics**: `mae`, `rmse`, `ic`, `directional_accuracy`
- **Trading Metrics**: `bt_sharpe`, `bt_sortino`, `bt_cagr`, `bt_max_drawdown`, `bt_calmar`, `bt_profit_factor`, `bt_win_rate`, `bt_num_trades`

### Agregadas (Summary Run)
- `total_tickers`: Total de tickers procesados
- `successful_tickers`: Tickers entrenados exitosamente
- `ic_mean`, `ic_median`: Estadísticas de IC
- `ic_positive_count`: Tickers con IC > 0
- `ic_above_threshold`: Tickers con IC > 0.05
- `mae_mean`, `mae_median`: Estadísticas de MAE

### Comparación (si habilitada)
- `baseline_ml_mae`, `gru_ml_mae`, `diff_ml_mae`
- `baseline_ml_ic`, `gru_ml_ic`, `diff_ml_ic`
- Similar para todas las métricas ML

## 📁 Outputs

```
runs/e1_baseline/<timestamp>/
├── config_used.yaml                     # Config completa usada
├── baseline_summary_all.csv             # Resumen todos los tickers
├── comparison_vs_gru.csv                # Comparación con GRU (si habilitada)
└── <TICKER>/
    ├── <TICKER>_baseline_predictions.csv   # Predicciones por fold
    ├── <TICKER>_baseline_folds.csv         # Métricas por fold
    ├── <TICKER>_baseline_backtest.csv      # Serie temporal backtest
    └── <TICKER>_baseline_summary.csv       # Resumen del ticker
```

## 🚀 Uso

### Desde Airflow UI

1. Navegar a: http://localhost:8080
2. Buscar DAG: `e1_baseline_linear_regression`
3. Click en "Trigger DAG"
4. (Opcional) Configurar parámetros:
   - `tickers`: Lista personalizada
   - `auto_compare_with_gru`: True/False
5. Click "Trigger"

### Desde CLI

```bash
# Trigger con parámetros default
docker exec -it airflow-webserver airflow dags trigger e1_baseline_linear_regression

# Trigger con tickers específicos
docker exec -it airflow-webserver airflow dags trigger e1_baseline_linear_regression \
  --conf '{"tickers": "AAPL, MSFT, GOOGL"}'

# Trigger sin comparación automática
docker exec -it airflow-webserver airflow dags trigger e1_baseline_linear_regression \
  --conf '{"auto_compare_with_gru": "False"}'
```

### Programáticamente (API)

```bash
curl -X POST "http://localhost:8080/api/v1/dags/e1_baseline_linear_regression/dagRuns" \
  -H "Content-Type: application/json" \
  -u "airflow:airflow" \
  -d '{
    "conf": {
      "tickers": "AAPL, MSFT",
      "auto_compare_with_gru": "True"
    }
  }'
```

## 🔍 Monitoreo

### Logs en Airflow
- Ver logs de cada task en Airflow UI → DAG → Task → Logs

### Métricas en MLflow
- Navegar a: http://localhost:5050
- Buscar experimento: `E1_Baseline_LinearRegression`
- Filtrar runs por fecha/ticker
- Comparar métricas en gráficos

### Archivos de Output
```bash
# Ver último run
ls -lht runs/e1_baseline/ | head -n 2

# Ver summary
cat runs/e1_baseline/<timestamp>/baseline_summary_all.csv

# Ver comparación con GRU
cat runs/e1_baseline/<timestamp>/comparison_vs_gru.csv
```

## ⚙️ Configuración

### Variables de Entorno
- `MLFLOW_TRACKING_URI`: URL de MLflow (default: `http://mlflow:5000`)

### Config Base
El DAG usa `src/config/base.yaml` para:
- Tickers por estrategia (`universe.tickers_by_strategy.e1_conservative`)
- Parámetros de modelo (lookback, horizon, thresholds)
- Costos de transacción

## 🔗 Relación con Otros DAGs

### `e1_conservative_pipeline` (GRU)
- **Schedule**: Lunes 2:00 AM (1 hora antes)
- **Relación**: El baseline puede comparar con su output
- **Datos**: Comparten descarga y limpieza

### Ejecución Simultánea
```bash
# Ejecutar GRU primero
docker exec -it airflow-webserver airflow dags trigger e1_conservative_pipeline

# Esperar ~30 min, luego ejecutar baseline
docker exec -it airflow-webserver airflow dags trigger e1_baseline_linear_regression
```

## 📈 Interpretación de Resultados

### IC (Information Coefficient)
- **IC > 0.05**: Significativo en finanzas
- **IC > 0**: Capacidad predictiva positiva
- **IC < 0**: Modelo sin valor o overfitting

### Comparación Baseline vs GRU
- **GRU >> Baseline**: Arquitectura recurrente añade valor
- **GRU ≈ Baseline**: Considerar simplificar modelo
- **Baseline > GRU**: Revisar overfitting o hiperparámetros

### Métricas de Trading
- **Sharpe > 0.9**: Objetivo E1 conservador
- **Max DD < 15%**: Límite de riesgo
- **Hit Rate > 55%**: Indicador de consistencia

## 🐛 Troubleshooting

### Error: "No tickers to train"
- Verificar `universe.tickers_by_strategy.e1_conservative` en config
- O pasar parámetro `tickers` explícitamente

### Error: "No se encontró summary del GRU"
- Ejecutar primero `e1_conservative_pipeline`
- O deshabilitar comparación: `auto_compare_with_gru=False`

### Error: Import errors
- Verificar que `src/` esté montado en `/opt/airflow` del contenedor
- Verificar dependencias en `requirements.txt`

## 📚 Referencias

- [README_E1_BASELINE.md](../../../README_E1_BASELINE.md) - Documentación del baseline
- [QUICKSTART_E1_BASELINE.md](../../../QUICKSTART_E1_BASELINE.md) - Guía rápida
- [README_AIRFLOW_USAGE.md](../../../README_AIRFLOW_USAGE.md) - Uso general de Airflow
- [README_E1.md](../../../README_E1.md) - Documentación del modelo GRU
