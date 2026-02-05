# E1 Simple - Pipeline Simplificado

**Versión simplificada de E1** diseñada para desarrollo rápido y experimentación. 80% más rápido que E1 Conservative.

---

## 🚀 Quick Start

```bash
# Instalar dependencias
pip install -r requirements.txt

# Entrenar (descarga y limpia datos automáticamente)
python -m src.train_e1_simple_pipeline --tickers AAPL

# Ver resultados
cat runs/e1_simple/*/summary_all.csv
```

⏱️ **Tiempo:** ~1 minuto por ticker  
📊 **Output:** Predicciones, backtest, modelo, métricas

---

## � TL;DR - Cómo Funciona E1 Simple

### 1. Modelo
- **Arquitectura:** GRU 1 capa (64 unidades) + Dropout + Dense(16)
- **Entrada:** 180 días de datos (27 features: tendencia, momentum, volumen)
- **Salida:** Predicción de retorno a 90 días
- **Validación:** Time split simple 70/15/15 (train/val/test)

### 2. Predicciones y Backtest
1. **Entrenar GRU** en 70% de datos
2. **Predecir retornos** en 15% test
3. **Generar señales de trading** usando predicciones:
   - BUY si predicción ≥ tau_buy (defecto 0.06 = +6%)
   - SELL si predicción ≤ tau_sell (defecto 0.00)
4. **Ejecutar backtest** con esas señales → obtener Sharpe, CAGR, etc.


### 3. Métricas Calculadas
- **IC (Information Coefficient):** correlación Spearman predicción-realidad. Mayor = mejor.
- **Directional Accuracy:** % aciertos de signo. Mayor = mejor.
- **Sharpe:** retorno/riesgo del backtest. Mayor = mejor.
- **MAE/RMSE:** error de predicción. Menor = mejor.

#### ¿Cómo se complementan estas métricas?

Las métricas de predicción (MAE, RMSE, Directional Accuracy, IC) evalúan la calidad del modelo para anticipar retornos:
- **MAE/RMSE**: Miden el error promedio de las predicciones. Menor error implica que el modelo predice retornos más cercanos a la realidad, pero no garantiza que esas predicciones sean útiles para ganar dinero.
- **Directional Accuracy**: Indica qué porcentaje de veces el modelo acierta la dirección (sube/baja). Es útil, pero puede ser engañoso si el modelo acierta en movimientos pequeños y falla en los grandes.
- **Information Coefficient (IC)**: Mide la correlación entre predicción y realidad. Un IC alto sugiere que el modelo captura bien la señal, pero no necesariamente que la estrategia sea rentable después de costos.

Las métricas de trading (Sharpe, max drawdown) resumen el impacto final de todas las decisiones (aciertos, errores y costos) en el capital:
- **Sharpe**: Mide el retorno ajustado por riesgo. Un Sharpe mínimo asegura que la estrategia no solo gana, sino que lo hace de forma eficiente y consistente.
- **Max drawdown**: Mide la peor caída desde un pico de capital. Limitar el drawdown protege contra pérdidas grandes y prolongadas.

Por eso, aunque un modelo tenga buen MAE/RMSE/IC, si el Sharpe es bajo o el drawdown es alto, la estrategia no es viable. Idealmente, todas las métricas deberían ser buenas: las de predicción aseguran que el modelo tiene sentido, y las de trading que es útil en la práctica.

### 4. Targets (Umbrales de Referencia)
- IC ≥ 0.05
- Directional Accuracy ≥ 0.55
- Sharpe ≥ 1.0
- MAE ≤ 0.03
- RMSE ≤ 0.05



### 5. Resumen Operativo
```
Datos (180d) → GRU → Predicción retorno
                  ↓
            Backtest (tau_buy/sell)
                  ↓
            Calcular IC, DirAcc, Sharpe, MAE, RMSE
      ↓
    Resumen de métricas
```

Resultados guardados en `summary_all.csv` con métricas para comparación rápida.

---

## �📊 Simplificaciones vs E1 Conservative

| Aspecto | E1 Simple | E1 Conservative |
|---------|-----------|-----------------|
| **Tiempo/ticker** | ~1 min | ~5 min |
| **Validación** | Time split (70/15/15) | Walk-forward (5 folds) |
| **GRU** | 1 capa (64 units) | 2 capas (128→64) |
| **Complejidad** | Baja | Media-Alta |
| **Uso ideal** | Dev/testing/baseline | Producción |

### Arquitectura Simplificada

```
Input (180 días, 27 features)
  ↓
GRU(64 units)
  ↓
Dropout(0.2)
  ↓
Dense(16)
  ↓
Output(1) - Predicción retorno 90d
```

**Beneficios:**
- ✅ 50% menos parámetros que E1 Conservative
- ✅ 2x más rápido de entrenar
- ✅ Menos propenso a overfitting
- ✅ Suficiente para horizonte 90 días

---

## 💻 Uso

### CLI Options

```bash
# Modo completo (download + clean + train)
python -m src.train_e1_simple_pipeline --tickers AAPL,MSFT

# Todos los tickers del config
python -m src.train_e1_simple_pipeline

# Skip download (usar datos existentes)
python -m src.train_e1_simple_pipeline --tickers AAPL --skip-download

# Skip cleaning (usar raw data)
python -m src.train_e1_simple_pipeline --tickers AAPL --skip-cleaning

# Solo entrenar (reutilizar datos descargados)
python -m src.train_e1_simple_pipeline --tickers AAPL --skip-download --skip-cleaning
```

### Output Esperado

```
============================================================
Pipeline E1 Simple - 1 tickers
Output: runs/e1_simple/20260118_004042
============================================================

Paso 1/3: Descargando datos...
------------------------------------------------------------
✓ Descargados 2 archivos

Paso 2/3: Limpiando datos...
------------------------------------------------------------
✓ Limpiados: 2 | Rechazados: 0

Paso 3/3: Entrenando modelos...
------------------------------------------------------------
Benchmark: SPY (2515 días)

[1/1] AAPL
  ✓ Usando datos limpios: AAPL_daily.csv
  Samples: 2046 | Features: 20 | Lookback: 180d
  Split: train=1432 val=307 test=307
  Entrenando GRU [64]...
  ✓ Epochs: 34/100 | Val Loss: 0.194798
  ✓ MAE=0.1628 RMSE=0.1862 IC=0.273 Dir=50.8% Sharpe=0.68
  ✓ Modelo guardado: AAPL_model.pth

============================================================
✓ Completado: 1/1 tickers
  Resultados en: runs/e1_simple/20260118_004042/
============================================================
```

---

## 📁 Estructura de Resultados

```
runs/e1_simple/20260118_004042/
├── config_used.yaml           # Config reproducible
├── summary_all.csv             # Resumen de todos los tickers
│
├── AAPL/
│   ├── AAPL_predictions.csv   # (timestamp, y_true, y_pred)
│   ├── AAPL_backtest.csv      # Serie PnL: pos, signal, equity, costs
│   ├── AAPL_scaler.csv        # Normalización features (mean/std)
│   ├── AAPL_model.pth         # Modelo PyTorch guardado
│   └── AAPL_summary.csv       # Métricas completas
│
└── MSFT/
    └── ...
```

### Archivos Clave

**`summary_all.csv`** - Métricas agregadas:
- Datos: `ticker`, `n_samples`, `n_train`, `n_val`, `n_test`
- ML: `ml_mae`, `ml_rmse`, `ml_ic`, `ml_directional_accuracy`
- Backtest: `bt_sharpe`, `bt_cagr`, `bt_max_drawdown`, `bt_num_trades`

**`{ticker}_predictions.csv`**:
```csv
timestamp,y_true,y_pred
2024-01-15,-0.0234,0.0156
2024-01-16,0.0421,0.0389
```

**`{ticker}_backtest.csv`** - Serie temporal:
- Columnas: `pos`, `signal`, `gross_ret`, `costs`, `net_ret`, `equity`, `turnover`

---

## ⚙️ Configuración (base.yaml)

```yaml
strategies:
  e1_simple:
    lookback_days: 180        # Ventana entrada (6 meses)
    horizon_days: 90          # Target: retorno a 90 días
    
    thresholds:
      tau_buy: 0.06           # +6% predicho → señal compra
      tau_sell: 0.00          # 0% o menos → cerrar posición
    
    model:
      gru_units: [64]         # 1 capa GRU
      dropout: 0.2
      dense_units: 16
      learning_rate: 0.001
      batch_size: 64
      max_epochs: 100
      early_stopping_patience: 10
      loss: "huber"
      huber_delta: 1.0
    
    backtest:
      holding_period_days: 90   # Holding period = horizon
      allow_short: false
      max_position: 1.0

costs:
  daily_round_trip_bps: 10    # 10 bps = 0.1% costos transacción
```

---

## 🔧 Personalización

### 1. Modificar arquitectura GRU

Para modelo más grande:
```yaml
strategies:
  e1_simple:
    model:
      gru_units: [128]        # GRU más grande
      dense_units: 32
      dropout: 0.3            # Más regularización
```

### 2. Ajustar thresholds de señales

Para cambiar la sensibilidad de entradas/salidas:
```yaml
strategies:
  e1_simple:
    thresholds:
      tau_buy: 0.06
      tau_sell: 0.00
```

### 3. Ajustar costos de transacción

Para escenarios más conservadores:
```yaml
costs:
  daily_round_trip_bps: 20
```

---

## 📊 Features (27 indicadores)

**Precio y retorno (9):**
- `ret_1d`, `ret_5d`, `ret_20d`, `ret_60d` - Retornos multihorizonte
- `vol_20d`, `vol_60d` - Volatilidad realizada
- `range_pct`, `atr_14`, `vol_zscore_60`

**Tendencia (14):**
- `sma_50`, `sma_200`, `ema_50` - Medias móviles
- `close_sma200_dist` - Distancia a SMA(200)
- `macd_line`, `macd_signal`, `macd_hist` - MACD
- `bb_pct_b`, `bb_bandwidth` - Bollinger Bands
- `adx_14` - Fuerza de tendencia
- `rsi_14` - Momentum
- `stoch_k`, `stoch_d` - Stochastic

**Contexto benchmark (2):**
- `bench_ret_1d`, `bench_ret_20d` - Retornos SPY

**Volumen (2):**
- `volume`, `volume_sma20_ratio`

---

## 🔬 Airflow DAG

El DAG `e1_simple_pipeline` ejecuta el pipeline completo:

### Tareas

1. **download_daily_data** - Descarga OHLCV diario
2. **clean_daily_data** - Limpia y valida datos
3. **train_e1_simple_models** - Entrena modelos + MLflow logging
4. **notify_api** - Notifica a FastAPI

### Trigger desde CLI

```bash
# Un ticker
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_simple_pipeline \
  --conf '{"tickers": "AAPL", "skip_download": "False", "skip_cleaning": "False"}'

# Reutilizando datos de E1 Conservative
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_simple_pipeline \
  --conf '{"tickers": "AAPL,MSFT", "skip_download": "True", "skip_cleaning": "True"}'

# Todos los tickers
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_simple_pipeline \
  --conf '{"tickers": ""}'
```

### Parámetros DAG

- `tickers`: Lista separada por comas (vacío = todos del config)
- `skip_download`: "True" = no descarga, reutiliza data/raw/daily
- `skip_cleaning`: "True" = no limpia, reutiliza data/clean

### MLflow Tracking

- **Experiment:** "E1_Simple_Strategy"
- **Métricas por ticker:** mae, rmse, ic, directional_accuracy, bt_sharpe, bt_cagr, bt_max_drawdown
- **Parámetros:** strategy, ticker, lookback_days, horizon_days, gru_units, etc.
- **Artifacts:** models/, predictions/, backtest/
- **Run summary:** summary_all.csv + métricas agregadas (IC mean/median/min/max)

**Ver en UI:**
- MLflow: http://localhost:5000 → Experiment "E1_Simple_Strategy"
- Airflow: http://localhost:8080 → DAG "e1_simple_pipeline"

---

## 🎯 Casos de Uso

### ✅ Cuándo usar E1 Simple

1. **Desarrollo inicial** - Probar features rápidamente
2. **Experimentación** - Iterar hiperparámetros
3. **Baseline** - Punto de comparación
4. **Debugging** - Verificar pipeline funciona
5. **Aprendizaje** - Entender flujo sin complejidad

### ❌ Cuándo usar E1 Conservative

1. **Producción** - Validación robusta con walk-forward
2. **Paper académico** - Métricas rigurosas
3. **Trading real** - Decisiones con dinero
4. **Portafolio final** - Versión optimizada

---

## 🔄 Workflow Recomendado

```
1. Desarrollo con E1 Simple
   ↓
   Itera rápido (arquitectura, features, targets)
   ↓
2. Validación con E1 Conservative
   ↓
   Walk-forward confirma robustez
   ↓
3. Optimización (Optuna)
   ↓
   Tune hyperparams por ticker
   ↓
4. Producción con E1 Conservative + tuned params
```

---

## 📚 Referencias

- [README_E1.md](README_E1.md) - E1 Conservative completo
- [README_AIRFLOW_USAGE.md](README_AIRFLOW_USAGE.md) - Uso de DAGs
- [src/train_e1_simple_pipeline.py](src/train_e1_simple_pipeline.py) - Código fuente
- [src/config/base.yaml](src/config/base.yaml) - Configuración

---

## 💡 Tips

1. **Empieza con E1 Simple** - Itera rápido
2. **Compara métricas clave** - Simple vs Conservative deberían ser consistentes
3. **Analiza summary_all.csv** - Distribución de métricas entre tickers
4. **Revisa backtest CSV** - Equity curve y drawdowns
5. **Skip download/clean** - Para experimentos rápidos

---

**Última actualización:** Febrero 2, 2026  
**Versión:** 2.1 (removido Decision Score en E1 Simple)
