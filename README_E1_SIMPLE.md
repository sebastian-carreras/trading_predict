# E1 Simple - Pipeline Simplificado

**Versión simplificada de E1** para desarrollo rápido y experimentación. ~80% más rápido que E1 Conservative.

---

## Quick Start

```bash
# Instalar dependencias
pip install -r requirements.txt

# Entrenar E1 completo (baseline + simple + conservador)
python -m src.train_e1_all
python -m src.train_e1_all --tickers AAPL,MSFT

# Al finalizar, ejecuta automáticamente la comparación de versiones E1
# y guarda: reports/tables/e1_versions_comparison.csv y .md

# Entrenar (descarga y limpia datos automáticamente)
python -m src.train_e1_simple_pipeline 
python -m src.train_e1_simple_pipeline --tickers AAPL

# Ver resultados
cat runs/e1_simple/*/summary_all.csv
```

**Output:** Predicciones, backtest, modelo, métricas

---

## TL;DR - Cómo Funciona E1 Simple

### 1. Modelo
- **Arquitectura:** GRU 1 capa (64 unidades) + Dropout(0.2) + Dense(16)
- **Entrada:** 360 días de datos (~15 features optimizadas para largo plazo)
- **Salida:** Predicción de retorno logarítmico a 90 días
- **Validación:** Time split simple 70/15/15 (train/val/test)

### 2. Pipeline de Entrenamiento
1. **Cargar datos** - CSV de `data/clean/` (o `data/raw/daily/`)
2. **Calcular features** - 15 indicadores técnicos optimizados
3. **Crear secuencias** - Ventanas de 360 días para RNN
4. **Crear target** - Retorno logarítmico forward a 90 días
5. **Split temporal** - 70% train, 15% val, 15% test
6. **Estandarizar** - Z-score (fit solo en train, aplicar a val/test)
7. **Entrenar GRU** - Con early stopping en validación
8. **Evaluar** - Métricas ML + backtest con costos

### 3. Generación de Señales
```
Predicción → Comparar con umbrales → Señal de trading

Si predicción ≥ tau_buy (0.06 = +6%)  → BUY
Si predicción ≤ tau_sell (0.00)       → SELL/HOLD
```

### 4. Métricas Calculadas

**Métricas ML (calidad de predicción):**
- **MAE:** Error absoluto promedio (menor = mejor)
- **RMSE:** Error cuadrático medio (menor = mejor)
- **IC:** Information Coefficient - correlación Spearman (mayor = mejor, >0.05 significativo)
- **Directional Accuracy:** % aciertos de dirección (mayor = mejor, >55% útil)

**Métricas de Trading (backtest):**
- **Sharpe:** Retorno ajustado por riesgo (>1.0 bueno, >1.5 excelente)
- **CAGR:** Retorno anualizado compuesto
- **Max Drawdown:** Máxima caída desde pico (menor = mejor, <20% ideal)

### 5. Targets de Referencia
| Métrica | Umbral |
|---------|--------|
| IC | ≥ 0.05 |
| Directional Accuracy | ≥ 55% |
| Sharpe | ≥ 1.0 |
| MAE | ≤ 0.03 |
| RMSE | ≤ 0.05 |

### 6. Flujo Resumido
```
Datos OHLCV (10 años)
    ↓
Features (15 indicadores)
    ↓
Secuencias (360 días × 15 features)
    ↓
GRU [64] → Dense [16] → Output [1]
    ↓
Predicción retorno 90d
    ↓
Señal (BUY/HOLD/SELL)
    ↓
Backtest → Métricas
```

---

## Comparación: E1 Simple vs E1 Conservative

| Aspecto | E1 Simple | E1 Conservative |
|---------|-----------|-----------------|
| **Tiempo/ticker** | ~1 min | ~5 min |
| **Validación** | Time split (70/15/15) | Walk-forward (5 folds) |
| **Arquitectura GRU** | 1 capa (64 units) | 2 capas (128→64) |
| **Parámetros modelo** | ~8K | ~16K |
| **Uso ideal** | Desarrollo, experimentación | Producción, validación robusta |

### Arquitectura GRU

```
Input (360 días, ~15 features)
  ↓
GRU(64 units)
  ↓
Dropout(0.2)
  ↓
Dense(16, relu)
  ↓
Output(1) - Predicción retorno 90d
```

---

## Features (15 indicadores optimizados)

Las features están optimizadas para predicción a largo plazo (horizon=90 días):

### Precio/Retorno (5 features)
| Feature | Descripción | Uso estratégico |
|---------|-------------|-----------------|
| `ret_1w` | Retorno semanal (5 días) | Momentum semanal sin ruido diario |
| `ret_4w` | Retorno mensual (20 días) | Momentum persistente, crítico para horizon=90d |
| `ret_13w` | Retorno trimestral (60 días) | Alineado con horizon, detecta tendencias largo plazo |
| `vol_4w` | Volatilidad mensual | Riesgo mensual, más estable que diaria |
| `vol_regime` | Cambio de volatilidad | Expansión/contracción de vol, predice reversiones |

### Volatilidad/ATR (2 features)
| Feature | Descripción | Uso estratégico |
|---------|-------------|-----------------|
| `atr_14` | Average True Range (14d) normalizado | Volatilidad "real" considerando gaps |
| `vol_zscore_60` | Z-score de volumen (60d) | Detecta volumen anormal (confirmación) |

### Tendencia (4 features)
| Feature | Descripción | Uso estratégico |
|---------|-------------|-----------------|
| `sma_50` | SMA de 50 días | Tendencia medio plazo |
| `sma_200` | SMA de 200 días | Tendencia largo plazo (filtro fundamental) |
| `sma50_sma200_ratio` | SMA50/SMA200 - 1 | Golden/Death Cross, fuerza de tendencia |
| `close_sma200_dist` | Close/SMA200 - 1 | Sobrecompra/sobreventa vs tendencia |

### Momentum (1 feature)
| Feature | Descripción | Uso estratégico |
|---------|-------------|-----------------|
| `macd_hist` | Histograma MACD | Divergencia momentum vs tendencia |

### Bandas de Bollinger (2 features)
| Feature | Descripción | Uso estratégico |
|---------|-------------|-----------------|
| `bb_pct_b` | %B Bollinger | Posición relativa en bandas (sobrecompra/venta) |
| `bb_bandwidth` | Ancho de bandas | Volatilidad, detecta "squeeze" |

### Fuerza de Tendencia (1 feature)
| Feature | Descripción | Uso estratégico |
|---------|-------------|-----------------|
| `adx_14` | Average Directional Index | Fuerza de tendencia (no dirección) |

**Nota:** Features de benchmark fueron removidas para evitar leakage temporal.

---

## Uso

### CLI Options

```bash
# Runner unificado de E1 (3 versiones)
python -m src.train_e1_all
python -m src.train_e1_all --tickers AAPL,MSFT

# Modo completo (download + clean + train)
python -m src.train_e1_simple_pipeline --tickers AAPL,MSFT

# Todos los tickers del config
python -m src.train_e1_simple_pipeline

# Skip download (usar datos existentes)
python -m src.train_e1_simple_pipeline --tickers AAPL --skip-download

# Skip cleaning (usar datos limpios existentes)
python -m src.train_e1_simple_pipeline --tickers AAPL --skip-cleaning

# Solo entrenar (reutilizar datos)
python -m src.train_e1_simple_pipeline --tickers AAPL --skip-download --skip-cleaning
```

### Output Esperado

```
============================================================
Pipeline E1 Simple - 1 tickers
Output: runs/e1_simple/20260205_120000
============================================================

Paso 1/3: Descargando datos...
------------------------------------------------------------
✓ Descargados 2 archivos

Paso 2/3: Limpiando datos...
------------------------------------------------------------
✓ Limpiados: 2 | Rechazados: 0

Paso 3/3: Entrenando modelos...
------------------------------------------------------------
[1/1] AAPL
  ✓ Usando datos limpios: AAPL_daily.csv
  Samples: 2046 | Features: 15 | Lookback: 360d
  Split: train=1432 val=307 test=307
  Entrenando GRU [64]...
  ✓ Epochs: 34/100 | Val Loss: 0.1948
  ✓ MAE=0.1628 RMSE=0.1862 IC=0.273 Dir=50.8% Sharpe=0.68

============================================================
✓ Completado: 1/1 tickers
  Resultados en: runs/e1_simple/20260205_120000/
============================================================
```

---

## Estructura de Resultados

```
runs/e1_simple/20260205_120000/
├── config_used.yaml           # Config reproducible
├── summary_all.csv            # Resumen de todos los tickers
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

**`summary_all.csv`** - Métricas agregadas por ticker:
```csv
ticker,n_samples,n_train,n_val,n_test,lookback_days,horizon_days,ml_mae,ml_rmse,ml_ic,ml_directional_accuracy,bt_sharpe,bt_cagr,bt_max_drawdown
AAPL,2046,1432,307,307,360,90,0.1628,0.1862,0.273,0.508,0.68,0.12,-0.18
```

**`{ticker}_predictions.csv`** - Predicciones vs realidad:
```csv
timestamp,y_true,y_pred
2024-01-15,-0.0234,0.0156
2024-01-16,0.0421,0.0389
```

**`{ticker}_model.pth`** - Modelo serializado con:
- Pesos del modelo (state_dict)
- Arquitectura (hidden_sizes, dropout, dense_units)
- Scalers (mean/std de features y target)
- Metadata (lookback_days, horizon_days, feature_names)

---

## Configuración (base.yaml)

```yaml
strategies:
  e1_simple:
    enabled: true
    frequency: "1D"
    lookback_days: 360        # 1 año de datos de entrada
    horizon_days: 90          # Target: retorno a 90 días
    rebalance: "monthly"

    thresholds:
      tau_buy: 0.06           # +6% predicho → señal compra
      tau_sell: 0.00          # 0% o menos → cerrar posición

    filters:
      regime:
        type: "sma200"
        sma200_required: true  # Solo operar si close > SMA200

    risk:
      stop_loss_pct: 0.10     # Stop loss del 10%

    model:
      type: "GRU"
      gru_units: [64]         # 1 capa GRU
      dropout: 0.2
      dense_units: 16
      loss: "huber"
      huber_delta: 1.0
      optimizer: "adamw"
      learning_rate: 0.001
      batch_size: 64
      max_epochs: 100
      early_stopping_patience: 10
      clipnorm: 1.0

    backtest:
      holding_period_days: 90   # Holding period = horizon
      allow_short: false
      max_position: 1.0

costs:
  daily_round_trip_bps: 10    # 10 bps = 0.1% costos transacción
```

---

## Airflow DAG

El DAG `e1_simple_pipeline` ejecuta el pipeline completo:

### Tareas
1. **download_daily_data** - Descarga OHLCV diario (yfinance)
2. **clean_daily_data** - Limpia y valida datos
3. **train_e1_simple_models** - Entrena modelos + MLflow logging
4. **notify_api** - Notifica a FastAPI (opcional)

### Trigger desde CLI

```bash
# Un ticker
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_simple_pipeline \
  --conf '{"tickers": "AAPL", "skip_download": "False", "skip_cleaning": "False"}'

# Reutilizando datos existentes
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_simple_pipeline \
  --conf '{"tickers": "AAPL,MSFT", "skip_download": "True", "skip_cleaning": "True"}'
```

### MLflow Tracking

- **Experiment:** "E1_Simple"
- **Metricas:** mae, rmse, ic, directional_accuracy, bt_sharpe, bt_cagr, bt_max_drawdown
- **Parametros:** strategy, ticker, lookback_days, gru_units, learning_rate, etc.
- **Artifacts:** models/, predictions/, backtest/, scalers/
- **Fallback robusto:** si `MLFLOW_TRACKING_URI` remoto no está disponible, el pipeline usa tracking local automáticamente.
- **Store aislado automático:** si `mlruns/` tiene experimentos malformados (por ejemplo, `meta.yaml` faltante), usa `runs/e1_simple/mlflow_store` para evitar errores ruidosos de inicialización.
- **Modo transparente offline→online:** por defecto, el fallback local prioriza SQLite en `runs/mlflow_local/mlflow.db` (artifacts en `runs/mlflow_local/artifacts`), para que luego MLflow pueda leer los mismos runs.

#### Modo Transparente (CLI sin Docker + UI con Docker)

Si quieres que los runs creados desde CLI (sin MLflow server activo) aparezcan luego al levantar MLflow en Docker, inicia el servicio con backend/artifacts apuntando al mismo storage local:

```bash
MLFLOW_BACKEND_STORE_URI=sqlite:////mlflow_data/mlflow_local/mlflow.db \
MLFLOW_DEFAULT_ARTIFACT_ROOT=file:///mlflow_data/mlflow_local/artifacts \
docker compose --profile mlflow up -d mlflow
```

`docker-compose.yaml` ya monta `./runs` en `/mlflow_data`, por lo que ese backend SQLite queda compartido entre CLI y servidor.

---

## Casos de Uso

### Cuándo usar E1 Simple
1. **Desarrollo inicial** - Probar features y arquitectura rápidamente
2. **Experimentación** - Iterar hiperparámetros
3. **Baseline** - Punto de comparación para modelos más complejos
4. **Debugging** - Verificar que el pipeline funciona
5. **Aprendizaje** - Entender el flujo sin complejidad de walk-forward

### Cuándo usar E1 Conservative
1. **Producción** - Validación robusta con walk-forward
2. **Paper académico** - Métricas rigurosas
3. **Trading real** - Decisiones con dinero real
4. **Validación final** - Antes de deploy

---

## Workflow Recomendado

```
1. Desarrollo con E1 Simple
   ↓
   Iterar rápido (arquitectura, features)
   ↓
2. Validación con E1 Conservative
   ↓
   Walk-forward confirma robustez
   ↓
3. Optimización (Optuna)
   ↓
   Tune hyperparams por ticker
   ↓
4. Producción con modelo optimizado
```

---

## Comparación de las 3 versiones de E1

Puedes comparar automáticamente **E1 Baseline**, **E1 Simple** y **E1 Conservative** con tiempo de entrenamiento y métricas clave.

```bash
# Usa el último run de cada versión
python scripts/evaluation/compare_e1_versions.py

# Filtrar por tickers específicos
python scripts/evaluation/compare_e1_versions.py --tickers AAPL,MSFT

# Comparación ticker-vs-ticker entre versiones
python scripts/evaluation/compare_e1_versions.py --tickers AAPL,MSFT --per-ticker
```

La comparación reporta:
- `train_time_seconds_avg`
- `train_time_seconds_total`
- `mae`, `rmse`, `ic`, `directional_accuracy`
- `bt_sharpe`, `bt_cagr`, `bt_max_drawdown`, `bt_num_trades`

Archivos de salida:
- `reports/tables/e1_versions_comparison/e1_versions_comparison_<YYYYMMDD_HHMMSS>.csv`
- `reports/tables/e1_versions_comparison/e1_versions_comparison_<YYYYMMDD_HHMMSS>.md`
- `reports/tables/e1_versions_comparison/e1_versions_comparison_<YYYYMMDD_HHMMSS>_per_ticker.csv`
- `reports/tables/e1_versions_comparison/e1_versions_comparison_<YYYYMMDD_HHMMSS>_per_ticker.md`

---

## Referencias

- [README_E1.md](README_E1.md) - E1 Conservative completo
- [COMPARISON_E1.md](COMPARISON_E1.md) - Comparación detallada Simple vs Conservative
- [README_AIRFLOW_USAGE.md](README_AIRFLOW_USAGE.md) - Uso de DAGs
- [src/train_e1_simple_pipeline.py](src/train_e1_simple_pipeline.py) - Código fuente
- [src/features/build_features_e1.py](src/features/build_features_e1.py) - Cálculo de features
- [src/config/base.yaml](src/config/base.yaml) - Configuración

---

## Tips

1. **Empieza con E1 Simple** - Itera rápido antes de validar con Conservative
2. **Revisa summary_all.csv** - Distribución de métricas entre tickers
3. **Analiza backtest CSV** - Equity curve y drawdowns
4. **Skip download/clean** - Para experimentos rápidos con datos existentes
5. **Compara IC vs Sharpe** - Un buen IC no garantiza buen Sharpe (y viceversa)

---

**Última actualización:** Febrero 5, 2026
**Versión:** 3.0 (features optimizadas, lookback 360d)
