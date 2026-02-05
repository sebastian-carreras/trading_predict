# E1 - Estrategia Conservadora (GRU)

**Horizonte**: 90 días | **Frecuencia**: diaria | **Perfil**: Bajo riesgo, largo plazo

Predicción de retornos acumulados a 90 días usando arquitectura GRU para inversión conservadora con rebalanceo mensual.

---

## 🎯 Versiones Disponibles

### **E1 Conservative (Producción)** - `train_e1_pipeline.py`
Pipeline completo con walk-forward validation y decision score multi-métrica para uso en producción.

**Características:**
- Walk-forward: 5 folds para validación robusta
- Decision profiles: conservative/moderate/aggressive
- Arquitectura: GRU 2 capas (128→64 units)
- Airflow DAG: `e1_conservative_pipeline`
- MLflow experiment: "E1_Conservative_Strategy"

**Uso:**
```bash
# CLI
python -m src.train_e1_pipeline --tickers AAPL,MSFT

# Airflow
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_conservative_pipeline --conf '{"tickers": "AAPL,MSFT"}'
```

**Resultados:** `runs/e1_conservative/<timestamp>/`

---

### **E1 Simple (Desarrollo)** ⭐ - `train_e1_simple_pipeline.py` 
**Versión simplificada** para desarrollo rápido, experimentación y baseline:

**Simplificaciones:**
- ✅ **Time split simple**: 70/15/15 (vs walk-forward)
- ✅ **Decision score**: 5 métricas fijas (vs perfiles)
- ✅ **GRU**: 1 capa 64 units (vs 2 capas 128→64)
- ✅ **80% más rápido**: ~1 min vs ~5 min por ticker

**Uso:**
```bash
# CLI (descarga y limpia automáticamente)
python -m src.train_e1_simple_pipeline --tickers AAPL,MSFT

# Airflow
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_simple_pipeline --conf '{"tickers": "AAPL,MSFT"}'
```

**Resultados:** `runs/e1_simple/<timestamp>/`

**📖 Documentación completa:** [README_E1_SIMPLE.md](README_E1_SIMPLE.md)

---

## 📋 Resumen de la Estrategia

**Objetivo**: Capturar tendencias de mediano-largo plazo con bajo riesgo y turnover reducido.

**Características**:
- **Target**: Retorno logarítmico acumulado a 90 días
- **Lookback**: 180 días (ventana de entrada)
- **Rebalanceo**: Mensual o trimestral
- **Umbrales**: τ_buy = 0.06 (6%), τ_sell = 0.00
- **Filtros**: RSI < 60, Close > SMA(200), Bollinger %B < 0.9
-- **Directional Accuracy mínima** (referencia): 58% como piso para aprobar señales

**Métricas sanitizadas:** Todos los valores NaN e Inf se convierten a valores válidos:
- NaN → 0.0
- Inf positivo → 1e8 (o 1e3 para sortino/profit_factor)
- Esto evita problemas en MLflow y análisis posterior.

## Arquitectura del Modelo

### GRU (Gated Recurrent Unit)

**Justificación**: GRU demuestra eficiencia computacional y buena generalización en predicción de retornos a mediano-largo plazo, con menor riesgo de overfitting que LSTM en horizontes largos.

```
Input: (sequence_length=180 días, features=27)
↓
GRU Layer 1 (128 units, return_sequences=True)
↓
Dropout (0.2)
↓
GRU Layer 2 (64 units, return_sequences=False)
↓
Dropout (0.2)
↓
Dense (32 units, activation='relu')
↓
Output (1): predicción de retorno a 90 días
```

**Hiperparámetros**:
- Optimizer: AdamW, lr=0.001
- Loss: Huber (δ=1.0) - robusta a outliers
- Batch size: 64
- Max epochs: 200
- Early stopping: patience=15
- Regularización: gradient clipping norm=1.0

## 📂 Archivos Implementados

| Archivo | Propósito | Estado |
|---------|-----------|--------|
| `src/data/download_daily.py` | Descarga OHLCV diario + benchmark SPY | ✅ |
| `src/data/clean_daily.py` | Limpieza y validación de datos | ✅ |
| `src/features/build_features_e1.py` | Cálculo de 27 features técnicos | ✅ |
| `src/features/build_sequences_e1e2.py` | Conversión a secuencias RNN | ✅ |
| `src/models/e1_gru.py` | Arquitectura GRU PyTorch | ✅ |
| `src/train_e1_pipeline.py` | Pipeline E1 Conservative (walk-forward) | ✅ |
| `src/train_e1_simple_pipeline.py` | Pipeline E1 Simple (time split) | ✅ |
| `src/backtest/daily.py` | Backtesting con costos + métricas sanitizadas | ✅ |
| `dockerfiles/airflow/dags/E1/e1_conservative_pipeline.py` | Airflow DAG E1 Conservative | ✅ |
| `dockerfiles/airflow/dags/E1/e1_simple_pipeline.py` | Airflow DAG E1 Simple | ✅ |

## Cómo Usar

### Paso 1: Instalar dependencias

```bash
cd trading_predict
pip install -r requirements.txt
```

### Paso 2: Entrenar E1

#### E1 Conservative (producción, walk-forward)

```bash
# Entrenar para todos los tickers E1 del config
python -m src.train_e1_pipeline

# O especificar tickers manualmente
python -m src.train_e1_pipeline --tickers AAPL,MSFT,GGAL.BA

# Con Airflow
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_conservative_pipeline --conf '{"tickers": "AAPL,MSFT"}'
```

#### E1 Simple (desarrollo, time split)

```bash
# Entrenar (descarga y limpia automáticamente)
python -m src.train_e1_simple_pipeline --tickers AAPL,MSFT

# Skip download/clean para experimentos rápidos
python -m src.train_e1_simple_pipeline --tickers AAPL --skip-download --skip-cleaning

# Con Airflow
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_simple_pipeline --conf '{"tickers": "AAPL,MSFT"}'
```

### Paso 3: Revisar resultados

**E1 Conservative:** `runs/e1_conservative/<timestamp>/`
**E1 Simple:** `runs/e1_simple/<timestamp>/`

```
runs/e1_conservative/20260118_153022/
├── config_used.yaml                       # Config usado (reproducibilidad)
├── summary_all.csv                        # Resumen de todos los tickers
├── AAPL/
│   ├── AAPL_predictions.csv              # Predicciones agregadas walk-forward
│   ├── AAPL_scaler.csv                   # Scaler (mean/std de features)
│   ├── AAPL_target_scaler.csv            # Scaler del target
│   ├── AAPL_walkforward_folds.csv        # Resumen de folds
│   ├── AAPL_walkforward_backtest.csv     # Backtest agregado walk-forward
│   ├── AAPL_walkforward_predictions.csv  # Predicciones por fold
│   ├── AAPL_walkforward_metrics.png      # Visualización de métricas
│   ├── AAPL_model.pth                    # Modelo del último fold
│   └── AAPL_summary.csv                  # Métricas ML + trading
├── MSFT/
│   └── ...
└── GGAL.BA/
    └── ...
```

**MLflow:** Si `MLFLOW_TRACKING_URI` está configurado, cada ticker registra:
- Métricas: mae, rmse, ic, directional_accuracy, bt_sharpe, bt_cagr, bt_max_drawdown
- Parámetros: strategy, ticker, lookback_days, gru_units
- Artifacts: models/, predictions/, backtest/

## Parámetros (desde base.yaml)

```yaml
splits:
  method: "walk_forward"
  folds: 5
        sortino: 0.35
        calmar: 0.25
        max_drawdown: 0.25
strategies:
  e1_conservative:
    lookback_days: 180        # Ventana de entrada
    horizon_days: 90          # Target: retorno a 90 días
    model:
      gru_units: [128, 64]
      dropout: 0.2
      dense_units: 32
      learning_rate: 0.001
      batch_size: 64
      max_epochs: 200
      early_stopping_patience: 15
      loss: "huber"
```

## Features Calculadas (27 features)

Según especificación del documento:

**Precio/retorno (7)**
- `ret_1d`, `ret_5d`, `ret_20d`, `ret_60d`
- `vol_20d`, `vol_60d`
- `range_pct`, `atr_14`, `vol_zscore_60`

**Tendencia (14)**
- `sma_50`, `sma_200`, `close_sma200_dist`, `ema_50`
- `macd_line`, `macd_signal`, `macd_hist`
- `bb_pct_b`, `bb_bandwidth`
- `adx_14`

**Contexto (2)**
- `bench_ret_1d`, `bench_ret_20d` (benchmark SPY)

**Opcional (agregable)**
- Fundamentales (P/E, dividend yield) con lag de publicación
- Sentimiento (score semanal) con ventana cerrada

## Features Calculadas (27 indicadores)

**Precio y retorno (9 features)**:
- `ret_1d`, `ret_5d`, `ret_20d`, `ret_60d` - Retornos a distintos horizontes
- `vol_20d`, `vol_60d` - Volatilidad realizada
- `range_pct` - (High-Low)/Close
- `atr_14` - Average True Range
- `vol_zscore_60` - Z-score de volumen (normalizado)

**Tendencia de largo plazo (14 features)**:
- `sma_50`, `sma_200` - Medias móviles simples
- `close_sma200_dist` - Distancia relativa: Close/SMA(200) - 1
- ` Lógica de Señales de Trading

**Criterio de compra** (entrada larga):
- **Condición principal**: `ŷ^(90) > τ_buy` (predicción > 6%)
- **Filtros de confirmación**:
  - RSI(14) < 60 - Evitar sobrecompra
  - Close > SMA(200) - Filtro de régimen alcista
  - Bollinger %B < 0.9 - No en extremo superior
  - ADX(14) > 20 (opcional) - Tendencia definida

**Criterio de venta** (salida):
- Por modelo: `ŷ^(90) < 0` o `< τ_sell`
- Stop-loss: -10% (8-12% según volatilidad)
- Take-profit: +15% (12-20% buscando R:R ≥ 1.5)
- Trailing stop: opcional para capturar tendencias

**Gestión de capital**:
- Portfolio: 8-12 activos de baja correlación
- Pesos: inversamente proporcionales a volatilidad (risk parity)
- Rebalanceo: mensual (reducir turnover)

## 📁 Estructura de Resultados

Cada ejecución genera outputs en `runs/e1_conservative/<timestamp>/`:

```
runs/e1_conservative/20260105_153022/
├── config_used.yaml              # Configuración exacta usada
├── summary_all.csv                # Resumen de todos los tickers
├── AAPL/
│   ├── AAPL_predictions.csv      # (timestamp, y_true, y_pred)
│   ├── AAPL_scaler.csv            # Parámetros de normalización
│   ├── AAPL_walkforward_folds.csv # Métricas por fold (walk-forward)
│   ├── AAPL_walkforward_backtest.csv # Serie PnL agregada walk-forward
│   ├── AAPL_walkforward_predictions.csv # Predicciones agregadas walk-forward
│   └── AAPL_summary.csv           # Métricas ML + trading
├── MSFT/
│   └── ...
└── YPF/
    └── ...
```

## 🔄 Próximos Pasos

**Estado actual**:
1.  Descarga de datos
2.  Feature engineering (27 indicadores)
3.  Modelo GRU con regularización
4.  Pipeline de entrenamiento
5.  Métricas ML (MAE/RMSE/IC)

**Pendiente**:
6.  **Backtest engine** - Convertir predicciones → señales → PnL
7.  **Reglas de trading** - Implementar filtros RSI/MACD/Bollinger
8.  **Reportes visuales** - Equity curves, drawdown, tablas LaTeX
9.  **Benchmarks** - Buy & Hold, SMA crossover con mismos costos

## Validación Rápida

```bash
# Probar con un ticker
python -m src.train_e1_pipeline --tickers AAPL
```

**Output esperado**:
```
Descargando AAPL... ✓ 1258 días
Calculando features... ✓ 27 features
Creando secuencias... ✓ 894 muestras
Split temporal: train=626 val=134 test=134
Entrenando GRU...
  Epoch 45/200, Val Loss: 0.002134 (early stop)
  Decision score 0.78 (threshold 0.75) → COMPRAR
✓ AAPL: MAE=0.0156 RMSE=0.0234 IC=0.342 Dir_Acc=58.2%
Guardado en runs/e1_conservative/20260105_153022/
```

## Referencias

- [README general](README.md) - Overview del proyecto
- [base.yaml](src/config/base.yaml) - Configuración completa
- [Especificación técnica](secciones%20del%20plan%20de%20trabajo/3.%20Implementación%20de%20modelos%20de%20machine%20learning/) - Detalles de arquitectura

---

**Estrategia E1 implementada según especificación del documento de diseño**
- **Max Drawdown**: Pérdida máxima desde peak
- **Calmar**: CAGR / Max Drawdown
- **Profit Factor**: Ganancia total / Pérdida total
- **Hit Rate**: % de trades ganadores
- **Turnover**: Frecuencia de rebalanceo

## Próximos Pasos

1.  **Datos + Features + Modelo** (COMPLETO)
2.  **Backtest con costos** (COMPLETO) - Implementado en `src/backtest/daily.py`
3.  **Reglas de trading** (COMPLETO) - Integradas en el backtest diario (taus + holding period)
4.  **Reportes** - Equity curves + tablas en `reports/`

## Validación Rápida

```bash
# Verificar que funciona con un ticker
cd trading_predict
python -m src.train_e1_pipeline --tickers AAPL
```

Deberías ver:
```
Entrenando GRU para AAPL...
  Epochs: 45, Val Loss: 0.002134
✓ AAPL: MAE=0.0156 IC=0.342
```

---

**Implementado según**: `3. Implementación de modelos de machine learning/Informacion basica de la implementacion.md`

---

## 📊 E1 Simple - Detalles de Implementación

### ¿Cuándo usar E1 Simple vs E1 Conservadora?

| Aspecto | E1 Simple | E1 Conservadora |
|---------|-----------|-----------------|
| **Tiempo de entrenamiento** | ~1 min | ~5 min |
| **Validación** | Time split (1 fold) | Walk-forward (5 folds) |
| **Decision score** | 5 métricas simples | 4-6 métricas por perfil |
| **Arquitectura** | GRU 1 capa (64 units) | GRU 2 capas (64→32 units) |
| **Uso recomendado** | Desarrollo, experimentación | Producción, validación robusta |
| **Resultados** | Baseline rápido | Robustez temporal |

### Decision Score Simplificado

E1 Simple usa **solo 5 métricas** con interpretación clara:

```yaml
decision:
  targets:
    ic_min: 0.05                    # IC > 0.05 es significativo
    directional_accuracy_min: 0.55  # >55% predice dirección correcta
    sharpe_min: 1.0                 # Sharpe ≥ 1.0 es bueno
    mae_max: 0.03                   # Error < 3% es aceptable
    rmse_max: 0.05                  # RMSE < 5% es aceptable
  
  weights:
    ic: 0.25                        # 25% peso - capacidad predictiva
    directional_accuracy: 0.20      # 20% peso - acierto direccional
    sharpe: 0.30                    # 30% peso - retorno ajustado por riesgo
    mae: 0.15                       # 15% peso - error absoluto
    rmse: 0.10                      # 10% peso - error cuadrático
  
```

### Arquitectura GRU Simplificada

```
Input: (180 días, 27 features)
↓
GRU(64 units)
↓
Dropout(0.2)
↓
Dense(16 units, relu)
↓
Output(1): retorno predicho a 90 días
```

**Ventajas vs arquitectura de 2 capas:**
- ✅ 50% menos parámetros → menos overfitting
- ✅ 2x más rápido de entrenar
- ✅ Más fácil de interpretar
- ✅ Suficiente para horizontes largos (90 días)

### Ejemplo de Output

```bash
$ python -m src.train_e1_simple_pipeline --tickers AAPL

============================================================
Pipeline E1 Simple - 1 tickers
Output: runs/e1_simple/20260116_143522
============================================================

Benchmark: SPY (2547 días)

[1/1] AAPL
  ✓ Usando datos limpios: AAPL_daily.csv
  Samples: 894 | Features: 27 | Lookback: 180d
  Split: train=626 val=134 test=134
  Entrenando GRU [64]...
  ✓ Epochs: 42/100 | Val Loss: 0.002156
  ✓ MAE=0.0148 RMSE=0.0221 IC=0.352 Dir=59.0% Sharpe=1.18
  ✓ Decision score 0.782 (threshold 0.70) → BUY
  ✓ Modelo guardado: AAPL_model.pth

============================================================
✓ Completado: 1/1 tickers
  Resultados en: runs/e1_simple/20260116_143522/
============================================================
```

### Estructura de Resultados

```
runs/e1_simple/20260116_143522/
├── config_used.yaml              # Config usado
├── summary_all.csv               # Resumen de todos los tickers
└── AAPL/
    ├── AAPL_predictions.csv      # (timestamp, y_true, y_pred)
    ├── AAPL_backtest.csv          # Serie de PnL
    ├── AAPL_scaler.csv            # Parámetros de normalización
    ├── AAPL_model.pth             # Modelo entrenado
    └── AAPL_summary.csv           # Métricas completas
```

---

### Posibles mejoras futuras:
#### Fase 1 - Corto Plazo (1-2 semanas) 
⭐⭐⭐ Hyperparameter tuning con Optuna 
- Optimizar tau_buy, tau_sell, arquitectura 
- Documentar proceso de búsqueda 
⭐⭐⭐ Walk-forward validation 
- Demostrar robustez temporal 
- Gráficos de IC y Sharpe por ventana 
#### Fase 2 - Mediano Plazo (2-3 semanas) 
⭐⭐ Feature importance
- Justificar selección de features
- Eliminar features ruidosas
⭐⭐⭐ Portfolio optimization
- Estrategia multi-ticker
- Comparar vs equal-weight
#### Fase 3 - Largo Plazo (si tiempo)
⭐⭐ Drift detection + retraining
- Sistema de monitoreo
⭐⭐ Ensemble de modelos
Mejorar estabilidad