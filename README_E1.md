# E1 - Estrategia Conservadora (GRU)

**Horizonte**: 90 días | **Frecuencia**: diaria | **Perfil**: Bajo riesgo, largo plazo

Predicción de retornos acumulados a 90 días usando arquitectura GRU para inversión conservadora con rebalanceo mensual.

## 📋 Resumen de la Estrategia

**Objetivo**: Capturar tendencias de mediano-largo plazo con bajo riesgo y turnover reducido.

**Características**:
- **Target**: Retorno logarítmico acumulado a 90 días
- **Lookback**: 180 días (ventana de entrada)
- **Rebalanceo**: Mensual o trimestral
- **Umbrales**: τ_buy = 0.06 (6%), τ_sell = 0.00
- **Filtros**: RSI < 60, Close > SMA(200), Bollinger %B < 0.9
- **Perfil de scoring**: `conservative` (configurable por estrategia)
- **Criterio de decisión**: `decision_score` ≥ 0.65 (perfil conservative)
- **Directional Accuracy mínima** (en el perfil): 58% como piso para aprobar señales

**Métricas objetivo (para `decision_score`, perfil conservative)**:
- Sortino ≥ 1.0
- Calmar ≥ 1.0
- Max Drawdown ≤ 20%
- Directional Accuracy ≥ 0.58
- `decision_score` ≥ 0.65 para emitir señal BUY

## 🧮 Decision Score Compuesto

- El pipeline calcula un `decision_score` por **perfil** (conservative/moderate/aggressive).
- Para E1 se usa por defecto el perfil `conservative` (ver `splits.strategy_profile`).
- Los objetivos y pesos se configuran en `splits.decision_profiles.<profile>.{target_metrics,weights}` y el umbral BUY/HOLD en `splits.decision_profiles.<profile>.threshold`.
- Se imprime en consola al final del walk-forward/time split, se persiste en `*_summary.csv` y se loguea en MLflow/Airflow.
- Los componentes se loguean de forma **dinámica** como `decision_component_<metric>` (ej: `decision_component_sortino`, `decision_component_calmar`, `decision_component_max_drawdown`, `decision_component_directional_accuracy`).
- También se registra `decision_profile` para poder comparar runs entre perfiles.

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
| `src/data/download_daily.py` | Descarga OHLCV diario + benchmark SPY |  |
| `src/features/build_features_e1.py` | Cálculo de 22 features técnicos |  |
| `src/features/build_sequences.py` | Conversión a secuencias RNN |  |
| `src/models/e1_gru.py` | Arquitectura GRU PyTorch |  |
| `src/train_e1_pipeline.py` | Pipeline end-to-end |  |
| `src/backtest/daily.py` | Backtesting diario con costos (señales por `tau_buy`/`tau_sell`) + métricas |  |

## Cómo Usar

### Paso 1: Instalar dependencias

```bash
cd trading_predict
pip install -r requirements.txt
```

### Paso 2: Descargar datos

```bash
python -m src.data.download_daily
# use this force a new download  python -m src.data.download_daily --force 
```

Esto descarga OHLCV diario para todos los tickers definidos en `src/config/base.yaml` y los guarda en `data/raw/daily/`.

### Paso 3: Entrenar E1

```bash
# Entrenar para todos los tickers E1 del config
python -m src.train_e1_pipeline

# O especificar tickers manualmente
python -m src.train_e1_pipeline --tickers YPF,GGAL,AAPL
```

El pipeline realiza validación walk-forward usando `temporal_train_val_split` internamente para preservar datos en train/val y aplica embargo configurable.

### Paso 4: Revisar resultados

Los resultados se guardan en `runs/e1_conservative/<timestamp>/`:

```
runs/e1_conservative/20260105_153022/
├── config_used.yaml          # Config usado (reproducibilidad)
├── summary_all.csv            # Resumen de todos los tickers
├── YPF/
│   ├── YPF_predictions.csv   # Predicciones (y_true, y_pred)
│   ├── YPF_scaler.csv         # Scaler (mean/std de features)
│   ├── YPF_walkforward_folds.csv      # Resumen de folds (si se usa walk-forward)
│   ├── YPF_walkforward_backtest.csv   # Backtest agregado walk-forward
│   ├── YPF_walkforward_predictions.csv# Predicciones agregadas walk-forward
│   └── YPF_summary.csv        # Métricas ML + trading + decision_score
├── GGAL/
│   └── ...
└── AAPL/
  └── ...

Si `MLFLOW_TRACKING_URI` está configurado, cada ticker registra métricas y parámetros (incluyendo `decision_score` y sus componentes) en el experimento `E1_Conservative`.
```

## Parámetros (desde base.yaml)

```yaml
splits:
  method: "walk_forward"
  folds: 5
  strategy_profile:
    e1_conservative: conservative

  decision_profiles:
    conservative:
      threshold: 0.65
      target_metrics:
        sortino_min: 1.0
        calmar_min: 1.0
        max_drawdown_max: 0.20
        directional_accuracy_min: 0.58
      weights:
        sortino: 0.35
        calmar: 0.25
        max_drawdown: 0.25
        directional_accuracy: 0.15

  # (fallback legacy)
  target_metrics:
    mae_max: 0.03
    ic_min: 0.05
    sharpe_min: 1.0
    directional_accuracy_min: 0.58
  decision_score:
    threshold: 0.75
    weights:
      mae: 0.35
      ic: 0.25
      sharpe: 0.25
      directional_accuracy: 0.15

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
├── summary_all.csv                # Resumen de todos los tickers (incluye decision_score promedio)
├── AAPL/
│   ├── AAPL_predictions.csv      # (timestamp, y_true, y_pred)
│   ├── AAPL_scaler.csv            # Parámetros de normalización
│   ├── AAPL_walkforward_folds.csv # Métricas por fold (walk-forward)
│   ├── AAPL_walkforward_backtest.csv # Serie PnL agregada walk-forward
│   ├── AAPL_walkforward_predictions.csv # Predicciones agregadas walk-forward
│   └── AAPL_summary.csv           # Métricas ML + trading + decision_score
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