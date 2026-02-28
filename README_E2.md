# E2 - Estrategia Moderada (LSTM)



**Variantes disponibles**:
- **E2 Moderate** (este documento): LSTM con 2 capas, walk-forward validation, filtros complejos
- **[E2 Simple](README_E2_SIMPLE.md)**: LSTM con 1 capa, split temporal simple, prototipado rápido

Elige **E2 Moderate** para producción robusta; **E2 Simple** para investigación rápida.

## 📋 Resumen de la Estrategia

**Objetivo**: Capturar tendencias de corto-mediano plazo con mayor rotación y sensibilidad al momentum.

**Características**:
- **Target**: Retorno logarítmico acumulado a 20 días (2-4 semanas)
- **Lookback**: 60 días (ventana de entrada)
- **Rebalanceo**: Semanal o quincenal
- **Umbrales**: τ_buy = 0.025 (2.5%), τ_sell = 0.00
- **Filtros**: MACD confirmación, RSI entre 35-70, Volume z-score > 0

**Métricas objetivo**:
- Sharpe neto > 0.8
- Max Drawdown < 20%
- Win rate > 50%
- Profit Factor > 1.2

## Arquitectura del Modelo

### LSTM (Long Short-Term Memory)

**Justificación**: LSTM demuestra mayor efectividad en capturar tendencias de mediano plazo y patrones de volatilidad, con R² scores superiores en predicciones de 5-30 días. Su capacidad de memoria selectiva es óptima para detectar cambios de momentum.

```
Input: (sequence_length=60 días, features=~25)
↓
LSTM Layer 1 (128 units, return_sequences=True)
↓
Dropout (0.2)
↓
LSTM Layer 2 (64 units, return_sequences=False)
↓
Dropout (0.2)
↓
Dense (32 units, activation='relu')
↓
Output (1): predicción de retorno a 20 días
```

**Hiperparámetros**:
- Optimizer: AdamW, lr=0.001
- Loss: Huber (δ=1.0) - robusta a outliers
- Batch size: 64
- Max epochs: 150
- Early stopping: patience=12
- Regularización: gradient clipping norm=1.0

## 📂 Archivos a Implementar

| Archivo | Propósito | Estado |
|---------|-----------|--------|
| `src/features/build_features_e2.py` | Features momentum/volatilidad |  TODO |
| `src/models/e2_lstm.py` | Arquitectura LSTM PyTorch |  TODO |
| `src/train_e2_pipeline.py` | Pipeline end-to-end |  TODO |
| `src/backtest/rules_e2.py` | Reglas de señal (MACD, RSI) |  TODO |

## Features a Calcular (~25 indicadores)

**Precio y retorno core**:
- `ret_1d`, `ret_5d`, `ret_10d`, `ret_20d` - Retornos múltiples horizontes
- `gap_open_close` - Gap apertura-cierre
- `range_hl` - (High-Low)/Close
- `atr_14` - Average True Range

**Momentum (principal diferenciador vs E1)**:
- `rsi_14` - Relative Strength Index
- `stoch_k`, `stoch_d` - Stochastic Oscillator (14,3)
- `roc_10` - Rate of Change
- `macd_line`, `macd_signal`, `macd_hist` - MACD completo (12,26,9)

**Tendencia corta/media**:
- `sma_7`, `sma_20` - SMAs de corto plazo
- `ema_12`, `ema_26` - EMAs
- `close_ema26_dist` - Distancia: Close/EMA(26) - 1

**Volatilidad**:
- `vol_20d` - Volatilidad realizada
- `bb_bandwidth` - Ancho de Bollinger Bands (20,2)

**Volumen**:
- `obv` - On-Balance Volume
- `volume_roc_10` - Volume Rate of Change
- `volume_zscore_20` - Z-score volumen normalizado

**Contexto**:
- Opcional: FX, sentimiento diario

## Lógica de Señales de Trading

**Criterio de compra** (entrada larga):
- **Condición principal**: `ŷ^(20) > τ_buy` (predicción > 2.5%)
- **Confirmación técnica** (reducir ruido):
  - MACD histograma > 0 y creciente **O** MACD cruce alcista
  - RSI entre 35 y 70 (evitar extremos)
  - Volume z-score(20) > 0 (participación confirmada)

**Criterio de venta** (salida):
- Por modelo: `ŷ^(20) < 0` o `< τ_sell`
- Stop-loss: -7% (5-8% según volatilidad del activo)
- Take-profit: +10% (7-12% buscando R:R ≥ 1.5)
- **Time stop**: salir si pasan 20 días sin alcanzar objetivo

**Gestión de capital**:
- Portfolio: 5-8 activos con rotación activa
- Selección: momentum relativo (ranking por predicción)
- Rebalanceo: semanal o quincenal

## Métricas de Evaluación

**ML (offline)**:
- **MAE**: Mean Absolute Error sobre retornos
- **RMSE**: Root Mean Squared Error
- **Directional Accuracy**: % de predicciones con signo correcto
- **Spearman IC**: Correlación rank (robusta a outliers)

**Trading (online)**:
- **CAGR**: Retorno anualizado
- **Sharpe/Sortino**: Retorno ajustado por riesgo downside
- **Max Drawdown**: Pérdida máxima desde peak
- **Profit Factor**: Ganancia total / Pérdida total
- **Average Holding Time**: Duración promedio de trades
- **Exposure**: % tiempo en mercado

## 📁 Estructura de Resultados

Similar a E1, cada ejecución generará:

```
runs/e2_moderate/<timestamp>/
├── config_used.yaml
├── summary_all.csv
├── NVDA/
│   ├── NVDA_predictions.csv
│   ├── NVDA_scaler.csv
│   └── NVDA_summary.csv
└── GOOGL/
    └── ...
```

## 🚀 Comandos de Uso

**Una vez implementado**:

```bash
# Descargar datos (usa mismo descargador que E1)
python -m src.data.download_daily

# Entrenar E2 para tickers específicos
python -m src.train_e2_pipeline --tickers NVDA GOOGL AMZN

# Todos los tickers E2
python -m src.train_e2_pipeline

# Ver resultados
cat runs/e2_moderate/*/summary_all.csv
```

## 🔄 Próximos Pasos para Implementación

1.  **Adaptar build_features_e2.py** desde E1:
   - Agregar features momentum (RSI, Stochastic, ROC)
   - Agregar MACD, OBV
   - Reducir lookback a 60 días

2.  **Crear e2_lstm.py**:
   - Copiar estructura desde `e1_gru.py`
   - Reemplazar GRU → LSTM layers
   - Ajustar hiperparámetros (patience=12, epochs=150)

3.  **Crear train_e2_pipeline.py**:
   - Copiar desde `train_e1_pipeline.py`
   - Cambiar imports (E2 features, LSTM model)
   - Ajustar horizon=20, lookback=60

4.  **Implementar rules_e2.py**:
   - Filtros MACD + RSI + volume z-score
   - Stops más ajustados (7% vs 10% de E1)

## Variantes Disponibles

### E2 Moderate (Este documento)
- ✅ Walk-forward validation (5 folds)
- ✅ 2 capas LSTM (128 → 64 units)
- ✅ Filtros complejos (MACD, RSI, volume zscore)
- ✅ Decision score con perfil "moderate" (4 métricas)
- 📋 Uso: Evaluación rigurosa, producción

### E2 Simple (Ver [README_E2_SIMPLE.md](README_E2_SIMPLE.md))
- ⚡ Split temporal simple (sin walk-forward)
- ⚡ 2 capas LSTM (128 → 64 units, misma arquitectura)
- ⚡ Filtros completos (RSI + MACD + volume zscore)
- ⚡ Decision score con 5 métricas (IC, accuracy, sharpe, MAE, RMSE)
- 🚀 Uso: Prototipado rápido, investigación

## Referencias

- [README general](README.md) - Overview del proyecto
- [README_E1.md](README_E1.md) - Estrategia E1 Conservative
- [README_E1_SIMPLE.md](README_E1_SIMPLE.md) - Estrategia E1 Simple (patrón análogo)
- [README_E2_SIMPLE.md](README_E2_SIMPLE.md) - Estrategia E2 Simple (esta versión simplificada)
