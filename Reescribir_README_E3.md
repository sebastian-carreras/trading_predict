# E3 - Estrategia Intradía (LSTM Ensemble)

**Horizonte**: 30 minutos | **Frecuencia**: 5-min barras | **Perfil**: Alto riesgo, alta frecuencia

Predicción de retornos a 30 minutos usando ensemble de LSTM para trading intradía con ejecución automatizada.

## 📋 Resumen de la Estrategia

**Objetivo**: Capturar movimientos de corto plazo intradía con adaptación rápida a no-estacionariedades del mercado.

**Características**:
- **Target**: Retorno a 6 barras de 5-min (30 minutos)
- **Lookback**: 96 barras (~8 horas de trading)
- **Rebalanceo**: Continuo intradía
- **Umbrales**: τ_buy = 0.001, τ_sell = 0.001 (~0.10% en log-retorno)
- **Costos**: 20 bps round-trip (comisión + slippage)
- **Delay ejecución**: 1 barra (realista)

**Métricas objetivo**:
- Profit Factor neto > 1.2-1.4
- Max Drawdown intradía < 5%
- Win rate > 50% (no obligatorio si payoff asimétrico)
- Control estricto de slippage

## Arquitectura del Modelo

### LSTM Ensemble

**Justificación**: Ensembles de LSTM con ponderación online según performance reciente demuestran capacidad de adaptarse a no-estacionariedades del mercado. Específicamente diseñados para trading de alta frecuencia.

```
Ensemble de 3 modelos LSTM independientes (diferentes seeds):

Modelo base:
Input: (sequence_length=96 barras, features=7)
↓
LSTM (hidden_size=64, num_layers=2)
↓
Dropout (0.2)
↓
Output (1): retorno a H=6 barras

Agregación: promedio simple (baseline)
Extensión: weighted average por performance reciente (validación rolling)
```

**Hiperparámetros**:
- Optimizer: Adam, lr=0.001
- Loss: Huber (SmoothL1) - robusta a spikes intradía
- Batch size: 256 (mayor por alta frecuencia)
- Max epochs: 30 (re-entrenamiento diario)
- Early stopping: patience=5
- Regularización: dropout 0.2 + gradient clipping

## 📂 Archivos Implementados

| Archivo | Propósito | Estado |
|---------|-----------|--------|
| `src/train_e3_pipeline.py` | Pipeline completo (download/run/backtest) |  |
| `src/data/intraday_yfinance.py` | Descarga OHLCV 5-min (Yahoo Finance) |  |
| `src/features/intraday.py` | Features intradía (7 indicadores) |  |
| `src/models/e3_lstm.py` | LSTMRegressor PyTorch |  |
| `src/backtest/intraday.py` | Backtesting con costos 20 bps |  |
| `src/reporting/intraday_metrics.py` | Métricas (MAE/RMSE/IC/Directional) |  |

## Features Calculadas (7 indicadores baseline)

**OHLCV 5-min** (siempre disponibles, sin leakage):

1. **`ret_1`** - Log-retorno 1 barra (5 min)
2. **`ret_12m`** - Retorno rolling ~1h (suma 12 barras)
3. **`vol_24`** - Volatilidad rolling ~2h (std 24 barras)
4. **`vol_96`** - Volatilidad rolling ~1 día (std 96 barras)
5. **`atr_14`** - Average True Range normalizado
6. **`vol_z_96`** - Z-score volumen (rolling 96 barras)
7. **`tod_sin`, `tod_cos`** - Time-of-day cíclico (captura patrones horarios)

**Nota**: Diseño minimalista para evitar overfitting en alta frecuencia. Features adicionales (order flow, bid-ask spread) requieren datos nivel 2.

## Lógica de Señales de Trading

**Criterio de compra** (posición larga):
- **Condición principal**: `ŷ^(6) > τ_buy` Y `ŷ^(6)` > (costos + slippage)
- **Umbral**: 0.10%–0.35% para 30 min (depende del activo)
- **Filtros**:
  - RSI(9) < 70 (evitar extremos)
  - Spread < umbral de liquidez
  - No operar primeros/últimos 30 min de sesión

**Criterio de salida**:
- **Stop-loss**: 1–1.5 × ATR(14) o fijo 0.15%–0.40%
- **Take-profit**: 1.5–3 × riesgo (R:R > 1.5)
- **Time-stop**: cerrar si pasan H barras sin materializar

**Gestión de capital**:
- Máximo 2-3 posiciones simultáneas
- Tamaño: 20-30% capital por operación
- **NO operar**: primeros 30 min (volatilidad de apertura), últimos 30 min (cierre)

## Métricas de Evaluación

**ML (offline)**:
- **Directional Accuracy**: % de predicciones con signo correcto (clave en intradía)
- **MAE/RMSE**: Error sobre retornos
- **IC**: Information Coefficient

**Trading (online)**:
- **Profit Factor**: Ganancia total / Pérdida total (muy sensible a costos)
- **Sharpe intradía**: Retorno/riesgo en ventanas de 1 día
- **Max Drawdown intradía**: Pérdida máxima en sesión
- **Slippage promedio**: Controlado y reportado (crítico)
- **Time-in-market**: % tiempo con posición abierta
- **Turnover**: Frecuencia de operaciones

## 📁 Estructura de Resultados

Cada ejecución genera outputs en `runs/e3_intraday/<timestamp>/`:

```
runs/e3_intraday/20260105_143022/
├── config_used.yaml              # Configuración exacta
├── SPY/
│   ├── SPY_predictions.csv       # (timestamp, y_true, y_pred)
│   ├── SPY_backtest.csv          # Serie temporal PnL
│   ├── SPY_metrics.json          # MAE/RMSE/IC/Dir_Acc
│   └── SPY_backtest_summary.json # Profit Factor, Sharpe, trades
├── QQQ/
│   └── ...
└── NVDA/
    └── ...
```

## 🚀 Comandos de Uso

### Modo 1: Descarga solo datos

```bash
# Descargar datos 5-min para tickers específicos
python -m src.train_e3_pipeline --mode download --tickers SPY QQQ

# Todos los tickers E3 (desde base.yaml)
python -m src.train_e3_pipeline --mode download
```

→ Guarda en `data/raw/intraday/<TICKER>_5min.csv`

### Modo 2: Entrenamiento + Backtest completo

```bash
# Un ticker
python -m src.train_e3_pipeline --mode run --tickers SPY

# Múltiples tickers
python -m src.train_e3_pipeline --mode run --tickers SPY QQQ AMD

# Todos los tickers E3
python -m src.train_e3_pipeline --mode run
```

**Qué hace internamente**:
1. Carga OHLCV 5-min (descarga si no existe)
2. Calcula 12 features intradía
3. Crea secuencias (96 barras → predice 6 barras)
4. Entrena ensemble de 3 LSTM
5. Genera predicciones out-of-sample
6. Ejecuta backtest con costos 20 bps + delay 1 barra
7. Reporta métricas ML + trading

### Ver resultados

```bash
# Métricas de backtest
cat runs/e3_intraday/*/SPY_backtest_summary.json

# Ver predicciones
head -20 runs/e3_intraday/*/SPY_predictions.csv

# PnL acumulado
tail -1 runs/e3_intraday/*/SPY_backtest.csv
```

## Validación Rápida

```bash
# Probar con SPY (activo líquido)
python -m src.train_e3_pipeline --mode run --tickers SPY
```

**Output esperado**:
```
Descargando SPY 5-min... ✓ 4890 barras (~60 días)
Calculando features... ✓ 7 features
Creando secuencias... ✓ 4794 muestras
Entrenando ensemble de 3 LSTM...
  Model 1: MAE=0.0012
  Model 2: MAE=0.0011
  Model 3: MAE=0.0013
  Ensemble: MAE=0.0011 Dir_Acc=52.3%
Ejecutando backtest (costos 20 bps)...
  Trades: 342
  Profit Factor: 1.15
  Sharpe: 0.78
  Max DD: -3.2%
✓ Guardado en runs/e3_intraday/20260105_143022/
```

## 🔧 Configuración

Parámetros en [`base.yaml`](src/config/base.yaml) sección `strategies.e3_intraday`:

```yaml
e3_intraday:
  frequency: "5min"
  lookback_bars: 96          # ~8 horas
  horizon_bars: 6            # 30 min
  thresholds:
    tau_buy: 0.001
    tau_sell: 0.001
  data:
    provider: "yfinance"
    period: "60d"            # Historial descargado
    interval: "5m"
  model:
    ensemble_members: 3
    lstm_hidden_size: 64
    lstm_num_layers: 2
    learning_rate: 0.001
    batch_size: 256
    max_epochs: 30
```

## ⚠ Consideraciones Importantes

**Limitaciones de datos Yahoo Finance**:
- Historial 5-min: típicamente 60 días (limitado)
- Calidad: puede haber gaps en pre-market/after-hours
- Solución producción: usar proveedor premium (Interactive Brokers, Alpaca)

**Riesgos intradía**:
- **Muy sensible a costos**: 20 bps puede eliminar edge
- **Slippage**: usar órdenes limit, no market
- **Overfitting**: validar con walk-forward estricto
- **Latencia**: delay de 1 barra es optimista (real: 1-5 barras)

## Referencias

- [README general](README.md) - Overview del proyecto
- [base.yaml](src/config/base.yaml) - Configuración E3
- [src/train_e3_pipeline.py](src/train_e3_pipeline.py) - Código fuente

---

**Estado**:  Completamente implementada y funcional
