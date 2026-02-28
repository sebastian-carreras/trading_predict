# E1 - Estrategia Conservadora (GRU)

**Horizonte**: 90 días | **Frecuencia**: diaria | **Perfil**: Bajo riesgo, largo plazo

Predicción de retornos logarítmicos acumulados a 90 días usando arquitectura GRU para inversión conservadora con rebalanceo mensual.

---

## Versiones Disponibles

| Versión | Archivo | Validación | Uso Recomendado |
|---------|---------|------------|-----------------|
| **E1 Baseline** | `src/e1/train_baseline.py`  | Time split (70/15/15)  |  Baseline académico, comparación de valor incremental |
| **E1 Conservative** | `train_e1_pipeline.py`  | Walk-forward (5 folds) | Producción, validación robusta |
| **E1 Simple** | `train_e1_simple_pipeline.py` | Time split (70/15/15)  | Desarrollo, experimentación |

---

## Quick Start

```bash
# Instalar dependencias
pip install -r requirements.txt

# E1 completo (3 versiones: baseline + simple + conservador)
python -m src.e1.train_all
python -m src.e1.train_all --tickers AAPL,MSFT

# E1 Conservative (walk-forward validation)
python -m src.e1.train_pipeline 
python -m src.e1.train_pipeline --tickers AAPL

# E1 Baseline (regresión lineal)
python -m src.e1.train_baseline --tickers AAPL

# E1 Simple (desarrollo rápido)
python -m src.e1.train_simple_pipeline --tickers AAPL

# Ver resultados
cat runs/e1_conservative/*/summary_all.csv
```

---

## TL;DR - Cómo Funciona E1 Conservative

### 1. Modelo
- **Arquitectura:** GRU 2 capas (64→32 unidades) + Dropout(0.3) + Dense(16)
- **Entrada:** 360 días de datos (~15 features optimizadas para largo plazo)
- **Salida:** Predicción de retorno logarítmico a 90 días
- **Validación:** Walk-forward (5 folds) con embargo de 90 días

### 2. Pipeline de Entrenamiento
1. **Cargar datos** - CSV de `data/clean/` (o `data/raw/daily/`)
2. **Calcular features** - 15 indicadores técnicos optimizados
3. **Crear secuencias** - Ventanas de 360 días para RNN
4. **Crear target** - Retorno logarítmico forward a 90 días
5. **Walk-forward split** - 5 folds con embargo temporal de 90 días
6. **Estandarizar** - Z-score (fit solo en train de cada fold)
7. **Entrenar GRU** - Con early stopping en validación interna
8. **Evaluar** - Métricas ML + backtest con costos por fold
9. **Agregar** - Métricas promedio de todos los folds

### 3. Generación de Señales
```
Predicción → Comparar con umbrales → Señal de trading

Si predicción ≥ tau_buy (0.02 = +2%)  → BUY
Si predicción ≤ tau_sell (0.00)       → SELL/HOLD
(Nota: valores por defecto; Optuna puede generar overrides por ticker)
```

### 4. Flujo Resumido
```
Datos OHLCV (10 años)
    ↓
Features (15 indicadores)
    ↓
Secuencias (360 días × 15 features)
    ↓
Walk-forward (5 folds)
    ↓
GRU [64→32] → Dense [16] → Output [1]
    ↓
Predicción retorno 90d
    ↓
Señal (BUY/HOLD/SELL)
    ↓
Backtest → Métricas agregadas
```

---

## Arquitectura GRU

### E1 Conservative (Producción)

```
Input: (360 días, ~15 features)
  ↓
GRU(64 units, return_sequences=True)
  ↓
Dropout(0.3)
  ↓
GRU(32 units, return_sequences=False)
  ↓
Dropout(0.3)
  ↓
Dense(16, relu)
  ↓
Output(1) - Predicción retorno 90d
```

**Hiperparámetros:**
- Optimizer: AdamW, lr=0.001
- Loss: Huber (δ=1.0) - robusta a outliers
- Batch size: 64
- Max epochs: 100
- Early stopping: patience=10
- Gradient clipping: norm=1.0

### E1 Simple (Desarrollo)

```
Input: (360 días, ~15 features)
  ↓
GRU(64 units)
  ↓
Dropout(0.2)
  ↓
Dense(16, relu)
  ↓
Output(1) - Predicción retorno 90d
```

### E1 Baseline

```
Input: (360 días, ~15 features)
  ↓
Flatten 3D→2D (secuencia completa como vector)
  ↓
LinearRegression (sklearn)
  ↓
Output(1) - Predicción retorno 90d
```

**Notas del baseline:**
- Usa el mismo target y esquema temporal que E1 Conservative para comparación justa.
- Es más rápido y simple, pero no modela dependencias temporales no lineales como GRU.

---

## Features (15 indicadores optimizados)

Las features están optimizadas para predicción a largo plazo (horizon=90 días).

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

---

## Validación Walk-Forward

E1 Conservative usa validación walk-forward para demostrar robustez temporal:

```
Fold 1: |-- Train --|-- Val --|-- Test --|
Fold 2:             |-- Train --|-- Val --|-- Test --|
Fold 3:                         |-- Train --|-- Val --|-- Test --|
Fold 4:                                     |-- Train --|-- Val --|-- Test --|
Fold 5:                                                 |-- Train --|-- Val --|-- Test --|
```

**Configuración:**
- **Folds:** 5
- **Embargo:** 90 días entre train y test (evita leakage)
- **Validación interna:** 15% del train para early stopping

**Ventajas:**
- Simula reentrenamiento periódico real
- Detecta degradación temporal del modelo
- Métricas más confiables que un solo split
- Estándar académico para predicción de series temporales financieras

---

## Métricas

### Métricas ML (calidad de predicción)
| Métrica | Descripción | Target |
|---------|-------------|--------|
| **MAE** | Error absoluto promedio | ≤ 0.03 |
| **RMSE** | Error cuadrático medio | ≤ 0.05 |
| **IC** | Information Coefficient (Spearman) | ≥ 0.05 |
| **Directional Accuracy** | % aciertos de dirección | ≥ 55% |

### Métricas de Trading (backtest)
| Métrica | Descripción | Target |
|---------|-------------|--------|
| **Sharpe** | Retorno ajustado por riesgo | ≥ 1.0 |
| **CAGR** | Retorno anualizado compuesto | > 0% |
| **Max Drawdown** | Máxima caída desde pico | ≤ 20% |
| **Profit Factor** | Ganancias / Pérdidas | > 1.5 |

---

## Uso

### CLI Options

```bash
# Entrenar todos los tickers del config
python -m src.train_e1_pipeline

# Especificar tickers
python -m src.train_e1_pipeline --tickers AAPL,MSFT,GGAL.BA

# Config personalizado
python -m src.train_e1_pipeline --config src/config/custom.yaml --tickers AAPL
```

### Airflow DAG

```bash
# Trigger desde CLI
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_conservative_pipeline \
  --conf '{"tickers": "AAPL,MSFT"}'
```

### Output Esperado

```
[1/3] AAPL
  ✓ Usando datos limpios: AAPL_daily.csv
  Samples: 2046 | Features: 15 | Lookback: 360d
  ▶ Ejecutando walk-forward (5 folds, test=auto)
    Fold 1: 2020-01-15 -> 2021-03-20 | MAE=0.0812 IC=0.182 Sharpe=0.95
    Fold 2: 2021-03-22 -> 2022-05-25 | MAE=0.0923 IC=0.156 Sharpe=0.78
    Fold 3: 2022-05-27 -> 2023-08-01 | MAE=0.0785 IC=0.203 Sharpe=1.12
    Fold 4: 2023-08-03 -> 2024-10-08 | MAE=0.0891 IC=0.168 Sharpe=0.89
    Fold 5: 2024-10-10 -> 2026-01-05 | MAE=0.0834 IC=0.191 Sharpe=1.05
  ✓ MAE=0.0849 IC=0.180 Sharpe=0.96

✓ Resultados guardados en runs/e1_conservative/20260205_120000/
```

---

## Estructura de Resultados

```
runs/e1_conservative/20260205_120000/
├── config_used.yaml                       # Config reproducible
├── summary_all.csv                        # Resumen de todos los tickers
│
├── AAPL/
│   ├── AAPL_model.pth                     # Modelo del último fold
│   ├── AAPL_predictions.csv               # Predicciones (timestamp, y_true, y_pred)
│   ├── AAPL_backtest.csv                  # Serie PnL (si time split)
│   ├── AAPL_scaler.csv                    # Normalización features (mean/std)
│   ├── AAPL_target_scaler.csv             # Normalización target
│   ├── AAPL_summary.csv                   # Métricas completas
│   │
│   │ # Walk-forward artifacts:
│   ├── AAPL_walkforward_folds.csv         # Métricas por fold
│   ├── AAPL_walkforward_predictions.csv   # Predicciones agregadas
│   ├── AAPL_walkforward_backtest.csv      # Backtest agregado
│   ├── AAPL_walkforward_metrics.png       # Gráfico IC/Sharpe por fold
│   ├── AAPL_fold1_backtest.csv            # Backtest fold 1
│   ├── AAPL_fold2_backtest.csv            # ...
│   └── ...
│
└── MSFT/
    └── ...
```

### Archivos Clave

**`summary_all.csv`** - Métricas agregadas por ticker:
```csv
ticker,n_samples,folds,lookback_days,horizon_days,ml_mae,ml_rmse,ml_ic,ml_directional_accuracy,bt_sharpe,bt_cagr,bt_max_drawdown
AAPL,2046,5,360,90,0.0849,0.1123,0.180,0.582,0.96,0.08,-0.15
```

**`{ticker}_walkforward_folds.csv`** - Métricas por fold:
```csv
fold,window,test_start,test_end,n_train,n_val,n_test,ml_mae,ml_ic,bt_sharpe
1,2020-01-15 -> 2021-03-20,2020-01-15,2021-03-20,1200,180,150,0.0812,0.182,0.95
```

**`{ticker}_model.pth`** - Modelo serializado con:
- Pesos del modelo (state_dict)
- Arquitectura (hidden_sizes, dropout, dense_units)
- Scalers (mean/std de features y target)
- Metadata (lookback_days, horizon_days, feature_names)
- Info del fold (window, test_size, gap_samples)

---

## Configuración (base.yaml)

```yaml
splits:
  method: "walk_forward"
  folds: 5
  embargo_days:
    e1: 90

strategies:
  e1_conservative:
    enabled: true
    frequency: "1D"
    lookback_days: 360        # 1 año de datos de entrada
    horizon_days: 90          # Target: retorno a 90 días
    rebalance: "monthly"

    thresholds:
      tau_buy: 0.02           # +2% predicho → señal compra (default; Optuna override por ticker)
      tau_sell: 0.00          # 0% o menos → cerrar posición (default; Optuna override por ticker)

    risk:
      stop_loss_pct: 0.10     # Stop loss del 10%
      take_profit_pct: 0.15   # Take profit del 15%

    model:
      type: "GRU"
      gru_units: [64, 32]     # 2 capas GRU
      dropout: 0.3
      recurrent_dropout: 0.2
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

## Comparación: E1 Baseline vs Conservative vs Simple

| Aspecto | E1 Baseline | E1 Conservative | E1 Simple |
|---------|-------------|-----------------|-----------|
| **Tiempo/ticker** | ~1-2 min | ~5 min | ~1 min |
| **Validación** | Walk-forward (5 folds) | Walk-forward (5 folds) | Time split (70/15/15) |
| **Modelo** | LinearRegression | GRU 2 capas (64→32) | GRU 1 capa (64) |
| **Capta no-linealidad temporal** | No | Sí | Parcial |
| **Uso ideal** | Baseline / control académico | Producción, validación robusta | Desarrollo, experimentación |
| **Robustez temporal** | Alta (múltiples períodos) | Alta (múltiples períodos) | Media (un solo período) |
| **Paper académico** | Sí (baseline obligatorio) | Sí | No |

**Cuándo usar cada uno:**
- **E1 Baseline:** establecer piso de desempeño e identificar si GRU agrega valor real.
- **E1 Conservative:** validación final y uso operativo dentro del alcance del proyecto.
- **E1 Simple:** iteración rápida, debugging y pruebas exploratorias.

---

## MLflow Tracking

Si `MLFLOW_TRACKING_URI` está configurado, el pipeline registra:

**Parámetros:**
- strategy, ticker, lookback_days, horizon_days
- gru_units, dropout, learning_rate, batch_size
- loss, early_stopping_patience, max_epochs
- split_method, seed

**Métricas:**
- val_loss
- ml_mae, ml_rmse, ml_ic, ml_directional_accuracy
- bt_sharpe, bt_cagr, bt_max_drawdown

**Run agregado (E1 Conservative):**
- `E1_Summary_<timestamp>` con artifact `summary_all.csv`
- Métricas agregadas `summary_<col>_mean` para columnas numéricas de `summary_all.csv`
- Métricas por ticker `ticker_<TICKER>_<col>` para auditoría rápida en la UI

**Artifacts:**
- models/{ticker}_model.pth
- predictions/{ticker}_predictions.csv
- backtests/{ticker}_backtest.csv
- walkforward/{ticker}_walkforward_*.csv

---

## Archivos Implementados

| Archivo | Propósito |
|---------|-----------|
| `src/e1/train_baseline.py` | Pipeline E1 Baseline (LinearRegression + walk-forward) |
| `src/e1/baseline_linear.py` | Implementación del modelo baseline y métricas asociadas |
| `src/lifecycle/promotion.py` | Lógica de scoring y decisión de promoción champion/challenger |
| `scripts/evaluation/promote_candidate.py` | CLI para evaluar y promover candidatos manualmente |
| `src/train_e1_pipeline.py` | Pipeline E1 Conservative (walk-forward) |
| `src/train_e1_simple_pipeline.py` | Pipeline E1 Simple (time split) |
| `src/features/build_features_e1.py` | Cálculo de 15 features técnicos |
| `src/features/build_sequences_e1e2.py` | Conversión a secuencias RNN |
| `src/models/e1_gru.py` | Arquitectura GRU PyTorch |
| `src/backtest/backtest_daily.py` | Backtesting con costos |
| `dockerfiles/airflow/dags/E1/` | DAGs de Airflow |

---

## Lógica de Señales de Trading

### Criterio de compra (entrada larga)
- **Condición principal**: `ŷ^(90) > τ_buy` (predicción > 2%, default; Optuna override por ticker)

### Criterio de venta (salida)
- Por modelo: `ŷ^(90) < 0` o `< τ_sell`
- Stop-loss: -10%
- Take-profit: +15%

### Gestión de capital
- Portfolio: 8-12 activos de baja correlación
- Pesos: inversamente proporcionales a volatilidad
- Rebalanceo: mensual

---

## Model Promotion (Champion/Challenger)

El sistema usa un patrón champion/challenger para gestionar modelos en producción:

**Ciclo de vida:** `candidate → (guardrails) → (comparación) → champion → retired`

### Composite Score

La decisión de promoción se basa en un score compuesto configurable:

$$\text{score} = 0.35 \cdot \text{Sharpe} + 0.25 \cdot \text{IC} + 0.20 \cdot \text{DirAcc} + 0.20 \cdot \text{Calmar}$$

- **Margen mínimo**: el candidato debe superar al champion por ≥5% en el composite score
- **Safety net**: el candidato debe tener `bt_sharpe > 0`
- **Granularidad**: cada ticker se evalúa y promueve independientemente
- **Bootstrap**: si no hay champion, el candidato se promueve automáticamente

### Uso

```bash
# Dry-run: ver decisiones sin modificar el registry
python -m scripts.evaluation.promote_candidate

# Ejecutar promociones
python -m scripts.evaluation.promote_candidate --execute

# Tickers específicos
python -m scripts.evaluation.promote_candidate --tickers AAPL,MSFT --execute

# Detalle por métrica
python -m scripts.evaluation.promote_candidate --verbose

# Auto-promoción durante entrenamiento
python -m src.e1.train_pipeline --auto-promote
```

### Configuración

Los pesos y umbrales se configuran en `src/config/base.yaml` bajo `lifecycle.promotion`:

```yaml
lifecycle:
  promotion:
    auto_promote: false
    min_improvement: 0.05
    require_positive_sharpe: true
    scoring_weights:
      bt_sharpe: 0.35
      ml_ic: 0.25
      ml_directional_accuracy: 0.20
      bt_calmar: 0.20
```

---

## Evaluación Continua

Para validar modelos en producción:

```bash
# Validación retrospectiva (out-of-time)
python scripts/evaluation/e1_retrospective_validation.py \
    --ticker AAPL \
    --train-days-ago 360 \
    --horizon 90
```

---

## Referencias

- [README_E1_SIMPLE.md](README_E1_SIMPLE.md) - E1 Simple detallada
- [COMPARISON_E1.md](COMPARISON_E1.md) - Comparación Simple vs Conservative
- [README_E1_BASELINE.md](README_E1_BASELINE.md) - Baseline para comparación
- [src/config/base.yaml](src/config/base.yaml) - Configuración
- [src/features/build_features_e1.py](src/features/build_features_e1.py) - Código de features

---

**Última actualización:** Febrero 25, 2026
**Versión:** 3.2 (incluye documentación de Model Promotion champion/challenger)
