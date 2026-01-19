# E2 Simple - Estrategia Moderada Simplificada (LSTM)

**Horizonte**: 20 días | **Frecuencia**: diaria | **Perfil**: Riesgo medio, mediano plazo, **simplificado**

Predicción de retornos acumulados a 20 días usando arquitectura LSTM simplificada (1 capa), sin validación walk-forward. Diseñado para rapidez de prototipado y evaluación rápida, manteniendo la esencia del momentum en mediano plazo.

## 📋 Resumen de la Estrategia

**Objetivo**: Capturar tendencias de corto-mediano plazo con arquitectura más simple, trade-off entre precisión y velocidad de entrenamiento.

**Características**:
- **Target**: Retorno logarítmico acumulado a 20 días (2-4 semanas)
- **Lookback**: 60 días (ventana de entrada)
- **Rebalanceo**: Semanal o quincenal
- **Umbrales**: τ_buy = 0.025 (2.5%), τ_sell = 0.00
- **Filtros**: RSI básico (35-70), sin MACD ni volume z-score
- **Split**: Temporal simple (70/15/15), **sin walk-forward**

**Métricas objetivo**:
- Information Coefficient (IC) > 0.05
- Directional Accuracy > 55%
- Sharpe neto > 1.0
- MAE < 0.03

## Arquitectura del Modelo

### LSTM (2 capas)

**Justificación**: Misma arquitectura que E2 Moderate pero sin walk-forward validation. El ciclo de iteración es mucho más rápido al usar split temporal simple, ideal para prototipado y experimentación.

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
Dense (16 units, activation='relu')
↓
Output (1): predicción de retorno a 20 días
```

**Hiperparámetros**:
- Optimizer: AdamW, lr=0.001
- Loss: Huber (δ=1.0) - robusta a outliers
- Batch size: 64
- Max epochs: 100 (vs 150 en E2 Moderate)
- Early stopping: patience=10 (vs 12 en E2 Moderate)
- Regularización: gradient clipping norm=1.0

## 📂 Archivos Asociados

| Archivo | Propósito | Estado |
|---------|-----------|--------|
| [src/train_e2_simple_pipeline.py](src/train_e2_simple_pipeline.py) | Pipeline simplificado (sin walk-forward) | ✅ Implementado |
| [dockerfiles/airflow/dags/e2_simple_pipeline.py](dockerfiles/airflow/dags/e2_simple_pipeline.py) | DAG Airflow para E2 Simple | ✅ Implementado |
| [src/config/base.yaml](src/config/base.yaml) | Configuración E2 Simple (added) | ✅ Actualizado |
| [src/features/build_features_e2.py](src/features/build_features_e2.py) | Features compartidas (E2 Moderate & Simple) | ✅ Existente |
| [src/models/e2_lstm.py](src/models/e2_lstm.py) | Modelo LSTM PyTorch | ✅ Existente |

## Features (~25 indicadores)

Mismos features que E2 Moderate (ver [README_E2.md](README_E2.md) para detalle completo):
- **Core**: Retornos (1d, 5d, 20d, 60d), volatilidad, rango, ATR, volume z-score
- **Momentum**: RSI(14), Stochastic, ROC, MACD, histograma
- **Tendencia**: SMA(50, 200), EMA(50), distancia a SMA(200), Bollinger %B, ancho
- **Contexto**: Benchmark (SPY) retornos

## Lógica de Señales de Trading

**Criterio de compra** (entrada larga):
- **Condición principal**: `ŷ^(20) > τ_buy` (predicción > 2.5%)
- **Confirmación técnica**:
  - MACD histograma > 0 y creciente **O** MACD cruce alcista
  - RSI entre 35 y 70 (evitar extremos)
  - Volume z-score(20) > 0 (participación confirmada)

**Criterio de venta** (salida):
- Por modelo: `ŷ^(20) < 0`
- Stop-loss: -7%
- Time stop: salir si pasan 20 días

**Gestión de capital**:
- Portfolio: 5-8 activos
- Selección: momentum relativo (ranking por predicción)
- Rebalanceo: semanal

## Métricas de Evaluación (Decision Score Simplificado)

**5 componentes sin walk-forward**:

1. **IC (Information Coefficient)**: Correlación Spearman entre y_true y y_pred
   - Target: IC_min = 0.05
   - Peso: 25%

2. **Directional Accuracy**: % de predicciones con signo correcto (up/down)
   - Target: directional_accuracy_min = 0.55
   - Peso: 20%

3. **Sharpe**: Retorno ajustado por riesgo del backtest simple
   - Target: sharpe_min = 1.0
   - Peso: 30%

4. **MAE (Mean Absolute Error)**: Magnitud promedio de errores
   - Target: mae_max = 0.03
   - Peso: 15%

5. **RMSE (Root Mean Squared Error)**: Penaliza errores grandes
   - Target: rmse_max = 0.05
   - Peso: 10%

**Decision Signal**: 
- Score >= 0.70 → "buy"
- Score < 0.70 → "hold"

## 📁 Estructura de Resultados

Cada ejecución genera:

```
runs/e2_simple/<timestamp>/
├── config_used.yaml              # Configuración utilizada
├── summary_all.csv               # Resumen agregado (todos los tickers)
├── NVDA/
│   ├── NVDA_predictions.csv      # Predicciones (y_true, y_pred)
│   ├── NVDA_backtest.csv         # Resultados del backtest
│   ├── NVDA_scaler.csv           # Escaladores (mean, std) de features
│   ├── NVDA_summary.csv          # Métricas por ticker (ml + bt + decision)
│   └── NVDA_model.pth            # Modelo LSTM entrenado (PyTorch)
├── GOOGL/
│   └── ...
└── AMZN/
    └── ...
```

### Contenido de `summary_all.csv`:
- `ticker`: Símbolo del activo
- `n_samples`, `n_train`, `n_val`, `n_test`: Tamaños de split
- `lookback_days`, `horizon_days`: Parámetros temporales
- `split_method`: "time_split" (no walk-forward)
- `epochs_ran`, `val_loss`: Entrenamiento
- `ml_mae`, `ml_rmse`, `ml_directional_accuracy`, `ml_ic`: Métricas ML
- `bt_sharpe`, `bt_cagr`, `bt_max_drawdown`, `bt_calmar`, `bt_profit_factor`, etc.: Métricas trading
- `decision_score`, `decision_signal`: Scoring y señal final

## 🚀 Comandos de Uso

### Línea de comandos

```bash
# Entrenar E2 Simple para tickers específicos
python -m src.train_e2_simple_pipeline --tickers NVDA,GOOGL,AMZN

# Todos los tickers E2 Simple del config
python -m src.train_e2_simple_pipeline

# Omitir descarga (reutilizar datos)
python -m src.train_e2_simple_pipeline --skip-download

# Omitir limpieza (usar raw data)
python -m src.train_e2_simple_pipeline --skip-cleaning

# Omitir ambos (máxima velocidad, si datos ya existen)
python -m src.train_e2_simple_pipeline --skip-download --skip-cleaning
```

### Airflow DAG

```bash
# Activar DAG en la UI de Airflow (schedule: Lunes 4 AM, después de E2 Moderate)
# O triggear manualmente:

airflow dags trigger e2_simple_pipeline \
  --conf '{"tickers": "NVDA,GOOGL,AMZN", "skip_download": false}'
```

**Parámetros DAG**:
- `tickers`: CSV de tickers a entrenar (default: config e2_simple)
- `skip_download`: Si True, usa datos descargados previamente (default: False)
- `skip_cleaning`: Si True, usa datos raw sin limpiar (default: False)

### Ver resultados

```bash
# Resumen agregado de última ejecución
head -5 runs/e2_simple/*/summary_all.csv | tail -10

# Predicciones de un ticker
cat runs/e2_simple/*/NVDA/NVDA_predictions.csv | head -20

# Métricas de trading (backtest)
cat runs/e2_simple/*/NVDA/NVDA_backtest.csv | tail -1

### MLflow Tracking

El DAG de E2 Simple loguea en MLflow:

**Params (25+)**:
- Model: `lstm_units`, `lookback_days`, `horizon_days`, `dropout`, `epochs`, etc.
- Filters: `rsi14_min`, `rsi14_max`, `macd_confirmation`, `volume_zscore_window`, `volume_zscore_min`
- Thresholds: `tau_buy`, `tau_sell`
- Targets: `ic_target_min`, `sharpe_target_min`, `decision_score_threshold` ⭐

**Metrics (12+)**:
- ML: `mae`, `rmse`, `ic`, `directional_accuracy`
- Backtest: `sharpe`, `cagr`, `max_drawdown`, `calmar`, `profit_factor`, `win_rate`, `num_trades`
- Decision: `decision_score`, `decision_signal`

**Summary Run Metrics** (agregadas de todos los tickers):
- `ic_mean`, `ic_median`, `ic_min`, `ic_max`, `ic_positive_count`, `ic_above_threshold`
- `sharpe_mean`, `sharpe_median`, `sharpe_min`, `sharpe_max`, `sharpe_above_threshold`
- `decision_score_mean`, `decision_score_median`, `decision_score_min`, `decision_score_max`
- `decision_buy_signals`, `total_tickers`, `successful_tickers`

**Artifacts (3+)**:
- `predictions.csv`: Predicciones LSTM con precios reales
- `model.pth`: Checkpoint del modelo PyTorch
- `scaler.csv`: Escalador MinMaxScaler para inferencia

**Targets en Summary** ⭐:
Los valores objetivo/mínimos se loguean como **params** en el Summary Run para facilitar la comparación:
- `ic_target_min`: 0.05 (IC mínimo esperado)
- `sharpe_target_min`: 1.0 (Sharpe Ratio mínimo esperado)
- `decision_score_threshold`: 0.7 (threshold para señal de compra)

```bash
# Ver runs en MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

# Filtrar runs con IC > target
# En la UI: Metrics -> ic_mean > 0.05
```
```

## 🔄 Comparación: E2 Simple vs E2 Moderate

| Aspecto | E2 Simple | E2 Moderate |
|---------|-----------|-------------|
| LSTM units | [128, 64] (2 capas) | [128, 64] (2 capas) |
| Dropout | 0.2 | 0.2 |
| Dense units | 16 | 32 |
| Max epochs | 100 | 150 |
| Early stopping patience | 10 | 12 |
| Validation | Time split simple | Walk-forward (5 folds) |
| Decision score | 5 métricas | Perfil "moderate" |
| Filtros | RSI + MACD + volume | RSI + MACD + volume |
| Tiempo training | ~5-10 min/ticker | ~15-30 min/ticker |
| **Uso ideal** | Prototipado rápido, experimentos | Evaluación robusta, producción |

## 📚 Diferencias vs E1 Simple

- **Target**: 20d (E2) vs 90d (E1)
- **Lookback**: 60d (E2) vs 180d (E1)
- **Features**: Momentum-centric (E2) vs tendencia-centric (E1)
- **Rebalanceo**: Semanal (E2) vs mensual (E1)
- **Modelo**: LSTM (E2) vs GRU (E1)

## 🔗 Referencias Relacionadas

- [README_E2.md](README_E2.md) - Estrategia E2 Moderate (versión con walk-forward)
- [README_E1_SIMPLE.md](README_E1_SIMPLE.md) - E1 Simple (patrón similar en E1)
- [README_E1.md](README_E1.md) - E1 Conservative
- [README_WALK_FORWARD.md](README_WALK_FORWARD.md) - Validación walk-forward (no usado en Simple)
- [base.yaml](src/config/base.yaml) - Configuración centralizada

## 🚀 Próximos Pasos

1. **Entrenar para todos los tickers**: Usar DAG Airflow o línea de comandos
2. **Monitorear resultados**: Comparar IC, Sharpe y decision scores entre tickers
3. **Ajustar hiperparámetros**: Si resultados pobres, tunear epsilon_buy, LSTM units, epochs
4. **Evaluar vs E2 Moderate**: Comparar validación simple vs walk-forward (ver [WALK_FORWARD_PARAMS_VERIFICATION.md](docs/WALK_FORWARD_PARAMS_VERIFICATION.md))
5. **Producción**: Usar E2 Moderate para production-ready; E2 Simple para investigación rápida

---

**Estado**: ✅ Implementado y listo para usar  
**Última actualización**: 2026 (matching E1 Simple & E2 Moderate patterns)
