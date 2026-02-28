# Baseline E1: Regresión Lineal Simple

Implementación de baseline para comparar contra el modelo GRU de la estrategia E1 (Conservadora).

## 📋 Objetivo

Establecer un punto de referencia simple usando **Regresión Lineal** para evaluar si la complejidad del modelo GRU aporta valor predictivo real.

## 🎯 Métricas de Comparación

### ML Metrics (Offline)
- **MAE** (Mean Absolute Error): Error absoluto promedio
- **RMSE** (Root Mean Squared Error): Error cuadrático medio
- **Directional Accuracy**: % de predicciones con signo correcto
- **IC** (Information Coefficient): Correlación de Spearman entre predicciones y valores reales
  - IC > 0.05: Significativo en finanzas
  - IC < 0: Indica overfitting o falta de capacidad predictiva

### Trading Metrics (Online - Backtest)
- **Sharpe Ratio**: Retorno ajustado por riesgo
- **Sortino Ratio**: Retorno ajustado por downside risk
- **CAGR**: Retorno anualizado compuesto
- **Max Drawdown**: Peor caída desde máximo
- **Calmar Ratio**: CAGR / Max Drawdown
- **Profit Factor**: Ganancias brutas / Pérdidas brutas
- **Hit Rate**: % operaciones ganadoras

## 🚀 Uso

### 1. Entrenar Baseline

```bash
# Entrenar baseline para un ticker
python -m src.train_e1_baseline --tickers AAPL

# Entrenar para múltiples tickers
python -m src.train_e1_baseline --tickers AAPL GOOGL MSFT

# Con comparación automática
python -m src.train_e1_baseline --tickers AAPL --compare-with-gru
```

### 2. Entrenar GRU (para comparación)

```bash
# Entrenar modelo GRU original
python -m src.train_e1_pipeline --tickers AAPL
```

### 3. Comparar Modelos

```bash
# Comparación automática (usa últimos runs)
python scripts/evaluation/compare_e1_models.py

# Especificar runs manualmente
python scripts/evaluation/compare_e1_models.py \
  --baseline-run runs/e1_baseline/20260115_120000 \
  --gru-run runs/e1_conservative/20260115_130000
```

## 📊 Outputs

### Baseline Training
```
runs/e1_baseline/<timestamp>/
├── <TICKER>/
│   ├── <TICKER>_baseline_predictions.csv    # Predicciones por fold
│   ├── <TICKER>_baseline_folds.csv          # Métricas por fold
│   ├── <TICKER>_baseline_backtest.csv       # Serie temporal de backtest
│   └── <TICKER>_baseline_summary.csv        # Resumen de métricas
├── baseline_summary_all.csv                 # Consolidado todos los tickers
└── config_used.yaml                         # Config usada
```

### Comparison Report
```
reports/
├── e1_model_comparison.csv                  # Tabla comparativa
└── e1_model_comparison.md                   # Reporte en Markdown
```

## 🧮 Arquitectura del Baseline

### Regresión Lineal Simple
- **Input**: Secuencias 3D (n_samples, 360 días, ~27 features) → aplanadas a 2D
- **Modelo**: `sklearn.LinearRegression` estándar
- **Features**: Mismas que GRU (indicadores técnicos, retornos, volatilidad, etc.)
- **Target**: Retorno acumulado a 90 días

### ¿Por qué Regresión Lineal?
✅ Simple y rápida de entrenar
✅ Interpretable (pesos lineales)
✅ Baseline clásico en ML
✅ Sin riesgo de overfitting por arquitectura compleja
✅ Demuestra si patrones no-lineales/temporales valen la pena

## 📈 Interpretación de Resultados

### Si GRU >> Baseline:
- ✅ La arquitectura recurrente captura patrones temporales valiosos
- ✅ Justifica la complejidad del modelo
- ✅ Vale la pena usar deep learning

### Si Baseline ≈ GRU:
- ⚠️ Relaciones lineales son suficientes
- ⚠️ GRU puede estar overfitting
- ⚠️ Considerar simplificar arquitectura

### Si Baseline > GRU:
- ❌ GRU está overfitting
- ❌ Revisar regularización, data leakage, o hiperparámetros
- ❌ Considerar más datos o features diferentes

## 🔍 Ejemplo de Output

```
COMPARACIÓN: Regresión Lineal Baseline vs GRU
====================================================================================================

📊 MÉTRICAS ML (Offline)
----------------------------------------------------------------------------------------------------
MAE                  | Base:     0.0342 | GRU:     0.0298 | Δ:    -0.0044 |  -12.87% ✓
RMSE                 | Base:     0.0521 | GRU:     0.0467 | Δ:    -0.0054 |  -10.36% ✓
Dir. Accuracy        | Base:     0.5423 | GRU:     0.5891 | Δ:    +0.0468 |   +8.63% ✓
IC (Spearman)        | Base:     0.0312 | GRU:     0.0589 | Δ:    +0.0277 |  +88.78% ✓

💰 MÉTRICAS TRADING (Online - Backtest)
----------------------------------------------------------------------------------------------------
Sharpe Ratio         | Base:     0.6734 | GRU:     0.9123 | Δ:    +0.2389 |  +35.47% ✓
CAGR                 | Base:     0.0823 | GRU:     0.1156 | Δ:    +0.0333 |  +40.46% ✓
Max Drawdown         | Base:    -0.1834 | GRU:    -0.1267 | Δ:    +0.0567 |  +30.91% ✓

RESUMEN
====================================================================================================
Total métricas comparadas: 11
GRU superior en: 9 métricas (81.8%)
Baseline superior en: 2 métricas (18.2%)

🏆 Top 3 mejoras del GRU:
  ✓ IC (Spearman): +88.78%
  ✓ CAGR: +40.46%
  ✓ Sharpe Ratio: +35.47%
```

## ⚙️ Configuración

El baseline usa la misma configuración que E1 del archivo `src/config/base.yaml`:

```yaml
strategies:
  e1_conservative:
    lookback_days: 360
    horizon_days: 90
    thresholds:
      tau_buy: 0.06
      tau_sell: 0.00
```

## 🔬 Validación Walk-Forward

- **Splits**: Mismo esquema temporal que GRU (5 folds por defecto)
- **Embargo**: Gap temporal para evitar data leakage
- **Normalización**: StandardScaler fit solo en train, aplicado a val/test
- **Sin shuffle**: Respeta orden temporal

## 📚 Referencias

- Baseline según especificación en [README.md](../README.md)
- Configuración en [src/config/base.yaml](../src/config/base.yaml)
- Documentación E1: [README_E1.md](../README_E1.md)

## ⚠️ Limitaciones

- **No captura dependencias temporales**: Aplana secuencias, pierde orden
- **Asume linealidad**: No modela interacciones complejas
- **Overfitting potencial**: Muchas features (360 días × 27 features = 9720 variables)
- **No usa memoria**: Cada predicción es independiente

## ✅ Próximos Pasos

1. Ejecutar baseline para mismo universo que GRU
2. Comparar resultados usando `scripts/evaluation/compare_e1_models.py`
3. Analizar diferencias en métricas ML vs Trading
4. Documentar conclusiones para tesis
5. Considerar otros baselines: SMA Crossover, Buy & Hold
