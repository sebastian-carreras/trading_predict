# Backtesting en Trading Predict

## ¿Qué es Backtesting?

El backtesting simula cómo habría funcionado tu estrategia de trading en el pasado usando datos históricos. Es esencial para:

- Validar que el modelo no solo predice bien, sino que **genera ganancias**
- Entender riesgos (drawdowns, volatilidad)
- Calcular costos de transacción realistas
- Comparar diferentes estrategias objetivamente

## Implementación E1 (Conservative)

### Flujo del Backtest

```
Predicciones → Señales → Posiciones → PnL → Métricas
```

**1. Predicciones del Modelo**```python
y_pred = [0.08, -0.03, 0.05, 0.12, ...]  # Retorno predicho a 90 días
```

**2. Generación de Señales**```python
# Umbral de compra: 6% (tau_buy = 0.06)
if y_pred >= 0.06:
    signal = BUY  # Comprar
elif y_pred <= 0.00:
    signal = SELL  # Cerrar posición
else:
    signal = HOLD  # Mantener
```

**3. Gestión de Posiciones**
- **Holding Period**: 90 días (igual que el horizonte de predicción)
- **Rebalanceo**: Solo después de 90 días o si estamos flat
- **Max Position**: 100% del capital (E1 es single-stock, no portfolio)
- **Shorts**: NO permitidos (estrategia conservadora)

**4. Cálculo de PnL**```python
# Retorno bruto diario
gross_ret = position * daily_return

# Costos de transacción (10 bps = 0.1%)
costs = position_change * 0.001

# Retorno neto
net_ret = gross_ret - costs
```

**5. Equity Curve**```python
equity = initial_capital * exp(cumsum(net_ret))
```

## Métricas de Backtesting

### Retorno
- **Total Return**: `(equity_final / equity_initial) - 1`
- **CAGR**: Retorno anualizado compuesto
  ```
  CAGR = (1 + total_return)^(1/years) - 1
  ```

### Riesgo
- **Sharpe Ratio**: Retorno ajustado por volatilidad
  ```
  Sharpe = (mean_ret / std_ret) * sqrt(252)
  ```
  - > 1.0 = Bueno
  - > 2.0 = Excelente
  - < 0 = Perdiendo dinero

- **Max Drawdown**: Máxima caída desde peak
  ```
  MDD = max((running_max - equity) / running_max)
  ```
  - < 10% = Muy bueno
  - 10-20% = Aceptable
  - > 30% = Riesgoso

- **Calmar Ratio**: CAGR / Max Drawdown
  - > 1.0 = Bueno (retorno > riesgo)

### Trading
- **Profit Factor**: Ganancias / Pérdidas
  ```
  PF = sum(returns > 0) / abs(sum(returns < 0))
  ```
  - > 1.5 = Bueno
  - > 2.0 = Excelente

- **Win Rate**: % de días con retorno positivo
  ```
  WR = wins / total_trades
  ```
  - > 55% = Bueno para acciones

- **Num Trades**: Cantidad de cambios de posición
  - Bajo = estrategia buy & hold
  - Alto = estrategia activa (más costos)

## 🔧 Configuración en base.yaml

```yaml
strategies:
  e1_conservative:
    thresholds:
      tau_buy: 0.06      # Comprar si predicción > 6%
      tau_sell: 0.00     # Vender si predicción <= 0%
    
    backtest:
      holding_period_days: 90   # Mantener 90 días
      allow_short: false        # Solo long
      max_position: 1.0         # 100% capital

costs:
  daily_round_trip_bps: 10  # 10 bps = 0.1% (entrada+salida)
```

## Outputs del Backtest

### `{ticker}_backtest.csv`
```csv
timestamp,pos,signal,gross_ret,costs,net_ret,equity,turnover
2024-01-02,1.0,1.0,0.0123,0.0001,0.0122,100122.50,1.0
2024-01-03,1.0,0.0,0.0056,0.0000,0.0056,100683.60,0.0
...
```

### `{ticker}_summary.csv`
```
ticker: AAPL
ml_ic: 0.145
bt_sharpe: 1.85
bt_cagr: 0.187
bt_max_drawdown: 0.123
bt_calmar: 1.52
bt_profit_factor: 2.34
bt_win_rate: 0.58
bt_num_trades: 12
```

## 🎨 Visualización en MLflow

En MLflow UI podrás:

1. **Comparar métricas de trading**:
   ```
   Filter: metrics.bt_sharpe > 1.5 AND metrics.bt_max_drawdown < 0.15
   ```

2. **Scatter Plot**: IC vs Sharpe
   - Ver si alta predictibilidad → alta rentabilidad

3. **Download backtest CSV** para análisis avanzado

## 🔍 Ejemplo de Análisis

### Ticker: AAPL

**Métricas ML**:
- IC = 0.12 (correlación predicción-realidad)
- MAE = 0.16 (error promedio)

**Métricas Backtesting**:
- Sharpe = 1.8  (buen retorno ajustado)
- CAGR = 15% 
- Max DD = 18% ⚠ (aceptable pero alto)
- Win Rate = 56% 
- Profit Factor = 2.1 

**Conclusión**: Modelo predice decentemente (IC=0.12) y genera retornos sólidos en backtest (Sharpe=1.8). Max DD de 18% requiere gestión de riesgo.

## ⚖ ML Metrics vs Trading Metrics

| Métrica ML | Trading Metric | Relación |
|------------|----------------|----------|
| IC (correlation) | Sharpe Ratio | Alta correlación ≠ alta rentabilidad |
| Directional Acc | Win Rate | Dirección correcta ≠ magnitud correcta |
| MAE/RMSE | Profit Factor | Error bajo ≠ ganancias |

**Lección**: Un modelo con IC bajo puede ser rentable si:
- Predice bien las grandes movidas
- Filtra bien las señales (thresholds)
- Gestiona bien el riesgo (stops, position sizing)

## 🚨 Limitaciones del Backtest

1. **Slippage**: No simulado (asume fills a precio close)
2. **Market Impact**: No considera que tus trades afecten el precio
3. **Shorting Costs**: Borrow fees no incluidos
4. **Dividendos**: No incluidos en retornos
5. **Survivorship Bias**: Solo tickers que sobrevivieron (no bankruptcies)

**Solución**: Costos conservadores (10 bps) compensan parcialmente.

## Próximos Pasos

1. **Ejecutar E1 con backtest**:
   ```bash
   # Trigger DAG en Airflow UI con ticker específico
   {"tickers": "AAPL"}
   ```

2. **Revisar en MLflow**:
   - Experimento: E1_Conservative_Strategy
   - Métricas: bt_sharpe, bt_max_drawdown, bt_cagr

3. **Iterar hiperparámetros**:
   - Ajustar tau_buy/tau_sell
   - Probar allow_short = true
   - Experimentar con holding_period

---

**Autor**: Sebastian Carreras - FIUBA AI Posgrado  
**Última actualización**: 2026-01-06
