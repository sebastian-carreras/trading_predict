# Trading Predict

Sistema de trading automatizado con Machine Learning implementando cuatro estrategias diferenciadas por horizonte temporal y perfil de riesgo.

## Objetivo del Proyecto

Desarrollar e implementar un sistema automatizado de predicción y ejecución de operaciones en mercados financieros, utilizando técnicas de Machine Learning (redes neuronales recurrentes GRU/LSTM, k-NN) y análisis técnico para cuatro estrategias con diferentes horizontes y perfiles de riesgo.

## Modos de Ejecución

### **Producción (Docker - RECOMENDADO)**

Stack completo con Airflow (orquestación) + MLflow (tracking) + FastAPI (API):

```bash
# Configurar variables de entorno
cp .env.example .env

# Levantar stack completo
docker-compose --profile all up -d

# Acceder a interfaces
# - Airflow UI: http://localhost:8080
# - MLflow UI: http://localhost:5000
# - FastAPI Docs: http://localhost:8800/docs
```

 **Ver documentación completa**: [README_DOCKER.md](README_DOCKER.md)

### **Desarrollo Local**

Para debugging y desarrollo rápido:

```bash
# Activar ambiente Python
conda activate ia_ceia_18co

# Ejecutar pipeline manualmente
python -m src.train_e1_pipeline --tickers AAPL
python -m src.train_e2_pipeline --tickers AAPL
```

## Estrategias Implementadas

| Estrategia | Modelo | Horizonte | Target | Perfil | README |
|------------|--------|-----------|--------|--------|---------|
| **E1** Conservadora | GRU | 90 días | Retorno acumulado | Bajo riesgo, largo plazo | [README_E1.md](README_E1.md) |
| **E2** Moderada | LSTM | 20 días | Retorno acumulado | Riesgo medio, mediano plazo | [README_E2.md](README_E2.md) |
| **E3** Intradía | LSTM Ensemble | 30 min | Retorno 6 barras | Alto riesgo, alta frecuencia | [README_E3.md](README_E3.md) |
| **E4** Pairs Trading | k-NN + OU | 10 días | Spread cointegrado | Market-neutral | [README_E4.md](README_E4.md) |

Ver documentación específica de cada estrategia para detalles de implementación, uso y configuración.

### Estructura del Proyecto
```
trading_predict/
├── data/              # Datos (raw → clean → features)
├── src/               # Código fuente modular
│   ├── config/        # Configuración centralizada (base.yaml)
│   ├── data/          # Descarga y limpieza
│   ├── features/      # Feature engineering
│   ├── models/        # Arquitecturas de redes neuronales
│   ├── backtest/      # Motor de backtesting
│   └── reporting/     # Métricas y visualizaciones
├── reports/           # Outputs finales (figuras/tablas)
├── runs/              # Historial de experimentos (timestamped)
└── notebooks/         # Análisis exploratorio
```

**Filosofía de diseño**: Separación clara datos → código → resultados para máxima reproducibilidad y trazabilidad.

## Configuración Centralizada

Todos los parámetros están en [`src/config/base.yaml`](src/config/base.yaml):

- **Universo de tickers** por estrategia
- **Costos de transacción**: 10 bps (diario), 20 bps (intradía)
- **Horizontes de predicción**: E1=90d, E2=20d, E3=30min, E4=10d
- **Umbrales de trading**: τ_buy, τ_sell por estrategia
- **Hiperparámetros de modelos**: arquitecturas, learning rates, regularización
- **Criterios de decisión**: perfiles de scoring (`decision_profiles`) con objetivos/pesos/threshold por estrategia

## Ejecutar Estrategias

Ver README específico de cada estrategia para comandos detallados:

**E1 - Conservadora (GRU)**```bash
python -m src.data.download_daily
python -m src.train_e1_pipeline --tickers AAPL
```
→ Ver [README_E1.md](README_E1.md)

**E2 - Moderada (LSTM)**```bash
python -m src.data.download_daily
python -m src.train_e2_pipeline --tickers AAPL
```
→ Ver [README_E2.md](README_E2.md)

La ejecución de E2 guarda artefactos por ticker en `runs/e2_moderate/<timestamp>/<TICKER>/`, incluyendo:
- `*_predictions.csv`, `*_summary.csv` y el modelo entrenado `*_model.pth`.

**E3 - Intradía (Ensemble)**```bash
python -m src.train_e3_pipeline --mode run --tickers SPY
```
→ Ver [README_E3.md](README_E3.md)

**E4 - Pairs Trading (k-NN)**```bash
# TODO: pendiente implementación
```
→ Ver [README_E4.md](README_E4.md)

## Métricas de Evaluación

**Métricas ML (offline)**:
- MAE, RMSE: error de predicción
 - El MAE mide el error promedio absoluto, siendo más robusto ante outliers, mientras que el RMSE penaliza más los errores grandes debido a la elevación al cuadrado.
- Directional Accuracy: % de predicciones con signo correcto
 - Directional Accuracy mide el porcentaje de predicciones donde el modelo acierta el signo correcto del movimiento (alza o baja), independientemente de la magnitud. Accuracy superior al 50% indica capacidad predictiva.
- IC (Information Coefficient): correlación predicción vs realidad
 - Information Coefficient (IC) representa la correlación de Spearman entre las predicciones del modelo y los retornos reales, evaluando la capacidad del modelo de rankear correctamente los activos. IC > 0.05 suele considerarse significativo en finanzas. IC < 0 indica overfitting o falta de capacidad predictiva.


**Criterio compuesto de decisión (`decision_score`)**:
- Se calcula por **perfil** y estrategia (ej: E1 → `conservative`, E2 → `moderate`, E3 → `aggressive`).
- La estrategia elige el perfil vía `splits.strategy_profile`, y cada perfil define `threshold`, `weights` y `target_metrics` en `splits.decision_profiles`.
- Los componentes se registran como `decision_component_<metric>` (dinámico) + `decision_profile`.
- Se mantiene compatibilidad con el esquema legacy `splits.target_metrics` + `splits.decision_score` si no hay perfiles configurados.

**Cómo conviven `tau_buy`/`tau_sell` con `decision_score`**:
- `tau_buy` y `tau_sell` (por estrategia) definen la lógica de **señales y trading** a partir de la predicción: cuándo entrar/salir en el backtest.
- `decision_score` es un criterio **macro de calidad (go/no-go)** del run: resume métricas ML + trading y produce `decision_signal` (BUY/HOLD) comparando contra un `threshold`.
- En otras palabras: los *taus* gobiernan el comportamiento del backtest (trades), y el `decision_score` gobierna si “habilitamos” recomendar/operar esa estrategia para ese ticker.


**Métricas de Trading (online)**:
- CAGR, Sharpe, Sortino: retorno ajustado por riesgo
 - El CAGR (Compound Annual Growth Rate) mide el retorno anualizado compuesto de la estrategia/inversión.
 - Nota: en ventanas menores a 1 año, el CAGR sigue siendo válido como **anualización**, pero puede volverse más ruidoso/volátil (la anualización amplifica retornos cortos). Por eso conviene interpretarlo junto con Max Drawdown, Sharpe/Sortino, Calmar, Profit Factor y hit rate.
 - El Sharpe Ratio evalúa el retorno ajustado por riesgo de una inversión o estrategia de trading, siendo una métrica estándar para comparar estrategias. Se compara con un retorno libre de riesgo y la volatilidad de la estrategia/inversión (objetivos típicos: E1 ≥0.9, E2 ≥0.8, E4 ≥1.0) 
 - El Sortino Ratio se enfoca en la volatilidad negativa (downside), siendo más relevante para inversionistas que solo se preocupan por pérdidas.
- Profit Factor, Hit Rate: calidad de operaciones
 - El Profit Factor es el ratio entre ganancias brutas totales y pérdidas brutas totales (objetivo mínimo 1.2-1.4 para estrategia intradía E3)
 - Hit Rate es el porcentaje de operaciones ganadoras sobre el total (idealmente > 50% para E1/E2, aunque depende de la estrategia) 
- Max Drawdown, Calmar: control de pérdidas
 - El Max Drawdown representa el peor escenario de pérdida que experimentó una estrategia de trading o inversión durante un período específico. Esta métrica cuantifica el "dolor" financiero máximo al que estuvo expuesto un inversionista entre un punto máximo y su subsecuente mínimo. Un Max Drawdown bajo es crucial para estrategias conservadoras (E1) y moderadas (E2), ya que refleja la capacidad de la estrategia para proteger el capital en mercados adversos.
 - El ratio de Calmar divide el CAGR por el Max Drawdown absoluto, proporcionando una medida de retorno ajustado por pérdida máxima, particularmente relevante para inversionistas con baja tolerancia al riesgo
- Turnover: frecuencia de rebalanceo
 - Turnover cuantifica la frecuencia de rebalanceo del portafolio, típicamente expresado como el valor total transaccionado dividido por el capital bajo gestión. Esta métrica es crítica porque alta frecuencia de operaciones incrementa los costos de transacción, pudiendo erosionar completamente la rentabilidad predicha por el modelo. El sistema evalúa todas las estrategias con costos de transacción incluidos (10 bps para estrategias diarias, 20 bps para intradía) en el backtesting neto, asegurando que las métricas reflejen performance ejecutable

Todas las estrategias se evalúan con **costos de transacción** incluidos (backtesting neto).

## Reproducibilidad

Cada ejecución del pipeline genera una carpeta timestamped en `runs/`:

```
runs/<estrategia>/<YYYYMMDD_HHMMSS>/
├── config_used.yaml       # Configuración exacta
├── predictions_*.csv      # Predicciones del modelo
├── summary.json           # Métricas ML y trading
└── backtest_*.csv         # Serie temporal de PnL (Profit and Loss)
```

**Ventajas**:
- Auditoría completa: saber qué parámetros generaron qué resultados
- Versionado automático de experimentos
- Facilita comparación entre runs

## Metodología Anti-Leakage

**Principios fundamentales** (críticos en tesis):

 **Features "as-of"**: toda variable en timestamp `t` usa solo datos ≤ `t`  
 **Normalización correcta**: scaler fit solo en train, aplicado a val/test  
 **Split temporal**: walk-forward (expanding/rolling) + helper `temporal_train_val_split` para evitar fugas y pérdida de muestras en validación, nunca shuffle  
 **Embargo**: gap de H días entre train y validación para evitar solapamiento  
 **Costos realistas**: backtesting incluye comisiones + slippage

## Estado del Proyecto

| Componente | Estado | Notas |
|------------|--------|-------|
| E1 - Descarga datos |  | `src/data/download_daily.py` |
| E1 - Features |  | 27 indicadores técnicos |
| E1 - Modelo GRU |  | `src/models/e1_gru.py` |
| E1 - Pipeline |  | `src/train_e1_pipeline.py` |
| E1 - Backtest |  | Implementado: `src/backtest/daily.py` (costos + señales por `tau_buy`/`tau_sell`) |
| E2 - Modelo + Pipeline |  | `src/models/e2_lstm.py`, `src/train_e2_pipeline.py` |
| E3 - Pipeline completo |  | `src/train_e3_pipeline.py` |
| E3 - Backtest |  | Implementado: `src/backtest/intraday.py` (costos intradía) |
| E4 - Implementación |  | Especificada, no codificada |

## Documentación Adicional

### Guías de Implementación
- **[README_HYPERPARAMETER_TUNING.md](README_HYPERPARAMETER_TUNING.md)** - Optimización de hiperparámetros con Optuna + MLflow
- **[README_WALK_FORWARD.md](README_WALK_FORWARD.md)** - Validación walk-forward para robustez temporal
- **[README_E1.md](README_E1.md)** - Estrategia E1 Conservadora (GRU)
- **[README_E2.md](README_E2.md)** - Estrategia E2 Moderada (LSTM)
- **[README_E3.md](README_E3.md)** - Estrategia E3 Intradía (Ensemble)
- **[README_E4.md](README_E4.md)** - Estrategia E4 Pairs Trading
- **[README_DOCKER.md](README_DOCKER.md)** - Deployment con Docker + Airflow
- **[README_BACKTESTING.md](README_BACKTESTING.md)** - Motor de backtesting
- **[README_DATA_CLEANING.md](README_DATA_CLEANING.md)** - Pipeline de datos

### Referencias Técnicas
- [Configuración base.yaml](src/config/base.yaml) - Parámetros del sistema
- [General High Level Overview.md](General%20High%20level%20Overview.md) - Visión arquitectónica
- [Especificación de modelos](secciones%20del%20plan%20de%20trabajo/3.%20Implementación%20de%20modelos%20de%20machine%20learning/) - Detalles técnicos

### Scripts y Herramientas
```bash
# Optimización de hiperparámetros (E1)
python scripts/optimize_e1_hyperparameters.py --n_trials 50

# Comparación de impacto de normalización
python scripts/compare_normalization_impact.py

# Verificación de scaling de features
python scripts/verify_scaling.py

# Limpieza de datos
python scripts/run_data_cleaning.py
```

Ver cada README específico para comandos detallados.


## Contexto Académico

Proyecto final - Posgrado en Inteligencia Artificial FIUBA  
**Fases de desarrollo** (plan de trabajo):

1.  Investigación y análisis
2.  Diseño de arquitectura
3.  Implementación de modelos ML (en progreso)
4.  Integración y pruebas
5.  Documentación y presentación

---

## Detalles de los Modelos de Machine Learning a Implementar
Este documento define una especificación **implementable** (datos → features → targets → entrenamiento → evaluación → reglas de trading) para cuatro estrategias. La prioridad es evitar **data leakage**, definir objetivos medibles y mantener coherencia entre: (a) lo que predice el modelo y (b) cómo se transforma en decisiones de compra/venta.

> Nota de enfoque: para toma de decisiones, en general es más estable modelar **retornos** (o spreads en pairs trading) que el **precio** directo. En lo que sigue, cuando se hable de “predicción de precio”, se recomienda reinterpretarlo como “predicción de retorno acumulado a horizonte $H$”.

### Convenciones y reglas comunes (aplican a los 4 modelos)

### Configuración cerrada para la tesis (supuestos y parámetros finales)

Para poder implementar y evaluar de punta a punta sin bloquearse por disponibilidad de datos, se fijan los siguientes supuestos. Si luego cambiás el mercado (ARG vs USA vs crypto), se ajustan costos y features “contexto”, pero la estructura queda igual.

**Universo de activos (cierre)**
- Estrategias 1 y 2 (diarias): acciones/ETFs líquidos (p. ej. universo de 20–50 tickers) + benchmark SPY.
- Estrategia 4 (pairs): pares dentro del mismo sector/universo anterior (p. ej. 20–100 candidatos, seleccionar top pares por cointegración).
- Estrategia 3 (intradía): OHLCV 5-min consistente. Implementación base con Yahoo Finance (vía `yfinance`, historial típico ~60 días) y universo definido en `base.yaml`.

**Costos (cierre, para backtesting neto)**
- Diario (E1/E2/E4): 10 bps round-trip (comisión + slippage), configurable como parámetro.
- Intradía (E3): 20 bps round-trip (comisión + slippage), configurable como parámetro.

**Targets y horizontes (cierre)**
- E1 Conservadora: $H=90$ días, lookback = 180 días.
- E2 Moderada: $H=20$ días, lookback = 60 días.
- E3 Intradía: $H=6$ barras de 5-min (30 min), lookback = 96 barras.
- E4 Pairs: $H=10$ días para predicción (solo como confirmación), y reglas basadas en Z-score.

**Umbrales y reglas (cierre, versión base reproducible)**
- En E1/E2 se usa umbral fijo inicial + chequeo de costos:
   - E1: $\tau_{buy}=0.06$ (6% a 90D), $\tau_{sell}=0.00$.
   - E2: $\tau_{buy}=0.025$ (2.5% a 20D), $\tau_{sell}=0.00$.
   - En ambos: solo operar si $\hat{y}^{(H)}$ excede (costos estimados).
- En E3: umbral simétrico sobre retorno predicho (30 min): $\tau_{buy}=0.001$, $\tau_{sell}=0.001$ (≈ 0.10% en log-retorno), con delay de ejecución de 1 barra.
- En E4: entrada por |Z| ≥ 2.0; salida por |Z| ≤ 0.25; stop por |Z| ≥ 3.0; time-stop 20 días.

**Selección del “mejor modelo” (cierre)**
- Selección por validación walk-forward maximizando una métrica de trading simple:
   - E1/E2: Sharpe neto en validación (con costos) con límite de turnover.
   - E3: Profit Factor neto y control de *time-in-market* (muy sensible a costos).
   - E4: Sharpe neto y estabilidad de cointegración.

**Definiciones**
- Precio: $P_t$ (idealmente **ajustado** por splits/dividendos si aplica).
- Log-retorno: $r_t = \ln(P_t / P_{t-1})$.
- Retorno a horizonte $H$ (días o barras): $y_t^{(H)} = \sum_{i=1}^{H} r_{t+i}$.

**Prevención de leakage (crítico en tesis)**
- Toda feature debe estar disponible **en el timestamp $t$**. Macro/fundamentales/sentimiento deben:
   - Usar el **último dato publicado** a $t$ (no el “final value” del período).
   - Aplicar **lag de publicación** (por ejemplo, fundamentals trimestrales con 30–60 días de retraso; macro con calendario; sentimiento con ventana cerrada).
- Normalización/estandarización: el scaler se ajusta **solo con train**, y se aplica a val/test.
- Cross-validation de series: usar **walk-forward** (expanding o rolling). Para alta frecuencia y/o múltiples activos: preferir *purged/embargo* para evitar fuga por solapamiento temporal.

**Costos y supuestos de ejecución (para evaluar estrategias)**
- Incorporar al menos: comisión + slippage (p. ej. 5–20 bps según mercado y horizonte).
- Medir métricas “offline” (MAE/RMSE) y métricas “de trading” (CAGR, Sharpe, MaxDD, Profit Factor, turnover). Un buen MAE no garantiza PnL.

**Benchmarks obligatorios (por estrategia)**
- Buy & Hold del activo/benchmark.
- Reglas técnicas simples (SMA crossover, RSI/MACD) con mismos costos.
- Para pairs: Z-score estático y Z-score con umbral dinámico.

## Modelo 1: Estrategia Conservadora (Largo Plazo, Bajo Riesgo)

### Arquitectura: GRU (Gated Recurrent Unit)

**Justificación técnica**: Los modelos GRU suelen ser competitivos en predicción de series financieras a mediano-largo plazo, con buena eficiencia computacional y generalización. En esta tesis se prioriza la predicción de **retornos a horizonte** (en lugar de precio directo) para tomar decisiones y evaluar PnL con costos.[2][3][4]

### Configuración del Modelo

**Objetivo (target) recomendado**
- Predicción de retorno acumulado $y_t^{(H)}$ con $H \in \{60, 90\}$ días (horizonte largo).
- Alternativa (si querés mantener “precio”): predecir $\ln(P_{t+H})$ y luego convertir a retorno. En evaluación reportar error sobre retorno.

**Ventana temporal**
- **Frecuencia**: diaria.
- **Training window**: 3–5 años (mínimo 2).
- **Sequence length (lookback)**: 120–252 días (≈ 6–12 meses). Recomendación inicial: **180 días**.
- **Prediction horizon**: 60 o 90 días.
- **Rebalanceo**: mensual (o trimestral si el turnover es alto).

**Features de entrada (sugerencia robusta y auditable)**

> Evitar inflar features por “agregados” ambiguos (por ejemplo “Bollinger Bands” es 2–3 variables). Es mejor listar variables concretas y su cálculo.

- **Precio/retorno (core)**
   - $r_t$ (log-retorno 1D)
   - Retornos rolling: $\sum r_{t-5}$, $\sum r_{t-20}$, $\sum r_{t-60}$
   - Volatilidad realizada: $\sigma_{20}$, $\sigma_{60}$ (std de $r$)
   - Rango intradía: $(High-Low)/Close$, y ATR(14)
   - Volumen: Volume, Dollar Volume, y *volume z-score* (rolling 60)

- **Tendencia (largo plazo)**
   - SMA(50), SMA(200), distancia relativa: $Close/SMA(200) - 1$
   - EMA(50)
   - MACD (12,26,9): línea, señal e histograma
   - Bollinger (20,2): %B y bandwidth
   - ADX(14) (si lo incluís, ayuda a filtrar “mercado lateral”)

- **Contexto de mercado (sin leakage)**
   - Retornos de benchmark (SPY / índice local): $r^{bench}_t$ y $\sum r^{bench}_{t-20}$
   - FX (si aplica) retorno y volatilidad rolling
   - Tasa libre de riesgo (si está disponible) y *term spread* (si aplica)

- **Fundamentales (opcionales, con timestamp real de publicación)**
   - Dividend yield y P/E como “último valor reportado” persistido hasta nuevo reporte
   - Flag de “edad del dato” (días desde última publicación)

- **Sentimiento (si hay pipeline)**
   - Score agregado semanal con ventana cerrada (p. ej. lunes 00:00–domingo 23:59) y se aplica desde el lunes siguiente

**Arquitectura de red (GRU)**```
Input: (sequence_length=180, features=F)
↓
GRU 1 (128 units, return_sequences=True, recurrent_dropout=0.1)
↓
Dropout (0.2)
↓
GRU 2 (64 units, return_sequences=False, recurrent_dropout=0.1)
↓
Dropout (0.2)
↓
Dense Layer (32 units, activation='relu')
↓
Output (1): predicción de retorno a H días
```

**Hiperparámetros de entrenamiento (recomendación inicial)**
- Optimizer: AdamW (o Adam) con $lr=1e-3$ (tune: $[3e-4, 3e-3]$)
- Batch size: 32–128 (iniciar en 64)
- Épocas: 50–200 con EarlyStopping (patience 10–20) + ReduceLROnPlateau
- Regularización: L2 (1e-5 a 1e-4) + gradient clipping (clipnorm 1.0)
- *Loss scaling*: entrenar sobre retornos estandarizados suele estabilizar.

**Función de pérdida (recomendada)**
- Huber Loss sobre retorno (robusta a outliers) con $\delta$ acorde a escala (p. ej. 1.0 si el target está estandarizado).
- Alternativa: MSE si el target está bien normalizado y sin outliers.

**Métricas de evaluación (ML + trading)**
- ML: MAE/RMSE sobre $y^{(H)}$, $R^2$, Directional Accuracy sobre $\operatorname{sign}(y^{(H)})$.
- Trading (backtest): CAGR, Sharpe/Sortino, Max Drawdown, Calmar, hit rate (win rate), *avg trade*, turnover.
- Importante: reportar métricas **netas de costos** y también “gross” (sin costos) para diagnóstico.

### Lógica de Generación de Señales

**Criterio de compra (ejemplo, coherente con target retorno)**
- Entrar **solo si el retorno esperado neto de costos es atractivo**:
   - $\hat{y}^{(H)} > \tau_{buy}$, con $\tau_{buy}$ inicial 6%–10% para 90D (ajustar por activo).
- Filtros (reducen falsas entradas):
   - RSI(14) < 60 (evitar sobrecompra fuerte)
   - Tendencia: $Close > SMA(200)$ (filtro de régimen) **o** $ADX(14) > 20$ (si se usa)
   - Evitar pico: bandwidth BB(20) no extremo + %B < 0.9
   - (Opcional) Sentimiento semanal ≥ 0

**Criterio de venta / salida**
- Salida por modelo: $\hat{y}^{(H)} < 0$ (o < $\tau_{sell}$)
- Salida por riesgo:
   - Stop-loss inicial: 8%–12% (depende de volatilidad del activo)
   - Take-profit: 12%–20% (buscar R:R ≥ 1.5)
   - Trailing stop opcional (para capturar tendencias largas)

**Diversificación**: Portfolio de 8-12 activos de baja correlación, pesos asignados inversamente proporcionales a volatilidad histórica (risk parity).

### Métricas Objetivo de la Estrategia

- **Retorno anual esperado**: 8–15% (dependiente del universo de activos)
- **Sharpe objetivo (neto de costos)**: > 0.9
- **Máximo Drawdown**: < 15%
- **Turnover**: bajo (p. ej. < 1 rotación/mes)
- **Win rate**: > 55% (no obligatorio si el payoff es asimétrico)

## Modelo 2: Estrategia Moderada/Intermedia (Medio Plazo, Riesgo Medio)

### Arquitectura: LSTM (Long Short-Term Memory)

**Justificación técnica**: LSTM demuestra mayor efectividad en capturar tendencias de mediano plazo y patrones de volatilidad, con R² scores superiores en predicciones de 5-30 días. Su capacidad de memoria selectiva es óptima para detectar cambios de momentum.[3][5][2]

### Configuración del Modelo

**Objetivo (target) recomendado**
- Predicción de retorno acumulado $y_t^{(H)}$ con $H \in \{10, 20\}$ días (≈ 2–4 semanas).

**Ventana temporal**
- Frecuencia: diaria.
- Training window: 1–2 años.
- Sequence length: 60 días (iniciar en 60; tune 30–90).
- Horizon: 10 o 20 días.
- Rebalanceo: semanal o cada 2 semanas.

**Features de entrada (momentum/volatilidad, concretas)**
- Precio/retorno: $r_t$, retornos rolling 5/10/20, gap open-close, rango (high-low)
- Momentum:
   - RSI(14)
   - Stochastic %K/%D (14,3)
   - ROC(10)
   - MACD (12,26,9): línea/señal/hist
- Tendencia corta/media: SMA(7), SMA(20), EMA(12), EMA(26), distancia $Close/EMA(26)-1$
- Volatilidad: ATR(14), Bollinger bandwidth (20,2), $\sigma_{20}$
- Volumen: OBV, Volume ROC(10), *volume z-score* (20)
- Contexto: retornos del benchmark y FX (mejor que “close” crudo)
- Sentimiento: si existe, usarlo con ventana cerrada diaria y/o agregación horaria (sin mirar futuro)

**Arquitectura de red (LSTM, alineada con target retorno)**```
Input: (sequence_length=60, features=F)
↓
LSTM 1 (128 units, return_sequences=True)
↓
Dropout (0.2)
↓
LSTM 2 (64 units, return_sequences=False)
↓
Dropout (0.2)
↓
Dense (32 units, activation='relu')
↓
Output (1): predicción de retorno a H días
```

**Hiperparámetros de entrenamiento (cierre)**
- Optimizer: AdamW/Adam, $lr=1e-3$ (tune: $[3e-4, 3e-3]$)
- Batch: 64
- EarlyStopping: patience 10–15; ReduceLROnPlateau
- Clipping: clipnorm 1.0

**Función de pérdida**
- Huber Loss sobre retorno (recomendada).
- Si el objetivo es captar colas (eventos extremos), considerar Quantile Loss (p. ej. q=0.25/0.5/0.75) pero manteniendo un único output si querés evitar multi-output.

**Métricas de evaluación**
- ML: MAE/RMSE (retorno), Directional Accuracy, Spearman IC (correlación rank entre $\hat{y}$ y $y$ por ventana)
- Trading: Sharpe/Sortino, MaxDD, Profit Factor, Exposure, hit rate, average holding time

### Lógica de Generación de Señales

**Criterio de compra**
- Señal por modelo: $\hat{y}^{(H)} > \tau_{buy}$ (p. ej. 2%–5% para 10–20D, depende del activo)
- Confirmación técnica (reduce ruido):
   - MACD hist > 0 y creciente **o** cruce MACD > signal
   - RSI entre 35 y 70
   - Volume z-score(20) > 0 (participación)

**Criterio de venta / salida**
- Por modelo: $\hat{y}^{(H)} < 0$ o < $\tau_{sell}$
- Por riesgo: stop-loss 5%–8% y take-profit 7%–12% (ajustar por volatilidad)
- Time stop: salir si pasan $H$ días sin alcanzar objetivo

**Diversificación**: Portfolio de 5-8 activos con rotación activa basada en momentum relativo.

### Métricas Objetivo de la Estrategia

- **Retorno anual esperado**: 12–25%
- **Sharpe objetivo (neto)**: > 0.8
- **Máximo Drawdown**: < 20%
- **Win rate**: > 50% (o Profit Factor > 1.2)
- **Frecuencia**: 1–6 trades/mes por activo (controlar turnover/costos)

## Modelo 3: Estrategia Agresiva (Intradía, Alto Riesgo)

### Arquitectura: Ensemble LSTM para Alta Frecuencia

**Justificación técnica**: Para predicción intradía, ensembles de LSTM con ponderación online según performance reciente demuestran capacidad de adaptarse a no-estacionariedades del mercado. Esta arquitectura es específicamente diseñada para trading de alta frecuencia.[6][7]

### Configuración del Modelo (implementación base)

**Objetivo (target) recomendado**
- Predicción de retorno a corto horizonte en barras de 5 minutos:
   - $H \in \{3, 6, 12\}$ barras (15, 30, 60 min)
- Recomendación inicial: **H = 6** (30 min) para balance ruido vs acción.

**Ventana temporal**
- Frecuencia: 5 minutos.
- Training window: ~60 días (limitación típica de 5-min en Yahoo Finance).
- Sequence length: 60–120 barras (5–10 horas). Iniciar en **96** (8 horas aprox).
- Re-entrenamiento: diario (nocturno) o cada X días con *rolling window*.

**Features de entrada (baseline implementado, auditable)**

- **OHLCV 5-min (siempre disponibles)**
   - Log-retorno 1 barra: $r_t$
   - Retorno rolling ~1h: $\sum_{i=1}^{12} r_{t-i}$
   - Volatilidad rolling: std($r$) en 24 barras (~2h) y 96 barras (~1 día)
   - Rango: $(High-Low)/Close$ y ATR(14) normalizado por precio
   - Volumen z-score rolling 96
   - *Time-of-day* cíclico (sin/cos)

- **Evitar leakage intradía**: no usar información posterior al cierre de la barra actual.

**Arquitectura Ensemble (alineada con ventanas y target)**```
Ensemble de 3 modelos LSTM independientes:

Modelo base (x5, distintas inicializaciones/seed):
Input: (sequence_length=96 barras de 5min, features=F)
↓
LSTM (hidden_size=64, num_layers=2)
↓
Dropout (0.2)
↓
Output (1): retorno a H barras

Agregación (cierre, simple y reproducible): promedio simple
Opcional: weighted average con pesos por performance reciente (validación rolling)
```

**Hiperparámetros de entrenamiento (cierre)**
- Optimizer: Adam, $lr=1e-3$
- Batch: 256
- Épocas: hasta 30, EarlyStopping patience 5
- Regularización: dropout 0.2 + clipping

**Función de pérdida (definida y reproducible)**
- Regresión de retorno con Huber (SmoothL1): robusta a outliers.
- Extensión opcional (no necesaria para el baseline): penalización direccional si se busca mejorar *Directional Accuracy*.

**Métricas de evaluación**
- ML: Directional Accuracy, MAE/RMSE sobre retorno, *AUC* solo si convertís a clasificación.
- Trading: Profit Factor, Sharpe intradía, MaxDD intradía, promedio de slippage, turnover, *time-in-market*.

### Lógica de Generación de Señales

**Criterio de compra (posición larga)**
- Umbral por retorno neto esperado:
   - $\hat{y}^{(H)} > \tau_{buy}$ y $\hat{y}^{(H)}$ > (costos + slippage estimado)
   - Inicial: 0.10%–0.35% para 30 min (depende activo)
- Filtros:
   - RSI(9) < 70
   - Spread < umbral de liquidez
   - (Si existe) imbalance/volume delta a favor

**Criterio de salida**
- Stop-loss: basado en volatilidad (p. ej. 1–1.5 × ATR(14) en 5-min) o umbral fijo 0.15%–0.40%
- Take-profit: 1.5–3 × riesgo (mantener R:R > 1.5)
- Time-stop: cerrar si pasan $H$ barras sin materializarse
- Regla operativa: no abrir trades cerca del cierre (p. ej. última media hora)

**Criterio de venta corta (short)**:
- Similar pero inverso, solo si la regulación del mercado lo permite

**Gestión de capital**:
- Máximo 2-3 posiciones simultáneas
- Tamaño de posición: 20-30% del capital por operación
- No operar en los primeros 30 min de mercado (evitar volatilidad de apertura)

### Métricas Objetivo de la Estrategia

- **Profit Factor (neto)**: > 1.2–1.4 (muy sensible a costos)
- **Máximo Drawdown intradía**: < 5%
- **Win rate**: > 50% (no obligatorio si el payoff es asimétrico)
- **Slippage promedio**: controlado y reportado
- **Frecuencia**: la define el umbral; priorizar calidad sobre cantidad

## Modelo 4: Estrategia de Arbitraje (Pairs Trading)

### Arquitectura: k-NN + Cointegración con Ornstein-Uhlenbeck

**Justificación técnica**: El arbitraje estadístico basado en cointegración y procesos mean-reverting (OU) es el método estándar en pairs trading. Algoritmos k-NN con aprendizaje local de series interdependientes han demostrado superar modelos AR y Granger en backtesting de pares benchmark (KO-PEP).[8]

### Configuración del Modelo

**Selección de Pares**:
- **Cointegración**: Test de Engle-Granger y Johansen para identificar pares cointegrados (p-value < 0.05).
- **Similitud fundamental**: Empresas del mismo sector, capitalización comparable.
- **Ejemplos**: YPF-PBR (petróleo), GGAL-BMA (bancos argentinos), KO-PEP (bebidas), AAPL-MSFT (tech).

**Objetivo (target) recomendado**
- Predecir el cambio del spread o el spread a horizonte $H$:
   - $y_t = S_{t+H} - S_t$ o $y_t = S_{t+H}$
- Horizon: 5–20 días (iniciar en 10).

**Ventana temporal**
- Frecuencia: diaria (o intradía si tenés data sólida, pero complica ejecución y costos).
- Training window: 1–2 años.
- Lookback para estimar $\beta$ y parámetros OU: 60–252 días (iniciar en 120).
- Recalibración: mensual, con chequeo de estabilidad semanal.

**Features de entrada (concretas y estandarizadas)**
- **Spread**: \( S_t = P_{A,t} - \beta \cdot P_{B,t} \), donde \(\beta\) es el coeficiente de cointegración
- **Z-score del spread**: \( Z_t = \frac{S_t - \mu_S}{\sigma_S} \)
- **Half-life**: Tiempo característico de mean-reversion del spread
- **Velocidad OU**: \( \theta \) (tasa de reversión a la media)
- **Volatilidad OU**: \( \sigma \) del proceso estocástico
- **Features auxiliares**: Volume ratio (Vol_A/Vol_B), Correlation rolling 30-day, Relative RSI (RSI_A - RSI_B)

**Preprocesamiento y estabilidad (recomendado)**
- Alinear calendarios (mismo timestamp/market hours) y tratar faltantes.
- Estimar $\beta$ en ventana rolling y fijar $S_t$ con ese $\beta$.
- Re-validar cointegración en cada recalibración; si falla, pausar el par.

**Algoritmo k-NN (especificación implementable)**```
1. Construcción del espacio de estados:
   - Estado: [S_t, \Delta S_t, Z_t, Corr_{30}, VolumeRatio]
   
2. Búsqueda de vecinos invariantes:
   - Estandarizar features (train-only)
   - Distancia: Euclidiana (baseline) o Mahalanobis (si hay colinealidad)
   - k \in {5, 10, 20} (tunable)
   
3. Predicción mutua del centroide:
   - Estimar y_{t} (cambio o nivel futuro del spread) con promedio ponderado por 1/distancia
   
4. Modelado OU:
   dS_t = θ(μ - S_t)dt + σdW_t
   - θ: velocidad de reversión (estimado por MLE)
   - μ: nivel medio del spread
   - σ: volatilidad del spread
```

**Función de pérdida**
- MAE sobre spread (o sobre $\Delta S$). Reportar también RMSE.

**Métricas de evaluación**
- Estadísticas: tasa de convergencia (Z vuelve a 0), tiempo a convergencia, estabilidad de $\beta$
- Trading: PnL por trade, Sharpe, MaxDD, exposición neta (debe ser ~0), costos, *hit rate*

### Lógica de Generación de Señales

**Criterio de entrada (long-short, dollar-neutral)**
- **Condición 1**: Z-score > +2.0 (spread anormalmente alto)
  - **Acción**: Short activo A, Long activo B (apostar a convergencia)
  
- **Condición 2**: Z-score < -2.0 (spread anormalmente bajo)
  - **Acción**: Long activo A, Short activo B

- **Filtros adicionales**:
   - Half-life < 20 días
   - Cointegración estable (p-value test actualizado < 0.05)
   - Predicción k-NN consistente con convergencia: $\hat{S}_{t+H}$ apunta hacia $\mu$

**Criterio de salida**
- **Exit normal**: Z-score cruza 0 (spread regresa a media)
- **Take-profit**: Z-score alcanza signo opuesto (ej. entrada en +2, salida en -0.5)
- **Stop-loss**: Z-score se aleja más (ej. entrada en +2, stop en +3, indica breakdown de cointegración)
- **Time-stop**: 30 días sin reversión (modelo no aplica)

**Gestión de capital**:
- Pesos balanceados dollar-neutral: Invertir $X$ en long y $X$ en short
- Máximo 2-3 pares activos simultáneamente
- Tamaño de posición: 15-25% del capital por par

### Métricas Objetivo de la Estrategia

- **Retorno anual esperado**: 6–12% (típicamente más estable)
- **Sharpe objetivo (neto)**: > 1.0–1.2
- **Máximo Drawdown**: < 10%
- **Beta del portfolio**: ~0 (market-neutral)
- **Holding promedio**: 5–20 días (según half-life)

## Resumen Comparativo de los Modelos

| Estrategia | Modelo | Target | Horizonte | Rebalanceo | Objetivo (trading) |
|------------|--------|--------|----------|------------|--------------------|
| Conservadora | GRU | Retorno acumulado | 60–90 días | Mensual/Trim. | Sharpe > 0.9, MaxDD < 15% |
| Moderada | LSTM | Retorno acumulado | 10–20 días | Semanal | Sharpe > 0.8, MaxDD < 20% |
| Agresiva intradía | Ensemble LSTM | Retorno a corto horizonte | 30 min (6×5-min) | Intradía | Profit Factor > 1.2, MaxDD intradía < 5% |
| Arbitraje | k-NN + OU | Spread / ΔSpread | 5–20 días | Mensual | Market-neutral, Sharpe > 1.0 |

## Pipeline de Desarrollo

Para la fase de diseño (15h), deberás:[1]

1. **Definir formalmente target por estrategia** (precio vs retorno vs spread) y fijar horizonte(s) $H$ (evitar cambiarlo durante backtests).

2. **Especificar arquitecturas + entrenamiento**: hiperparámetros iniciales, regularización, early stopping, clipping, y criterio de selección del mejor modelo (por métrica de validación).

3. **Definir datasets y splits**: time-series split. Conservadora/Moderada: 70/15/15 en el tiempo (o walk-forward). Agresiva: rolling. Arbitraje: walk-forward.

> Nota: en esta entrega se implementan y evalúan E1/E2/E3/E4 (E3 intradía con OHLCV 5-min).

4. **Establecer benchmarks**: Buy & Hold + reglas técnicas simples + (pairs) z-score estándar.

5. **Diseñar métricas**: además de MAE/RMSE, incluir Sharpe, Sortino, Calmar, MaxDD, hit rate, Profit Factor, turnover, exposición, slippage.

6. **Documentar lógica de señales**: reglas explícitas (umbrales), manejo de costos, gestión de posición (size), y stops.

7. **Plan de gestión de riesgo**: stop-loss/take-profit, límites por activo, límite de pérdida diaria (intradía), y control de correlación entre estrategias.

Esta arquitectura modular te permite entrenar y evaluar cada modelo independientemente (sin multitask ni híbridos), y comparar su aporte incremental contra benchmarks en backtesting con costos.[2][6][8][1]

## Checklist de implementación (E1/E2/E3/E4)

La idea de esta checklist es que cada punto se pueda “tildar” con un entregable verificable (código + reporte + métricas), y te sirva como guía para capítulos/secciones de tesis.

### 1) Datos (adquisición y versionado)

**Objetivo**: tener un dataset diario limpio y reproducible para un universo fijo de tickers.

- [ ] Definir universo final (lista de tickers) y período total (mínimo 5–8 años si se puede; si no, lo máximo disponible).
- [ ] Descargar OHLCV ajustado (si aplica) + benchmark SPY.
- [ ] (E3) Descargar OHLCV 5-min (p. ej. Yahoo Finance) para el universo intradía y guardar en `data/raw/intraday/`.
- [ ] Estandarizar calendario: solo días de mercado; tratar faltantes (forward-fill solo donde tenga sentido, nunca en el target).
- [ ] Guardar “raw” y “clean” con versionado (parquet/csv + checksum + fecha de descarga).

**Criterio de aceptación**
- Para cada ticker: no hay timestamps duplicados, no hay valores negativos imposibles (p. ej. volumen < 0), y el porcentaje de faltantes está reportado.

### 2) Ingeniería de features (sin leakage)

**Objetivo**: implementar un generador de features determinístico (mismo input → mismo output).

- [ ] Implementar retornos/log-retornos ($r_t$) y features rolling (retornos, volatilidad, ATR, medias móviles, MACD, RSI, Bollinger %B/bandwidth).
- [ ] Verificar “as-of” time: toda feature en $t$ usa solo datos ≤ $t$.
- [ ] (Opcional) Macro/fundamentales/sentimiento: incorporar solo si se puede garantizar timestamp de publicación + lag.
- [ ] Normalización: definir qué se escala (features y/o target) y asegurar fit solo en train.

**Criterio de aceptación**
- Test simple: para un rango temporal, recalcular features y confirmar igualdad bit a bit (o tolerancia numérica).
- No se usan valores futuros (validación manual en 2–3 features clave).

### 3) Construcción de targets y datasets por estrategia

**Objetivo**: armar datasets (X, y) por estrategia con ventanas temporales.

- [ ] E1: construir $y_t^{(90)}$ y secuencias lookback=180.
- [ ] E2: construir $y_t^{(20)}$ y secuencias lookback=60.
- [ ] E3: construir $y_t^{(H)}$ con $H=6$ barras (30 min) y secuencias lookback=96 (5-min).
- [ ] E4: construir spread $S_t$, Z-score $Z_t$, half-life, $\beta$ rolling y dataset k-NN.
- [ ] Evitar solapamiento indebido: al usar $y_t^{(H)}$, recordar que utiliza $t+1..t+H$; eso afecta cómo se hace el split.

**Criterio de aceptación**
- Dimensiones correctas: (n_samples, lookback, n_features) y targets alineados.
- Reporte de distribución de targets (media, std, percentiles, outliers).

### 4) Splits y validación walk-forward (core metodológico)

**Objetivo**: evaluar como serie temporal (sin shuffle) y evitar fuga por solapamiento.

- [ ] Implementar walk-forward (expanding o rolling) con 3–5 folds.
- [ ] Agregar embargo/purga simple: entre train y val/test dejar un gap de al menos $H$ días (para E1/E2).
- [ ] Definir una métrica de selección por estrategia (cierre):
   - E1/E2: Sharpe neto en validación, con tope de turnover.
   - E3: Profit Factor neto y control de *time-in-market* (muy sensible a costos).
   - E4: Sharpe neto + estabilidad cointegración.

**Criterio de aceptación**
- Para cada fold: fechas de train/val/test documentadas y no solapadas.

### 5) Entrenamiento de modelos (E1/E2)

**Objetivo**: entrenar GRU (E1) y LSTM (E2) sobre retornos.

- [ ] Implementar GRU E1 con los hiperparámetros cerrados (y logging de seeds, versiones, y config).
- [ ] Implementar LSTM E2 con los hiperparámetros cerrados.
- [ ] EarlyStopping + ReduceLROnPlateau + gradient clipping.
- [ ] Guardar checkpoints y el scaler del fold.

**Criterio de aceptación**
- Para cada fold: métricas ML (MAE/RMSE/Directional Acc) y curva de entrenamiento guardadas.
- Reproducibilidad: correr dos veces con seed fija y obtener resultados muy cercanos.

### 6) Selección de pares y modelo (E4)

**Objetivo**: tener un pipeline estable de selección y trading de pares.

- [ ] Generar candidatos por sector/universo.
- [ ] Test de cointegración (Engle-Granger y/o Johansen) y filtro por estabilidad.
- [ ] Estimar $\beta$ rolling, calcular Z-score y half-life.
- [ ] Implementar k-NN (k ∈ {5,10,20}) como confirmación (no reemplaza la regla Z).

**Criterio de aceptación**
- Lista de pares finales con p-values, half-life, y periodo de validez.

### 7) Motor de backtesting (con costos)

**Objetivo**: transformar predicciones/reglas en PnL reproducible.

- [ ] Implementar simulador diario con: costo round-trip=10 bps, posiciones, rebalanceo y stops (si aplican).
- [ ] (E3) Implementar simulador intradía con costo round-trip=20 bps, delay de ejecución 1 barra, y soporte long/short.
- [ ] E1/E2: reglas por umbral ($\tau_{buy}$/$\tau_{sell}$) + filtros (RSI/MACD) + time stop.
- [ ] E4: entradas/salidas por Z-score (±2.0 / 0.25 / stop 3.0 / time-stop 20 días), dollar-neutral.
- [ ] Reportar métricas netas: CAGR, Sharpe/Sortino, MaxDD, Calmar, hit rate, Profit Factor, turnover.

**Criterio de aceptación**
- Backtest benchmark (Buy&Hold + SMA crossover) disponible para comparación en el mismo framework.

### 8) Reporte final (capítulo de resultados)

**Objetivo**: dejar “listo para tesis” el paquete de resultados.

- [ ] Tablas por estrategia: métricas ML y métricas de trading (train/val/test).
- [ ] Curvas: equity curve, drawdown, distribución de retornos por trade, turnover en el tiempo.
- [ ] Análisis de sensibilidad: variar 1–2 hiperparámetros/umbrales (por ejemplo $\tau_{buy}$) y mostrar impacto.
- [ ] Discusión de riesgos: no-estacionariedad, sobreajuste, costos, limitaciones de datos.

**Criterio de aceptación**
- Un reporte que se pueda pegar en la tesis con metodología + resultados + comparación con baselines.

## Estructura recomendada del repo (Opción A)

Esta estructura está pensada para que puedas ejecutar el pipeline “end-to-end” con pocos entrypoints y buena trazabilidad (config + seeds + outputs). Adaptala si ya tenés módulos existentes.
### Explicación de la estructura por carpetas

La organización separa claramente **datos → código → resultados**, siguiendo un flujo lógico de trabajo:

**`data/`** - Datos descargados y procesados (raw → clean → features)
- `raw/` → Datos originales descargados (OHLCV diario y 5-min), **nunca se modifican** (principio de inmutabilidad)
- `clean/` → Datos limpios y alineados temporalmente (fechas validadas, faltantes tratados)
- `features/` → Dataset con indicadores técnicos ya calculados (RSI, MACD, retornos, volatilidad, etc.)
- `pairs/` → Artefactos específicos de pairs trading: spreads, betas rolling, Z-scores

**Ventaja**: trazabilidad completa - siempre se puede volver a los datos originales si algo falla en pasos posteriores.

**`src/`** - Código fuente (motor del sistema)
- `config/` → Archivos YAML con parámetros de configuración (universo de tickers, costos, umbrales, hiperparámetros)
- `data/` → Scripts para descarga y limpieza de datos
- `features/` → Scripts para calcular indicadores técnicos y construir targets
- `models/` → Definiciones de arquitecturas de redes neuronales (GRU, LSTM) y scripts de entrenamiento
- `pairs/` → Código para selección de pares cointegrados y cálculo de spreads
- `backtest/` → Motor de backtesting (simula operaciones de compra/venta con costos de transacción)
- `reporting/` → Generación de tablas de métricas y visualizaciones finales
- `train_e3_pipeline.py` → **Entrypoint completo para E3** (descarga → entrena → backtest en un comando)

**Ventaja**: código modular y reutilizable - cada carpeta tiene una responsabilidad única y bien definida.

**`reports/`** - Outputs finales para documentación
- `figures/` → Gráficos generados (equity curves, drawdown, distribuciones)
- `tables/` → Tablas de métricas en formato publicable (CSV/LaTeX)

**Ventaja**: todo lo que va en el documento de tesis queda centralizado y listo para usar.

**`runs/`** - Historial de experimentos (versionado automático)
- Cada ejecución del pipeline crea una carpeta timestamped: `YYYYMMDD_HHMMSS/`
- Dentro se guardan: config usado, predicciones, métricas, backtest CSV, checkpoints de modelos
- Ejemplo para E3: `runs/e3_intraday/20260105_143022/GGAL_backtest.csv`

**Ventaja**: reproducibilidad total - se puede volver a cualquier experimento anterior y saber exactamente qué parámetros se usaron.

**📓 `notebooks/`** - Jupyter notebooks para análisis exploratorio
- Para explorar datos, validar ausencia de leakage, revisar resultados interactivamente
- **No son la fuente de verdad del pipeline** (solo consultan `data/` y `runs/`)

**Ventaja**: análisis interactivo sin contaminar el código de producción.

### Flujo de trabajo típico

```
1. Descargar datos → guardados en data/raw/
2. Limpiar datos → guardados en data/clean/
3. Crear features → guardados en data/features/
4. Entrenar modelo → lee de data/features/, escribe en runs/
5. Backtest → lee predicciones de runs/, genera métricas
6. Reportes → lee de runs/, genera figuras/tablas en reports/
```

**Para E3 intradía (pipeline automatizado):**```bash
# Paso 1: Descarga OHLCV 5-min → data/raw/intraday/
python -m src.train_e3_pipeline --mode download

# Paso 2: Entrena + backtest → runs/e3_intraday/<timestamp>/
python -m src.train_e3_pipeline --mode run
```

### Ventajas de esta arquitectura

 **Reproducible**: mismo config + mismos datos = mismos resultados  
 **Auditable**: cada corrida queda guardada con su configuración exacta  
 **Modular**: se puede cambiar una parte sin romper las demás  
 **Thesis-friendly**: `reports/` tiene todo listo para copiar al documento final  
 **Escalable**: fácil agregar nuevas estrategias o fuentes de datos
**Árbol sugerido**```
trading_predict/
   data/
      raw/                      # descargas originales (no modificar)
      clean/                    # OHLCV limpio y alineado
      features/                 # dataset con features (por fecha/ticker)
      pairs/                    # spreads, betas, cointegración
   src/
      train_e3_pipeline.py   # entrypoint E3 (descarga/train/backtest)
      config/
         base.yaml               # universo, costos, horizontes, umbrales
         experiment_e1.yaml
         experiment_e2.yaml
         experiment_e4.yaml
      data/
         download.py             # descarga diaria (OHLCV + SPY)
         clean.py                # limpieza + calendario
         splits.py               # walk-forward + embargo
         intraday_yfinance.py    # descarga/carga OHLCV 5-min (baseline)
      features/
         build_features.py       # indicadores técnicos sin leakage
         build_targets.py        # y^(H) para E1/E2
         build_sequences.py      # ventanas (lookback) para RNN
         intraday.py             # features/target/secuencias intradía (baseline)
      models/
         e1_gru.py               # definición GRU
         e2_lstm.py              # definición LSTM
         e3_lstm.py              # LSTM regressor (baseline)
         train.py                # entrenamiento (fold-aware)
         predict.py              # predicciones out-of-sample
      pairs/
         select_pairs.py         # cointegración + filtros
         build_spread.py         # beta rolling + Z-score + half-life
         knn_confirm.py          # k-NN confirmatorio
      backtest/
         daily.py                # backtest diario E1/E2 (señales por tau + costos + métricas)
         intraday.py             # backtest intradía E3 (costos + métricas)
         rules_e2.py             # reglas E2 (si aplica)
      reporting/
         make_report.py          # tablas + gráficos finales
   notebooks/
      01_data_audit.ipynb       # auditoría de datos (faltantes/outliers)
      02_features_sanity.ipynb  # sanity checks de leakage
      03_results_review.ipynb   # lectura de resultados + figuras
   reports/
      figures/
      tables/
   runs/
      YYYYMMDD_HHMMSS/          # outputs por corrida (config + métricas)
         config_used.yaml
         metrics.json
         equity_curve.csv
         predictions.parquet
```

**Entrypoints prácticos (baseline)**
- E3 (intradía): `python -m src.train_e3_pipeline --mode download` y luego `python -m src.train_e3_pipeline --mode run`.

**Entry points mínimos (orden recomendado)**
1. `src/data/download.py` → baja OHLCV + SPY
2. `src/data/clean.py` → limpia y alinea
3. `src/features/build_features.py` → features
4. `src/features/build_targets.py` + `src/features/build_sequences.py` → datasets E1/E2
5. `src/models/train.py` → entrena walk-forward y guarda predicciones OOS
6. `src/pairs/select_pairs.py` + `src/pairs/build_spread.py` (+ `src/pairs/knn_confirm.py`) → pipeline E4
7. Backtesting: `src/backtest/daily.py` (E1/E2) / `src/backtest/intraday.py` (E3)
8. `src/reporting/make_report.py` → tablas/figuras finales

**Convenciones prácticas**
- Toda corrida escribe en `runs/` y guarda el `config_used.yaml`.
- Los modelos se evalúan siempre con el mismo motor de backtest y mismos costos.
- Las notebooks solo “consumen” outputs (no son la fuente de verdad del pipeline).

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_82975ff3-1e2e-4b75-bfc9-583824ada2bc/26800206-c5cf-44a1-90d9-a5cb6bab97f3/Sebastian-Carreras-Proyecto-final-v4_2.pdf)
[2](https://arxiv.org/pdf/2411.05790.pdf)
[3](https://journals.scholarpublishing.org/index.php/TMLAI/article/view/18843)
[4](https://journalwjaets.com/sites/default/files/fulltext_pdf/WJAETS-2025-0167.pdf)
[5](https://www.sciencedirect.com/science/article/pii/S1566253524003944)
[6](https://research.vu.nl/en/publications/an-ensemble-of-lstm-neural-networks-for-high-frequency-stock-mark/)
[7](https://onlinelibrary.wiley.com/doi/full/10.1002/for.2585)
[8](https://www.linkedin.com/pulse/universal-cointegration-pairs-trading-via-machine-nikolay-nikolaev)
[9](https://dl.acm.org/doi/abs/10.1145/3700058.3700075)
[10](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5023739)
[11](https://arxiv.org/pdf/2407.16103.pdf)

## Especificación de datos para los 4 modelos (alineada con la implementación)

Según la planificación del proyecto y la literatura especializada en predicción financiera con ML, los datos requeridos para cada estrategia deben estructurarse considerando horizontes temporales, indicadores técnicos y características específicas del perfil de riesgo.[1][2]

### Datos base comunes

**Fuentes de datos históricas (cierre Opción A)**
- Diario (E1/E2/E4): OHLCV (idealmente *adjusted*), benchmark (SPY) y calendario de trading.
- Intradía (E3): OHLCV 5-min vía Yahoo Finance (vía `yfinance`), con historial típico ~60 días.

**Split y validación (cierre)**
- Walk-forward (expanding o rolling) con validación temporal.
- Scalers fit en train y aplicados a val/test.

### Estrategia 1: Conservadora (E1)

**Horizonte temporal:** Datos diarios, análisis de tendencias de 3-12 meses[1]

**Variables predictoras (features):**
- Medias móviles de largo plazo: SMA 50, 100, 200 días[5]
- Indicadores de tendencia: ADX (Average Directional Index), Parabolic SAR
- Indicadores de volumen: OBV (On-Balance Volume), Volume Rate of Change
- Volatilidad histórica (20-60 días)[6]
- Ratios fundamentales: P/E ratio, dividend yield, ROE (si están disponibles)
- Variables macroeconómicas (opcionales y con lag de publicación): tasas de interés, inflación e índices de referencia (p. ej. S&P500 o índice local)[7]

**Target (cierre):** retorno acumulado a 90 días.

### Estrategia 2: Moderada/Intermedia (E2)

**Horizonte temporal:** Datos diarios/horarios, análisis de 1-4 semanas[1]

**Variables predictoras (features):**
- Medias móviles de medio plazo: EMA 12, 26, 50 días[5]
- MACD (Moving Average Convergence Divergence): línea MACD, línea señal, histograma[8][5]
- RSI (Relative Strength Index) períodos 14 y 21[8][5]
- Bandas de Bollinger (20 períodos): banda superior, inferior, ancho de banda[5][8]
- Stochastic Oscillator (%K, %D)
- ATR (Average True Range) para volatilidad[6]
- Momentum indicators: ROC (Rate of Change), Williams %R
- Volumen relativo y cambios porcentuales de volumen[8]

**Target (cierre):** retorno acumulado a 20 días.

### Estrategia 3: Agresiva intradía (E3)

**Horizonte temporal:** Datos de alta frecuencia (5-min), operaciones intradía[4][1]

**Variables predictoras (features):**
- OHLCV 5-min
- Retornos: 1 barra y rolling (p. ej. 12 barras ~1h)
- Volatilidad rolling (24 y 96 barras)
- Rango (high-low)/close y ATR(14) normalizado
- Volumen z-score rolling
- Time-of-day (sin/cos)

**Target (cierre):** retorno a 30 minutos (6 barras de 5-min).

### Estrategia 4: Arbitraje (pairs trading) (E4)

**Horizonte temporal:** Datos de alta frecuencia (minuto a minuto) o diarios según implementación[10][4]

**Variables predictoras (features):**
- Precios históricos de ambos activos (par seleccionado)
- Spread normalizado: diferencia de precios estandarizada[4]
- Z-score del spread (precio relativo)[4]
- Cointegración estadística entre pares: test de Engle-Granger, Johansen
- Coeficiente de correlación rolling (ventanas de 20, 60, 390 minutos)[4]
- Distancia euclidiana entre series de precios normalizadas[4]
- Volatilidad del spread: desviación estándar rolling[4]
- Ratio de precios entre activos
- Half-life del spread (velocidad de reversión a la media)
- Beta dinámica entre activos

**Método de selección de pares:**
- Análisis de cointegración estadística[4]
- Correlación histórica >0.8[4]
- Similitud sectorial y de capitalización[10]

**Target (cierre):** el modelo k-NN predice spread/Δspread como confirmación; las entradas/salidas se disparan por Z-score con umbrales fijos.

### Estructura general de datos para descarga

**Formato recomendado:**```
timestamp, ticker, open, high, low, close, volume, [indicadores técnicos calculados]
```

**Consideraciones técnicas:**
- Implementar pipeline de ingeniería de features que calcule automáticamente indicadores técnicos[1]
- Normalizar/estandarizar todas las variables antes del entrenamiento[11]
- Crear ventanas temporales (time windows) para capturar patrones secuenciales[3]
- Split temporal: 70% entrenamiento, 15% validación, 15% test (sin mezclar temporalmente)[11]
- Para alta frecuencia: considerar datos de al menos 3-6 meses[4]
- Para estrategias de largo plazo: datos de 2-5 años[3]

**Próximos pasos sugeridos:**
1. Definir los tickers específicos a operar (acciones argentinas, estadounidenses, bonos)
2. Seleccionar proveedor de datos según presupuesto (Yahoo Finance gratuito, APIs premium como Alpha Vantage, Bloomberg)[1]
3. Implementar módulo de adquisición y limpieza de datos (Fase 4 del proyecto, 80h)[1]
4. Calcular y almacenar indicadores técnicos para cada estrategia
5. Crear datasets específicos por estrategia con las features definidas

### Referencias
[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_82975ff3-1e2e-4b75-bfc9-583824ada2bc/26800206-c5cf-44a1-90d9-a5cb6bab97f3/Sebastian-Carreras-Proyecto-final-v4_2.pdf)
[2](https://questdb.com/glossary/machine-learning-for-market-prediction/)
[3](https://www.simplilearn.com/tutorials/machine-learning-tutorial/stock-price-prediction-using-machine-learning)
[4](https://www.econjournals.com/index.php/ijefi/article/download/5127/pdf/13716)
[5](https://www.swastika.co.in/blog/intraday-trading-using-rsi-macd-and-bollinger-bands)
[6](https://robotwealth.com/machine-learning-financial-prediction-david-aronson/)
[7](https://www.itransition.com/machine-learning/stock-prediction)
[8](https://www.binance.com/en/square/post/12997454358994)
[9](https://www.youtube.com/watch?v=bPlk9oqkbmw)
[10](https://www.sciencedirect.com/science/article/abs/pii/S0957417425021153)
[11](https://www.sciencedirect.com/science/article/pii/S2667305324001236)