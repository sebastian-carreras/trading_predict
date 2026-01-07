# E4 - Pairs Trading (k-NN + Ornstein-Uhlenbeck)

**Horizonte**: 10 días | **Frecuencia**: diaria | **Perfil**: Market-neutral, bajo riesgo sistemático

Arbitraje estadístico basado en cointegración y procesos mean-reverting para trading de pares con exposición neta cercana a cero.

## 📋 Resumen de la Estrategia

**Objetivo**: Capturar convergencias de spreads cointegrados con riesgo de mercado minimizado (beta ~0).

**Características**:
- **Target**: Spread a horizonte H=10 días (confirmación k-NN)
- **Pares**: Activos cointegrados del mismo sector
- **Entrada**: Z-score > ±2.0 (spread desviado)
- **Salida**: Z-score cruza 0 (convergencia)
- **Stop**: Z-score > ±3.0 (breakdown cointegración)
- **Time-stop**: 20 días sin reversión

**Métricas objetivo**:
- Sharpe neto > 1.0–1.2
- Max Drawdown < 10%
- Beta del portfolio ~0 (market-neutral)
- Holding promedio: 5–20 días (según half-life)

## 🏗️ Arquitectura del Modelo

### k-NN + Cointegración + Ornstein-Uhlenbeck

**Justificación**: El arbitraje estadístico basado en cointegración y procesos mean-reverting (OU) es el método estándar en pairs trading. Algoritmos k-NN con aprendizaje local de series interdependientes han demostrado superar modelos AR y Granger en backtesting de pares benchmark (KO-PEP).

### Componentes

**1. Selección de pares** (cointegración):
- Test Engle-Granger y/o Johansen: p-value < 0.05
- Similitud fundamental: mismo sector, cap comparable
- Ejemplos: YPF-PBR, GGAL-BMA, KO-PEP, AAPL-MSFT

**2. Construcción del spread**:
```
S_t = P_A,t - β × P_B,t

donde β = coeficiente de cointegración (rolling window)
```

**3. Modelado Ornstein-Uhlenbeck**:
```
dS_t = θ(μ - S_t)dt + σdW_t

θ: velocidad de reversión (estimado por MLE)
μ: nivel medio del spread
σ: volatilidad del spread
```

**4. k-NN confirmación** (opcional):
- Estado: [S_t, ΔS_t, Z_t, Corr_30, VolumeRatio]
- Distancia: Euclidiana o Mahalanobis
- k ∈ {5, 10, 20}
- Predicción: promedio ponderado 1/distancia

## 📂 Archivos a Implementar

| Archivo | Propósito | Estado |
|---------|-----------|--------|
| `src/pairs/select_pairs.py` | Test cointegración + filtros | ⏳ TODO |
| `src/pairs/build_spread.py` | Cálculo β rolling + Z-score + half-life | ⏳ TODO |
| `src/pairs/knn_confirm.py` | k-NN confirmatorio (opcional) | ⏳ TODO |
| `src/backtest/rules_e4.py` | Reglas entrada/salida Z-score | ⏳ TODO |
| `src/models/ou_process.py` | Estimación parámetros OU (θ, μ, σ) | ⏳ TODO |

## 📊 Features y Variables

**Spread y derivados**:
- **`spread`**: S_t = P_A - β × P_B
- **`z_score`**: (S_t - μ_S) / σ_S (normalizado)
- **`delta_spread`**: ΔS_t = S_t - S_{t-1}
- **`half_life`**: Tiempo característico de mean-reversion (días)

**Parámetros OU**:
- **`theta`**: Velocidad de reversión a la media (θ)
- **`mu`**: Nivel medio del spread (μ)
- **`sigma`**: Volatilidad del proceso (σ)

**Features auxiliares**:
- **`volume_ratio`**: Vol_A / Vol_B
- **`correlation_30`**: Correlación rolling 30 días
- **`relative_rsi`**: RSI_A - RSI_B
- **`beta_rolling`**: β estimado en ventana móvil (60-252 días)

**Validación de estabilidad**:
- **`coint_pvalue`**: p-value test Engle-Granger (actualizado)
- **`beta_std`**: Estabilidad de β (desviación estándar rolling)

## 🎯 Lógica de Señales de Trading

### Entrada (long-short, dollar-neutral)

**Condición 1**: Z-score > +2.0 (spread anormalmente alto)
- **Acción**: Short activo A, Long activo B
- **Razón**: Apostar a que spread converge a μ

**Condición 2**: Z-score < -2.0 (spread anormalmente bajo)
- **Acción**: Long activo A, Short activo B

**Filtros adicionales** (reducir falsas señales):
- Half-life < 20 días (reversión rápida)
- Cointegración estable: p-value test < 0.05 (actualizado)
- k-NN confirmación: predicción apunta a convergencia (opcional)

### Salida

**Exit normal**: Z-score cruza 0 (spread regresa a media)

**Take-profit**: Z-score alcanza signo opuesto
- Ejemplo: entrada en +2.0, salida en -0.5

**Stop-loss**: Z-score se aleja más (breakdown cointegración)
- Ejemplo: entrada en +2.0, stop en +3.0

**Time-stop**: 30 días sin reversión (modelo no aplica)

### Gestión de capital

**Dollar-neutral**:
- Invertir $X en long y $X en short (exposición neta ~$0)
- Ejemplo: Long $10k AAPL, Short $10k MSFT

**Sizing**:
- Máximo 2-3 pares activos simultáneamente
- Tamaño por par: 15-25% del capital
- Ajustar por volatilidad del spread

## 📈 Métricas de Evaluación

**Estadísticas de cointegración**:
- **Tasa de convergencia**: % de veces que Z vuelve a 0
- **Tiempo a convergencia**: Promedio días hasta salida
- **Estabilidad de β**: Coeficiente de variación

**Trading**:
- **PnL por trade**: Ganancia promedio por operación
- **Sharpe**: Retorno/riesgo (objetivo > 1.0)
- **Max Drawdown**: Pérdida máxima desde peak (objetivo < 10%)
- **Beta del portfolio**: Debe estar ~0 (market-neutral)
- **Hit rate**: % de trades ganadores
- **Exposure neta**: Debe ser cercano a cero

**Costos**:
- Considerar 10 bps round-trip por par (20 bps total: 2 activos)
- Posibles costos de borrow para shorts (según broker)

## 📁 Estructura de Resultados

```
runs/e4_pairs/<timestamp>/
├── config_used.yaml
├── pairs_selection.csv           # Pares seleccionados + stats
├── GGAL_BMA/
│   ├── spread_timeseries.csv     # (date, spread, z_score)
│   ├── ou_params.json             # θ, μ, σ, half-life
│   ├── trades.csv                 # Historial de operaciones
│   └── backtest_summary.json      # Métricas del par
├── YPF_PAMP/
│   └── ...
└── summary_all_pairs.csv         # Resumen agregado
```

## 🚀 Comandos de Uso

**Una vez implementado**:

```bash
# Paso 1: Seleccionar pares cointegrados
python -m src.pairs.select_pairs --sector energy --min_pvalue 0.05

# Paso 2: Construir spreads y calcular Z-scores
python -m src.pairs.build_spread --pairs GGAL,BMA YPF,PAMP

# Paso 3: Entrenar k-NN (opcional)
python -m src.pairs.knn_confirm --pairs GGAL,BMA --k 10

# Paso 4: Backtest con reglas Z-score
python -m src.backtest.rules_e4 --pairs GGAL,BMA YPF,PAMP
```

## 🔄 Próximos Pasos para Implementación

1. ⏳ **select_pairs.py**:
   - Generar candidatos por sector/correlación
   - Test Engle-Granger (statsmodels.tsa.stattools.coint)
   - Test Johansen (opcional, multivariate)
   - Filtrar por p-value < 0.05 y half-life < 20 días

2. ⏳ **build_spread.py**:
   - Estimar β rolling (OLS en ventana móvil 60-252 días)
   - Calcular spread S_t = P_A - β × P_B
   - Calcular Z-score normalizado
   - Estimar parámetros OU (MLE o discrete approximation)

3. ⏳ **knn_confirm.py**:
   - Construir espacio de estados [S_t, ΔS_t, Z_t, ...]
   - Implementar k-NN con sklearn.neighbors
   - Predicción: spread futuro o dirección de convergencia

4. ⏳ **rules_e4.py**:
   - Lógica entrada: Z > ±2.0 + filtros
   - Lógica salida: Z cruza 0, stop ±3.0, time-stop 20 días
   - Dollar-neutral sizing

5. ⏳ **ou_process.py**:
   - Maximum Likelihood Estimation de θ, μ, σ
   - Cálculo half-life: ln(2) / θ
   - Validación estacionariedad

## 🔧 Configuración

Parámetros en [`base.yaml`](src/config/base.yaml) sección `strategies.e4_pairs`:

```yaml
e4_pairs:
  training_window_days: 252      # 1 año para cointegración
  beta_lookback_days: 120         # Rolling β
  horizon_days: 10                # k-NN confirmación
  
  entry_exit:
    entry_z: 2.0                  # |Z| >= 2.0 para entrar
    exit_z: 0.25                  # |Z| <= 0.25 para salir
    stop_z: 3.0                   # |Z| >= 3.0 stop-loss
    time_stop_days: 20
  
  filters:
    cointegration_pvalue_max: 0.05
    half_life_days_max: 20
  
  knn:
    k_candidates: [5, 10, 20]
    distance: "euclidean"
  
  pairs:
    - ["GGAL", "BMA"]
    - ["YPF", "PAMP"]
    - ["EDN", "CEPU"]
    - ["KO", "PEP"]
    - ["XLE", "XLF"]
```

## ⚠️ Consideraciones Importantes

**Riesgos específicos**:
- **Breakdown de cointegración**: Validar p-value mensualmente
- **Eventos corporativos**: Splits, dividendos rompen spread
- **Costos de borrow**: Shorts pueden ser caros o inviables
- **Liquidez**: Evitar pares ilíquidos (spreads anchos)

**No-estacionariedad**:
- Parámetros OU pueden cambiar (crisis, cambios regulatorios)
- Re-estimar θ, μ, σ mensualmente
- Pausar pares si p-value > 0.05

**Correlación ≠ Cointegración**:
- Correlación alta es necesaria pero no suficiente
- Test formal de cointegración es obligatorio

## 📚 Referencias

- [README general](README.md) - Overview del proyecto
- [base.yaml](src/config/base.yaml) - Configuración E4
- Paper: "Universal Cointegration Pairs Trading via Machine Learning" (Nikolaev)

## 🎓 Ejemplos de Pares por Sector

**Energía**:
- YPF - PBR (petróleo Argentina-Brasil)
- XLE - XLF (ETFs sectores)

**Bancos Argentina**:
- GGAL - BMA (Galicia - Macro)

**Utilities**:
- EDN - CEPU (energía Argentina)

**Tech USA**:
- AAPL - MSFT (mega-caps tech)

**Bebidas**:
- KO - PEP (Coca-Cola - Pepsi, benchmark clásico)

---

**Estado**: ⏳ Especificada, pendiente codificación
