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

## Arquitectura del Modelo

### k-NN + Cointegración + Ornstein-Uhlenbeck

**Justificación**: El arbitraje estadístico basado en cointegración y procesos mean-reverting (OU) es el método estándar en pairs trading. Algoritmos k-NN con aprendizaje local de series interdependientes han demostrado superar modelos AR y Granger en backtesting de pares benchmark (KO-PEP).

### Componentes

**1. Selección de pares** (cointegración):
- Test Engle-Granger y/o Johansen: p-value < 0.05
- Similitud fundamental: mismo sector, cap comparable
- Ejemplos: YPF-VIST, GGAL-BMA, KO-PEP, AAPL-MSFT

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
| `src/pairs/select_pairs.py` | Test cointegración + filtros | ✅ IMPLEMENTADO |
| `src/pairs/build_spread.py` | Cálculo β rolling + Z-score + half-life | ✅ IMPLEMENTADO |
| `src/pairs/knn_confirm.py` | k-NN confirmatorio (opcional) | ✅ IMPLEMENTADO |
| `src/backtest/rules_e4.py` | Reglas entrada/salida Z-score | ✅ IMPLEMENTADO |
| `src/models/ou_process.py` | Estimación parámetros OU (θ, μ, σ) | ✅ IMPLEMENTADO |
| `src/train_e4_pipeline.py` | Pipeline principal E4 | ✅ IMPLEMENTADO |

## Features y Variables

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

## Lógica de Señales de Trading

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

## Métricas de Evaluación

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

### Ejecución Directa (Python)

```bash
# Ejecutar pipeline completo con todos los pares del config
python -m src.train_e4_pipeline

# Ejecutar con pares específicos
python -m src.train_e4_pipeline --pairs GGAL.BA,BMA.BA YPFD.BA,PAMP.BA

# Ejecutar con tag personalizado para output
python -m src.train_e4_pipeline --output-tag my_test

# Test rápido con un par
python scripts/test_e4_simple.py
```

### Ejecución con Airflow (Producción)

**DAG 1: Pipeline Principal** (Semanal - Lunes 4 AM)
```bash
# Trigger manual
airflow dags trigger e4_pairs_trading_pipeline

# Con parámetros custom
airflow dags trigger e4_pairs_trading_pipeline \
  --conf '{"use_knn": "True", "entry_z": "2.5", "exit_z": "0.2"}'
```

**DAG 2: Re-calibración Mensual** (Día 15 de cada mes)
```bash
airflow dags trigger e4_monthly_recalibration
```

**UI de Airflow**: http://localhost:8080 → DAG `e4_pairs_trading_pipeline`

**MLflow**: http://localhost:5050 → Experiment "E4_Pairs_Trading_Strategy"

📖 **Documentación completa de DAGs**: [docs/AIRFLOW_E4_DAGS.md](docs/AIRFLOW_E4_DAGS.md)


**Estructura de salida generada**:

```
runs/e4_pairs/<timestamp>/
├── config_used.yaml              # Configuración utilizada
├── summary_all_pairs.csv         # Resumen de todos los pares
├── GGAL.BA_BMA.BA/
│   ├── spread_timeseries.csv    # Serie temporal del spread
│   ├── ou_params.json            # Parámetros OU estimados
│   ├── trades.csv                # Historial de operaciones
│   └── backtest_summary.json    # Métricas del par
└── YPFD.BA_PAMP.BA/
    └── ...
```

##  Configuración

Parámetros en [`base.yaml`](src/config/base.yaml) sección `strategies.e4_pairs`:

```yaml
e4_pairs:
  training_window_days: 252       # 1 año para cointegración
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
    - ["GGAL.BA", "BMA.BA"] - Bancos Argentina
    - ["YPFD.BA", "VIST.BA"] - Energía Argentina
    - ["PAMP.BA", "CEPU.BA"] - Energía Argentina
    - ["TGSU2.BA", "TGNO4.BA"] - Transporte de Gas Argentina
    - ["TXAR.BA", "LOMAD.BA"] - Utilities Argentina
    - ["TECO2.BA", "CVHD.BA"] - Servicios Argentina
    - ["CVHD.BA","GCLA.BA"] - Medios Argentina
    - ["KO", "PEP"] - Consumo
    - ["MSFT", "GOOGL"] - Tech
    - ["XOM", "CVX"] - Energía
    - ["JPM", "BAC"] - Bancos/financiero
    - ["CAT", "DE"] - Industriales

```

## ⚠ Consideraciones Importantes

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

## Referencias

- [README general](README.md) - Overview del proyecto
- [base.yaml](src/config/base.yaml) - Configuración E4
- Paper: "Universal Cointegration Pairs Trading via Machine Learning" (Nikolaev)

## Ejemplos de Pares por Sector

**Energía**:
- YPF - PBR (petróleo Argentina-Brasil)
- XLE - XLF (ETFs sectores)
✅ **IMPLEMENTADO** - Listo para ejecutar y evaluar

## 📝 Notas de Implementación

La estrategia E4 ha sido completamente implementada siguiendo las especificaciones de este documento:

**Módulos implementados**:
1. **`src/pairs/select_pairs.py`**: Selección de pares cointegrados usando tests de Engle-Granger y Johansen
2. **`src/pairs/build_spread.py`**: Construcción de spreads con hedge ratio rolling y cálculo de z-scores
3. **`src/pairs/ou_process.py`**: Estimación de parámetros Ornstein-Uhlenbeck (θ, μ, σ) y half-life
4. **`src/pairs/knn_confirm.py`**: Modelo k-NN para confirmación de señales (opcional)
5. **`src/backtest/rules_e4.py`**: Reglas de trading basadas en z-score con gestión dollar-neutral
6. **`src/train_e4_pipeline.py`**: Pipeline completo que integra todos los componentes

**Características implementadas**:
- ✅ Tests de cointegración (Engle-Granger y Johansen)
- ✅ Cálculo de spread con hedge ratio rolling (β)
- ✅ Z-score normalizado para señales de entrada/salida
- ✅ Estimación de parámetros OU via MLE discreto
- ✅ Cálculo de half-life y validación de estacionariedad
- ✅ k-NN confirmatorio con cross-validation
- ✅ Señales de trading: entrada (|z| ≥ 2.0), salida (|z| ≤ 0.25), stop (|z| ≥ 3.0)
- ✅ Backtest con gestión dollar-neutral (exposición neta ~0)
- ✅ Métricas de evaluación: Sharpe, CAGR, drawdown, win rate, etc.
- ✅ Soporte para todos los pares definidos en `base.yaml`

**Pares configurados en `base.yaml`**:
- Bancos Argentina: GGAL.BA - BMA.BA
- Energía Argentina: YPFD.BA - PAMP.BA, EDN.BA - CEPU.BA
- Consumo USA: KO - PEP
- Sectores USA: XLE - XLF

**Próximos pasos**:
1. Ejecutar el pipeline con los pares configurados
2. Analizar resultados de cointegración y parámetros OU
3. Evaluar performance del backtest (Sharpe, drawdown, win rate)
4. Ajustar parámetros si es necesario (entry_z, stop_z, time_stop)
5. Comparar con estrategias E1 y E2
**Bancos Argentina**:
- GGAL - BMA (Galicia - Macro)

**Utilities**:
- EDN - CEPU (energía Argentina)

**Tech USA**:
- AAPL - MSFT (mega-caps tech)

**Bebidas**:
- KO - PEP (Coca-Cola - Pepsi, benchmark clásico)

---

**Estado**:  Especificada, pendiente codificación
