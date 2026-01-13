# Guía Rápida: Estrategia E4 - Pairs Trading

## Resumen de la Implementación

La estrategia E4 de pairs trading ha sido completamente implementada siguiendo las especificaciones del [README_E4.md](../README_E4.md).

## Archivos Implementados

### Módulo de Pares (`src/pairs/`)
- **`select_pairs.py`**: Selección y validación de pares cointegrados
  - Test Engle-Granger y Johansen
  - Filtros de correlación y estabilidad
  
- **`build_spread.py`**: Construcción de spreads
  - Hedge ratio (β) rolling via OLS
  - Cálculo de z-score normalizado
  - Features del par (correlación, volume ratio, etc.)

- **`ou_process.py`**: Modelado Ornstein-Uhlenbeck
  - Estimación de parámetros (θ, μ, σ) via MLE
  - Cálculo de half-life
  - Test de estacionariedad (ADF)

- **`knn_confirm.py`**: Confirmación k-NN (opcional)
  - Predicción de convergencia del spread
  - Cross-validation para selección de k
  - Señales de confirmación

### Backtest (`src/backtest/`)
- **`rules_e4.py`**: Reglas de trading y backtest
  - Señales basadas en z-score (entrada/salida/stop)
  - Gestión dollar-neutral
  - Métricas de evaluación

### Pipeline Principal
- **`src/train_e4_pipeline.py`**: Orquestación completa
  - Flujo end-to-end desde datos hasta resultados
  - Procesamiento de múltiples pares
  - Guardado de resultados y métricas

## Uso Básico

### 1. Ejecutar con todos los pares del config
```bash
python -m src.train_e4_pipeline
```

### 2. Ejecutar con pares específicos
```bash
python -m src.train_e4_pipeline --pairs GGAL.BA,BMA.BA YPFD.BA,PAMP.BA
```

### 3. Test rápido
```bash
python scripts/test_e4_simple.py
```

## Pares Configurados

Los siguientes pares están definidos en `src/config/base.yaml`:

**Argentina**:
- GGAL.BA - BMA.BA (Bancos)
- YPFD.BA - PAMP.BA (Energía)
- EDN.BA - CEPU.BA (Energía)

**USA**:
- KO - PEP (Consumo)
- XLE - XLF (Sectores)

## Parámetros Clave

Configurados en `base.yaml` bajo `strategies.e4_pairs`:

```yaml
entry_exit:
  entry_z: 2.0          # Umbral de entrada
  exit_z: 0.25          # Umbral de salida
  stop_z: 3.0           # Stop-loss
  time_stop_days: 20    # Time-stop

filters:
  cointegration_pvalue_max: 0.05
  half_life_days_max: 20

knn:
  enabled: true
  k_candidates: [5, 10, 20]
```

## Resultados Generados

Para cada ejecución se crea un directorio en `runs/e4_pairs/<timestamp>/`:

```
runs/e4_pairs/<timestamp>/
├── config_used.yaml              # Configuración utilizada
├── summary_all_pairs.csv         # Métricas de todos los pares
├── GGAL.BA_BMA.BA/
│   ├── spread_timeseries.csv    # Serie temporal del spread
│   ├── ou_params.json            # Parámetros OU (θ, μ, σ, half-life)
│   ├── trades.csv                # Trades detallados
│   └── backtest_summary.json    # Métricas del backtest
└── ...
```

## Métricas de Evaluación

El backtest calcula:
- **Total Return**: Retorno total del período
- **CAGR**: Retorno anualizado compuesto
- **Sharpe Ratio**: Retorno ajustado por riesgo
- **Max Drawdown**: Máxima pérdida desde peak
- **Win Rate**: % de trades ganadores
- **Num Trades**: Número de operaciones
- **Net Exposure**: Exposición neta promedio (debe ser ~0 para market-neutral)

## Validaciones Automáticas

El pipeline valida automáticamente:
1. ✅ Cointegración (p-value < 0.05)
2. ✅ Half-life razonable (< 20 días)
3. ✅ Estacionariedad del spread
4. ✅ Correlación mínima entre activos
5. ✅ Disponibilidad de datos

## Próximos Pasos

1. **Ejecutar el pipeline** con los pares configurados
2. **Analizar resultados** en `summary_all_pairs.csv`
3. **Revisar pares individuales** para entender comportamiento
4. **Ajustar parámetros** si es necesario (entry_z, stop_z, etc.)
5. **Comparar con E1/E2** en términos de Sharpe, drawdown, y consistencia

## Comparación con E1 y E2

| Característica | E1 Conservative | E2 Moderate | E4 Pairs |
|----------------|----------------|-------------|----------|
| Horizonte | 90 días | 20 días | 10 días |
| Tipo | Direccional | Direccional | Market-neutral |
| Beta de mercado | ~1.0 | ~1.0 | ~0.0 |
| Sharpe objetivo | > 1.0 | > 0.8 | > 1.0-1.2 |
| Max Drawdown | < 20% | < 30% | < 10% |
| Modelo | GRU | LSTM | Cointegración + k-NN |
| Complejidad | Media | Media | Alta |

## Referencias

- **Documentación completa**: [README_E4.md](../README_E4.md)
- **Configuración**: `src/config/base.yaml`
- **Tests**: `scripts/test_e4_simple.py`
