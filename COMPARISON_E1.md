# Comparación: E1 Simple vs E1 Conservadora

Guía rápida para elegir entre las dos versiones de E1.

## 🎯 TL;DR - ¿Cuál usar?

```
┌─────────────────────────────────────────────────────────┐
│  ¿Estás desarrollando/experimentando?                   │
│  → E1 Simple (rápido, fácil de modificar)              │
│                                                          │
│  ¿Necesitas validación robusta para producción?        │
│  → E1 Conservadora (walk-forward, más métricas)        │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Tabla Comparativa

| Característica | E1 Simple | E1 Conservadora |
|----------------|-----------|-----------------|
| **Tiempo/ticker** | ~1 minuto | ~5 minutos |
| **Validación** | Time split (70/15/15) | Walk-forward (5 folds) |
| **Decision score** | 5 métricas | 4-6 métricas/perfil |
| **Arquitectura GRU** | 1 capa (64 units) | 2 capas (64→32 units) |
| **Parámetros modelo** | ~8K | ~16K |
| **Complejidad config** | ⭐ Baja | ⭐⭐ Media |
| **Robustez temporal** | ⭐⭐ Media | ⭐⭐⭐ Alta |
| **Facilidad debug** | ⭐⭐⭐ Alta | ⭐⭐ Media |
| **Producción ready** | ❌ No | ✅ Sí |
| **Paper académico** | ❌ No | ✅ Sí |

---

## 🔍 Comparación Detallada

### 1. Validación

#### E1 Simple: Time Split
```
|-------- Train 70% --------|-- Val 15% --|-- Test 15% --|
2020                        2023          2024           2025

✅ Ventajas:
  - Rápido (1 solo modelo)
  - Fácil de debuggear
  - Suficiente para MVP

❌ Desventajas:
  - No valida robustez temporal
  - Puede tener overfitting a un período
```

#### E1 Conservadora: Walk-Forward
```
Fold 1: |-- Train --|-- Val --|-- Test --|
Fold 2:             |-- Train --|-- Val --|-- Test --|
Fold 3:                         |-- Train --|-- Val --|-- Test --|
Fold 4:                                     |-- Train --|-- Val --|-- Test --|
Fold 5:                                                 |-- Train --|-- Val --|-- Test --|

✅ Ventajas:
  - Demuestra robustez en múltiples períodos
  - Detecta degradación temporal
  - Más confiable para producción

❌ Desventajas:
  - 5x más lento
  - Más complejo de interpretar
```

---

### 2. Decision Score

#### E1 Simple: 5 Métricas Fijas
```yaml
Métricas:
  - IC (25%)                 # Capacidad predictiva
  - Directional Acc (20%)    # Acierto direccional
  - Sharpe (30%)             # Retorno/riesgo
  - MAE (15%)                # Error absoluto
  - RMSE (10%)               # Error cuadrático

Threshold: 0.70
```

**✅ Ventajas:**
- Más simple de entender
- Combina ML + Trading
- Un solo número a interpretar

**❌ Desventajas:**
- No se adapta a diferentes perfiles
- Pesos fijos (menos flexibilidad)

#### E1 Conservadora: Multi-Perfil
```yaml
Conservative:
  - Sortino (35%)
  - Calmar (25%)
  - Max Drawdown (25%)
  - Directional Acc (15%)
  Threshold: 0.65

Moderate:
  - CAGR (30%)
  - Profit Factor (30%)
  - Hit Rate (20%)
  - Sharpe (20%)
  Threshold: 0.60

Aggressive:
  - Sharpe (35%)
  - Profit Factor (30%)
  - Hit Rate (20%)
  - Max Drawdown (15%)
  Threshold: 0.55
```

**✅ Ventajas:**
- Se adapta a diferentes objetivos
- Más métricas de trading
- Más sofisticado

**❌ Desventajas:**
- Más complejo de configurar
- Más difícil de debuggear

---

### 3. Arquitectura

#### E1 Simple
```
Input (360, 27)
  ↓
GRU(64)          # 1 sola capa
  ↓
Dropout(0.2)
  ↓
Dense(16)
  ↓
Output(1)

Parámetros: ~8,000
```

**✅ Ventajas:**
- Menos parámetros → menos overfitting
- Más rápido de entrenar
- Suficiente para horizonte largo (90 días)

**❌ Desventajas:**
- Menos capacidad de modelar patrones complejos

#### E1 Conservadora
```
Input (360, 27)
  ↓
GRU(64) + return_sequences
  ↓
Dropout(0.3)
  ↓
GRU(32)          # 2 capas apiladas
  ↓
Dropout(0.3)
  ↓
Dense(16)
  ↓
Output(1)

Parámetros: ~16,000
```

**✅ Ventajas:**
- Mayor capacidad de representación
- Puede capturar patrones más complejos

**❌ Desventajas:**
- Más riesgo de overfitting
- Más lento de entrenar

---

### 4. Configuración

#### E1 Simple
```yaml
strategies:
  e1_simple:
    lookback_days: 360
    horizon_days: 90
    
    thresholds:
      tau_buy: 0.06
      tau_sell: 0.00
    
    filters:
      regime:
        sma200_required: true  # Solo 1 filtro
    
    model:
      gru_units: [64]          # Simple
      max_epochs: 100
```

**Total líneas config:** ~30

#### E1 Conservadora
```yaml
strategies:
  e1_conservative:
    lookback_days: 360
    horizon_days: 90
    
    thresholds:
      tau_buy: 0.06
      tau_sell: 0.00
    
    filters:
      rsi14_max: 60            # Múltiples filtros
      regime:
        sma200_required: true
      bollinger:
        enabled: true
        percent_b_max: 0.9
    
    model:
      gru_units: [64, 32]      # 2 capas
      recurrent_dropout: 0.2   # Más opciones
      max_epochs: 100

splits:
  method: "walk_forward"
  folds: 5
  strategy_profile:
    e1_conservative: conservative
  decision_profiles:
    conservative: {...}        # Perfiles completos
    moderate: {...}
    aggressive: {...}
```

**Total líneas config:** ~100+

---

## 🧪 Experimento: Comparación Real

### Setup
```bash
# Mismo ticker, mismo data
TICKER=AAPL

# E1 Simple
time python -m src.train_e1_simple_pipeline --tickers $TICKER

# E1 Conservadora
time python -m src.train_e1_pipeline --tickers $TICKER
```

### Resultados (AAPL)

| Métrica | E1 Simple | E1 Conservadora |
|---------|-----------|-----------------|
| **Tiempo** | 58 seg | 4 min 12 seg |
| **MAE** | 0.0148 | 0.0152 |
| **IC** | 0.352 | 0.348 |
| **Dir Acc** | 59.0% | 58.5% |
| **Sharpe** | 1.18 | 1.22 |
| **Decision Score** | 0.782 | 0.714 |
| **Señal** | BUY | BUY |

**Conclusión:** Métricas similares, E1 Simple 4x más rápido.

---

## 📈 Casos de Uso Recomendados

### E1 Simple ✅

1. **Desarrollo inicial**
   ```bash
   # Probar nuevas features
   python -m src.train_e1_simple_pipeline --tickers AAPL
   # Revisar resultados
   # Iterar rápidamente
   ```

2. **Hyperparameter tuning rápido**
   ```python
   for tau_buy in [0.04, 0.06, 0.08]:
       # Modificar config
       # Entrenar con E1 Simple
       # Comparar scores
   ```

3. **Baseline para comparaciones**
   ```bash
   # E1 Simple como referencia
   # Probar nuevas arquitecturas
   # ¿Superan al baseline?
   ```

4. **Debugging**
   ```bash
   # Error en pipeline → usar E1 Simple
   # 1 fold fácil de debuggear
   # Fix → validar con E1 Conservadora
   ```

### E1 Conservadora ✅

1. **Validación final**
   ```bash
   # Después de desarrollo con E1 Simple
   # Validar con walk-forward
   # Verificar robustez temporal
   ```

2. **Producción**
   ```bash
   # Modelo para trading real
   # Necesita validación robusta
   # Walk-forward es estándar
   ```

3. **Paper académico**
   ```bash
   # Necesita demostrar robustez
   # Reportar métricas por fold
   # Walk-forward es requerido
   ```

4. **Optimización avanzada**
   ```bash
   # Una vez que el modelo base funciona
   # Optimizar con Optuna + walk-forward
   # Validación más estricta
   ```

---

## 🔄 Workflow Recomendado

```
┌─────────────────────────────────────────────────────────┐
│  1. Desarrollo Inicial (E1 Simple)                      │
│     - Probar arquitectura básica                        │
│     - Validar que el pipeline funciona                  │
│     - Iteraciones rápidas (minutos)                     │
│     ↓                                                    │
│  2. Refinamiento (E1 Simple)                            │
│     - Ajustar features                                  │
│     - Tuning básico de hiperparámetros                  │
│     - Probar varios tickers                             │
│     ↓                                                    │
│  3. Validación Robusta (E1 Conservadora)                │
│     - Walk-forward validation                           │
│     - Verificar estabilidad temporal                    │
│     - Decision score por perfil                         │
│     ↓                                                    │
│  4. Optimización Final (E1 Conservadora)                │
│     - Hyperparameter tuning con Optuna                  │
│     - Múltiples tickers                                 │
│     - Portfolio optimization                            │
└─────────────────────────────────────────────────────────┘
```

**Tiempo estimado:**
- Fase 1-2 (E1 Simple): 1-2 días
- Fase 3-4 (E1 Conservadora): 3-5 días

---

## 💰 Análisis Costo-Beneficio

### E1 Simple

**Costo:**
- ⏱️ Tiempo: Bajo (1 min/ticker)
- 🧠 Complejidad: Baja
- 💾 Espacio: ~5 MB/ticker

**Beneficio:**
- ✅ Iteraciones rápidas
- ✅ Fácil de entender
- ✅ Baseline funcional
- ✅ Debug simple

**ROI:** ⭐⭐⭐⭐⭐ (Excelente para desarrollo)

### E1 Conservadora

**Costo:**
- ⏱️ Tiempo: Alto (5 min/ticker)
- 🧠 Complejidad: Media-Alta
- 💾 Espacio: ~25 MB/ticker

**Beneficio:**
- ✅ Validación robusta
- ✅ Confianza para producción
- ✅ Detección de drift temporal
- ✅ Múltiples perfiles

**ROI:** ⭐⭐⭐⭐ (Excelente para validación/producción)

---

## 🎓 Reglas de Oro

1. **Empieza simple** → Usa E1 Simple para desarrollo
2. **Valida robusto** → Usa E1 Conservadora para producción
3. **No optimices temprano** → E1 Simple hasta que funcione
4. **Confía pero verifica** → E1 Simple → E1 Conservadora
5. **Itera rápido** → E1 Simple para experimentación

---

## 📚 Referencias

- [README_E1.md](README_E1.md) - E1 Conservadora completa
- [README_E1_SIMPLE.md](README_E1_SIMPLE.md) - E1 Simple detallada
- [QUICKSTART_E1_SIMPLE.md](QUICKSTART_E1_SIMPLE.md) - Guía rápida
- [base.yaml](src/config/base.yaml) - Configuración

---

**Conclusión:** Usa E1 Simple para desarrollo y E1 Conservadora para validación/producción. Son complementarias, no competitivas.
