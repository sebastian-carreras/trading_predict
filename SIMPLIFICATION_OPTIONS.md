# Opciones de Simplificación para E1

Documento de diseño con todas las opciones discutidas para simplificar E1.

## 🎯 Estado Actual

### ✅ Implementadas en E1 Simple

1. **Decision Score Simplificado** (Opción 1)
   - 5 métricas: IC, Directional Accuracy, Sharpe, MAE, RMSE
   - Un solo perfil (sin conservative/moderate/aggressive)
   - Threshold único: 0.70

2. **Sin Walk-Forward** (Opción 4)
   - Time split simple: 70/15/15
   - 1 fold en vez de 5
   - 80% más rápido

3. **Arquitectura Simplificada** (Opción 3 - parcial)
   - GRU de 1 capa (64 units)
   - Dense(16)
   - ~50% menos parámetros

---

## 🔮 Opciones Futuras

### Opción 2: Reducir Features (27 → 10-12)

**Estado:** ❌ No implementado

**Propuesta:**
```python
# En vez de 27 features, usar solo:
essential_features = [
    # Retornos (4)
    'ret_1d', 'ret_5d', 'ret_20d', 'vol_20d',
    
    # Tendencia (3)
    'sma_50', 'sma_200', 'rsi_14',
    
    # Momentum (2)
    'macd_line', 'macd_signal',
    
    # Benchmark (2)
    'bench_ret_1d', 'bench_ret_20d'
]
# Total: 11 features
```

**Beneficios:**
- ✅ Menos riesgo de overfitting
- ✅ Más rápido de calcular
- ✅ Más interpretable
- ✅ Menos correlación entre features

**Cómo implementar:**
```python
# En src/features/build_features_e1.py
def compute_e1_features_minimal(ohlcv, benchmark_df=None):
    """Versión con solo 11 features esenciales."""
    df = pd.DataFrame(index=ohlcv.index)
    
    # Retornos
    df['ret_1d'] = ohlcv['close'].pct_change()
    df['ret_5d'] = ohlcv['close'].pct_change(5)
    df['ret_20d'] = ohlcv['close'].pct_change(20)
    df['vol_20d'] = df['ret_1d'].rolling(20).std()
    
    # Tendencia
    df['sma_50'] = ohlcv['close'].rolling(50).mean()
    df['sma_200'] = ohlcv['close'].rolling(200).mean()
    df['rsi_14'] = compute_rsi(ohlcv['close'], 14)
    
    # Momentum
    df['macd_line'], df['macd_signal'], _ = compute_macd(ohlcv['close'])
    
    # Benchmark
    if benchmark_df is not None:
        df['bench_ret_1d'] = benchmark_df['close'].pct_change()
        df['bench_ret_20d'] = benchmark_df['close'].pct_change(20)
    
    return df.dropna()
```

**Archivo de config:**
```yaml
# src/config/base.yaml
strategies:
  e1_minimal:
    features:
      mode: "minimal"  # En vez de "full"
      count: 11
```

---

### Opción 5: Simplificar Reglas de Trading

**Estado:** ⚠️ Parcialmente implementado (solo SMA200)

**Propuesta actual (E1 Simple):**
```python
# Solo 1 filtro
if y_pred > tau_buy and close > sma_200:
    signal = "BUY"
```

**Simplificación adicional posible:**
```python
# Opción A: Sin filtros
if y_pred > tau_buy:
    signal = "BUY"

# Opción B: Solo stop loss
if y_pred > tau_buy:
    signal = "BUY"
    stop_loss = -0.10  # Solo este control de riesgo
```

**Beneficios:**
- ✅ Más trades (menos restrictivo)
- ✅ Más simple de explicar
- ✅ Menos parámetros que optimizar

**Comparación:**

| Versión | Filtros | Complejidad | Trades/año (est.) |
|---------|---------|-------------|-------------------|
| E1 Conservadora | 4 (RSI, SMA200, Bollinger, ADX) | Alta | ~8 |
| E1 Simple | 1 (SMA200) | Baja | ~12 |
| E1 Minimal (propuesta) | 0 | Muy baja | ~15-20 |

---

### Opción 3 Extended: Modelos más Simples

**Estado:** ⚠️ Parcialmente implementado (GRU 1 capa)

**Arquitecturas alternativas:**

#### A) GRU más pequeño
```python
# Actual E1 Simple: GRU(64)
# Propuesta: GRU(32)
model:
  gru_units: [32]  # En vez de [64]
  dense_units: 8   # En vez de 16
```

**Pros:** Más rápido, menos overfitting  
**Cons:** Menos capacidad de modelar

#### B) Modelo lineal (baseline fuerte)
```python
Input (360 días, 11 features) → Flatten (3960)
  ↓
Dense(128, relu)
  ↓
Dropout(0.3)
  ↓
Dense(32, relu)
  ↓
Dropout(0.2)
  ↓
Dense(1)
```

**Pros:** Muy rápido, muy interpretable  
**Cons:** No captura dependencias temporales

**Cómo implementar:**
```python
# src/models/e1_linear.py
import torch.nn as nn

class LinearRegressor(nn.Module):
    def __init__(self, input_dim, hidden_sizes=[128, 32]):
        super().__init__()
        self.flatten = nn.Flatten()
        
        layers = []
        prev_size = input_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.flatten(x)  # (batch, seq_len * features)
        return self.net(x)   # (batch, 1)
```

---

## 📊 Matriz de Complejidad

| Versión | Features | Modelo | Validación | Decision | Trading | Tiempo | Complejidad |
|---------|----------|--------|------------|----------|---------|--------|-------------|
| **E1 Conservadora** | 27 | GRU 2 capas | Walk-forward | Multi-perfil | 4 filtros | 5 min | ⭐⭐⭐⭐⭐ |
| **E1 Simple** ✅ | 27 | GRU 1 capa | Time split | 5 métricas | 1 filtro | 1 min | ⭐⭐⭐ |
| **E1 Minimal** (propuesta) | 11 | GRU 1 capa | Time split | 5 métricas | 0 filtros | 45 seg | ⭐⭐ |
| **E1 Baseline** (propuesta) | 11 | Linear | Time split | 3 métricas | 0 filtros | 20 seg | ⭐ |

---

## 🚀 Roadmap de Simplificación

### Fase 1: MVP ✅ (Implementado)
- [x] Decision score simplificado (5 métricas)
- [x] Sin walk-forward (time split)
- [x] Arquitectura GRU simple (1 capa)
- [x] Pipeline funcional
- [x] Documentación completa

**Resultado:** E1 Simple operativo

---

### Fase 2: Minimal (Opcional)
- [ ] Reducir features a 11-12
- [ ] Remover filtros de trading
- [ ] GRU más pequeño (32 units)
- [ ] Decision score con 3 métricas

**Objetivo:** E1 Minimal ~45 segundos/ticker

**Configuración propuesta:**
```yaml
strategies:
  e1_minimal:
    features:
      mode: "minimal"
      count: 11
    
    thresholds:
      tau_buy: 0.06
      tau_sell: 0.00
    
    filters: {}  # Sin filtros
    
    model:
      gru_units: [32]
      dense_units: 8
      max_epochs: 80

decision_minimal:
  targets:
    ic_min: 0.05
    sharpe_min: 1.0
    directional_accuracy_min: 0.55
  weights:
    ic: 0.40
    sharpe: 0.40
    directional_accuracy: 0.20
  threshold: 0.65
```

---

### Fase 3: Baseline (Experimental)
- [ ] Modelo lineal (sin RNN)
- [ ] 11 features
- [ ] Decision score ultra-simple (2-3 métricas)
- [ ] Sin filtros

**Objetivo:** Baseline máximo simple ~20 segundos/ticker

**Uso:** Punto de comparación absoluto. Si GRU no supera a este baseline, hay problemas.

**Configuración propuesta:**
```yaml
strategies:
  e1_baseline:
    features:
      mode: "minimal"
      count: 11
    
    model:
      type: "linear"
      hidden_sizes: [128, 32]
      max_epochs: 50

decision_baseline:
  targets:
    sharpe_min: 0.8
    directional_accuracy_min: 0.53
  weights:
    sharpe: 0.60
    directional_accuracy: 0.40
  threshold: 0.60
```

---

## 🎯 Decisión: ¿Qué implementar?

### Criterios de Decisión

| Fase | ¿Cuándo implementar? |
|------|----------------------|
| **E1 Simple** ✅ | Siempre (ya está listo) |
| **E1 Minimal** | Si necesitas iterar MUY rápido (ej: grid search manual) |
| **E1 Baseline** | Si quieres comparar GRU vs modelo simple |

### Recomendación Actual

**Para desarrollo normal:** E1 Simple es suficiente

**Razones:**
1. Ya es 80% más rápido que E1 Conservadora
2. Mantiene todas las features (27)
3. Arquitectura GRU de 1 capa es simple pero efectiva
4. Decision score con 5 métricas es interpretable

**E1 Minimal solo si:**
- Necesitas ejecutar 100+ experimentos
- Tienes límites de tiempo estrictos
- Quieres el mínimo viable absoluto

**E1 Baseline solo si:**
- Quieres validar que GRU agrega valor vs modelo lineal
- Paper académico que requiere ablation study
- Debugging: sospechas que el modelo temporal no está aprendiendo

---

## 📈 Análisis Costo-Beneficio

### E1 Simple (actual) ✅

**Costo de implementación:** Completo  
**Ganancia vs E1 Conservadora:** 
- 80% tiempo ⬆️
- 70% config ⬆️
- 50% parámetros ⬆️

**Pérdida:**
- Robustez temporal ⬇️ (sin walk-forward)

**Veredicto:** ⭐⭐⭐⭐⭐ Excelente trade-off

---

### E1 Minimal (propuesta)

**Costo de implementación:** ~2-3 horas
**Ganancia adicional vs E1 Simple:**
- 25% tiempo ⬆️ (45 seg vs 1 min)
- 55% features ⬆️ (11 vs 27)
- 20% config ⬆️

**Pérdida:**
- Información ⬇️ (menos features)
- Capacidad predictiva ⬇️ (posible)

**Veredicto:** ⭐⭐⭐ Útil pero no esencial

---

### E1 Baseline (propuesta)

**Costo de implementación:** ~4-5 horas
**Ganancia adicional vs E1 Simple:**
- 66% tiempo ⬆️ (20 seg vs 1 min)
- Interpretabilidad ⬆️ (modelo lineal)

**Pérdida:**
- Capacidad temporal ⬇️ (no RNN)
- Probablemente peor performance

**Veredicto:** ⭐⭐ Solo para comparación/ablation

---

## 💡 Recomendación Final

### Para tu proyecto actual:

```
┌──────────────────────────────────────────────────┐
│  Usa E1 Simple (ya implementado)                │
│                                                  │
│  ✅ Suficiente para desarrollo                  │
│  ✅ 80% más rápido que E1 Conservadora          │
│  ✅ Mantiene todas las features                 │
│  ✅ Decision score interpretable                │
│  ✅ Documentación completa                      │
│                                                  │
│  NO implementes E1 Minimal/Baseline a menos que:│
│  - Necesites grid search masivo (Minimal)       │
│  - Paper académico requiera ablation (Baseline) │
└──────────────────────────────────────────────────┘
```

### Workflow Sugerido:

```
Desarrollo → E1 Simple (1 min/ticker)
    ↓
Validación → E1 Conservadora (5 min/ticker)
    ↓
Producción → E1 Conservadora optimizada
```

---

## 🔧 Cómo Implementar E1 Minimal (Si lo necesitas)

### 1. Crear features mínimas

```bash
# src/features/build_features_e1_minimal.py
cp src/features/build_features_e1.py src/features/build_features_e1_minimal.py
# Editar para solo calcular 11 features
```

### 2. Actualizar config

```yaml
# src/config/base.yaml
strategies:
  e1_minimal:
    features:
      mode: "minimal"
    model:
      gru_units: [32]
      dense_units: 8
```

### 3. Crear pipeline

```bash
# src/train_e1_minimal_pipeline.py
cp src/train_e1_simple_pipeline.py src/train_e1_minimal_pipeline.py
# Modificar para usar build_features_e1_minimal
```

### 4. Ejecutar

```bash
python -m src.train_e1_minimal_pipeline --tickers AAPL
```

**Tiempo estimado implementación:** 2-3 horas

---

## 📚 Referencias

- [README_E1_SIMPLE.md](README_E1_SIMPLE.md) - Documentación de E1 Simple
- [COMPARISON_E1.md](COMPARISON_E1.md) - Comparación detallada
- [QUICKSTART_E1_SIMPLE.md](QUICKSTART_E1_SIMPLE.md) - Guía rápida

---

**Última actualización:** 16 Enero 2026  
**Estado:** E1 Simple implementado y listo para usar
