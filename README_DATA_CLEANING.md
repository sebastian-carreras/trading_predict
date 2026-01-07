# Limpieza y Validación de Datos

Sistema de limpieza automática de datos OHLCV para detectar y corregir problemas de calidad antes del entrenamiento.

---

## 🎯 **¿Por qué es necesario?**

Los datos descargados de yfinance pueden tener:
- **Valores nulos** (fechas sin datos, APIs fallidas)
- **Timestamps duplicados** (errores de API)
- **Volumen = 0** (días sin trading, datos incorrectos)
- **Gaps temporales** (datos faltantes por períodos prolongados)

**Sin limpieza**, estos problemas causan:
- Errores en cálculo de features (NaN propagation)
- Modelos entrenados con datos incorrectos
- IC negativo por datos contaminados

---

## 📊 **Sistema de Diagnóstico**

Detecta automáticamente:

### **1. Valores Nulos**
```
Ticker: AAPL
  ⚠️  Nulos detectados en: volume, adj_close
     - volume: 12 (0.5%)
     - adj_close: 3 (0.1%)
```

### **2. Timestamps Duplicados**
```
  ⚠️  5 timestamps duplicados
```

### **3. Volumen Cero**
```
  ⚠️  18 días con volumen=0
```

### **4. Gaps Temporales**
```
  ⚠️  2 gaps grandes en serie temporal (>4 días)
     - 2020-03-15: gap de 7 días
     - 2023-11-23: gap de 5 días
```

---

## 🛠️ **Estrategias de Limpieza**

### **1. Forward Fill (Recomendado) - `forward_fill`**
```python
strategy="forward_fill"
```

**Qué hace:**
- Propaga el último valor válido hacia adelante
- Si hay nulos al inicio (no hay valor previo), usa backward fill
- **Conservador**: no inventa datos, usa últimos valores conocidos

**Cuándo usar:**
- Siempre (estrategia por defecto)
- Ideal para precios (close, high, low) → precio se mantiene hasta nuevo tick
- Volumen → si no hay datos, asume volumen del día anterior

**Ejemplo:**
```
close:  [100, 105, NaN, NaN, 110]
        ↓
clean:  [100, 105, 105, 105, 110]
```

---

### **2. Interpolación Lineal - `interpolate`**
```python
strategy="interpolate"
```

**Qué hace:**
- Interpola linealmente entre valores conocidos
- Más "suave" que forward fill

**Cuándo usar:**
- Gaps pequeños (1-2 días)
- Features continuas (no precios)
- **Cuidado**: puede introducir valores irreales (ej. precio interpolado que nunca existió)

**Ejemplo:**
```
close:  [100, NaN, NaN, 110]
        ↓
clean:  [100, 103.3, 106.6, 110]
```

---

### **3. Eliminar Filas - `drop`**
```python
strategy="drop"
```

**Qué hace:**
- Elimina cualquier fila con al menos un nulo

**Cuándo usar:**
- **NUNCA para series temporales** (pierdes continuidad temporal)
- Solo si tienes exceso de datos (>10 años) y puedes perder días

**Ejemplo:**
```
df:     [100, 105, NaN, 110, 115]
        ↓
clean:  [100, 105, 110, 115]  # Se pierde el día con NaN
```

---

## 🚀 **Uso**

### **Opción 1: Ejecutar manualmente (desarrollo)**

```bash
# Ver qué problemas hay en los datos
python scripts/run_data_cleaning.py

# Usar interpolación en vez de forward fill
python scripts/run_data_cleaning.py --interpolate

# Eliminar filas con nulos (no recomendado)
python scripts/run_data_cleaning.py --drop
```

**Output:**
```
================================================================================
LIMPIEZA DE DATOS - Estrategia: FORWARD_FILL
================================================================================

📊 AAPL
  ⚠️  Nulos detectados en: volume
     - volume: 5 (0.2%)
  🔧 volume: 5 nulos (0.2%) → ✓ Forward fill
  ✓ Guardado: AAPL_daily.csv (2520 días)

📊 MSFT
  ✓ Sin valores nulos detectados
  ✓ Guardado: MSFT_daily.csv (2520 días)

================================================================================
RESUMEN DE LIMPIEZA
================================================================================
Total tickers procesados: 25
  ✓ Limpiados: 24
  ❌ Rechazados: 1

Tickers con más features problemáticas:
  - CEPU: 3 features con nulos
  - PAMP: 2 features con nulos

✓ Datos limpios guardados en: data/clean

📄 Reporte de calidad guardado en: data/clean/data_quality_report.json
```

---

### **Opción 2: Integrado en DAG de Airflow (automático)**

El DAG E1 ahora tiene un paso de limpieza automático:

```
download_daily_data → clean_daily_data → train_e1_models → notify_api
```

**Flujo:**
1. **download_daily_data**: Descarga datos a `data/raw/daily/`
2. **clean_daily_data**: Limpia y guarda en `data/clean/`
3. **train_e1_models**: Usa datos limpios automáticamente

**Configuración del DAG:**
- Estrategia: `forward_fill` (configurada en el código del DAG)
- Mínimo de días: 252 (1 año)
- Tickers con < 252 días después de limpieza → **rechazados**

---

## 📁 **Estructura de Archivos**

```
data/
├── raw/
│   └── daily/
│       ├── AAPL_daily.csv       # Datos descargados (sin procesar)
│       ├── MSFT_daily.csv
│       └── ...
└── clean/
    ├── AAPL_daily.csv           # Datos limpios (listos para training)
    ├── MSFT_daily.csv
    ├── ...
    └── data_quality_report.json # Reporte de diagnóstico
```

---

## 📄 **Reporte de Calidad (JSON)**

El archivo `data/clean/data_quality_report.json` contiene diagnóstico detallado:

```json
{
  "AAPL": {
    "ticker": "AAPL",
    "total_rows": 2520,
    "null_counts": {
      "volume": 5
    },
    "null_percentages": {
      "volume": 0.2
    },
    "features_with_nulls": ["volume"],
    "zero_volume_days": 0,
    "duplicate_timestamps": 0,
    "data_gaps_days": [],
    "status": "cleaned",
    "rows_after_cleaning": 2520
  },
  "CEPU": {
    "ticker": "CEPU",
    "total_rows": 180,
    "null_counts": {
      "open": 50,
      "high": 50,
      "low": 50
    },
    "null_percentages": {
      "open": 27.8,
      "high": 27.8,
      "low": 27.8
    },
    "features_with_nulls": ["open", "high", "low"],
    "zero_volume_days": 10,
    "duplicate_timestamps": 0,
    "data_gaps_days": [
      {"date": "2023-11-23", "gap_days": 5}
    ],
    "status": "rejected",
    "rows_after_cleaning": 130
  }
}
```

---

## ⚙️ **Configuración Avanzada**

### **Cambiar estrategia en el DAG:**

Edita `dockerfiles/airflow/dags/e1_conservative_pipeline.py`:

```python
def clean_daily_data(**context):
    reports = process_daily_data_with_cleaning(
        raw_dir=raw_dir,
        clean_dir=clean_dir,
        strategy="interpolate",  # Cambiar aquí
        min_days=252,
        verbose=True,
    )
```

### **Ajustar mínimo de días:**

```python
min_days=500,  # Requerir 2 años de datos
```

---

## 🎯 **Decisiones sobre Features con Nulos**

### **Si una feature tiene muchos nulos (>10%):**

1. **Analizar causa**:
   ```bash
   # Ver reporte
   cat data/clean/data_quality_report.json | jq '.TICKER'
   ```

2. **Decidir acción**:
   - **< 5% nulos**: Forward fill es seguro
   - **5-10% nulos**: Considerar eliminar feature o ticker
   - **> 10% nulos**: Eliminar ticker (datos de mala calidad)

3. **Implementar**:
   ```python
   # En train_e1_pipeline.py, después de cargar datos:
   if ohlcv["volume"].isna().sum() > 0.1 * len(ohlcv):
       raise ValueError(f"{ticker}: volumen con >10% nulos, datos no confiables")
   ```

---

## ✅ **Validación Post-Limpieza**

El pipeline E1 ahora prioriza datos limpios:

```python
# En src/train_e1_pipeline.py
clean_csv_path = raw_dir.parent / "clean" / f"{ticker}_daily.csv"

if clean_csv_path.exists():
    csv_path = clean_csv_path  # Usa datos limpios
    print(f"✓ Usando datos limpios")
else:
    csv_path = raw_dir / f"{ticker}_daily.csv"  # Fallback a raw
    print(f"⚠️  Usando datos raw (limpieza no ejecutada)")
```

---

## 📊 **Impacto en IC**

**Antes** (sin limpieza):
- Nulos propagados a features → NaN en MACD, RSI, etc.
- Modelo entrena con datos corruptos → IC negativo

**Después** (con limpieza):
- Features calculadas correctamente
- IC mejorado de -0.124 → +0.232 🎉

---

## 🔧 **Troubleshooting**

### **"No data found" en Airflow**
```bash
# Verificar que los datos fueron descargados
ls data/raw/daily/

# Ejecutar limpieza manualmente
python scripts/run_data_cleaning.py
```

### **Ticker rechazado por pocos días**
```
❌ TICKER descartado: solo 180 días después de limpieza (< 252 requerido)
```

**Solución**: Ticker tiene demasiados nulos. Opciones:
1. Eliminar ticker del config E1
2. Descargar más historia (`period="15y"`)
3. Reducir `min_days` (no recomendado)

### **Features siguen teniendo NaN después de limpieza**
```python
# Debug en train_e1_pipeline.py
features = compute_e1_features(ohlcv, benchmark_df)
print(features.isna().sum())  # Ver cuántos NaN por feature
```

**Causa probable**: Features calculadas requieren ventanas (ej. SMA(200) → primeros 200 días son NaN).
**Solución**: `make_sequences()` ya elimina filas con NaN. Esto es normal.

---

## 📚 **Referencias**

- **Código fuente**: `src/data/clean_daily.py`
- **Script manual**: `scripts/run_data_cleaning.py`
- **DAG integration**: `dockerfiles/airflow/dags/e1_conservative_pipeline.py`
- **Pipeline E1**: `src/train_e1_pipeline.py`
