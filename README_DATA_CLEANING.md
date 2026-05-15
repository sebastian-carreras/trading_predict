# Limpieza y Validación de Datos

Sistema de limpieza automática de datos OHLCV para detectar y corregir problemas de calidad antes del entrenamiento.

---

## **¿Por qué es necesario?**

Los datos descargados de yfinance/IOL pueden tener:
- **Valores nulos** (fechas sin datos, APIs fallidas)
- **Timestamps duplicados** (errores de API)
- **Volumen = 0** (días sin trading, datos incorrectos)
- **Gaps temporales** (datos faltantes por períodos prolongados)

**Sin limpieza**, estos problemas causan:
- Errores en cálculo de features (NaN propagation)
- Modelos entrenados con datos incorrectos
- IC negativo por datos contaminados

---

## **Fuentes de Datos**

### **Datos Diarios (E1/E2)**

| Fuente | Cobertura | Uso |
|--------|-----------|-----|
| **yfinance** | Todos los tickers | Fuente primaria para USA tickers |
| **IOL API** | Activos argentinos (`.BA`, `AL*`, `GD*`, `AE*`) | Fuente primaria para Arg tickers |

### **Datos Intraday (E3)**

| Fuente | Cobertura | Frecuencia |
|--------|-----------|------------|
| **Alpaca Markets** | Acciones US (AAPL, NVDA, SPY, etc.) | Barras de 5 min |
| **yfinance** | Fallback cuando Alpaca no disponible | Barras de 5 min |

---

## **Sistema de Diagnóstico**

Detecta automáticamente:

### **1. Valores Nulos**
```
Ticker: AAPL
  ⚠  Nulos detectados en: volume, adj_close
     - volume: 12 (0.5%)
     - adj_close: 3 (0.1%)
```

### **2. Timestamps Duplicados**
```
  ⚠  5 timestamps duplicados
```

### **3. Volumen Cero**
```
  ⚠  18 días con volumen=0
```

### **4. Gaps Temporales**
```
  ⚠  2 gaps grandes en serie temporal (>4 días)
     - 2020-03-15: gap de 7 días
     - 2023-11-23: gap de 5 días
```

---

## **Estrategias de Limpieza**

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

## **Uso**

### **Descarga de datos**
```bash
# Descarga diaria (yfinance + IOL fallback)
python -m src.data.download_daily

# Forzar re-descarga completa
python -m src.data.download_daily --force
```

### **Limpieza de datos**
```bash
# Ver qué problemas hay en los datos (forward fill por defecto)
python scripts/data/run_data_cleaning.py

# Usar interpolación en vez de forward fill
python scripts/data/run_data_cleaning.py --interpolate

# Eliminar filas con nulos (no recomendado)
python scripts/data/run_data_cleaning.py --drop
```

**Output:**
```
================================================================================
LIMPIEZA DE DATOS - Estrategia: FORWARD_FILL
================================================================================

 AAPL
  ✓ Sin valores nulos detectados
  ✓ Guardado: AAPL_daily.csv (2515 días)

 YPFD.BA
  ⚠  Nulos detectados en: volume
     - volume: 5 (0.2%)
  🔧 volume: 5 nulos (0.2%) → ✓ Forward fill
  ✓ Guardado: YPFD.BA_daily.csv (1980 días)

================================================================================
RESUMEN DE LIMPIEZA
================================================================================
Total tickers procesados: 101
  ✓ Limpiados: 99
   Rechazados: 2

📄 Reporte de calidad guardado en: data/clean/data_quality_report.json
```

---

## **Estructura de Archivos**
```
data/
├── raw/
│   ├── daily/                   # ~105 CSVs OHLCV diarios (yfinance + IOL)
│   │   ├── AAPL_daily.csv
│   │   ├── YPFD.BA_daily.csv
│   │   └── ...
│   └── intraday/                # 14 CSVs de barras 5-min (Alpaca)
│       ├── AAPL_5min.csv
│       ├── NVDA_5min.csv
│       └── ...
├── clean/                       # ~101 CSVs limpios (listos para training)
│   ├── AAPL_daily.csv
│   ├── YPFD.BA_daily.csv
│   ├── ...
│   └── data_quality_report.json # Reporte de diagnóstico por ticker
├── cache/                       # Caché temporario
├── features/                    # Salidas de feature engineering
└── snapshots/                   # Outputs de EDA
    ├── e1_conservative_eda_v1/
    ├── e2_moderate_eda_v1/
    └── e3_intraday_eda_v1/
```

---

## **Tickers Soportados (~101 en `data/clean/`)**

### Activos Argentinos (~65)
- **Acciones blue-chip**: YPFD.BA, GGAL.BA, PAMP.BA, BYMA.BA, CEPU.BA, BBAR.BA, BMA.BA, EDN.BA, LOMA.BA, ALUA.BA, AGRO.BA, METR.BA, TGSU2.BA, TGNO4.BA
- **Bonos soberanos**: AL29, AL30, AL35, AL41, GD29, GD30, GD35, GD38, GD41, GD46, AE38
- **Otros**: A3.BA, CECO2.BA, CELU.BA, ETHA.BA, YPF

### Activos Internacionales (~36)
- **Tecnología**: AAPL, NVDA, GOOGL, AMZN, META, NFLX, AMD, TSLA
- **Finanzas**: JPM, BAC, CAT, DE, XOM, CVX
- **Consumo/Salud**: JNJ, PG, V, KO, PEP
- **ETFs de índices**: SPY, QQQ

El universo de tickers por estrategia está definido en `src/config/base.yaml`.

---

## **Reporte de Calidad (JSON)**

El archivo `data/clean/data_quality_report.json` contiene diagnóstico detallado:

```json
{
  "AAPL": {
    "ticker": "AAPL",
    "total_rows": 2515,
    "null_counts": {},
    "null_percentages": {},
    "features_with_nulls": [],
    "zero_volume_days": 0,
    "duplicate_timestamps": 0,
    "data_gaps_days": [],
    "status": "cleaned",
    "rows_after_cleaning": 2515
  },
  "CEPU.BA": {
    "ticker": "CEPU.BA",
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

## **Configuración**

El universo de tickers y parámetros se leen de `src/config/base.yaml`:

```yaml
data:
  start_date: "2016-03-03"
  end_date: "2026-03-03"
  timezone: "America/New_York"
```

### **Umbral mínimo de días:**
```python
min_days=252  # 1 año bursátil — tickers con menos son rechazados
```

---

## **Decisiones sobre Features con Nulos**

### **Si una feature tiene muchos nulos (>10%):**

1. **Analizar causa**:
   ```bash
   cat data/clean/data_quality_report.json | jq '.TICKER'
   ```

2. **Decidir acción**:
   - **< 5% nulos**: Forward fill es seguro
   - **5-10% nulos**: Considerar eliminar feature o ticker
   - **> 10% nulos**: Eliminar ticker (datos de mala calidad)

3. **Implementar**: En el pipeline de training, rechazar el ticker antes de entrenar.

---

## **Validación Post-Limpieza**

Los pipelines E1 y E2 priorizan datos limpios automáticamente:

```python
# En src/e1/train_pipeline.py / src/e2/train_pipeline.py
clean_csv_path = data_dir / "clean" / f"{ticker}_daily.csv"

if clean_csv_path.exists():
    csv_path = clean_csv_path  # Usa datos limpios
else:
    csv_path = data_dir / "raw" / "daily" / f"{ticker}_daily.csv"  # Fallback a raw
```

---

## **Impacto en IC**

**Antes** (sin limpieza):
- Nulos propagados a features → NaN en MACD, RSI, etc.
- Modelo entrena con datos corruptos → IC negativo

**Después** (con limpieza):
- Features calculadas correctamente
- IC mejorado de -0.124 → +0.232

---

## **Troubleshooting**

### **Ticker rechazado por pocos días**
```
 TICKER descartado: solo 180 días después de limpieza (< 252 requerido)
```

**Solución**: Ticker tiene demasiados nulos. Opciones:
1. Eliminar ticker del config en `src/config/base.yaml`
2. Descargar más historia (`period="15y"`)
3. Reducir `min_days` (no recomendado)

### **Features siguen teniendo NaN después de limpieza**
```python
# Debug en el pipeline
features = compute_features(ohlcv)
print(features.isna().sum())
```

**Causa probable**: Features calculadas requieren ventanas (ej. SMA(200) → primeros 200 días son NaN).
**Solución**: `make_sequences()` ya elimina filas con NaN al construir las secuencias. Esto es normal.

### **Datos intraday faltantes (E3)**
```bash
# Verificar archivos existentes
ls data/raw/intraday/

# Re-descargar desde Alpaca (requiere credenciales en .env)
python -m src.e3.intraday_data
```

---

## **Referencias**

| Componente | Archivo |
|------------|---------|
| Descarga diaria | `src/data/download_daily.py` |
| Limpieza diaria | `src/data/clean_daily.py` |
| Script manual | `scripts/data/run_data_cleaning.py` |
| Datos intraday (E3) | `src/e3/intraday_data.py` |
| IOL API | `src/data/iol_api.py` |
| Pipeline E1 | `src/e1/train_pipeline.py` |
| Pipeline E2 | `src/e2/train_pipeline.py` |
| Configuración | `src/config/base.yaml` |
| Reporte de calidad | `data/clean/data_quality_report.json` |
