# Trading en Vivo con E1 Simple + API IOL

Script para ejecutar predicciones en vivo con E1 Simple y operar automáticamente en IOL (Invertir Online) usando su ambiente de prueba.

## 🚀 Quick Start

### 1. Configurar Credenciales

Agregar al archivo `.env`:
```bash
IOL_USERNAME=tu_usuario_iol
IOL_PASSWORD=tu_password_iol
```

### 2. Entrenar Modelo E1 Simple

```bash
# Entrenar para un ticker argentino (IOL)
python -m src.train_e1_simple_pipeline --tickers GGAL

# O para ticker US (si tienes modelo entrenado)
python -m src.train_e1_simple_pipeline --tickers AAPL
```

### 3. Ejecutar Trading en Vivo

```bash
# Dry run (simular sin ejecutar)
python scripts/e1_simple_iol_live_trade.py \
  --ticker GGAL \
  --model runs/e1_simple/20260119_120000/GGAL/GGAL_model.pth \
  --quantity 10 \
  --dry-run

# Ejecutar orden real en IOL (ambiente de prueba)
python scripts/e1_simple_iol_live_trade.py \
  --ticker GGAL \
  --model runs/e1_simple/20260119_120000/GGAL/GGAL_model.pth \
  --quantity 10
```

## 📋 Flujo del Script

1. **Autenticación IOL**
   - POST a `https://api.invertironline.com/token`
   - Headers: `Content-Type: application/x-www-form-urlencoded`
   - Body: `grant_type=password&username=xxx&password=yyy`
   - Obtiene `access_token` con validez ~1 hora

2. **Cargar Modelo E1 Simple**
   - Lee archivo `.pth` generado por el pipeline
   - Extrae: arquitectura GRU, scalers (X e y), lookback_days

3. **Descargar Datos Recientes**
   - Últimos 365 días del ticker via yfinance
   - Benchmark (SPY) si está disponible

4. **Calcular Features y Predecir**
   - Calcula 27 features (momentum, tendencia, volumen)
   - Normaliza usando scalers del entrenamiento
   - Predice retorno a 90 días

5. **Decision Score Simplificado**
   - Compara predicción vs `tau_buy` (default 0.06 = +6%)
   - Si pred ≥ tau_buy → score alto → BUY
   - Si score ≥ 0.70 → ejecutar orden

6. **Ejecutar Orden en IOL**
   - POST a `/api/v2/operar/Comprar` si BUY
   - Mercado: `bCBA` (Bolsa argentina)
   - Plazo: `t2`, Validez: `dia`

## 🔧 Parámetros

```bash
--ticker GGAL                 # Ticker a operar (argentino para IOL)
--model path/to/model.pth     # Modelo E1 Simple entrenado
--quantity 10                 # Cantidad de acciones (default: 1)
--config src/config/base.yaml # Configuración (default)
--dry-run                     # Simular sin ejecutar orden
```

## 📊 Output Esperado

```
======================================================================
E1 Simple - Trading en vivo con API IOL (ambiente de prueba)
======================================================================
Ticker: GGAL
Modelo: runs/e1_simple/20260119_120000/GGAL/GGAL_model.pth
Cantidad: 10
Dry run: False
======================================================================

1️⃣  Autenticando con IOL...
✓ Autenticación exitosa. Token expira en 3600s
✓ Cuenta conectada

2️⃣  Cargando modelo E1 Simple...
Cargando modelo: GGAL_model.pth
✓ Modelo cargado: GGAL - e1_simple
  Lookback: 360d | Horizon: 90d
  Features: 27

3️⃣  Descargando datos recientes...
Descargando 365 días de GGAL...
✓ Descargados 252 días (desde 2025-01-19 hasta 2026-01-18)
Descargando 365 días de SPY...
✓ Descargados 252 días (desde 2025-01-19 hasta 2026-01-18)

4️⃣  Calculando predicción...
✓ Predicción: retorno esperado = +8.34%

5️⃣  Evaluando decisión...
✓ Decision score: 0.851 (threshold: 0.70) → BUY

6️⃣  Ejecutando acción...
🟢 Señal: COMPRAR 10 GGAL
   Ejecutando orden en IOL...
✓ Orden ejecutada: COMPRA 10 GGAL
  Detalles: {...}
✓ Orden ejecutada exitosamente

======================================================================
✓ Proceso completado
======================================================================
```

## 🛡️ Consideraciones de Seguridad

### Ambiente de Prueba
- El script usa `https://api.invertironline.com` (ambiente de prueba por defecto)
- Para producción, cambiar a URL de producción en `IOLClient.BASE_URL_TEST`

### Credenciales
- **Nunca** comitear el `.env` con credenciales reales
- Usar variables de entorno o secret managers en producción

### Validación
- Siempre probar con `--dry-run` primero
- Verificar que el modelo esté entrenado con el ticker correcto
- Revisar que `quantity` sea razonable

## 📈 Limitaciones Actuales

### Decision Score Simplificado
El script usa un **decision score simplificado** basado solo en la predicción:
- No calcula IC, Sharpe, MAE, RMSE en tiempo real (requeriría datos out-of-sample)
- Usa heurística: `pred_return ≥ tau_buy` → score alto

**Para producción**, considerar:
- Calcular métricas rolling con ventana móvil
- Usar backtesting reciente (últimos 30-60 días)
- Implementar sistema de confianza basado en performance histórica

### Tickers Argentinos vs US
- IOL opera principalmente tickers argentinos (GGAL, YPFD, etc.)
- Modelo E1 puede estar entrenado con tickers US (AAPL, MSFT)
- Asegurar coherencia ticker modelo ↔ ticker IOL

## 🔄 Workflow Recomendado

### Desarrollo
```bash
# 1. Entrenar modelo con datos históricos
python -m src.train_e1_simple_pipeline --tickers GGAL

# 2. Probar predicción en vivo (dry run)
python scripts/e1_simple_iol_live_trade.py \
  --ticker GGAL \
  --model runs/e1_simple/latest/GGAL/GGAL_model.pth \
  --dry-run

# 3. Si decision = BUY y score alto, ejecutar orden pequeña
python scripts/e1_simple_iol_live_trade.py \
  --ticker GGAL \
  --model runs/e1_simple/latest/GGAL/GGAL_model.pth \
  --quantity 1
```

### Automatización (Cron/Airflow)
```bash
# Ejecutar diariamente a las 10 AM (antes de apertura del mercado)
0 10 * * 1-5 /path/to/venv/bin/python /path/to/scripts/e1_simple_iol_live_trade.py --ticker GGAL --model /path/to/model.pth --quantity 5
```

## 📚 Documentación API IOL

- **Autenticación:** https://api.invertironline.com/Help/Autenticacion
- **Endpoints:** https://api.invertironline.com/Help
- **Ambiente de prueba:** Incluido en la documentación oficial

## 🐛 Troubleshooting

### Error: "No autenticado"
- Verificar credenciales en `.env`
- Revisar que `IOL_USERNAME` y `IOL_PASSWORD` estén correctos

### Error: "Modelo no encontrado"
- Verificar path al `.pth`
- Entrenar modelo si no existe: `python -m src.train_e1_simple_pipeline --tickers GGAL`

### Error: "Insuficientes datos"
- El modelo requiere al menos 360 días de historia
- Verificar que el ticker tenga datos suficientes en yfinance

### Error al ejecutar orden
- Revisar logs de respuesta de IOL
- Verificar saldo disponible en cuenta IOL
- Confirmar que el ticker esté disponible en IOL

## ⚠️ Disclaimer

Este script es para **fines educativos y de prueba**. 

- Usar **ambiente de prueba** de IOL para experimentar
- **No** operar con dinero real sin validación exhaustiva
- **No** garantizamos rentabilidad ni precisión de predicciones
- El trading algorítmico tiene riesgos significativos

---

**Última actualización:** Enero 19, 2026  
**Versión:** 1.0
