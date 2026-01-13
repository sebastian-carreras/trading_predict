# Configuración de Fallback con IOL API

Este proyecto usa **Yahoo Finance** como fuente principal de datos, pero incluye un **fallback automático con la API de InvertirOnline (IOL)** para tickers argentinos cuando Yahoo Finance falla.

## ¿Por qué IOL como Fallback?

- Yahoo Finance ocasionalmente tiene problemas con tickers argentinos (.BA)
- IOL proporciona datos directos del mercado argentino (BCBA)
- El fallback es automático y transparente

## Configuración (Opcional)

El fallback de IOL **solo se activa si configuras las credenciales**. Si no las configuras, el sistema funcionará normalmente con Yahoo Finance únicamente.

### 1. Requisitos

- Cuenta activa en [InvertirOnline](https://www.invertironline.com/)
- Activación de APIs (solicitar desde la sección de Mensajes)
- Aceptar términos y condiciones en: Mi Cuenta > Personalización > APIs

### 2. Configurar Variables de Entorno

Agrega tus credenciales de IOL como variables de entorno:

**Opción A: Archivo `.env` (desarrollo local)**

Crea un archivo `.env` en la raíz del proyecto:

```bash
IOL_USERNAME=tu_usuario_iol
IOL_PASSWORD=tu_contraseña_iol
```

**Opción B: Variables de entorno del sistema**

```bash
export IOL_USERNAME="tu_usuario_iol"
export IOL_PASSWORD="tu_contraseña_iol"
```

**Opción C: Docker Compose (producción)**

Edita `docker-compose.yaml` y agrega las variables en el servicio de Airflow:

```yaml
services:
  airflow-scheduler:
    environment:
      - IOL_USERNAME=${IOL_USERNAME}
      - IOL_PASSWORD=${IOL_PASSWORD}
```

Luego crea un archivo `.env` en la raíz con las credenciales.

### 3. Verificar Configuración

Ejecuta un test de descarga:

```bash
python -c "from src.data.iol_api import IOLClient; client = IOLClient(); print('✓ Credenciales IOL configuradas correctamente')"
```

Si aparece el mensaje de éxito, el fallback está listo.

## Uso

El fallback es **completamente automático**. Cuando ejecutes cualquier pipeline:

```bash
# Descarga manual
python -m src.data.download_daily

# O desde Airflow DAGs
# El DAG automáticamente intentará IOL si Yahoo Finance falla
```

### Flujo de Fallback

Para cada ticker `.BA`:

1. **Intento 1**: Yahoo Finance
   - ✓ Si funciona → guarda datos y continúa
   - ✗ Si falla → intenta paso 2

2. **Intento 2**: IOL API (solo si credenciales están configuradas)
   - ✓ Si funciona → guarda datos y continúa
   - ✗ Si falla → marca ticker como fallido

### Ejemplo de Logs

```
Descargando GGAL.BA desde Yahoo Finance...
  ⚠️  YFinance falló para GGAL.BA: HTTPError 404
  🔄 Intentando fallback con IOL API...
  📡 Descargando GGAL.BA desde IOL API...
  ✓ IOL: 2520 días guardados en GGAL.BA_daily.csv
```

## Desactivar Fallback

Si por alguna razón quieres desactivar el fallback de IOL:

**Opción 1**: No configurar las variables de entorno (se desactiva automáticamente)

**Opción 2**: Modificar el código en `src/data/download_daily.py`:

```python
written = download_daily_ohlcv(
    tickers, 
    out_dir=out_dir, 
    period="10y",
    skip_existing=True,
    use_iol_fallback=False,  # ← Cambiar a False
)
```

## API de IOL - Límites y Consideraciones

- **Autenticación**: Tokens válidos por 15 minutos (se renuevan automáticamente)
- **Rate Limits**: No documentados públicamente, pero razonables para uso normal
- **Mercados Soportados**: BCBA (Buenos Aires), NYSE (via ADRs)
- **Datos Ajustados**: La API soporta datos ajustados por splits/dividendos
- **Documentación**: https://api.invertironline.com/

## Seguridad

⚠️ **NUNCA** subas credenciales al repositorio:

- El archivo `.env` está en `.gitignore`
- Usa variables de entorno o secretos de Docker
- En producción, usa servicios de gestión de secretos (AWS Secrets Manager, HashiCorp Vault, etc.)

## Troubleshooting

### Error: "Se requieren credenciales IOL"

**Causa**: Variables de entorno no configuradas

**Solución**: Configura `IOL_USERNAME` y `IOL_PASSWORD` como se describe arriba

### Error: "401 Unauthorized"

**Causa**: Credenciales incorrectas o APIs no activadas en IOL

**Solución**: 
1. Verifica usuario/contraseña en IOL
2. Asegúrate de haber solicitado activación de APIs
3. Acepta los términos en: Mi Cuenta > Personalización > APIs

### IOL devuelve datos vacíos

**Causa**: Ticker no existe en IOL o formato incorrecto

**Solución**: Verifica que el ticker existe en el mercado argentino (BCBA)

### Timeout en autenticación

**Causa**: Problemas de red o servidor IOL caído

**Solución**: 
1. Verifica tu conexión a internet
2. Intenta más tarde (servidor IOL puede estar en mantenimiento)
3. Revisa el status en https://www.invertironline.com/

## Soporte

- IOL API Docs: https://api.invertironline.com/
- IOL Help: https://api.invertironline.com/Help/Autenticacion
- Soporte IOL: 0810-1222-IOL(465)
