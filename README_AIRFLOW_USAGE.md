# Guía: Ejecutar DAGs para Tickers Específicos

## E1 Conservative Pipeline - Ejecución Selectiva

### Desde Airflow UI (Recomendado)

1. **Acceder**: http://localhost:8080

2. **Trigger con Configuración**:
   - Click en `e1_conservative_pipeline`
   - Click en **"Play"** (▶) → **"Trigger DAG w/ config"**
   - En el JSON editor:
   
   ```json
   {
     "tickers": "AAPL,MSFT,GOOGL"
   }
   ```
   
   - Click **"Trigger"**

### Desde CLI

```bash
# Un ticker específico
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_conservative_pipeline \
  --conf '{"tickers": "AAPL"}'

# Múltiples tickers
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_conservative_pipeline \
  --conf '{"tickers": "AAPL,MSFT,NVDA"}'

# Todos los tickers (campo vacío)
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_conservative_pipeline \
  --conf '{"tickers": ""}'
```

## Ejemplos de Uso

### Debugging rápido (1 ticker)
```json
{"tickers": "AAPL"}
```
⏱ ~2-3 min | Útil para probar cambios

### Tech stocks comparison
```json
{"tickers": "AAPL,MSFT,GOOGL,META,NVDA"}
```
⏱ ~10-15 min | Compara ICs entre empresas

### Portfolio completo
```json
{"tickers": ""}
```
⏱ ~1-2 horas | Todos los tickers del config

## 🔍 Ver Resultados en MLflow

1. **MLflow UI**: http://localhost:5000
2. **Experimento**: "E1_Conservative_Strategy"
3. **Filtrar**: `params.ticker = "AAPL"`
4. **Comparar**: Seleccionar runs → "Compare"

## Notas

- SPY (benchmark) siempre se descarga automáticamente
- Solo entrena tickers válidos de E1 en `base.yaml`
- Cada ticker = 1 run en MLflow
- Run de resumen con métricas agregadas (IC mean, median, etc.)

---

**Ver más**: [README_DOCKER.md](README_DOCKER.md)
