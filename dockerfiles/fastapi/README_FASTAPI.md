# API de Predicciones — FastAPI

**Servicio HTTP** que expone la última predicción del modelo *champion* por estrategia/ticker y la convierte en una señal accionable **BUY / SELL / HOLD**. Pensado como superficie para la **demostración** (Swagger en `/docs`, sin frontend).

> **Cómo produce la predicción (Approach B):** la API **no hace inferencia en vivo**. Lee la última fila del walk-forward del champion (`<TICKER>_walkforward_predictions.csv`) ya calculado y la convierte en señal con los umbrales del registry. Es robusto, instantáneo y uniforme para E1/E2/E3. No usa `torch` ni `mlflow`.

Código: [dockerfiles/fastapi/app.py](dockerfiles/fastapi/app.py) · Fuente de verdad: [models/registry.json](models/registry.json) (vía `ModelRegistry`).

---

## Quick Start

### Opción A — Local (sin Docker, la más rápida)

```bash
conda activate ia_ceia_18co
# desde la raíz del repo:
uvicorn dockerfiles.fastapi.app:app --port 8800

# probar (otra terminal o navegador):
open http://localhost:8800/docs          # Swagger interactivo
curl http://localhost:8800/predict/e1/AAPL
```

### Opción B — Docker (stack del demo)

```bash
docker compose --profile all up -d fastapi   # arranca solo la API
open http://localhost:8800/docs
```

> El servicio `fastapi` no declara `depends_on`: arranca independiente y funciona aunque MLflow/Airflow estén caídos.

### Opción C — Python / notebook (test programático)

```python
from fastapi.testclient import TestClient
from dockerfiles.fastapi.app import app   # desde la raíz del repo

client = TestClient(app)
print(client.get("/predict/e1/AAPL").json())
```

---

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Health check básico |
| `GET` | `/health` | Health detallado (ruta del registry, nº de champions) |
| `GET` | `/models/status` | Champions registrados, agrupados por estrategia |
| `GET` | `/predict/{strategy}/{ticker}` | Predicción + señal del champion (`strategy` ∈ `e1\|e2\|e3`) |

**Documentación interactiva:** http://localhost:8800/docs (Swagger) · http://localhost:8800/redoc

---

## Ejemplo de respuesta

```bash
curl http://localhost:8800/predict/e1/AAPL
```

```json
{
  "ticker": "AAPL",
  "strategy": "e1",
  "variant": "e1_conservative",
  "signal": "BUY",
  "predicted_return": 0.0508,
  "prediction_date": "2025-10-06 00:00:00+00:00",
  "thresholds": { "tau_buy": 0.02, "tau_sell": 0.0 },
  "champion_metrics": {
    "bt_sharpe": 1.0181,
    "bt_total_return": 4.3668,
    "ml_directional_accuracy": 0.6996,
    "ml_ic": 0.1710,
    "horizon_days": 90.0,
    "horizon_bars": null
  },
  "note": "predicted_return es la última predicción walk-forward del champion ..."
}
```

> `pred_std` aparece solo en **E3** (desviación del ensemble, indicador de confianza).
> `horizon_bars` aparece en **E3** (intradía); `horizon_days` en **E1/E2**.

---

## Lógica de la señal

`predicted_return` es el retorno esperado del champion al horizonte de la estrategia. Se convierte en señal con los umbrales:

- `predicted_return >= tau_buy`  → **BUY**
- `predicted_return <= -tau_sell` → **SELL**
- en otro caso → **HOLD**

Los umbrales se resuelven en cadena: primero el registry (Optuna por ticker — `tau_*` en E3, `hp_tau_*` en E1/E2) y, si el champion no los tiene, los defaults de [src/config/base.yaml](src/config/base.yaml) (`strategies.<variant>.thresholds`).

> **Nota para la defensa:** `prediction_date` es la fecha del último dato de *test* del champion (no "hoy"). La API lo expone explícitamente para no presentarlo como tiempo real.

---

## Tickers disponibles

Listado completo en vivo: `GET /models/status`.

| Estrategia | Modelo | Horizonte | Tickers champion |
|-----------|--------|-----------|------------------|
| **E1** | GRU | 90 días | YPFD.BA, GGAL.BA, PAMP.BA, BYMA.BA, CEPU.BA, AAPL, MSFT, JNJ, PG, V |
| **E2** | LSTM | 20 días | AAPL, NVDA, GOOGL, AMZN, META, NFLX, BBAR.BA, BMA.BA, EDN.BA, LOMA.BA, TGSU2.BA |
| **E3** | LSTM ensemble | 30 min (intradía) | AAPL, SPY, NVDA, QQQ |

**Casos de error:** ticker sin champion → `404` · estrategia inválida (≠ e1/e2/e3) → `400`.

---

## Dependencias

`fastapi`, `uvicorn`, `pandas`, `pyyaml` ([requirements.txt](dockerfiles/fastapi/requirements.txt)). La imagen Docker es liviana: **no** instala `torch`/`mlflow` porque la API solo lee el registry y los CSV de `runs/`.

**Volúmenes que necesita** (montados en [docker-compose.yaml](docker-compose.yaml)): `./models` (registry) y `./runs` (predicciones guardadas).
