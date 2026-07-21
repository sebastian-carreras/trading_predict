# Prediction API — FastAPI

An **HTTP service** that exposes the latest prediction from the *champion* model per
strategy/ticker and turns it into an actionable **BUY / SELL / HOLD** signal. Built as a
demonstration surface — Swagger at `/docs`, no frontend.

> **How the prediction is produced:** the API does **not** run live inference. It reads the last
> row of the champion's already-computed walk-forward output
> (`<TICKER>_walkforward_predictions.csv`) and converts it into a signal using the thresholds from
> the registry. That's robust, instant, and uniform across E1/E2/E3 — and it means the image needs
> neither `torch` nor `mlflow`.

Code: [`app.py`](app.py) · Source of truth: [`models/registry.json`](../../models/registry.json)
(via `ModelRegistry`).

---

## Quick start

### Option A — local, no Docker (fastest)

```bash
conda activate ia_ceia_18co
# from the repo root:
uvicorn dockerfiles.fastapi.app:app --port 8800

# then, in another terminal or the browser:
open http://localhost:8800/docs          # interactive Swagger
curl http://localhost:8800/predict/e1/AAPL
```

### Option B — Docker

```bash
docker compose --profile all up -d fastapi   # start just the API
open http://localhost:8800/docs
```

> The `fastapi` service declares no `depends_on`: it starts independently and works even if MLflow
> and Airflow are down.

### Option C — Python / notebook

```python
from fastapi.testclient import TestClient
from dockerfiles.fastapi.app import app   # from the repo root

client = TestClient(app)
print(client.get("/predict/e1/AAPL").json())
```

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Basic health check |
| `GET` | `/health` | Detailed health (registry path, champion count) |
| `GET` | `/models/status` | Registered champions, grouped by strategy |
| `GET` | `/predict/{strategy}/{ticker}` | Champion prediction + signal (`strategy` ∈ `e1｜e2｜e3`) |

Interactive docs: http://localhost:8800/docs (Swagger) · http://localhost:8800/redoc

---

## Example response

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
  "note": "predicted_return is the champion's latest walk-forward prediction ..."
}
```

> `pred_std` appears only for **E3** (ensemble dispersion, a confidence indicator).
> `horizon_bars` appears for **E3** (intraday); `horizon_days` for **E1/E2**.

---

## Signal logic

`predicted_return` is the champion's expected return at the strategy's horizon. It becomes a
signal via the thresholds:

- `predicted_return >= tau_buy` → **BUY**
- `predicted_return <= -tau_sell` → **SELL**
- otherwise → **HOLD**

Thresholds resolve in a chain: first the registry (per-ticker Optuna values — `tau_*` for E3,
`hp_tau_*` for E1/E2), then, if the champion doesn't carry them, the defaults in
[`src/config/base.yaml`](../../src/config/base.yaml) under `strategies.<variant>.thresholds`.

> **Honesty note:** `prediction_date` is the date of the champion's last *test* observation, not
> today. The API exposes it explicitly so the response is never mistaken for a real-time signal.

---

## Available tickers

Live list: `GET /models/status`.

| Strategy | Model | Horizon | Champion tickers |
|---|---|---|---|
| **E1** | GRU | 90 days | YPFD.BA, GGAL.BA, PAMP.BA, BYMA.BA, CEPU.BA, AAPL, MSFT, JNJ, PG, V |
| **E2** | LSTM | 20 days | AAPL, NVDA, GOOGL, AMZN, META, NFLX, BBAR.BA, BMA.BA, EDN.BA, LOMA.BA, TGSU2.BA |
| **E3** | LSTM ensemble | 30 min (intraday) | AAPL, SPY, NVDA, QQQ |

Error cases: a ticker with no champion returns `404`; an invalid strategy (not e1/e2/e3) returns
`400`.

---

## Dependencies

`fastapi`, `uvicorn`, `pandas`, `pyyaml` — see [`requirements.txt`](requirements.txt). The Docker
image stays light: it deliberately does **not** install `torch` or `mlflow`, because the API only
reads the registry and the CSVs under `runs/`.

Required volumes, mounted in [`docker-compose.yaml`](../../docker-compose.yaml): `./models` (the
registry) and `./runs` (saved predictions).
