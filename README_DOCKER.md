# Docker stack: Airflow + MLflow + FastAPI + PostgreSQL

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Trading Predict Stack                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌──────────┐    ┌─────────────┐         │
│  │   Airflow   │───▶│  MLflow  │───▶│   FastAPI   │         │
│  │  Scheduler  │    │ Tracking │    │     API     │         │
│  │  Webserver  │    │  Server  │    │   (8800)    │         │
│  │   (8080)    │    │  (5050)  │    └─────────────┘         │
│  └─────────────┘    └──────────┘           │                │
│        │                  │                │                │
│        │           ┌──────┴──────┐         │                │
│        │           │             │         │                │
│        ▼           ▼             ▼         ▼                │
│  ┌──────────────────────────────────────────────┐           │
│  │          PostgreSQL Database                 │           │
│  │  - Airflow metadata                          │           │
│  │  - MLflow experiments & runs                 │           │
│  │            (5432)                            │           │
│  └──────────────────────────────────────────────┘           │
│        │                                                    │
│        ▼                                                    │
│  ┌──────────────────────────────────────────────┐           │
│  │          MinIO (S3-compatible)               │           │
│  │  - MLflow artifacts (models, metrics)        │           │
│  │  - Datasets                                  │           │
│  │            (9000, UI: 9001)                  │           │
│  └──────────────────────────────────────────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Quick start

### 1. Configure environment variables

```bash
cp .env.example .env
```

All ports are configurable there (`AIRFLOW_PORT`, `MLFLOW_PORT`, `FASTAPI_PORT`). The values
below are the ones this project uses; `docker-compose.yaml` falls back to the container-internal
defaults if a variable is left empty.

### 2. Start the services

```bash
docker-compose --profile mlflow up -d    # MLflow + PostgreSQL only
docker-compose --profile airflow up -d   # Airflow only
docker-compose --profile all up -d       # full stack (recommended)
```

### 2b. Transparent MLflow mode (offline → online)

To make training runs executed *without* Docker show up in the MLflow UI afterwards:

```bash
bash scripts/mlflow/up_transparent_mlflow.sh
```

The script aligns the container's MLflow version with your local environment, rebuilds the
`mlflow` service, and points it at `runs/mlflow_local` — the same store the offline runs write to.

### 3. Interfaces

| Service | URL | Credentials |
|---|---|---|
| Airflow UI | http://localhost:8080 | `_AIRFLOW_WWW_USER_USERNAME` / `_AIRFLOW_WWW_USER_PASSWORD` |
| MLflow UI | http://localhost:5050 | — |
| FastAPI docs | http://localhost:8800/docs | — |
| MinIO UI | http://localhost:9001 | `MINIO_ACCESS_KEY` / `MINIO_SECRET_ACCESS_KEY` |

> Credentials come from your `.env` (see `.env.example` for the variable names and their local
> development defaults). They are not reproduced here on purpose — see [Security](#security).

## Scheduling — how the DAGs actually run

The three strategies share a single `models/registry.json`, which `ModelRegistry` loads whole and
rewrites whole, with no locking or merging. Running them concurrently silently clobbers writes —
this happened for real on 2026-07-17. The schedule is built to make concurrency impossible:

| DAG | Schedule | Why |
|---|---|---|
| `e2_moderate_pipeline` | `0 8 * * *` (08:00 UTC / 05:00 ART) | The **cron anchor** — the only time-triggered retraining DAG |
| `e1_conservative_pipeline` | Dataset-triggered on `trading://registry/e2_moderate` | Starts only once E2 has fully finished, however long it takes |
| `e3_intraday_pipeline` | `None` — **manual only** | E3 is a documented negative result (median Sharpe −2.6, doesn't beat intraday costs). Not worth daily compute, and not worth the registry collision risk |
| `reporting/daily_report` | Dataset-triggered on both E1 and E2 | Data-aware, not clock-scheduled |

Trigger any of them by hand:

```bash
docker exec -it airflow_webserver airflow dags trigger e1_conservative_pipeline
docker exec -it airflow_webserver airflow dags trigger e3_intraday_pipeline
```

### What the E1 pipeline does

1. Download daily data (Yahoo Finance or IOL)
2. Compute technical features
3. Train a GRU model per ticker
4. Log experiments to MLflow
5. Store artifacts in MinIO
6. Evaluate and promote against the current champion

Outputs land in `runs/e1_conservative/<timestamp>/`, the MLflow experiment
`E1_Conservative_Strategy`, and the `s3://mlflow/` MinIO bucket.

### What the E3 pipeline does

1. Download 5-minute bars (last 60 days)
2. Train an ensemble of 3 LSTMs
3. Backtest with 20 bps intraday costs
4. Log to MLflow

Outputs land in `runs/e3_intraday/<timestamp>/` and the `E3_Intraday_Strategy` experiment.

## Local development

Run pipelines without Docker, for debugging:

```bash
conda activate ia_ceia_18co

python -m src.e1.train_pipeline --tickers AAPL
python -m src.e3.train_pipeline --tickers SPY

# with MLflow tracking against the containerized server
export MLFLOW_TRACKING_URI=http://localhost:5050
python -m src.e1.train_pipeline
```

### Airflow logs

```bash
docker logs -f airflow_scheduler

# test a single task of a DAG
docker exec -it airflow_webserver airflow tasks test \
  e1_conservative_pipeline download_daily_data 2026-01-06
```

## API endpoints (FastAPI)

```bash
# Health check
curl http://localhost:8800/health

# Register models (called from Airflow)
curl -X POST http://localhost:8800/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "e1_conservative",
    "run_dir": "/opt/airflow/runs/e1_conservative/20260106_020530",
    "timestamp": "2026-01-06T02:05:30"
  }'

# E1 prediction
curl -X POST http://localhost:8800/predict/e1/AAPL \
  -H "Content-Type: application/json" \
  -d '{"use_latest_data": true}'

# Model status
curl http://localhost:8800/models/status
```

## PostgreSQL

```bash
docker exec -it postgres psql -U airflow

\l                                            # list databases
\c mlflow_db                                  # connect to the MLflow DB
\dt                                           # list tables
SELECT experiment_id, name FROM experiments;  # list experiments
```

## MinIO (S3)

```bash
docker exec -it minio mc ls s3/mlflow                          # list buckets
docker exec -it minio mc ls s3/mlflow/0/<run_id>/artifacts/    # artifacts of a run
```

## Stopping

```bash
docker-compose --profile all down       # stop
docker-compose --profile all down -v    # stop and delete volumes (destroys data)
```

## Monitoring

```bash
docker stats                            # resource usage
docker-compose --profile all logs -f    # combined logs
docker-compose logs -f mlflow           # one service
```

## Security

The defaults above are for local development only. Before exposing this anywhere:

1. Change every password in `.env`
2. Use external secret management — never commit `.env`
3. Configure SSL/TLS for public endpoints
4. Restrict network access at the firewall
5. Add authentication to FastAPI (OAuth2 or API keys)

This repo runs `detect-secrets` as a pre-commit hook to keep credentials out of the history.

## Volume layout

```
docker volumes:
├── db_data/              # PostgreSQL data
├── minio_data/           # MinIO buckets
└── airflow/
    ├── dags/             # Airflow DAGs (one subfolder per strategy)
    │   ├── E1/
    │   ├── E2/
    │   ├── E3/
    │   └── E4/
    ├── logs/             # execution logs
    ├── plugins/          # custom plugins
    └── config/           # Airflow config
```

## Troubleshooting

**Airflow won't start** — reinitialize:

```bash
docker-compose --profile all down
docker volume rm trading_predict_db_data
docker-compose --profile all up airflow-init
docker-compose --profile all up -d
```

**MLflow can't connect to PostgreSQL** — make sure the database exists:

```bash
docker exec -it postgres psql -U airflow -c "CREATE DATABASE mlflow_db;"
```

**MinIO buckets aren't created** — create them by hand:

```bash
docker exec -it minio mc mb s3/mlflow
docker exec -it minio mc mb s3/data
```

**A DAG doesn't appear** — validate the DAG files parse:

```bash
python scripts/airflow/validate_dags.py
```

---

*Stack: Airflow 2.8.1 + MLflow + FastAPI + PostgreSQL 13 + MinIO. Trading Predict — AI
Specialization, FIUBA. Last updated 2026-07-21.*
