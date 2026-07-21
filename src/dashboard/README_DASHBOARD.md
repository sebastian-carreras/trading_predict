# Monitoring dashboard — MLflow UI

A supervision layer over training/prediction timings, ML metrics and trading metrics, with
per-strategy alert thresholds.

---

## Quick start

### Option A — local, no Docker

```bash
# 1. Train — MLflow writes to the local SQLite store automatically
python -m src.e1.train_pipeline --tickers AAPL

# 2. Start the MLflow UI against that local DB
mlflow ui --backend-store-uri sqlite:///runs/mlflow_local/mlflow.db --port 5050
open http://localhost:5050

# 3. Generate the alert report (summary view)
python -m src.dashboard.checker

# 4. Per-ticker view for one strategy
python -m src.dashboard.checker --view ticker --strategy e1_conservative

# 5. History for a specific ticker
python -m src.dashboard.checker --view history --strategy e1_conservative --ticker AAPL
```

> You don't need an MLflow server running in order to train. Pipelines write metrics to
> `runs/mlflow_local/mlflow.db` automatically, and starting the UI later shows every prior run.

### Option B — Docker (Postgres + MinIO)

```bash
docker compose --profile dashboard up -d
open http://localhost:5050   # MLflow UI
open http://localhost:9001   # MinIO UI — credentials from .env (MINIO_ACCESS_KEY / MINIO_SECRET_ACCESS_KEY)
python -m src.dashboard.checker
```

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Pipelines  │────▶│    MLflow    │────▶│  MLflow UI   │
│ E1/E2/E3/E4  │     │    Server    │     │  Dashboard   │
└──────┬───────┘     └──────┬───────┘     └──────────────┘
       │                    │
       │  timing JSONL      │  backend store
       ▼                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   reports/   │     │  PostgreSQL  │     │    MinIO     │
│   timing/    │     │  (metadata)  │     │ (artifacts)  │
└──────────────┘     └──────────────┘     └──────────────┘
       │
       ▼
┌──────────────┐
│   Checker    │──── reports/dashboard/
│   CLI        │     ├── dashboard_report_*.csv
│              │     ├── ticker_report_*.csv
│              │     └── history_report_*.csv
└──────────────┘
```

1. **Pipelines** train models and log metrics plus timings to MLflow.
2. **MLflow Server** stores metadata in Postgres and artifacts in MinIO.
3. **MLflow UI** shows experiments per strategy, with filters and charts.
4. **Checker CLI** joins MLflow data with the JSONL timing log and produces reports with alerts.

---

## Metrics

| Category | Metric | Description |
|---|---|---|
| Timing | `train_seconds` | Training time per ticker |
| Timing | `predict_seconds` | Prediction time per ticker |
| ML | `ml_mae` | Mean absolute error |
| ML | `ml_rmse` | Root mean squared error |
| ML | `ml_ic` | Information Coefficient (Spearman) |
| ML | `ml_directional_accuracy` | Share of correct direction calls |
| Trading | `bt_sharpe` | Sharpe ratio |
| Trading | `bt_sortino` | Sortino ratio |
| Trading | `bt_cagr` | Compound annual growth rate |
| Trading | `bt_max_drawdown` | Largest peak-to-trough loss |
| Trading | `bt_calmar` | Calmar ratio (CAGR / max drawdown) |
| Trading | `bt_profit_factor` | Gross profit / gross loss |

For each metric the checker reports the **average** across runs, the **latest** value, **p50/p95**
percentiles (timing only), and an alert level: 🟢 / 🟡 / 🔴.

---

## Alert thresholds

Calibrated against the real metric distribution (February 2026). Configured in
[`../config/dashboard_thresholds.yaml`](../config/dashboard_thresholds.yaml).

### E1 Simple / E1 Conservative

| Metric | 🟢 Green | 🟡 Yellow | 🔴 Red |
|---|---|---|---|
| MAE | ≤ 0.15 | 0.15–0.35 | > 0.35 |
| RMSE | ≤ 0.20 | 0.20–0.45 | > 0.45 |
| IC | ≥ 0.10 | −0.05–0.10 | < −0.05 |
| Directional accuracy | ≥ 55% | 50–55% | < 50% |
| Sharpe | ≥ 1.0 | 0.3–1.0 | < 0.3 |
| Sortino | ≥ 1.2 | 0.3–1.2 | < 0.3 |
| CAGR (Simple) | ≥ 8% | 0–8% | < 0% |
| CAGR (Conservative) | ≥ 0% | −5%–0% | < −5% |
| Max drawdown | ≤ 30% | 30–55% | > 55% |
| Calmar | ≥ 0.5 | 0.1–0.5 | < 0.1 |
| Profit factor | ≥ 1.3 | 1.0–1.3 | < 1.0 |
| Train time (Simple) | ≤ 2 min | 2–5 min | > 5 min |
| Train time (Conservative) | ≤ 10 min | 10–20 min | > 20 min |

### E2 Simple / E2 Moderate

| Metric | 🟢 Green | 🟡 Yellow | 🔴 Red |
|---|---|---|---|
| MAE | ≤ 0.02 | 0.02–0.04 | > 0.04 |
| RMSE | ≤ 0.03 | 0.03–0.06 | > 0.06 |
| IC | ≥ 0.05 | 0.02–0.05 | < 0.02 |
| Directional accuracy | ≥ 55% | 50–55% | < 50% |
| Sharpe | ≥ 0.8 | 0.4–0.8 | < 0.4 |
| Max drawdown | ≤ 15% | 15–20% | > 20% |

### E3 Intraday

| Metric | 🟢 Green | 🟡 Yellow | 🔴 Red |
|---|---|---|---|
| MAE | ≤ 0.001 | 0.001–0.003 | > 0.003 |
| RMSE | ≤ 0.002 | 0.002–0.005 | > 0.005 |
| IC | ≥ 0.03 | 0.01–0.03 | < 0.01 |
| Directional accuracy | ≥ 52% | 48–52% | < 48% |
| Profit factor | ≥ 1.4 | 1.0–1.4 | < 1.0 |
| Max drawdown (intraday) | ≤ 3% | 3–5% | > 5% |

### E4 Pairs

| Metric | 🟢 Green | 🟡 Yellow | 🔴 Red |
|---|---|---|---|
| Sharpe | ≥ 1.0 | 0.5–1.0 | < 0.5 |
| Max drawdown | ≤ 10% | 10–20% | > 20% |
| Win rate | ≥ 55% | 45–55% | < 45% |
| Total return | ≥ 5% | 0–5% | < 0% |

---

## Checker CLI

Three views: **summary**, **ticker** and **history**.

### Summary view (default)

Average and latest value per strategy, with alerts.

```bash
python -m src.dashboard.checker                            # all strategies
python -m src.dashboard.checker --strategy e1_conservative # one strategy
```

```
======================================================================
  E1 Conservative
  GRU 2 layers, walk-forward 5 folds, horizon 90d
======================================================================

  Timing
  ────────────────────────────────────────────────────────────────
  Metric                            Avg     Last  Alert
  ────────────────────────────────────────────────────────────────
  train_seconds                    45.2     42.1  🟢
  predict_seconds                   0.3      0.2  🟢

  ML
  ────────────────────────────────────────────────────────────────
  ml_mae                         0.0280   0.0250  🟢
  ml_ic                          0.0650   0.0720  🟢
  ml_directional_accuracy        0.5600   0.5800  🟢

  Trading
  ────────────────────────────────────────────────────────────────
  bt_sharpe                      0.8500   1.0200  🟢
  bt_cagr                        0.0900   0.1100  🟢
  bt_max_drawdown                0.1200   0.1400  🟢
```

### Ticker view

Metrics from the latest training run (or the average of the last N) for every ticker in a
strategy, including the **composite score** and separate ML/Trading alerts.

```bash
python -m src.dashboard.checker --view ticker --strategy e1_conservative
python -m src.dashboard.checker --view ticker --strategy e1_conservative --last 3
```

```
================================================================================
  E1 Conservative — per ticker (latest run)
  Score = 0.35·bt_sharpe + 0.25·ml_ic + 0.20·ml_directional_accuracy + 0.20·bt_calmar
================================================================================

  ML
  ──────────────────────────────────────────────────────────────────────────
  Ticker              MAE       RMSE         IC    Dir Acc
  ──────────────────────────────────────────────────────────────────────────
  AAPL             0.0820     0.1050     0.2100     0.5800  🟢
  GGAL.BA          0.1240     0.1680    -0.0300     0.4900  🟡

  Trading
  ──────────────────────────────────────────────────────────────────────────
  Ticker           Sharpe    Sortino       CAGR     Max DD   Score
  ──────────────────────────────────────────────────────────────────────────
  AAPL             1.2300     1.8100     0.1500     0.1200  0.5640  🟢
  GGAL.BA          0.4500     0.5200     0.0200     0.3800  0.1820  🟡

  ML summary:      1🟢 1🟡 0🔴 of 2 tickers
  Trading summary: 1🟢 1🟡 0🔴 of 2 tickers
```

The score uses `compute_score()` from `src/lifecycle/promotion.py` with the weights configured in
`src/config/base.yaml` under `lifecycle.promotion.scoring_weights` — deliberately the **same**
score that drives champion/challenger promotion, so the dashboard and the promotion logic can
never disagree about which model is better.

### History view

The last N runs for one ticker, with the score trend.

```bash
python -m src.dashboard.checker --view history --strategy e1_conservative --ticker AAPL
python -m src.dashboard.checker --view history --strategy e1_conservative --ticker AAPL --last 5
```

```
================================================================================
  E1 Conservative — AAPL (last 5 runs)
================================================================================
  Date             MAE     RMSE       IC  Dir Acc  Sharpe  Sortino     CAGR   Max DD    Score  ML  Trad
  ────────────────────────────────────────────────────────────────────────────────
  2026-02-25    0.0820   0.1050   0.2100   0.5800  1.2300   1.8100   0.1500   0.1200   0.5640  🟢 🟢
  2026-02-20    0.0900   0.1150   0.1800   0.5600  1.1000   1.6500   0.1200   0.1400   0.5120  🟢 🟢
  2026-02-15    0.1100   0.1400   0.0800   0.5300  0.8500   1.2000   0.0600   0.2200   0.3680  🟢 🟡
  2026-02-10    0.1250   0.1600   0.0500   0.5100  0.6200   0.8800   0.0300   0.2800   0.2560  🟡 🟡
  2026-02-05    0.1400   0.1800   0.0200   0.5000  0.4000   0.5500   0.0100   0.3500   0.1520  🟡 🟡

  Score trend: 0.1520 → 0.5640 (+271.1%) ↑
```

### CLI arguments

| Argument | Description | Default |
|---|---|---|
| `--view` | `summary`, `ticker` or `history` | `summary` |
| `--strategy` | Strategy key (e.g. `e1_conservative`). Required for ticker/history. | all |
| `--ticker` | Ticker (e.g. `AAPL`). Required for history. | — |
| `--last` | ticker: average the last N. history: number of runs. | ticker 1, history 10 |
| `--save` | Write CSV | `True` |
| `--quiet` | Suppress console output | `False` |

Outputs: a formatted table on stdout, plus CSVs under `reports/dashboard/` —
`dashboard_report_latest.csv`, `ticker_report_{strategy}_latest.csv`,
`history_report_{strategy}_{ticker}_latest.csv`, and timestamped historical copies.

---

## Using the MLflow UI

Each strategy has its own experiment:

| Strategy | Experiment name |
|---|---|
| E1 Simple | `E1_Simple` |
| E1 Conservative | `E1_Conservative_Strategy` |
| E2 Simple | `E2_Simple` |
| E2 Moderate | `E2_Moderate` |
| E3 Intraday | `E3_Intraday` |
| E4 Pairs | `E4_Pairs` |

### Filtering by tags

Every run carries dashboard tags:

- `dashboard.strategy` — strategy key
- `dashboard.ticker` — asset ticker
- `dashboard.run_type` — `train`, `predict`, `evaluate`, `aggregate`
- `dashboard.alert.overall` — 🟢/🟡/🔴
- `dashboard.alert.{metric}` — per-metric alert

```
tags.dashboard.strategy = "e1_simple" AND tags.dashboard.alert.overall = "🔴"
```

Note the tag is `dashboard.strategy`, with a dot — not `dashboard_strategy`.

### Useful charts

- **Scatter** `ml_ic` vs `bt_sharpe` — does predictive power translate into trading performance?
- **Line** `ml_mae` per run — quality over time
- **Bar** `dashboard.train_seconds` per ticker — find the bottlenecks

---

## Integrating into a pipeline

Use the context manager:

```python
from src.dashboard.integration import with_dashboard_logging

with mlflow.start_run(run_name=f"E2Simple_{ticker}_{ts}"):
    with with_dashboard_logging("e2_simple", ticker=ticker) as ctx:
        # ... train and evaluate ...
        ctx["train_seconds"] = train_duration
        ctx["predict_seconds"] = pred_duration
        ctx["metrics"] = {"ml_mae": mae, "bt_sharpe": sharpe}
```

It automatically logs strategy and ticker tags, training/prediction timings (falling back to
wall-clock if not set), and a threshold alert for every metric.

Additional metrics logged: `dashboard.train_seconds`, `dashboard.predict_seconds`, plus the same
values under the `timing_*` namespace.

---

## Infrastructure

### Local mode (development)

```bash
mlflow ui --backend-store-uri sqlite:///runs/mlflow_local/mlflow.db --port 5050
```

Backend store: `runs/mlflow_local/mlflow.db` (SQLite). Artifact store:
`runs/mlflow_local/artifacts/`. Suited to individual development and experimentation.

### Docker mode (local production)

```bash
docker compose --profile dashboard up -d
```

Backend store: PostgreSQL (`postgres:5432/mlflow_db`). Artifact store: MinIO (`s3://mlflow/`).
Suited to local production and multiple users.

### Scaling out

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Pipelines  │────▶│  MLflow Server   │────▶│  MLflow UI   │
│  (local/CI) │     │  (EC2/GKE/ECS)   │     │  (web)       │
└─────────────┘     └────────┬─────────┘     └──────────────┘
                             │
                    ┌────────┴─────────┐
              ┌─────▼──────┐    ┌──────▼─────┐
              │ PostgreSQL │    │     S3     │
              │ (RDS/Cloud │    │ (AWS/GCS/  │
              │  SQL)      │    │  MinIO)    │
              └────────────┘    └────────────┘
```

1. **External Postgres** — `MLFLOW_BACKEND_STORE_URI=postgresql://user:pass@host:5432/mlflow_db`
2. **Real S3 or remote MinIO** — `MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://my-mlflow-bucket/` plus AWS credentials
3. **Authentication** — MLflow behind an nginx reverse proxy with basic auth, or MLflow's own OAuth (2.10+)
4. **Kubernetes** — the MLflow Helm chart or a custom deployment; Postgres → CloudSQL/RDS, MinIO → S3/GCS

---

## File layout

```
src/
├── config/
│   ├── base.yaml                      # main config (includes scoring_weights)
│   └── dashboard_thresholds.yaml      # per-strategy alert thresholds
├── dashboard/
│   ├── __init__.py                    # helpers: tags, timing, alerts
│   ├── checker.py                     # CLI: health report, 3 views
│   └── integration.py                 # context manager for pipelines
└── lifecycle/
    └── promotion.py                   # compute_score() — used by the ticker view

runs/mlflow_local/
├── mlflow.db                          # SQLite backend store (single source of truth)
└── artifacts/                         # run artifacts

reports/
├── timing/timing_log.jsonl            # timing log
└── dashboard/                         # generated reports
```

---

## FAQ

**Why the MLflow UI instead of Streamlit or Dash?** Because the pipelines already log to MLflow,
run comparison is built in, tag and metric filtering is powerful, artifacts (models, predictions,
backtests) are reachable from the same UI, and the same code scales from local to a remote server.

**How do I see only runs with red alerts?** Filter on
`tags.dashboard.alert.overall = "🔴"`.

**What is the composite score?** A weighted average of trading metrics, configured in
`src/config/base.yaml` under `lifecycle.promotion.scoring_weights`. It drives both this dashboard
and champion/challenger promotion.

**Are thresholds editable?** Yes — edit
[`../config/dashboard_thresholds.yaml`](../config/dashboard_thresholds.yaml) and re-run the checker.

**Where does MLflow store data?** In `runs/mlflow_local/mlflow.db` (SQLite), the single source of
truth for metrics. Pipelines and the checker use the same fallback cascade: remote server
(`MLFLOW_TRACKING_URI`) if configured and reachable → local SQLite → default file store
(`mlruns/`) as a last resort.

> The `mlruns/` folder is no longer used. It was removed because it contained corrupted
> experiments; all tracking now goes through local SQLite or a remote server.

---

## See also

- [../lifecycle/README_LIFECYCLE.md](../lifecycle/README_LIFECYCLE.md) — model lifecycle and promotion
- [../../README_DOCKER.md](../../README_DOCKER.md) — Docker infrastructure
- [../../scripts/README.md](../../scripts/README.md) — evaluation and leaderboard scripts
