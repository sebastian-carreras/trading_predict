# Running DAGs for specific tickers

Every E1 DAG accepts a `tickers` parameter, so you can retrain a single symbol instead of the
whole universe. This guide covers the three E1 pipelines.

**DAG location:** [`dags/`](dags), with one subfolder per strategy — [E1](dags/E1),
[E2](dags/E2), [E3](dags/E3), [E4](dags/E4). Airflow scans subfolders automatically.

> **On scheduling:** only `e2_moderate_pipeline` runs on a cron (`0 8 * * *`).
> `e1_conservative_pipeline` is Dataset-triggered off E2's completion, and `e3_intraday_pipeline`
> is manual-only. The three share one `models/registry.json` and must never run concurrently —
> see [README_DOCKER.md](../../README_DOCKER.md). Everything below is manual triggering, which is
> always safe as long as another retraining DAG isn't already running.

---

## E1 Conservative pipeline

### From the Airflow UI

1. Go to http://localhost:8080
2. Click `e1_conservative_pipeline` → **Play** (▶) → **Trigger DAG w/ config**
3. In the JSON editor:
   ```json
   { "tickers": "AAPL,MSFT,GOOGL" }
   ```
4. Click **Trigger**

### From the CLI

```bash
# one ticker
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_conservative_pipeline --conf '{"tickers": "AAPL"}'

# several tickers
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_conservative_pipeline --conf '{"tickers": "AAPL,MSFT,NVDA"}'

# all tickers from the config (empty field)
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_conservative_pipeline --conf '{"tickers": ""}'
```

### Typical runs

| Config | Duration | Use |
|---|---|---|
| `{"tickers": "AAPL"}` | ~2–3 min | Quick debugging after a change |
| `{"tickers": "AAPL,MSFT,GOOGL,META,NVDA"}` | ~10–15 min | Compare ICs across companies |
| `{"tickers": ""}` | ~1–2 h | Full configured universe |

Only tickers valid for E1 in `base.yaml` are trained. Each ticker produces one MLflow run, plus a
summary run with aggregate metrics.

---

## E1 Simple pipeline

A simplified pipeline (single-layer GRU, no walk-forward) that downloads, cleans and trains in one
go. Useful for fast iteration — but note that without walk-forward its metrics are **not**
comparable to the conservative pipeline's.

### From the Airflow UI

Trigger `e1_simple_pipeline` with config:

```json
{
  "tickers": "AAPL,MSFT,GOOGL",
  "skip_download": "False",
  "skip_cleaning": "False"
}
```

### From the CLI

```bash
# one ticker, full pipeline
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_simple_pipeline \
  --conf '{"tickers": "AAPL", "skip_download": "False", "skip_cleaning": "False"}'

# reuse data already downloaded and cleaned by E1 Conservative
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_simple_pipeline \
  --conf '{"tickers": "AAPL,MSFT", "skip_download": "True", "skip_cleaning": "True"}'
```

### Parameters

| Parameter | Values | Meaning |
|---|---|---|
| `tickers` | comma-separated, or empty | Empty means all E1 Simple tickers from `base.yaml`, falling back to E1 Conservative's list if undefined |
| `skip_download` | `"True"` / `"False"` | `True` reuses the existing `data/raw/daily/`; `False` downloads with `skip_existing` and a minimum-freshness check |
| `skip_cleaning` | `"True"` / `"False"` | `True` reuses the existing `data/clean/`; `False` runs cleaning (forward fill, filters, 252-day minimum) |

Per-ticker artifacts go to `runs/e1_simple/<timestamp>/<ticker>/` (`models/`, `predictions/`,
`backtest/`), plus a `summary_all.csv` with aggregate metrics.

---

## E1 Baseline (Linear Regression)

The baseline is the sanity floor: if the GRU can't beat a linear model on the same features, the
extra complexity isn't earning its keep. It is **never promoted** to champion.

### From the CLI

```bash
# baseline for specific tickers, with automatic comparison
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_baseline_linear_regression \
  --conf '{"tickers": "AAPL,MSFT", "auto_compare_with_gru": "True"}'

# without automatic comparison
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_baseline_linear_regression \
  --conf '{"tickers": "AAPL", "auto_compare_with_gru": "False"}'

# all E1 tickers
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_baseline_linear_regression --conf '{"tickers": ""}'
```

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `tickers` | `"YPFD.BA, GGAL.BA, PAMP.BA, BYMA.BA, CEPU.BA, AAPL, MSFT, JNJ, PG, V"` | Comma-separated; empty means all E1 tickers |
| `auto_compare_with_gru` | `"True"` | `True` compares against the latest GRU run automatically |

### Recommended sequence

```bash
# 1. run the GRU first
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_conservative_pipeline --conf '{"tickers": "AAPL,MSFT"}'

# 2. wait for it to finish (~10–15 min)

# 3. run the baseline with comparison
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_baseline_linear_regression \
  --conf '{"tickers": "AAPL,MSFT", "auto_compare_with_gru": "True"}'
```

The comparison lands in `runs/e1_baseline/<timestamp>/comparison_vs_gru.csv`:

```bash
ls -lht runs/e1_baseline/ | head -n 2
cat runs/e1_baseline/<timestamp>/comparison_vs_gru.csv
```

The baseline uses the same features as the GRU, a plain scikit-learn linear regression, the same
metrics (MAE, RMSE, Spearman IC, directional accuracy) and the same backtest parameters
(`tau_buy`, `tau_sell`, costs) — so the comparison is apples to apples.

---

## Inspecting results in MLflow

Open http://localhost:5050 and pick the experiment:

| Pipeline | Experiment |
|---|---|
| E1 Conservative | `E1_Conservative_Strategy` |
| E1 Simple | `E1_Simple` |
| E1 Baseline | `E1_Baseline_LinearRegression` |

Filter with `params.ticker = "AAPL"`, then select runs and hit **Compare**.

---

## See also

- [docs/AIRFLOW_E1_BASELINE_DAG.md](../../docs/AIRFLOW_E1_BASELINE_DAG.md) — baseline DAG documentation
- [README_DOCKER.md](../../README_DOCKER.md) — Docker setup and the full scheduling picture
- [src/lifecycle/README_LIFECYCLE.md](../../src/lifecycle/README_LIFECYCLE.md) — what happens to a model after training
