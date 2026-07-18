# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Context

Academic final project for FIUBA's AI Specialization. Builds an AI system to support buy/sell decisions on financial assets using deep learning. **Not** a production trading system — no automatic order execution. Audience: thesis director and academic evaluators.

**Key constraints:**
- Prefer simple, proven solutions over cutting-edge complexity
- End-to-end functionality before marginal accuracy optimization
- Keep infrastructure lightweight (local + cloud, no over-engineering)

---

## Environment & Commands

```bash
# Activate Python environment
conda activate ia_ceia_18co

# Train E1 (Conservative, GRU, 90-day) — descarga datos nuevos incrementales por defecto
python -m src.e1.train_pipeline --tickers AAPL,MSFT
python -m src.e1.train_pipeline --tickers YPFD.BA --auto-promote
python -m src.e1.train_pipeline --tickers AAPL --skip-download   # usa datos existentes, sin descargar

# Train E2 (Moderate, LSTM, 20-day)
python -m src.e2.train_pipeline --tickers AAPL

# Train E3 (Intraday, LSTM, 30-min) — needs rework, ver PORTFOLIO.md (resultado negativo)
python -m src.e3.train_pipeline --tickers SPY
python -m src.e3.train_pipeline --download-only   # solo descarga datos intradiarios

# Train baselines
python -m src.e1.train_baseline
python -m src.e2.train_baseline

# Runner E1 completo (baseline + simple + conservative)
python -m src.e1.train_all
python -m src.e1.train_all --tickers AAPL,MSFT

# Run tests
pytest tests/

# Run a single test file
pytest tests/test_registry_atomicity.py -v

# What CI actually runs (.github/workflows/ci.yml, Python 3.10 & 3.12 matrix)
python -m pytest -v

# Docker (full stack: Airflow + MLflow + FastAPI)
cp .env.example .env
docker-compose --profile all up -d      # Airflow + MLflow + FastAPI + Postgres + MinIO
docker-compose --profile mlflow up -d   # MLflow + Postgres only
docker-compose --profile airflow up -d  # Airflow only
# Airflow UI: http://localhost:8080 | MLflow UI: http://localhost:5050 | API: http://localhost:8800/docs | MinIO: http://localhost:9001

bash scripts/mlflow/up_transparent_mlflow.sh   # sync local (non-Docker) MLflow runs into the Docker MLflow UI

# Pre-commit (secret scanning), one-time setup
pip install pre-commit detect-secrets
detect-secrets scan > .secrets.baseline
pre-commit install
```

**No Makefile** — all commands are run as `python -m src.<module>` (or `python -m scripts.<module>` / `python scripts/<path>.py`, see below).

---

## Architecture Overview

### Strategy Map

| Strategy | Model | Horizon | Freq | Status |
|----------|-------|---------|------|--------|
| E1 Conservative | GRU [64,32] | 90-day | daily | Champion |
| E2 Moderate | LSTM [128,64] | 20-day | daily | Champion |
| E3 Intraday | LSTM | 30-min | 5-min | need to rework (TODO) |
| E4 Pairs Trading | k-NN + OU | — | daily | Disabled (TODO) |

See `PORTFOLIO.md` for current out-of-sample walk-forward results per strategy (E1 median Sharpe 1.09, E2 0.75, E3 −3.2 — documented negative result, doesn't beat intraday costs).

### Training Pipeline Flow (E1/E2)

```
base.yaml config
    → Load OHLCV (data/clean/ or yfinance fallback)
    → Build technical features (13–16 features per strategy)
    → Create target (forward return at horizon days)
    → Walk-forward CV (5 folds, embargo >= horizon days)
        → Z-score scaling (stats from train only)
        → Train GRU/LSTM with early stopping
        → Predict + backtest + compute metrics
    → Save to runs/<variant>/<YYYYMMDD_HHMMSS>/<TICKER>/
    → Register as CANDIDATE in models/registry.json
    → [optional] Auto-promote if --auto-promote
    → MLflow tracking (remote with SQLite fallback)
```

### Model Lifecycle Registry

**Single source of truth**: `models/registry.json` (atomic writes via tempfile + os.replace).

Per `(strategy, ticker)`:
- `baseline` — fixed reference (Linear Regression for E1); never promoted
- `champion` — current best model in production
- `candidate` — latest trained model, awaiting promotion decision
- `retired[]` — historical models

**Key module**: `src/lifecycle/registry.py` → `ModelRegistry` class

**Promotion logic** (`src/lifecycle/promotion.py`):
- Phase 1 guardrails (permissive): reject NaN/Inf, Sharpe < 0, IC worse than baseline
- Composite score = 0.35×Sharpe + 0.25×IC + 0.20×DirectionalAcc + 0.20×Calmar
- Promote only if improvement ≥ 5% over champion
- Audit trail: `models/metrics_log.jsonl`

### Orchestration — Airflow (`dockerfiles/airflow/dags/`)

- Per-strategy retraining DAGs: `E1/e1_conservative_pipeline.py`, `e1_simple_pipeline.py`, `e1_baseline_linear_regression.py`, `e1_optuna_tuning.py`; `E2/e2_moderate_pipeline.py`, `e2_simple_pipeline.py`, `e2_optuna_tuning.py`; `E3/e3_intraday_pipeline.py`; `E4/e4_pairs_trading_pipeline.py`, `e4_monthly_recalibration.py`.
- E1/E2/E3 retrain **daily but chained sequentially, never in parallel** — they share a single `models/registry.json` written via `ModelRegistry` (loads the whole file into memory once, `_save()` overwrites it whole, no lock/merge — see `src/lifecycle/registry.py`), so concurrent runs can silently clobber each other's `train_data_end`/`recent_metrics` writes. Confirmed real overlap + lost writes on 2026-07-17 when E1/E2 both ran on the same `0 8 * * *` cron. Fix: **E2** is the cron anchor (`schedule_interval='0 8 * * *'`, 08:00 UTC/05:00 ART); **E1** is Dataset-triggered on `Dataset("trading://registry/e2_moderate")` (starts only once E2 fully finishes, regardless of duration). Each run calls `evaluate_and_promote(..., ohlcv_loader=..., full_config=...)`, so the fair-window champion/candidate comparison and `recent_metrics` refresh (see Leaderboard & Reporting below) happen daily for E1/E2.
- **E3 is manual-only** (`schedule_interval=None`) — status is "needs rework" (documented negative result, Sharpe −3.2, doesn't beat intraday costs — see PORTFOLIO.md), not worth auto-retraining daily. Trigger by hand: `airflow dags trigger e3_intraday_pipeline`.
- `reporting/daily_report.py` is **data-aware**, not clock-scheduled — it triggers only after both E1 and E2 finish retraining (`Dataset("trading://registry/e1_conservative")`, `.../e2_moderate`). Task chain: `refresh_dashboard` → `leaderboard` → `refresh_history_reports` → `stability_analysis`.
- Trigger a DAG manually: `docker exec -it airflow_webserver airflow dags trigger e1_conservative_pipeline`
- Validate DAGs: `python scripts/airflow/validate_dags.py`

### Leaderboard & Reporting

`scripts/evaluation/leaderboard.py` ranks all champions in `models/registry.json` using the **same composite score** as promotion (`src.lifecycle.promotion.compute_score`), diffs against `reports/dashboard/leaderboard_history.jsonl` to show score/rank deltas, flags champions "stale" (>10 days since training), and writes `reports/dashboard/leaderboard_latest.csv`.
```bash
python -m scripts.evaluation.leaderboard --strategies e1,e2 --top 10
```

### Key Files

| File | Purpose |
|------|---------|
| `src/config/base.yaml` | Centralized config for all strategies (universe, hyperparams, lifecycle rules) |
| `src/utils.py` | `project_root()`, `load_yaml()`, `log_timing_event()` |
| `src/lifecycle/registry.py` | ModelRegistry CRUD |
| `src/lifecycle/promotion.py` | Champion/challenger scoring |
| `src/lifecycle/guardrails.py` | Phase 1 validation rules |
| `models/registry.json` | Live model registry (do not edit manually) |
| `runs/<variant>/<ts>/<TICKER>/` | Immutable per-run outputs (model.pth, predictions.csv, metrics.json, plots/) |

### `scripts/` layout

Reorganized into subfolders (see `scripts/README.md` for the authoritative, current list — the root `README.md` still references pre-reorg script paths, don't trust those):
```
scripts/
├── data/          run_data_cleaning.py
├── optimization/  optimize_e1/e2/e3_hyperparameters.py, analyze_optuna_db.py
├── evaluation/    leaderboard.py, stability_analysis.py, compare_*.py, promote_candidate.py
├── trading/       e1_simple_iol_live_trade.py, iol_list_e1_prices.py
├── airflow/       validate_dags.py
└── mlflow/        up_transparent_mlflow.sh, cleanup_init_test_runs.py
```

> Several README_E*.md / QUICKSTART_*.md files linked from the root `README.md` no longer exist in the repo — that content was consolidated into `PORTFOLIO.md` and `docs/`. Prefer those over chasing dead links in `README.md`.

### Custom Claude Skills (`.claude/commands/`)

These skills use the model lifecycle system:
- `/model-status` — show champion/candidate/baseline for a strategy/ticker
- `/promote-model` — manually promote candidate to champion
- `/compare-models` — compare metrics across variants
- `/retire-model` — retire a model
- `/new-experiment` — launch a training run with custom hyperparameters

---

## Key Design Conventions

- **Walk-forward CV with embargo**: embargo period ≥ horizon days to prevent leakage
- **Feature scaling**: Z-score computed only on train fold (no data snooping)
- **Loss**: Huber loss (robust to price outliers), not MSE
- **Run directories**: immutable (`runs/<variant>/<YYYYMMDD_HHMMSS>/<TICKER>/`), never overwritten
- **Registry writes**: always atomic — use `ModelRegistry` methods, never write `registry.json` directly
- **Config access**: use `src/utils.py::load_yaml()` and `get_nested()` for safe nested access
- **Baseline**: separate track from champion/candidate; acts as a sanity floor, never promoted

---

## Strategies Currently Active

- **E1**: champion = `e1_conservative`, baseline = `e1_baseline`, retired = `e1_simple`
- **E2**: champion = `e2_moderate`, baseline = `e2_baseline`
- **E3**: champion = `e3_intraday`, baseline = `e3_baseline`
- **E4**: Not implemented, out of scope for the moment
