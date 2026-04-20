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

# Train E1 (Conservative, GRU, 90-day)
python -m src.e1.train_pipeline --tickers AAPL,MSFT
python -m src.e1.train_pipeline --tickers YPFD.BA --refresh-data --auto-promote

# Train E2 (Moderate, LSTM, 20-day)
python -m src.e2.train_pipeline --tickers AAPL

# Train baselines
python -m src.e1.train_baseline
python -m src.e2.train_baseline

# Run tests
pytest tests/

# Run a single test file
pytest tests/test_registry.py -v

# Docker (full stack: Airflow + MLflow + FastAPI)
cp .env.example .env
docker-compose --profile all up -d
# Airflow UI: http://localhost:8080 | MLflow UI: http://localhost:5050 | API: http://localhost:8800/docs
```

**No Makefile** — all commands are run as `python -m src.<module>`.

---

## Architecture Overview

### Strategy Map

| Strategy | Model | Horizon | Freq | Status |
|----------|-------|---------|------|--------|
| E1 Conservative | GRU [64,32] | 90-day | daily | Champion |
| E2 Moderate | LSTM [128,64] | 20-day | daily | Champion |
| E3 Intraday | LSTM | 30-min | 5-min | need to rework (TODO) |
| E4 Pairs Trading | k-NN + OU | — | daily | Disabled (TODO) |

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
