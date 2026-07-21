# trading_predict — Deep Learning for Buy/Sell Decision Support

![CI](https://github.com/sebastian-carreras/trading_predict/actions/workflows/ci.yml/badge.svg)

**▶️ Live demo:** https://huggingface.co/spaces/chatoxz/trading-predict-demo

An end-to-end **machine-learning system** that turns OHLCV market data into buy/sell/hold
signals across multiple strategies and horizons — with a full **MLOps lifecycle** around the
models: walk-forward validation, a champion/challenger model registry, automated promotion
guardrails, hyperparameter tuning, experiment tracking, a serving API, and CI.

> **Scope, stated honestly:** this is a decision-**support** research system, not an automated
> trading bot. There is no live order execution. All results below are **out-of-sample
> backtests, net of transaction costs**, from walk-forward cross-validation — not live P&L.
> It's the final project of the AI Specialization at FIUBA (Universidad de Buenos Aires).

---

## Why this project (what it demonstrates)

Most "stock prediction" repos are a single notebook that leaks future data and reports a
suspiciously good accuracy. This one is built to show the opposite skill set — **applied ML
engineering**:

- **Leakage-aware validation.** Walk-forward CV with an embargo ≥ horizon, scalers fit on the
  train fold only, strictly as-of features. No shuffle, no peeking.
- **A real model lifecycle.** A single-source-of-truth registry with baseline / candidate /
  champion / retired stages, atomic writes, automated promotion scoring, and hard guardrails.
- **Reproducibility.** Every run is an immutable, timestamped directory (config + predictions +
  metrics + model weights). Same config + same data → same result.
- **Honest evaluation.** Every strategy is compared against a baseline and reported **net of
  costs**, including the strategies that *didn't* work.

---

## Results at a glance

> **Metrics as of 2026-07-21.** These models retrain daily, so the numbers move. They are
> generated from `models/registry.json` — regenerate with `python demo/bundle_assets.py`, which
> writes `demo/assets/headline.json`. Never quote them from memory.

Backtest metrics are **out-of-sample (walk-forward) and net of transaction costs** (10 bps
round-trip daily, 20 bps intraday). Values shown are the **median across all per-ticker
champion models** in each strategy.

| Strategy | Model | Horizon | Champions | Median Sharpe | Median Dir. Acc | Verdict |
|---|---|---|:--:|:--:|:--:|---|
| **E1 — Conservative** | GRU, 2-layer (Optuna-tuned) | 90 d | 10 | **1.09** | **68%** | ✅ Strongest; beats baseline on every metric |
| **E2 — Moderate** | LSTM, 2-layer | 20 d | 31 | 0.55 | 55% | 🟡 Mixed — directional edge, but risk-adjusted return sits below baseline |
| **E3 — Intraday** | LSTM ensemble | 30 min | 4 | **−2.6** | 49% | 🔴 Negative result — does not beat costs; needs rework |
| **E4 — Pairs trading** | k-NN + Ornstein-Uhlenbeck | 10 d | — | — | — | ⚪ Specified, not implemented (roadmap) |

**45 champion models** in production across strategies, over a mixed US + Argentine (BYMA)
equity universe.

### Champion vs. baseline

Median across champions vs. the fixed baseline (Linear Regression):

| Metric | E1 champion | E1 baseline | E2 champion | E2 baseline |
|---|:--:|:--:|:--:|:--:|
| Sharpe | **1.09** | 0.87 | 0.55 | 0.60 |
| Directional accuracy | **68.2%** | 57.6% | **55.2%** | 50.8% |
| Information Coefficient | **0.124** | −0.016 | **0.030** | −0.017 |
| Calmar | **1.25** | 0.75 | 0.38 | 0.54 |

E1 clearly adds value over the baseline on every metric. E2, across a broad 31-ticker universe,
leads only on directional accuracy and rank-IC — on risk-adjusted return (Sharpe, Calmar) it sits
**below** the linear baseline. Reported as-is.

E1's best per-ticker champions: **BYMA.BA** (Sharpe 1.63), **PAMP.BA** (1.38), **GGAL.BA**
(1.30), **CEPU.BA** (1.24).

---

## What makes this an ML *engineering* project

### Model lifecycle & registry
- **Single source of truth:** `models/registry.json`, written atomically (tempfile +
  `os.replace`) so a crash can never corrupt it.
- **Stages per `(strategy, ticker)`:** `baseline` (fixed reference, never promoted) →
  `candidate` (latest trained) → `champion` (in production) → `retired[]` (history).
- **Automated promotion (`src/lifecycle/promotion.py`).** A composite score
  `0.35·Sharpe + 0.25·IC + 0.20·DirAcc + 0.20·Calmar`; a candidate only replaces the champion
  if it improves the score by **≥ 5%**.
- **Guardrails (`src/lifecycle/guardrails.py`).** Before anything is promoted: reject NaN/Inf,
  reject Sharpe < 0, reject IC worse than the baseline.
- **Audit trail:** every promotion/rejection is appended to `models/metrics_log.jsonl`.

### Training pipeline (E1/E2)
```
base.yaml config
  → load OHLCV (data/clean/ or yfinance)
  → build technical features (12–16 per strategy, all as-of)
  → build target (forward return at horizon H)
  → walk-forward CV (5 folds, embargo ≥ H days)
       → z-score scaling (stats from train fold only)
       → train GRU/LSTM with early stopping + Huber loss
       → predict → backtest with costs → compute metrics
  → write immutable run dir  runs/<variant>/<timestamp>/<TICKER>/
  → register as candidate → (optional) auto-promote
  → MLflow tracking
```

### Supporting infrastructure
- **Hyperparameter tuning:** Optuna, per ticker (GRU/LSTM units, dropout, LR, weight decay,
  batch size, trade thresholds), logged to MLflow.
- **Experiment tracking:** MLflow (remote, with SQLite fallback).
- **Serving:** FastAPI endpoint returning a BUY/SELL/HOLD signal per ticker (Swagger at `/docs`).
- **Orchestration:** Airflow DAGs per strategy. E2 is the daily cron anchor and E1 is
  Dataset-triggered off it — they share one registry file, so they must never run concurrently.
- **CI:** GitHub Actions runs the `pytest` suite (registry atomicity, guardrails, promotion
  scoring, walk-forward splits, feature-scaling no-leakage) on Python 3.10 and 3.12 on every
  push and PR.

---

## Methodology — anti-leakage by construction

Leakage is the #1 way backtests lie. The guardrails here are structural:

- **As-of features** — every feature at time *t* uses only data ≤ *t*.
- **Train-only scaling** — z-score statistics are computed on the training fold and applied to
  val/test; never on the full set.
- **Walk-forward + embargo** — expanding folds with a gap ≥ the prediction horizon between
  train and test, so overlapping-horizon labels can't leak.
- **Costs always on** — backtests are net of commission + slippage (10 bps daily, 20 bps
  intraday); a good MAE is worthless if turnover eats the P&L.
- **Robust loss** — Huber loss on standardized returns (robust to price outliers), not MSE.

Details in [README_WALK_FORWARD.md](README_WALK_FORWARD.md).

---

## Per-strategy status (the honest version)

- **E1 — Conservative (GRU, 90-day).** The strongest strategy. Median Sharpe ~1.1 and 68%
  directional accuracy across 10 tickers, beating the linear baseline on every metric.
  *Caveat:* max drawdowns are large (median ~37%, higher on volatile Argentine names) — well
  above the strategy's original <15% design target. Sharpe is good; capital-preservation is not.
- **E2 — Moderate (LSTM, 20-day).** Broadened to 31 tickers. Across that wider universe the
  median Sharpe (~0.55) and Calmar (0.38) sit **below** the linear baseline; the edge is in
  directional accuracy (55% vs 51%) and a modest positive rank-IC (0.03), not in risk-adjusted
  return. The weakest of the "working" strategies — a shorter horizon with a noisier signal,
  reported honestly.
- **E3 — Intraday (LSTM ensemble, 30-min).** **A documented negative result.** Net of the 20 bps
  intraday cost, Sharpe is strongly negative and directional accuracy sits below 50% — the
  strategy does not beat its costs. Kept in the repo transparently as a strategy that needs a
  rework (feature set, cost model, and label horizon), not as a win.
- **E4 — Pairs trading (k-NN + OU).** Fully specified (cointegration selection, Z-score entry/exit,
  OU mean-reversion) but not yet implemented. On the roadmap.

---

## Repo layout

```
trading_predict/
├── src/
│   ├── config/       # Centralized configuration (base.yaml)
│   ├── data/         # Download and cleaning
│   ├── features/     # Feature engineering
│   ├── e1/ … e4/     # One package per strategy (pipelines, models, baselines)
│   ├── backtest/     # Backtesting engine (costs, turnover, drawdown)
│   ├── lifecycle/    # Registry, promotion, guardrails, drift
│   └── dashboard/    # Reporting dashboard
├── models/           # registry.json — the model registry (never edit by hand)
├── runs/             # Immutable per-run outputs, timestamped
├── reports/          # Final figures and tables
├── demo/             # Static demo bundle + asset generator
├── dockerfiles/      # Airflow / MLflow / FastAPI images and DAGs
├── scripts/          # Operational scripts (data, optimization, evaluation, …)
├── tests/            # pytest suite (what CI runs)
├── docs/             # Academic specification and technical notes
└── notebooks/        # Exploratory analysis
```

**Design philosophy:** a clean split between data → code → results, so every number is traceable
back to the run that produced it.

---

## Configuration

Everything is driven by [`src/config/base.yaml`](src/config/base.yaml): the per-strategy ticker
universe, transaction costs (10 bps daily / 20 bps intraday), prediction horizons (E1 90 d,
E2 20 d, E3 30 min, E4 10 d), trade thresholds, model hyperparameters, and the decision-scoring
profiles used at promotion time. No magic numbers in the code.

---

## Tech stack

`Python` · `PyTorch` (GRU/LSTM) · `scikit-learn` · `pandas`/`numpy` · `Optuna` · `MLflow` ·
`FastAPI` · `Airflow` · `Docker` / docker-compose · `pytest` · `GitHub Actions` · `yfinance`

---

## Run it

```bash
# Full stack (recommended)
cp .env.example .env
docker-compose --profile all up -d
# Airflow → :8080 · MLflow → :5050 · API → :8800/docs

# Local training
conda activate ia_ceia_18co
python -m src.data.download_daily
python -m src.e1.train_pipeline --tickers AAPL,MSFT
python -m src.e2.train_pipeline --tickers NVDA
pytest tests/
```

Rank the current champions by the same composite score used for promotion:

```bash
python -m scripts.evaluation.leaderboard --strategies e1,e2 --top 10
```

---

## Documentation

| Document | What's in it |
|---|---|
| [README_WALK_FORWARD.md](README_WALK_FORWARD.md) | Walk-forward CV vs. a fixed split, concept drift, generated artifacts |
| [README_DOCKER.md](README_DOCKER.md) | Docker profiles, MLflow modes, DAG triggers, troubleshooting |
| [README_DATA_CLEANING.md](README_DATA_CLEANING.md) | OHLCV validation, cleaning strategies, quality reports |
| [scripts/README.md](scripts/README.md) | Authoritative list of operational scripts |
| [docs/SPEC.md](docs/SPEC.md) | Full academic specification of the four strategies *(Spanish)* |
| [docs/METRICS.md](docs/METRICS.md) | Evaluation metric glossary — what each one means and when it matters |

---

## Roadmap

- [x] Walk-forward CV + embargo, per-strategy champion/challenger lifecycle
- [x] Optuna tuning + MLflow tracking
- [x] Dockerized stack (Airflow + MLflow + FastAPI)
- [x] CI (GitHub Actions + pytest)
- [x] Drift detection (KS + PSI) — `src/lifecycle/drift.py` + tests (live Airflow wiring pending)
- [x] Hosted live demo — [huggingface.co/spaces/chatoxz/trading-predict-demo](https://huggingface.co/spaces/chatoxz/trading-predict-demo)
- [ ] Rework E3 (currently a negative result)
- [ ] Implement E4 (pairs trading)

---

*Final project — AI Specialization, FIUBA (Universidad de Buenos Aires). Built as a portfolio
piece demonstrating applied ML engineering and MLOps, not as investment advice.*
