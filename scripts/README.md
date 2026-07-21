# Scripts

Operational scripts, organized by function. This is the authoritative list — if a script exists
in `scripts/` and isn't here, this file is out of date.

```
scripts/
├── data/           # data acquisition and cleaning
├── optimization/   # hyperparameter optimization (Optuna)
├── evaluation/     # model evaluation, comparison, lifecycle operations
├── trading/        # live trading (IOL API)
├── airflow/        # DAG validation
└── mlflow/         # MLflow utilities
```

---

## `data/` — data management

### `run_data_cleaning.py`
Diagnoses and cleans historical OHLCV data.

```bash
python scripts/data/run_data_cleaning.py                # forward fill (default)
python scripts/data/run_data_cleaning.py --interpolate
python scripts/data/run_data_cleaning.py --drop
```

Writes cleaned CSVs to `data/clean/` plus a per-ticker diagnostic report at
`data/clean/data_quality_report.json`. Full detail in
[README_DATA_CLEANING.md](../README_DATA_CLEANING.md).

---

## `optimization/` — hyperparameter tuning

### `optimize_e1_hyperparameters.py`
Optuna optimization for E1 Conservative (GRU), with concise per-trial logging.

Optimized parameters: `tau_buy`, `tau_sell`, `gru_units_1`, `gru_units_2`, `dropout`,
`learning_rate`, `weight_decay`, `batch_size`.

Split parameters (`n_folds`, `internal_val_fraction`) are **not** optimized — they're fixed in the
pipeline configuration, because tuning the validation scheme against the objective is a
leakage vector.

```bash
# single ticker (smoke test)
python scripts/optimization/optimize_e1_hyperparameters.py --ticker YPFD.BA --n_trials 1

# quick mode (subset)
python scripts/optimization/optimize_e1_hyperparameters.py --quick --n_trials 10

# per-ticker optimization (recommended — produces a consumable YAML)
python scripts/optimization/optimize_e1_hyperparameters.py --per_ticker --n_trials 50
```

Per-trial output:

```text
[Trial 0007] obj=+0.6042 ic=+0.3121 sharpe=+0.8963 trades=8/10 time=42.5s | tau=(0.050,0.000) gru=[112,48] do=0.25 lr=0.000200 wd=0.000120 bs=64
```

`weight_decay` is optimized on a log scale, default range `1e-6 → 1e-2`; `1e-5 → 1e-3` is a
reasonable starting range.

Artifacts land in `reports/hyperparameter_optimization/`: `best_params_e1.yaml`,
`e1_all_trials.csv`, `e1_tuned_params_by_ticker.yaml` (consumable) and its `.meta.yaml`.

To feed the tuned parameters back into training:

```bash
export E1_TUNED_PARAMS_PATH="reports/hyperparameter_optimization/e1_tuned_params_by_ticker.yaml"
```

E1 training then applies per-ticker `thresholds` and `model` overrides.

### `optimize_e2_hyperparameters.py` · `optimize_e3_hyperparameters.py`
Optuna optimization for E2 Moderate and E3 Intraday.

```bash
python scripts/optimization/optimize_e2_hyperparameters.py --ticker MSFT --n-trials 100
```

### `analyze_optuna_db.py`
Analyzes results stored in the Optuna database.

```bash
python scripts/optimization/analyze_optuna_db.py --study-name e1_conservative_optimization
```

See also [docs/HYPERPARAMETER_TUNING_CHEATSHEET.md](../docs/HYPERPARAMETER_TUNING_CHEATSHEET.md).

---

## `evaluation/` — evaluation and lifecycle

### Lifecycle operations

**`promote_candidate.py`** — evaluate candidates against champions and promote them. Dry run by
default; see [src/lifecycle/README_LIFECYCLE.md](../src/lifecycle/README_LIFECYCLE.md).

```bash
python -m scripts.evaluation.promote_candidate            # dry run
python -m scripts.evaluation.promote_candidate --execute
```

**`leaderboard.py`** — ranks every champion in the registry using the same composite score as
promotion, diffs against `reports/dashboard/leaderboard_history.jsonl` to show score and rank
deltas, and flags champions as stale (>10 days since training).

```bash
python -m scripts.evaluation.leaderboard --strategies e1,e2 --top 10
```

**`rollback_forced_promotions.py`** — reverts promotions that bypassed the scoring guardrails.
Written after a real incident where 29 models were force-promoted despite scoring worse than the
champions they replaced.

**`backfill_train_data_end.py`** — backfills `train_data_end` on older registry entries, which is
what lets legacy champions reach the fair-window comparison path instead of falling back to
stored metrics.

**`replay_lifecycle_e3.py`** — replays the lifecycle for E3 against historical runs.

### Comparison and analysis

| Script | What it does |
|---|---|
| `compare_e1_versions.py` | Aggregate comparison of the three E1 variants (baseline, simple, conservative) |
| `compare_e1_models.py` · `compare_e2_models.py` · `compare_e3_models.py` | Compare trained runs within one strategy |
| `compare_strategies.py` | Cross-strategy comparison |
| `compare_us_only.py` | Restrict comparison to the US ticker subset |
| `consolidate_strategy_metrics.py` | Consolidate per-strategy metrics into one table |
| `stability_analysis.py` | Champion stability over time |
| `plot_walkforward_stability.py` | Plots walk-forward stability |
| `plot_conclusions.py` | Generates the conclusion figures |
| `recompute_ic_e3_spearman.py` | Recomputes E3 IC using Spearman correlation |
| `requirements_validation.py` | Validates project requirements against results |
| `e1_retrospective_validation.py` | Out-of-time validation with the real GRU architecture |

```bash
python scripts/evaluation/compare_e1_versions.py                                # latest run per version
python scripts/evaluation/compare_e1_versions.py --tickers AAPL,MSFT --per-ticker
python scripts/evaluation/e1_retrospective_validation.py --ticker AAPL --train-days-ago 360 --horizon 90
```

`compare_e1_versions.py` reports training time (`train_time_seconds_avg/total`), ML metrics
(`mae`, `rmse`, `ic`, `directional_accuracy`) and trading metrics (`bt_sharpe`, `bt_cagr`,
`bt_max_drawdown`, `bt_num_trades`), writing timestamped CSV and Markdown to
`reports/tables/e1_versions_comparison/`.

---

## `trading/` — live trading

> These scripts touch a real broker API. `--production` places real orders.

### `e1_simple_iol_live_trade.py`
Live trading through the IOL API using E1 Simple models. Predicts with the GRU model, computes a
decision score for BUY/HOLD/SELL, and logs to MLflow.

```bash
# dry run — no orders placed
python scripts/trading/e1_simple_iol_live_trade.py --ticker GGAL.BA --action BUY --market bCBA

# real execution
python scripts/trading/e1_simple_iol_live_trade.py \
    --ticker AAPL --action BUY --market aNYS --production --quantity 10
```

See [docs/E1_SIMPLE_IOL_LIVE_TRADING.md](../docs/E1_SIMPLE_IOL_LIVE_TRADING.md).

### `iol_list_e1_prices.py` · `iol_get_portfolio.py`
List current IOL prices for E1 tickers, and fetch the current portfolio. Output goes to the
console and to `reports/iol_prices_YYYYMMDD_HHMMSS.json`.

---

## `airflow/` — DAG validation

### `validate_dags.py`
Validates that every DAG file parses and imports cleanly.

```bash
python scripts/airflow/validate_dags.py
```

---

## `mlflow/` — MLflow utilities

### `up_transparent_mlflow.sh`
Brings up MLflow in transparent offline → online mode: detects your local `mlflow` version,
rebuilds the container to match, and starts it with backend and artifacts shared at
`runs/mlflow_local`.

```bash
bash scripts/mlflow/up_transparent_mlflow.sh
MLFLOW_VERSION=3.8.1 bash scripts/mlflow/up_transparent_mlflow.sh   # pin a version
```

### `cleanup_init_test_runs.py`
Removes initialization and test runs that pollute the MLflow UI.

---

## Important paths

| What | Where |
|---|---|
| Cleaned data | `data/clean/*.csv` |
| Model runs | `runs/<variant>/<timestamp>/<TICKER>/` |
| Model registry | `models/registry.json` |
| MLflow (local) | `runs/mlflow_local/mlflow.db` |
| Optuna | `runs/optuna_trials/optuna.db` |
