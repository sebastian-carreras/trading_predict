# Hyperparameter optimization with Optuna

Automated hyperparameter search for strategies E1, E2 and E3, using Optuna for Bayesian
optimization and MLflow for experiment tracking.

---

## Overview

Every script follows the same loop:

```
Optuna proposes hyperparameters (TPE sampler)
    → the training pipeline runs across all tickers
    → aggregate metrics are computed (Sharpe, IC, Calmar, DirAcc)
    → a composite objective is returned
    → Optuna learns from the result and proposes the next trial
```

Studies persist to SQLite (`runs/optuna_trials/optuna_studies.db`), so a run can be interrupted
and resumed without losing progress.

---

## Strategy comparison

| Aspect | E1 Conservative | E2 Moderate | E3 Intraday |
|---|---|---|---|
| Model | GRU, 2 layers | LSTM, 2 layers | LSTM ensemble |
| Horizon | 90 days | 20 days | 30 min (5-min bars) |
| Script | `optimize_e1_hyperparameters.py` | `optimize_e2_hyperparameters.py` | `optimize_e3_hyperparameters.py` |
| Default study name | `e1_hyperparameter_optimization` | `e2_hyperparameter_optimization_timesplit` | `e3_hyperparameter_optimization` |

---

## Search space

### E1 — GRU Conservative

| Parameter | Range | Meaning |
|---|---|---|
| `tau_buy` | [0.02, 0.10] step 0.01 | Buy signal threshold |
| `tau_sell` | [−0.02, 0.02] step 0.01 | Sell signal threshold |
| `gru_units_1` | [32, 128] step 16 | Units, GRU layer 1 |
| `gru_units_2` | [16, 64] step 16 | Units, GRU layer 2 |
| `dropout` | [0.20, 0.50] step 0.05 | Regularization |
| `learning_rate` | [1e-4, 1e-2] | Log-uniform |
| `weight_decay` | [1e-6, 1e-2] | Log-uniform |
| `batch_size` | {32, 64, 128} | Categorical |

**Objective:** `0.35×Sharpe + 0.25×IC + 0.20×DirAcc + 0.20×Calmar + trade_penalty`

### E2 — LSTM Moderate

| Parameter | Range | Meaning |
|---|---|---|
| `tau_buy` | [0.015, 0.05] step 0.005 | Buy signal threshold |
| `tau_sell` | [−0.01, 0.01] step 0.005 | Sell signal threshold |
| `lstm_units_1` | [64, 256] step 32 | Units, LSTM layer 1 |
| `lstm_units_2` | [32, 128] step 16 | Units, LSTM layer 2 |
| `dropout` | [0.10, 0.40] step 0.05 | Regularization |
| `learning_rate` | [5e-5, 5e-3] | Log-uniform |
| `batch_size` | {32, 64, 128} | Categorical |

**Objective:** `0.30×Sharpe + 0.20×IC + 0.30×DirAcc + 0.20×Calmar + trade_penalty`

### E3 — LSTM ensemble, intraday

| Parameter | Range | Meaning |
|---|---|---|
| `tau_buy` | [0.001, 0.005] step 0.0005 | Buy signal threshold (in bps) |
| `tau_sell` | [0.001, 0.005] step 0.0005 | Sell signal threshold |
| `lstm_hidden_size` | [64, 256] step 32 | Hidden LSTM layer size |
| `dense_units` | [16, 64] step 16 | Dense layer units |
| `dropout` | [0.10, 0.40] step 0.05 | Regularization |
| `learning_rate` | [1e-4, 5e-3] | Log-uniform |
| `weight_decay` | [1e-6, 1e-2] | Log-uniform |
| `batch_size` | {64, 128, 256, 512} | Categorical |
| `ensemble_members` | [2, 5] step 1 | Number of ensemble members |
| `consensus_tol` | [0.0001, 0.002] step 0.0001 | Consensus filter across members |

**Objective:** `0.30×Sharpe + 0.20×IC + 0.25×DirAcc + 0.25×Calmar + trade_penalty`

> `trade_penalty = −5` when fewer than 50% of tickers produce any trades. Without it, the
> optimizer learns that the safest way to maximize risk-adjusted return is to never trade —
> thresholds so high that no signal ever fires.

---

## Usage

```bash
# E1 — full universe (50 trials by default)
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 50

# E2
python scripts/optimization/optimize_e2_hyperparameters.py --n_trials 50

# E3
python -m scripts.optimization.optimize_e3_hyperparameters --n_trials 50
```

### Common options (all three scripts)

```bash
--ticker AAPL                      # single ticker (fastest, good for smoke tests)
--tickers AAPL,MSFT,GOOG           # explicit list
--quick                            # 3 random tickers from the universe (2 for E3)
--per_ticker                       # independent optimization per ticker, emits per-ticker YAML
--study_name <name>                # resume an existing study
--timeout 3600                     # total timeout in seconds
--mlflow_uri http://localhost:5050 # remote MLflow (Docker)
--use-latest-data                  # use data up to today (default is the config's fixed window)
```

### Narrowing ranges at runtime

```bash
# E1: constrain tau_buy and learning_rate
python scripts/optimization/optimize_e1_hyperparameters.py \
  --tau_buy_min 0.03 --tau_buy_max 0.07 \
  --learning_rate_min 1e-4 --learning_rate_max 1e-3 \
  --n_trials 30

# E3: limit ensemble size and batch sizes
python -m scripts.optimization.optimize_e3_hyperparameters \
  --ensemble_members_min 2 --ensemble_members_max 3 \
  --batch_sizes 64,128 \
  --n_trials 20
```

### Validation method

The default is `walk_forward` with 3 folds, configurable in `src/config/base.yaml` under
`optuna.<strategy>.validation`, and overridable from the CLI:

```bash
--validation_method time_split                        # single split, faster
--validation_method walk_forward --optuna_folds 5
```

Note that tuning against walk-forward folds is slower but keeps the search honest — optimizing
against a single split invites picking hyperparameters that happen to suit one window.

---

## Outputs

Everything lands in `reports/hyperparameter_optimization/`:

| File | Contents |
|---|---|
| `best_params_e{1,2,3}.yaml` | Best hyperparameters of the study |
| `e{1,2,3}_all_trials.csv` | Every trial in the study |
| `e{1,2,3}_run_trials.csv` | Only the trials from the current run |
| `e{1,2,3}_tuned_params_by_ticker.yaml` | Per-ticker overrides, consumable by training |
| `e{1,2,3}_tuned_params_by_ticker.meta.yaml` | Same, with study metadata |
| `e{1,2,3}_optimization_summary.txt` | Text summary with the top 10 trials |
| `figures/` | History, parameter importances, parallel coordinates |

With `--per_ticker`, per-ticker results are additionally written to `by_ticker/<TICKER>/`.

---

## Persistence and resuming

Trials accumulate in `runs/optuna_trials/optuna_studies.db`. If a run is interrupted, resume it
with the same `--study_name` and the previous trials are preserved:

```bash
python scripts/optimization/optimize_e1_hyperparameters.py \
  --study_name e1_hyperparameter_optimization \
  --n_trials 20
```

---

## MLflow tracking

Tracking uses an automatic cascade fallback:

1. Remote server (`MLFLOW_TRACKING_URI` in `.env`, or `--mlflow_uri`)
2. Local SQLite (`runs/mlflow_local/mlflow.db`)
3. File store (`mlruns/`)

To browse locally without Docker:

```bash
mlflow ui --backend-store-uri runs/mlflow_local/mlflow.db   # → http://localhost:5000
```

To browse the Optuna studies:

```bash
optuna-dashboard sqlite:///runs/optuna_trials/optuna_studies.db
```

---

## Feeding results back into training

Optimized parameters reach the training pipeline through the generated YAML. Its path is
registered in `src/config/base.yaml` under `optuna.<strategy>.tuned_params_path`; the pipeline
reads it automatically when the file exists, overriding the base config values.
