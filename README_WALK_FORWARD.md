# Walk-forward validation — strategies E1 and E2

## Goal

Validate the **temporal robustness** of E1 (Conservative GRU) and E2 (Moderate LSTM) through
walk-forward validation, showing that the models keep predictive power across multiple time
windows never seen during training.

## What is walk-forward validation?

A validation technique built specifically for time series:

1. **Splits data chronologically** into N consecutive folds
2. **Trains progressively**: each fold trains on all preceding data
3. **Validates forward**: each fold is evaluated on a future test window
4. **Respects causality**: never uses information from the future

### Why not a simple 70/15/15 split

| Aspect | Simple split | Walk-forward |
|---|---|---|
| **Validation windows** | 1 single window (15%) | N independent windows |
| **Temporal robustness** | Can't detect *concept drift*\* | Detects degradation over time |
| **Data usage** | Tests on 15% only | Uses the full history efficiently |
| **Academic rigor** | Basic | Gold standard for time series |

(\*) **Concept drift** is when the relationship between your input variables and the target
changes over time, so the model you trained stops representing reality and starts performing
worse. You train a model today assuming the "world of the dataset" will look similar tomorrow.
Concept drift is when the world changes: the rules linking X → y are no longer the same.

## Why walk-forward matters here

For financial series specifically, walk-forward earns its cost:

- **Realistic temporal evaluation** — trains on the past, evaluates on the future, respecting causality.
- **Less risk of overstating results** — doesn't depend on one favorable test window.
- **Robustness across regime changes** — shows model stability in different market conditions.
- **More defensible evidence** — reports multi-window out-of-sample performance, not a single split.
- **Better use of history** — every fold adds out-of-sample evidence without breaking time order.
- **A common yardstick** — E1 and E2 are compared under one homogeneous validation methodology.

> Practical note: use walk-forward for final validation and evaluation; keep lighter
> configurations for fast iteration during tuning when compute cost is a constraint.

### Visual scheme

```
Walk-forward with 5 folds (expanding window):

Fold 1: [train.............] -> [test1]
Fold 2: [train..................] -> [test2]
Fold 3: [train.......................] -> [test3]
Fold 4: [train................................] -> [test4]
Fold 5: [train.......................................] -> [test5]

Full OOS window: [test1][test2][test3][test4][test5]
```

## Configuration

### In `src/config/base.yaml`

```yaml
splits:
  method: "walk_forward"  # enables walk-forward (vs "time_split")
  folds: 5                # number of validation windows

  # Embargo, to prevent temporal leakage
  embargo_days:
    e1: 90   # gap between train and test (= horizon_days for E1)
    e2: 20   # gap between train and test (= horizon_days for E2)
    e3: 6    # horizon_bars=6; in 5-min bars, not days — same principle as E1/E2
```

### Key parameters

- **`folds`** — number of test windows (default 5).
  More folds means finer temporal granularity but a smaller test set per fold. 5–10 is
  reasonable for long series (>1000 samples).

- **`embargo_days.e1`** — temporal gap between train and test.
  **Critical: must be ≥ `horizon_days`** (90 for E1). Without it, predicting a 90-day forward
  return at time *t* means the target is already known inside the training window — textbook
  leakage.

- **`test_size`** — size of each test window.
  Defaults to `n_samples / (folds + 1)`, an even distribution. Override with `splits.test_size`.

## Running it

Make sure `splits.method = "walk_forward"` in `base.yaml`, then:

```bash
python -m src.e1.train_pipeline --tickers AAPL   # E1 — Conservative, GRU
python -m src.e2.train_pipeline --tickers NVDA   # E2 — Moderate, LSTM
python -m src.e3.train_pipeline --tickers AAPL   # E3 — Intraday, LSTM ensemble (5-min bars)
```

Expected output shape:

```
✓ Using clean data: AAPL_daily.csv
▶ Running walk-forward (5 folds, test=auto)
  Fold 1: 2018-11-19 -> 2020-03-27 | MAE=0.1372 IC=0.319 Sharpe=0.61
  Fold 2: 2020-03-30 -> 2021-08-03 | MAE=0.2355 IC=0.766 Sharpe=2.49
  Fold 3: 2021-08-04 -> 2022-12-07 | MAE=0.2646 IC=0.108 Sharpe=-0.08
  Fold 4: 2022-12-08 -> 2024-04-17 | MAE=0.1014 IC=0.692 Sharpe=0.59
  Fold 5: 2024-04-18 -> 2025-08-26 | MAE=0.1249 IC=0.274 Sharpe=0.79
✓ AAPL: MAE=0.1727 IC=-0.254
```

## Generated files

Each walk-forward run writes to `runs/<strategy>/<timestamp>/<ticker>/` — `runs/e1_conservative/…`
for E1, `runs/e2_moderate/…` for E2.

### 1. `<ticker>_walkforward_folds.csv`
Per-fold metrics.

| fold | window | ml_mae | ml_ic | bt_sharpe | bt_calmar | bt_profit_factor | … |
|---|---|---|---|---|---|---|---|
| 1 | 2018-11-19 -> 2020-03-27 | 0.137 | 0.319 | 0.61 | 1.07 | 1.23 | … |
| 2 | 2020-03-30 -> 2021-08-03 | 0.236 | 0.766 | 2.49 | 9.16 | 2.87 | … |

Key columns: `test_start` / `test_end` (the fold's window), `n_train` / `n_val` / `n_test`
(split sizes), `ml_*` (machine-learning metrics — MAE, IC, directional accuracy), `bt_*`
(trading metrics — Sharpe, Calmar, max drawdown, profit factor).

### 2. `<ticker>_walkforward_predictions.csv`
Predictions from all folds, concatenated: `timestamp`, `fold`, `y_true`, `y_pred`.

### 3. `<ticker>_walkforward_backtest.csv`
Full backtest across all folds: `timestamp`, `pos`, `signal`, `gross_ret`, `costs`, `net_ret`,
`equity`, `turnover`.

### 4. `<ticker>_walkforward_metrics.png`
IC and Sharpe per fold — IC on the top panel, Sharpe on the bottom, time windows on the x-axis.

### 5. `<ticker>_fold<N>_backtest.csv`
Detailed backtest for each individual fold (one file per fold). Useful for debugging performance
in a specific window.

### 6. `summary_all.csv`
Aggregate summary:

```csv
ticker,n_samples,n_test,folds,ml_mae,ml_ic,bt_sharpe,bt_calmar,...
AAPL,2045,1700,5,0.173,-0.254,0.635,0.581,...
```

Here `ml_ic` is computed over the concatenated predictions from all folds, and `bt_sharpe` is the
Sharpe of the combined backtest.

## Reading the results

### ML metrics

**Information Coefficient (IC)** — `> 0.05` is the minimum bar for positive predictive power;
`> 0.10` is good; `< 0` signals overfitting or no predictive ability.

**Directional accuracy** — above 50% means the model gets the direction right more often than
not. Below 50% is worse than a coin flip and worth investigating.

### Trading metrics

**Sharpe ratio** — `> 1.0` excellent risk-adjusted; `0.5–1.0` good; `< 0` not profitable.

**Calmar ratio** — annualized return over max drawdown; `> 1.0` indicates good risk control.

**Profit factor** — gross profits over gross losses; `> 1.5` robust, `1.0–1.5` acceptable,
`< 1.0` losing.

### Worked example

```
Fold 1: IC=0.319, Sharpe=0.61  → model works, moderate Sharpe
Fold 2: IC=0.766, Sharpe=2.49  → excellent (2020 bull market)
Fold 3: IC=0.108, Sharpe=-0.08 → concept drift (2022 bear market)
Fold 4: IC=0.692, Sharpe=0.59  → predictive power recovers
Fold 5: IC=0.274, Sharpe=0.79  → stable

Aggregate: IC=-0.254, Sharpe=0.635
```

The model shows predictive power in 4 of 5 folds, with a severe degradation in fold 3 (the
2021–2022 bear market).

Note the aggregate IC is **negative** while every individual fold is positive — that is not a
contradiction, it's Simpson's paradox showing up in the concatenation. Each fold ranks well
internally, but the folds sit at different return levels, so pooling them and ranking across the
whole series destroys the signal. It argues for periodic retraining and drift detection, features
that capture market regime, and treating per-fold IC as the honest number rather than the pooled one.

## Walk-forward vs. simple split, technically

### Simple split (`time_split`)

```python
idx_train, idx_val, idx_test = time_split(len(X))  # 70/15/15
X_train, y_train = X[idx_train], y[idx_train]
X_test, y_test = X[idx_test], y[idx_test]

model.fit(X_train, y_train, X_val, y_val)
y_pred = model.predict(X_test)

ic = np.corrcoef(y_test, y_pred)[0, 1]
```

Output: one IC value, one backtest.

### Walk-forward (`walk_forward`)

```python
from sklearn.model_selection import TimeSeriesSplit

splitter = TimeSeriesSplit(n_splits=5, test_size=340, gap=90)

for fold, (train_idx, test_idx) in enumerate(splitter.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]

    model = GRURegressor(...)          # a fresh model per fold
    model.fit(X_train, y_train, X_val, y_val)

    y_pred = model.predict(X_test)

    ic_fold = np.corrcoef(y_test, y_pred)[0, 1]
    sharpe_fold = compute_sharpe(backtest(y_pred))

    print(f"Fold {fold}: IC={ic_fold:.3f} Sharpe={sharpe_fold:.2f}")

ic_combined = compute_ic(all_y_true, all_y_pred)
```

Output: N per-fold IC values plus one aggregate.

## Why this is the standard

Walk-forward validation is the gold standard in trading research because it **replicates real
production** (models retrain on a growing history), **detects concept drift** (it shows when a
model stops working), **prevents cherry-picking** (you can't select the most favorable test
window), and **maximizes data usage** (validating across the whole series instead of a single 15%
slice).

### References

- **López de Prado, M. (2018).** *Advances in Financial Machine Learning*. Wiley. Ch. 7,
  "Cross-Validation in Finance".
- **Aronson, D. (2006).** *Evidence-Based Technical Analysis*. Wiley. Ch. 9, "Walk-Forward Analysis".

## Troubleshooting

**`No samples available for walk-forward`** — the dataset is too small after dropping NaNs and
applying the lookback. Reduce the per-fold test size:

```yaml
splits:
  test_size: 100   # instead of the automatic size
```

**`IC=nan in fold X`** — all predictions or all targets are constant in that fold (std = 0).
Check that the fold has enough variability in returns; it can also indicate poor data quality in
that window.

**`test_size <= gap_samples`** — the test window is smaller than the embargo. Either raise
`test_size` or lower the embargo, the latter carefully — the embargo is what prevents leakage:

```yaml
splits:
  test_size: 200
  embargo_days:
    e1: 60   # only with good reason
```

## Differences by strategy

| Aspect | E1 (GRU Conservative) | E2 (LSTM Moderate) | E3 (LSTM Intraday) |
|---|---|---|---|
| **Horizon** | 90 days | 20 days | 30 min (6 × 5-min bars) |
| **Embargo** | 90 days (= horizon) | 20 days (= horizon) | 6 bars (= horizon_bars) |
| **Model** | GRU, 2 layers, 64–32 units | LSTM, 2 layers, 128–64 units | LSTM ensemble, 3 members |
| **Objective** | Information Coefficient | Sharpe + profit factor + CAGR | Sharpe + profit factor |
| **Filters** | none | optional `rsi14_min/max` | none |
| **Complexity** | ~70K parameters | ~100K parameters | 3× the base model |
| **Speed** | ~15–20% faster | slower but more expressive | slowest (ensemble + 5-min data) |

**When to use which:** E1 for long-horizon strategies where temporal stability matters; E2 for
tactical strategies that need to capture more complex patterns; E3 for intraday, which requires
5-minute bar data.

---

*Trading Predict — AI Specialization, FIUBA (CEIA 18co). Last updated 2026-07-21.*
