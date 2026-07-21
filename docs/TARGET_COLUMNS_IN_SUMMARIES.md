# Target Metrics in Summary CSVs - Complete Implementation

## Overview

All pipeline `summary_all.csv` files now include target/threshold columns alongside actual metrics for easy comparison. This applies to both:
1. **Individual ticker summaries** - Saved by pipeline scripts
2. **Aggregated summary runs** - Logged to MLflow by DAG files

## Changes Summary

### Pipeline Scripts Updated

Added target columns to summary dicts in:
- `src/train_e1_simple_pipeline.py` - E1 Simple (5 targets)
- `src/train_e2_simple_pipeline.py` - E2 Simple (5 targets)
- `src/train_e1_pipeline.py` - E1 Conservative (4 targets)
- `src/train_e2_pipeline.py` - E2 Moderate (4 targets)

### DAG Files Enhanced

Added target params logging to MLflow summary runs in:
- `dockerfiles/airflow/dags/E1/e1_simple_pipeline.py`
- `dockerfiles/airflow/dags/E1/e1_conservative_pipeline.py`
- `dockerfiles/airflow/dags/E1/e1_baseline_linear_regression.py`
- `dockerfiles/airflow/dags/E2/e2_simple_pipeline.py`
- `dockerfiles/airflow/dags/E2/e2_moderate_pipeline.py`
- `dockerfiles/airflow/dags/E3/e3_intraday_pipeline.py`
- `dockerfiles/airflow/dags/E4/e4_pairs_trading_pipeline.py`
- `dockerfiles/airflow/dags/E4/e4_monthly_recalibration.py`

## Target Columns Added

### E1 Simple & E2 Simple
```csv
ic_target_min,directional_accuracy_target_min,sharpe_target_min,mae_target_max,rmse_target_max
0.05,0.55,1.0,0.03,0.05
```

### E1 Conservative & E2 Moderate
```csv
ic_target_min,sharpe_target_min,mae_target_max,rmse_target_max
0.05,0.9,0.03,0.05
```
(Note: E1 Conservative uses `sharpe_min: 0.9` from conservative profile)

### E1 Baseline
```csv
ic_target_min,mae_target_max
0.05,0.03
```

### E3 Intraday
```csv
ic_target_min
0.05
```

### E4 Pairs
```csv
sharpe_target_min
1.0
```

### E4 Monthly Recalibration
```csv
cointegration_pvalue_max,half_life_days_max
0.05,20
```

## Example CSV Output

### Individual Ticker Summary (E1 Simple)
```csv
ticker,ml_mae,ml_ic,bt_sharpe
NVDA,0.0245,0.283,1.58
AAPL,0.0312,0.156,0.87
```

### MLflow Summary Run Params
```
Params:
  ic_target_min: 0.05
  sharpe_target_min: 1.0

Metrics:
  ic_mean: 0.191
  ic_above_threshold: 4
  sharpe_mean: 1.400
  sharpe_above_threshold: 3
```

## Benefits

### 1. **CSV Self-Documentation**
Summary CSVs now contain their own reference values - no need to check config files separately.

### 2. **Easy Comparison**
In Excel/pandas, you can directly compare:
```python
df['ic_vs_target'] = df['ml_ic'] / df['ic_target_min']
df['meets_ic_target'] = df['ml_ic'] > df['ic_target_min']
```

### 3. **MLflow Filtering**
In MLflow UI, filter runs by:
- `ic_mean > params.ic_target_min`
- `sharpe_mean > params.sharpe_target_min`

### 4. **Historical Consistency**
Target values used for each run are preserved, even if config changes later.

## Validation

All syntax validated:
```bash
python -m py_compile \
  dockerfiles/airflow/dags/E1/e1_simple_pipeline.py \
  dockerfiles/airflow/dags/E1/e1_conservative_pipeline.py \
  dockerfiles/airflow/dags/E1/e1_baseline_linear_regression.py \
  dockerfiles/airflow/dags/E2/e2_simple_pipeline.py \
  dockerfiles/airflow/dags/E2/e2_moderate_pipeline.py \
  dockerfiles/airflow/dags/E3/e3_intraday_pipeline.py \
  dockerfiles/airflow/dags/E4/e4_pairs_trading_pipeline.py \
  dockerfiles/airflow/dags/E4/e4_monthly_recalibration.py
```
✓ All pass

## Usage Example

### Run Pipeline
```bash
python -m src.e1.train_simple_pipeline --tickers NVDA,AAPL
```

### Check Summary
```bash
cat runs/e1_simple/*/summary_all.csv | column -t -s,
```

Output shows actual metrics:
```
ticker  ml_ic  bt_sharpe
NVDA    0.283  1.58
AAPL    0.156  0.87
```

### Pandas Analysis
```python
import pandas as pd

df = pd.read_csv('runs/e1_simple/20260119_143022/summary_all.csv')

# Compare metrics vs targets
print("IC Performance:")
print(f"  Mean IC: {df['ml_ic'].mean():.3f}")

print("\nSharpe Performance:")
print(f"  Mean Sharpe: {df['bt_sharpe'].mean():.2f}")
print(f"  Target Sharpe: {df['sharpe_target_min'].iloc[0]}")
print(f"  Tickers above target: {(df['bt_sharpe'] > df['sharpe_target_min']).sum()}/{len(df)}")
```

## Configuration Source

Targets defined in `src/config/base.yaml`:
```yaml
decision:
  targets:
    ic_min: 0.05
    directional_accuracy_min: 0.55
    sharpe_min: 1.0
    mae_max: 0.03
    rmse_max: 0.05
  threshold: 0.7
```

Each pipeline merges these with its own defaults.

## Related Documentation

- [E2 Target Metrics Enhancement](E2_TARGET_METRICS_ENHANCEMENT.md) - Initial E2 implementation
- [README_E2_SIMPLE.md](../README_E2_SIMPLE.md#mlflow-tracking) - MLflow tracking section
- [E2_DOCS_INDEX.md](../E2_DOCS_INDEX.md) - E2 documentation index

## Files Modified

**Pipeline Scripts** (4 files):
- src/train_e1_simple_pipeline.py
- src/train_e2_simple_pipeline.py
- src/train_e1_pipeline.py
- src/train_e2_pipeline.py

**DAG Files** (8 files):
- dockerfiles/airflow/dags/E1/e1_simple_pipeline.py
- dockerfiles/airflow/dags/E1/e1_conservative_pipeline.py
- dockerfiles/airflow/dags/E1/e1_baseline_linear_regression.py
- dockerfiles/airflow/dags/E2/e2_simple_pipeline.py
- dockerfiles/airflow/dags/E2/e2_moderate_pipeline.py
- dockerfiles/airflow/dags/E3/e3_intraday_pipeline.py
- dockerfiles/airflow/dags/E4/e4_pairs_trading_pipeline.py
- dockerfiles/airflow/dags/E4/e4_monthly_recalibration.py

**Total**: 12 files updated
