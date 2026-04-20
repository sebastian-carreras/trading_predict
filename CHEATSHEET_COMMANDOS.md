# Cheat Sheet — Dashboard y Lifecycle

Comandos listos para copiar y pegar para monitoreo, reportes y promoción de modelos.

## 1. Dashboard

### Levantar MLflow UI local

```bash
mlflow ui --backend-store-uri sqlite:///runs/mlflow_local/mlflow.db --port 5050
open http://localhost:5050
```

### Levantar dashboard con Docker

```bash
docker compose --profile dashboard up -d
open http://localhost:5050
open http://localhost:9001
```

### Dashboard Checker CLI — vistas principales

```bash
python -m src.dashboard.checker
python -m src.dashboard.checker --strategy e1_conservative
python -m src.dashboard.checker --view ticker --strategy e1_conservative
python -m src.dashboard.checker --view ticker --strategy e1_conservative --last 3
python -m src.dashboard.checker --view history --strategy e1_conservative --ticker AAPL
python -m src.dashboard.checker --view history --strategy e1_conservative --ticker AAPL --last 5
```

### Strategy keys válidas para el checker

```bash
e1_simple
e1_conservative
e2_baseline
e2_simple
e2_moderate
e3_baseline
e3_intraday
e4_pairs
```

### Checker — ejemplos rápidos por estrategia

```bash
python -m src.dashboard.checker --strategy e1_simple
python -m src.dashboard.checker --strategy e1_conservative
python -m src.dashboard.checker --strategy e2_baseline
python -m src.dashboard.checker --strategy e2_simple
python -m src.dashboard.checker --strategy e2_moderate
python -m src.dashboard.checker --strategy e3_baseline
python -m src.dashboard.checker --strategy e3_intraday
python -m src.dashboard.checker --strategy e4_pairs
```

### Vista per-ticker por estrategia

```bash
python -m src.dashboard.checker --view ticker --strategy e1_simple
python -m src.dashboard.checker --view ticker --strategy e1_conservative
python -m src.dashboard.checker --view ticker --strategy e2_baseline
python -m src.dashboard.checker --view ticker --strategy e2_simple
python -m src.dashboard.checker --view ticker --strategy e2_moderate
python -m src.dashboard.checker --view ticker --strategy e3_baseline
python -m src.dashboard.checker --view ticker --strategy e3_intraday
python -m src.dashboard.checker --view ticker --strategy e4_pairs
```

### Historial por estrategia y ticker

```bash
python -m src.dashboard.checker --view history --strategy e1_simple --ticker AAPL
python -m src.dashboard.checker --view history --strategy e1_conservative --ticker AAPL
python -m src.dashboard.checker --view history --strategy e2_baseline --ticker NVDA
python -m src.dashboard.checker --view history --strategy e2_simple --ticker NVDA
python -m src.dashboard.checker --view history --strategy e2_moderate --ticker NVDA
python -m src.dashboard.checker --view history --strategy e3_baseline --ticker SPY
python -m src.dashboard.checker --view history --strategy e3_intraday --ticker SPY
```

## 2. Lifecycle

### Promoción de candidatos — genérico

```bash
python -m scripts.evaluation.promote_candidate
python -m scripts.evaluation.promote_candidate --execute
python -m scripts.evaluation.promote_candidate --verbose
python -m scripts.evaluation.promote_candidate --show-log 20
python -m scripts.evaluation.promote_candidate --tickers AAPL,MSFT --execute
python -m scripts.evaluation.promote_candidate --tickers YPFD.BA --force --execute
```

### Promoción filtrada por estrategia

```bash
python -m scripts.evaluation.promote_candidate --strategy e1
python -m scripts.evaluation.promote_candidate --strategy e1 --execute
python -m scripts.evaluation.promote_candidate --strategy e2
python -m scripts.evaluation.promote_candidate --strategy e2 --execute
python -m scripts.evaluation.promote_candidate --strategy e3
python -m scripts.evaluation.promote_candidate --strategy e3 --execute
python -m scripts.evaluation.promote_candidate --strategy e4
python -m scripts.evaluation.promote_candidate --strategy e4 --execute
```

### Promoción filtrada por estrategia y tickers

```bash
python -m scripts.evaluation.promote_candidate --strategy e1 --tickers AAPL,MSFT --execute
python -m scripts.evaluation.promote_candidate --strategy e2 --tickers NVDA,GOOGL --execute
python -m scripts.evaluation.promote_candidate --strategy e3 --tickers SPY,AAPL --execute
python -m scripts.evaluation.promote_candidate --strategy e4 --tickers GGAL.BA,BMA.BA --execute
```

### Skills / comandos interactivos documentados

```text
/model-status
/compare-models e1 AAPL
/promote-model e1 AAPL
/retire-model e1 AAPL
/new-experiment e1 attention_gru
```

## 3. Entrenar y alimentar Dashboard/Lifecycle

### E1 — Conservative / Baseline / Simple

```bash
python -m src.e1.train_all
python -m src.e1.train_all --tickers AAPL,MSFT

python -m src.e1.train_pipeline
python -m src.e1.train_pipeline --tickers AAPL
python -m src.e1.train_pipeline --tickers AAPL,MSFT --refresh-data
python -m src.e1.train_pipeline --tickers AAPL --auto-promote

python -m src.e1.train_baseline --tickers AAPL

python -m src.e1.train_simple_pipeline --tickers AAPL
python -m src.e1.train_simple_pipeline --auto-promote
```

### E2 — Moderate / Baseline

```bash
python -m src.e2.train_pipeline
python -m src.e2.train_pipeline --tickers NVDA,GOOGL
python -m src.e2.train_pipeline --tickers NVDA,GOOGL --refresh-data
python -m src.e2.train_pipeline --tickers NVDA,GOOGL --auto-promote

python -m src.e2.train_baseline --tickers NVDA
```

### E3 — Intraday / Baseline

```bash
python -m src.e3.train_pipeline --mode download
python -m src.e3.train_pipeline --mode download --tickers SPY,AAPL,NVDA,QQQ

python -m src.e3.train_pipeline
python -m src.e3.train_pipeline --tickers SPY,AAPL

python -m src.e3.train_baseline --tickers SPY
python -m src.e3.train_baseline --tickers SPY,AAPL,NVDA
python -m src.e3.train_baseline --tickers SPY --skip-download
```

### E4 — Pairs Trading

```bash
python -m src.e4.train_pipeline
python -m src.e4.train_pipeline --pairs GGAL.BA,BMA.BA YPFD.BA,PAMP.BA
python -m src.e4.train_pipeline --output-tag my_test
```

## 4. Optimización

### E1 — Optuna

```bash
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 10 --quick
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 50
python scripts/optimization/optimize_e1_hyperparameters.py --ticker AAPL --n_trials 30
python scripts/optimization/optimize_e1_hyperparameters.py --tickers AAPL,MSFT,JNJ --n_trials 20
python scripts/optimization/optimize_e1_hyperparameters.py --per_ticker --n_trials 30
python scripts/optimization/optimize_e1_hyperparameters.py --study_name e1_hyperparameter_optimization --n_trials 20
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 50 --mlflow_uri http://localhost:5050
```

### E2 — Optuna

```bash
python scripts/optimization/optimize_e2_hyperparameters.py --n_trials 10 --quick
python scripts/optimization/optimize_e2_hyperparameters.py --n_trials 50
python scripts/optimization/optimize_e2_hyperparameters.py --ticker NVDA --n_trials 30
python scripts/optimization/optimize_e2_hyperparameters.py --tickers NVDA,GOOGL,AMZN --n_trials 20
python scripts/optimization/optimize_e2_hyperparameters.py --per_ticker --n_trials 30
python scripts/optimization/optimize_e2_hyperparameters.py --study_name e2_hyperparameter_optimization_timesplit --n_trials 20
python scripts/optimization/optimize_e2_hyperparameters.py --n_trials 50 --mlflow_uri http://localhost:5050
```

### Reanudar / limitar búsqueda

```bash
python scripts/optimization/optimize_e1_hyperparameters.py --study_name e1_hyperparameter_optimization --timeout 3600
python scripts/optimization/optimize_e2_hyperparameters.py --study_name e2_hyperparameter_optimization_timesplit --timeout 3600
```

### Ajustar rangos de búsqueda

```bash
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 20 --tau_buy_min 0.03 --tau_buy_max 0.08 --dropout_min 0.20 --dropout_max 0.40
python scripts/optimization/optimize_e2_hyperparameters.py --n_trials 20 --tau_buy_min 0.015 --tau_buy_max 0.03 --dropout_min 0.10 --dropout_max 0.25
```

### Consumir parámetros tuneados en training

```bash
export E1_TUNED_PARAMS_PATH=reports/hyperparameter_optimization/e1_tuned_params_by_ticker.yaml
python -m src.e1.train_pipeline --tickers AAPL,MSFT --refresh-data

export E2_TUNED_PARAMS_PATH=reports/hyperparameter_optimization/e2_tuned_params_by_ticker.yaml
python -m src.e2.train_pipeline --tickers NVDA,GOOGL --refresh-data
```

### Dashboard / inspección de Optuna

```bash
optuna-dashboard sqlite:///runs/optuna_trials/optuna_studies.db
python scripts/optimization/analyze_optuna_db.py
```

### Archivos útiles de optimización

```bash
cat reports/hyperparameter_optimization/best_params_e1.yaml
cat reports/hyperparameter_optimization/best_params_e2.yaml
cat reports/hyperparameter_optimization/e1_tuned_params_by_ticker.yaml
cat reports/hyperparameter_optimization/e2_tuned_params_by_ticker.yaml
open reports/hyperparameter_optimization/figures/
tail -n 20 reports/hyperparameter_optimization/e1_all_trials.csv
tail -n 20 reports/hyperparameter_optimization/e2_all_trials.csv
```

## 5. Flujo corto recomendado

### E1

```bash
python -m src.e1.train_pipeline --tickers AAPL --refresh-data
python -m src.dashboard.checker --strategy e1_conservative
python -m scripts.evaluation.promote_candidate --strategy e1 --tickers AAPL --verbose
python -m scripts.evaluation.promote_candidate --strategy e1 --tickers AAPL --execute
```

### E2

```bash
python -m src.e2.train_pipeline --tickers NVDA,GOOGL --refresh-data
python -m src.dashboard.checker --strategy e2_moderate
python -m scripts.evaluation.promote_candidate --strategy e2 --verbose
python -m scripts.evaluation.promote_candidate --strategy e2 --execute
```

### E3

```bash
python -m src.e3.train_pipeline --mode download --tickers SPY,AAPL
python -m src.e3.train_pipeline --tickers SPY,AAPL
python -m src.dashboard.checker --strategy e3_intraday
python -m scripts.evaluation.promote_candidate --strategy e3 --verbose
python -m scripts.evaluation.promote_candidate --strategy e3 --execute
```

### E4

```bash
python -m src.e4.train_pipeline --pairs GGAL.BA,BMA.BA YPFD.BA,PAMP.BA
python -m src.dashboard.checker --strategy e4_pairs
python -m scripts.evaluation.promote_candidate --strategy e4 --verbose
python -m scripts.evaluation.promote_candidate --strategy e4 --execute
```

## 6. Archivos de salida útiles

```bash
cat reports/dashboard/dashboard_report_latest.csv
cat reports/dashboard/ticker_report_e1_conservative_latest.csv
cat reports/dashboard/history_report_e1_conservative_AAPL_latest.csv
cat runs/e1_conservative/*/summary_all.csv
cat runs/e2_moderate/*/summary_all.csv
cat runs/e3_baseline/*/baseline_summary_all.csv
cat models/registry.json
tail -n 20 models/promotion_log.jsonl
tail -n 20 models/metrics_log.jsonl
```

## 7. Notas rápidas

- `src.dashboard.checker --view history` exige `--strategy` y `--ticker`.
- `scripts.evaluation.promote_candidate` usa `--strategy e1` por default si no se especifica `--strategy`.
- En dashboard, las claves válidas de estrategia son las de `src/config/dashboard_thresholds.yaml`.
- En lifecycle, el filtro `--strategy` usa prefijos de registry: `e1`, `e2`, `e3`, `e4`.
- E4 trabaja por pares; el checker puede depender de cómo se haya logueado el identificador del par en MLflow.