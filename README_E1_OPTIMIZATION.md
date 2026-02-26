# Optimización de Hiperparámetros - Estrategia E1 (GRU Conservadora)

Guía operativa de optimización para E1 usando **Optuna** + **MLflow**.

## Alcance actual

La optimización E1 ajusta únicamente:

- **Thresholds de trading**: `tau_buy`, `tau_sell`
- **Arquitectura GRU**: `gru_units_1`, `gru_units_2`, `dropout`
- **Entrenamiento**: `learning_rate`, `batch_size`

### Importante (cambio vigente)

En E1 **ya no** se optimizan parámetros de split de validación temporal:

- `n_folds`
- `internal_val_fraction`

Esos parámetros quedan fijos por configuración del pipeline y no se sobreescriben desde tuned params.

---

## Espacio de búsqueda (default)

- `tau_buy`: 0.02 a 0.10 (step 0.01)
- `tau_sell`: -0.02 a 0.02 (step 0.01)
- `gru_units_1`: 32 a 128 (step 16)
- `gru_units_2`: 16 a 64 (step 16)
- `dropout`: 0.20 a 0.50 (step 0.05)
- `learning_rate`: 1e-4 a 1e-2 (log scale)
- `batch_size`: {32, 64, 128}

---

## Función objetivo

Se maximiza:

```python
objective = 0.5 * ic_mean + 0.5 * sharpe_mean + trade_penalty
```

Donde:

- `trade_penalty = -5` si menos del 50% de tickers generan trades
- `trade_penalty = 0` en caso contrario

Esto balancea calidad predictiva (IC) y performance de trading (Sharpe), evitando configuraciones sin señales.

---

## Uso rápido

### 1) Smoke test (1 ticker)

```bash
python scripts/optimization/optimize_e1_hyperparameters.py --ticker YPFD.BA --n_trials 1
```

### 2) Modo rápido

```bash
python scripts/optimization/optimize_e1_hyperparameters.py --quick --n_trials 10
```

### 3) Optimización completa

```bash
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 50
```

### 4) Optimización por ticker (recomendada)

```bash
python scripts/optimization/optimize_e1_hyperparameters.py --per_ticker --n_trials 50
```

---

## Logging por trial (modo conciso)

Ejemplo de salida:

```text
[Trial 0073] obj=+0.6042 ic=+0.3121 sharpe=+0.8963 trades=8/10 time=40.2s | tau=(0.050,0.000) gru=[112,48] do=0.25 lr=0.000200 bs=64
```

Con `weight_decay` habilitado, la línea de trial también incluye `wd`:

```text
[Trial 0073] obj=+0.6042 ic=+0.3121 sharpe=+0.8963 trades=8/10 time=40.2s | tau=(0.050,0.000) gru=[112,48] do=0.25 lr=0.000200 wd=0.000120 bs=64
```

Interpretación rápida:

- `obj`: valor objetivo del trial (lo que Optuna maximiza)
- `ic`: Information Coefficient promedio
- `sharpe`: Sharpe promedio de backtest
- `trades=8/10`: 8 tickers con al menos 1 trade sobre 10 evaluados
- `time`: duración total del trial

---

## Rango recomendado para weight_decay

- Rango default del optimizador: `1e-6 → 1e-2` (log scale)
- Rango recomendado inicial: `1e-5 → 1e-3`

Ejemplo:

```bash
python scripts/optimization/optimize_e1_hyperparameters.py \
  --per_ticker --n_trials 50 \
  --weight_decay_min 1e-5 \
  --weight_decay_max 1e-3
```

---

## Outputs

### Global

- `reports/hyperparameter_optimization/best_params_e1.yaml`
- `reports/hyperparameter_optimization/e1_all_trials.csv`
- `reports/hyperparameter_optimization/figures/`

### Por ticker

- `reports/hyperparameter_optimization/by_ticker/<TICKER>/best_params_e1.yaml`

### Agregados para training

- `reports/hyperparameter_optimization/e1_tuned_params_by_ticker.yaml` (consumible)
- `reports/hyperparameter_optimization/e1_tuned_params_by_ticker.meta.yaml` (con metadata)

---

## Integración con entrenamiento E1

Exportar la ruta del YAML por ticker:

```bash
export E1_TUNED_PARAMS_PATH="reports/hyperparameter_optimization/e1_tuned_params_by_ticker.yaml"
```

El pipeline E1 toma overrides por ticker para:

- `thresholds`
- `model`

No sobreescribe parámetros de split.

---

## MLflow y Optuna

### MLflow local

Por default se usa SQLite local:

- `runs/mlflow_local/mlflow.db`

Levantar UI:

```bash
mlflow ui --backend-store-uri sqlite:///runs/mlflow_local/mlflow.db
```

### Optuna storage

Por default:

- `sqlite:///runs/optuna_trials/optuna_studies.db`

Continuar un estudio existente:

```bash
python scripts/optimization/optimize_e1_hyperparameters.py \
  --study_name e1_hyperparameter_optimization \
  --n_trials 20
```

---

## Troubleshooting breve

### Optimización lenta

- Usar `--quick` o `--ticker` para iteraciones cortas
- Reducir `--n_trials` para smoke tests

### Problemas de gráficos (kaleido)

Si falla la exportación PNG, revisar dependencias de visualización o usar los resultados tabulares (`CSV`/`YAML`) para análisis.

### MLflow con errores de DB local

Recrear DB local cuando sea necesario:

```bash
rm -f runs/mlflow_local/mlflow.db
```

Luego relanzar optimización.

---

## Referencias

- `scripts/optimization/optimize_e1_hyperparameters.py`
- `README_HYPERPARAMETER_TUNING.md`
- `scripts/README.md`
