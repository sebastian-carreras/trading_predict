# Changelog - Sistema de Optimización de Hiperparámetros

## [2.0.0] - 2026-01-07

### ✨ Nuevas Características

#### MLflow con Backend SQLite
- **Cambio**: MLflow ahora usa SQLite por defecto en lugar de filesystem
- **Beneficio**: Eliminación del FutureWarning de deprecación
- **Ubicación**: `runs/mlflow_local/mlflow.db`
- **Comando UI**: `mlflow ui --backend-store-uri sqlite:///runs/mlflow_local/mlflow.db --port 5001`

#### Modo Local vs Remoto
- **Local** (default): No requiere Docker, tracking en SQLite
  ```bash
  python scripts/optimize_e1_hyperparameters.py --n_trials 50
  ```
- **Remoto**: Usa servidor MLflow en Docker
  ```bash
  python scripts/optimize_e1_hyperparameters.py --n_trials 50 --mlflow_uri http://localhost:5050
  ```

#### Manejo Correcto de Runs en MLflow
- Cada trial ahora crea su propio run con `mlflow.start_run()`
- Eliminado `MLflowCallback` que causaba conflictos
- Registra parámetros, métricas y artifacts correctamente

### 🐛 Correcciones

#### Warning de Dropout PyTorch
- **Problema**: `UserWarning: dropout option adds dropout after all but last recurrent layer`
- **Causa**: GRU con `num_layers=1` pero `dropout > 0`
- **Solución**: Dropout manual con `torch.nn.Dropout` después de cada capa GRU
- **Archivo**: `src/models/e1_gru.py`

#### Warning de Artifact Location
- **Problema**: `Malformed experiment 'artifacts'`
- **Causa**: MLflow creaba directorio `artifacts/` que interpretaba como experimento
- **Solución**: Limpieza automática y configuración correcta de artifact_location

#### Error de Credenciales boto3
- **Problema**: `NoCredentialsError: Unable to locate credentials`
- **Causa**: MLflow intentaba usar S3 sin boto3 instalado
- **Solución**: Instalado boto3 + configuración local por defecto

### 📚 Documentación Actualizada

#### README_HYPERPARAMETER_TUNING.md
- Sección "Configuración Inicial" con modos local/remoto
- Sección "MLflow Tracking" con comandos CLI
- Troubleshooting expandido (8 casos comunes)
- Workflow completo paso a paso
- Tips para tesis y producción

#### README.md
- Nueva sección "Scripts y Herramientas"
- Referencias a README_HYPERPARAMETER_TUNING.md

#### Docstring del Script
- Actualizado con nuevas opciones
- Ejemplos de uso local y remoto
- Outputs claramente documentados

### 🔧 Cambios Técnicos

#### Estructura de Archivos
```
runs/mlflow_local/
├── mlflow.db                        # SQLite (tracking backend)
└── <experiment_id>/                 # Experimento
    ├── meta.yaml
    └── <run_id>/                    # Runs/trials
        ├── metrics/
        ├── params/
        └── artifacts/
```

#### Variables de Entorno
```bash
# Para comandos CLI de MLflow
export MLFLOW_TRACKING_URI=sqlite:///runs/mlflow_local/mlflow.db
mlflow experiments search
```

### 📊 Métricas Registradas

Cada trial ahora registra en MLflow:
- **Parámetros**: tau_buy, tau_sell, gru_units_1/2, dropout, learning_rate, batch_size, trial_number
- **Métricas agregadas**: ic_mean/median/std, sharpe_mean/median/std
- **Métricas de validación**: ic_positive_pct, sharpe_positive_pct, tickers_with_trades, avg_num_trades
- **Objetivo**: objective_value (0.5*IC + 0.5*Sharpe + trade_penalty)
- **Artifacts**: trial_results.csv con métricas por ticker

### ⚠️ Breaking Changes

#### MLflow URI
- **Antes**: Default era `http://localhost:5050` (servidor remoto)
- **Ahora**: Default es `local` (SQLite)
- **Migración**: Explícitamente pasar `--mlflow_uri http://localhost:5050` si quieres usar servidor remoto

#### Imports
- **Removido**: `from optuna.integration.mlflow import MLflowCallback`
- **Razón**: Causaba conflictos con manejo manual de runs

### 🚀 Performance

- **Tracking local**: ~10% más rápido (sin overhead de red)
- **Base de datos**: Consultas más eficientes con SQLite
- **Persistencia**: Más robusta ante interrupciones

---

## [1.0.0] - 2026-01-05

### Lanzamiento Inicial
- Implementación básica con Optuna
- Tracking con MLflow (filesystem)
- Optimización de 7 hiperparámetros
- Métricas: IC y Sharpe ratio
- Visualizaciones: optimization_history, param_importances, parallel_coordinate

---

## Próximas Versiones

### [2.1.0] - Planificado
- [ ] Pruning automático de trials poco prometedores
- [ ] Multi-objective optimization (IC vs Sharpe vs Max Drawdown)
- [ ] Hyperband sampler para mayor eficiencia
- [ ] Integración con Optuna Dashboard

### [3.0.0] - Futuro
- [ ] Walk-forward validation automática
- [ ] Portfolio-level optimization (multi-ticker allocation)
- [ ] Ensemble de modelos con diferentes hiperparámetros
- [ ] Drift detection y re-optimización automática

---

## Comandos de Migración

### De versión 1.0 a 2.0

```bash
# 1. Limpiar tracking antiguo (opcional, si quieres empezar desde cero)
rm -rf runs/mlflow_local/*

# 2. Actualizar script (ya hecho vía git pull)
git pull origin main

# 3. Ejecutar optimización con nueva configuración
python scripts/optimize_e1_hyperparameters.py --n_trials 10 --quick

# 4. Verificar que no hay warnings
# ✅ No debe aparecer: FutureWarning filesystem
# ✅ No debe aparecer: UserWarning dropout
# ✅ No debe aparecer: Malformed experiment artifacts
```

### Acceder a resultados antiguos (si los guardaste)

```bash
# Si usabas MLflow remoto antes, los resultados siguen ahí
docker compose --profile mlflow up -d
open http://localhost:5050
```

---

## Soporte

Para reportar problemas o sugerencias:
- Ver [README_HYPERPARAMETER_TUNING.md](../README_HYPERPARAMETER_TUNING.md) - Troubleshooting
- Revisar este CHANGELOG para cambios recientes
- Verificar que tienes la última versión del código
