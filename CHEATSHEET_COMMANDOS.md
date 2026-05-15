# Cheat Sheet — Dashboard y Lifecycle

Comandos listos para copiar y pegar para monitoreo, reportes y promoción de modelos.

## 1. Dashboard

### Levantar MLflow UI local

```bash
# Inicia el servidor web de MLflow leyendo experimentos del archivo SQLite local (puerto 5050)
mlflow ui --backend-store-uri sqlite:///runs/mlflow_local/mlflow.db --port 5050
# Abre el navegador en la UI de MLflow para explorar runs, métricas e hiperparámetros
open http://localhost:5050
```

### Levantar dashboard con Docker

```bash
# Levanta en background el perfil "dashboard" del docker-compose (MLflow + MinIO/S3)
docker compose --profile dashboard up -d
# Abre la UI de MLflow corriendo en el contenedor
open http://localhost:5050
# Abre la consola web de MinIO (almacenamiento de artefactos S3-compatible)
open http://localhost:9001
```

### Dashboard Checker CLI — vistas principales

```bash
# Vista resumen global: muestra champion/candidate/baseline de todas las estrategias y tickers
python -m src.dashboard.checker
# Vista resumen filtrada solo para la estrategia e1_conservative
python -m src.dashboard.checker --strategy e1_conservative
# Vista per-ticker de e1_conservative: una fila por ticker con sus métricas del champion
python -m src.dashboard.checker --view ticker --strategy e1_conservative
# Igual que el anterior pero muestra solo los últimos 3 runs por ticker
python -m src.dashboard.checker --view ticker --strategy e1_conservative --last 3
# Vista historial: muestra todos los runs registrados de e1_conservative para el ticker AAPL
python -m src.dashboard.checker --view history --strategy e1_conservative --ticker AAPL
# Igual que el anterior pero limitado a los últimos 5 runs del ticker AAPL
python -m src.dashboard.checker --view history --strategy e1_conservative --ticker AAPL --last 5
```

### Strategy keys válidas para el checker

```bash
# Variante E1 simple (retirada, referencia histórica)
e1_simple
# Variante E1 conservadora con GRU [64,32] y horizonte 90 días — champion actual de E1
e1_conservative
# Baseline de E2: regresión lineal usada como piso de comparación
e2_baseline
# Variante E2 simple (retirada)
e2_simple
# Variante E2 moderada con LSTM [128,64] y horizonte 20 días — champion actual de E2
e2_moderate
# Baseline de E3: modelo de referencia para intradía
e3_baseline
# Variante E3 intradía con ensemble LSTM, horizonte 30 min, barras de 5 min
e3_intraday
# Variante E4 de pairs trading con k-NN + Ornstein-Uhlenbeck
e4_pairs
```

### Checker — ejemplos rápidos por estrategia

```bash
# Resumen de todos los tickers registrados bajo e1_simple
python -m src.dashboard.checker --strategy e1_simple
# Resumen de todos los tickers registrados bajo e1_conservative
python -m src.dashboard.checker --strategy e1_conservative
# Resumen de todos los tickers registrados bajo e2_baseline
python -m src.dashboard.checker --strategy e2_baseline
# Resumen de todos los tickers registrados bajo e2_simple
python -m src.dashboard.checker --strategy e2_simple
# Resumen de todos los tickers registrados bajo e2_moderate
python -m src.dashboard.checker --strategy e2_moderate
# Resumen de todos los tickers registrados bajo e3_baseline
python -m src.dashboard.checker --strategy e3_baseline
# Resumen de todos los tickers registrados bajo e3_intraday
python -m src.dashboard.checker --strategy e3_intraday
# Resumen de todos los tickers (o pares) registrados bajo e4_pairs
python -m src.dashboard.checker --strategy e4_pairs
```

### Vista per-ticker por estrategia

```bash
# Una fila por ticker con métricas del champion actual de e1_simple
python -m src.dashboard.checker --view ticker --strategy e1_simple
# Una fila por ticker con métricas del champion actual de e1_conservative
python -m src.dashboard.checker --view ticker --strategy e1_conservative
# Una fila por ticker con métricas del champion actual de e2_baseline
python -m src.dashboard.checker --view ticker --strategy e2_baseline
# Una fila por ticker con métricas del champion actual de e2_simple
python -m src.dashboard.checker --view ticker --strategy e2_simple
# Una fila por ticker con métricas del champion actual de e2_moderate
python -m src.dashboard.checker --view ticker --strategy e2_moderate
# Una fila por ticker con métricas del champion actual de e3_baseline
python -m src.dashboard.checker --view ticker --strategy e3_baseline
# Una fila por ticker con métricas del champion actual de e3_intraday
python -m src.dashboard.checker --view ticker --strategy e3_intraday
# Una fila por par con métricas del champion actual de e4_pairs
python -m src.dashboard.checker --view ticker --strategy e4_pairs
```

### Historial por estrategia y ticker

```bash
# Todos los runs históricos de e1_simple para el ticker AAPL
python -m src.dashboard.checker --view history --strategy e1_simple --ticker AAPL
# Todos los runs históricos de e1_conservative para el ticker AAPL
python -m src.dashboard.checker --view history --strategy e1_conservative --ticker AAPL
# Todos los runs históricos de e2_baseline para NVDA
python -m src.dashboard.checker --view history --strategy e2_baseline --ticker NVDA
# Todos los runs históricos de e2_simple para NVDA
python -m src.dashboard.checker --view history --strategy e2_simple --ticker NVDA
# Todos los runs históricos de e2_moderate para NVDA
python -m src.dashboard.checker --view history --strategy e2_moderate --ticker NVDA
# Todos los runs históricos de e3_baseline para SPY
python -m src.dashboard.checker --view history --strategy e3_baseline --ticker SPY
# Todos los runs históricos de e3_intraday para SPY
python -m src.dashboard.checker --view history --strategy e3_intraday --ticker SPY
```

## 2. Lifecycle

### Promoción de candidatos — genérico

```bash
# Modo dry-run: evalúa todos los candidates contra su champion y muestra quién sería promovido (sin cambiar el registry)
python -m scripts.evaluation.promote_candidate
# Ejecuta realmente la promoción: escribe los cambios en registry.json para los candidates que superan el umbral
python -m scripts.evaluation.promote_candidate --execute
# Dry-run con detalle extendido de métricas y scores de cada candidate
python -m scripts.evaluation.promote_candidate --verbose
# Muestra las últimas 20 entradas del log de auditoría de promociones (models/promotion_log.jsonl)
python -m scripts.evaluation.promote_candidate --show-log 20
# Ejecuta la promoción solo para los tickers AAPL y MSFT
python -m scripts.evaluation.promote_candidate --tickers AAPL,MSFT --execute
# Fuerza la promoción de YPFD.BA aunque no supere el umbral mínimo de mejora del 5%
python -m scripts.evaluation.promote_candidate --tickers YPFD.BA --force --execute
```

### Promoción filtrada por estrategia

```bash
# Dry-run de promoción solo para los candidates de la familia E1 (e1_conservative, e1_simple)
python -m scripts.evaluation.promote_candidate --strategy e1
# Ejecuta la promoción para todos los candidates de la familia E1
python -m scripts.evaluation.promote_candidate --strategy e1 --execute
# Dry-run de promoción solo para los candidates de la familia E2
python -m scripts.evaluation.promote_candidate --strategy e2
# Ejecuta la promoción para todos los candidates de la familia E2
python -m scripts.evaluation.promote_candidate --strategy e2 --execute
# Dry-run de promoción solo para los candidates de la familia E3
python -m scripts.evaluation.promote_candidate --strategy e3
# Ejecuta la promoción para todos los candidates de la familia E3
python -m scripts.evaluation.promote_candidate --strategy e3 --execute
# Dry-run de promoción solo para los candidates de la familia E4
python -m scripts.evaluation.promote_candidate --strategy e4
# Ejecuta la promoción para todos los candidates de la familia E4
python -m scripts.evaluation.promote_candidate --strategy e4 --execute
```

### Promoción filtrada por estrategia y tickers

```bash
# Ejecuta la promoción de E1 solo para los tickers AAPL y MSFT
python -m scripts.evaluation.promote_candidate --strategy e1 --tickers AAPL,MSFT --execute
# Ejecuta la promoción de E2 solo para NVDA y GOOGL
python -m scripts.evaluation.promote_candidate --strategy e2 --tickers NVDA,GOOGL --execute
# Ejecuta la promoción de E3 solo para SPY y AAPL
python -m scripts.evaluation.promote_candidate --strategy e3 --tickers SPY,AAPL --execute
# Ejecuta la promoción de E4 para los pares GGAL.BA y BMA.BA
python -m scripts.evaluation.promote_candidate --strategy e4 --tickers GGAL.BA,BMA.BA --execute
```

### Skills / comandos interactivos documentados

```text
# Muestra el estado actual (champion / candidate / baseline) del registry para todas las estrategias y tickers
/model-status
# Compara métricas de champion vs candidate vs baseline para la estrategia E1 y el ticker AAPL
/compare-models e1 AAPL
# Promueve manualmente el candidate de E1/AAPL a champion en el registry
/promote-model e1 AAPL
# Retira (mueve a retired[]) el modelo de E1/AAPL que está actualmente como champion
/retire-model e1 AAPL
# Lanza un nuevo run de entrenamiento E1 con una arquitectura experimental llamada "attention_gru"
/new-experiment e1 attention_gru
```

## 3. Entrenar y alimentar Dashboard/Lifecycle

### E1 — Conservative / Baseline / Simple

```bash
# Entrena e1_conservative para todos los tickers del universo definido en base.yaml
python -m src.e1.train_all
# Entrena e1_conservative solo para AAPL y MSFT
python -m src.e1.train_all --tickers AAPL,MSFT

# Entrena e1_conservative para todos los tickers del universo (sin refrescar datos)
python -m src.e1.train_pipeline
# Entrena e1_conservative solo para AAPL
python -m src.e1.train_pipeline --tickers AAPL
# Descarga datos frescos de yfinance antes de entrenar para AAPL y MSFT
python -m src.e1.train_pipeline --tickers AAPL,MSFT --refresh-data
# Entrena AAPL y promueve automáticamente el candidate a champion si supera el umbral
python -m src.e1.train_pipeline --tickers AAPL --auto-promote

# Entrena el baseline de E1 (Linear Regression) para AAPL — fija el piso de comparación
python -m src.e1.train_baseline --tickers AAPL

# Entrena la variante e1_simple (GRU sin búsqueda de umbral) para AAPL
python -m src.e1.train_simple_pipeline --tickers AAPL
# Entrena e1_simple para todo el universo y promueve automáticamente si mejora
python -m src.e1.train_simple_pipeline --auto-promote
```

### E2 — Moderate / Baseline

```bash
# Entrena e2_moderate para cada ticker del universo (sin refrescar datos)
python -m src.e2.train_pipeline
# Entrena e2_moderate solo para NVDA y GOOGL
python -m src.e2.train_pipeline --tickers NVDA,GOOGL
# Descarga datos frescos de yfinance antes de entrenar para NVDA y GOOGL
python -m src.e2.train_pipeline --tickers NVDA,GOOGL --refresh-data
# Entrena NVDA y GOOGL y promueve automáticamente los candidates que superen el umbral
python -m src.e2.train_pipeline --tickers NVDA,GOOGL --auto-promote

# Entrena el baseline de E2 (Linear Regression) para NVDA — fija el piso de comparación
python -m src.e2.train_baseline --tickers NVDA
```

### E3 — Intraday / Baseline

```bash
# Solo descarga datos intradía (barras de 5 min) para el universo completo sin entrenar
python -m src.e3.train_pipeline --mode download
# Descarga datos intradía para los tickers SPY, AAPL, NVDA y QQQ
python -m src.e3.train_pipeline --mode download --tickers SPY,AAPL,NVDA,QQQ

# Entrena e3_intraday para todos los tickers del universo (datos ya descargados)
python -m src.e3.train_pipeline
# Entrena e3_intraday solo para SPY y AAPL
python -m src.e3.train_pipeline --tickers SPY,AAPL
# Descarga datos intradía frescos hasta hoy y entrena e3_intraday (rango extendido)
python -m src.e3.train_pipeline --tickers SPY,AAPL --use-latest-data
# Entrena SPY y AAPL y promueve automáticamente los candidates que superen el composite score
python -m src.e3.train_pipeline --tickers SPY,AAPL --auto-promote
# Combina datos frescos + auto-promoción para SPY
python -m src.e3.train_pipeline --tickers SPY --use-latest-data --auto-promote

# Entrena el baseline de E3 para SPY (lee CSVs ya descargados en data/raw/intraday)
python -m src.e3.train_baseline --tickers SPY
# Entrena el baseline de E3 para SPY, AAPL y NVDA
python -m src.e3.train_baseline --tickers SPY,AAPL,NVDA
# Entrena el baseline de E3 con rango extendido hasta hoy (no descarga; correr antes intraday_data si hace falta)
python -m src.e3.train_baseline --tickers SPY --use-latest-data
# Entrena el baseline de E3 con un directorio de salida custom
python -m src.e3.train_baseline --tickers SPY --output-dir runs/e3_baseline/manual_test
```

### E4 — Pairs Trading

```bash
# Entrena e4_pairs para todos los pares definidos en base.yaml
python -m src.e4.train_pipeline
# Entrena solo los pares GGAL.BA/BMA.BA y YPFD.BA/PAMP.BA
python -m src.e4.train_pipeline --pairs GGAL.BA,BMA.BA YPFD.BA,PAMP.BA
# Entrena con un tag personalizado en el directorio de salida (útil para experimentos)
python -m src.e4.train_pipeline --output-tag my_test
```

## 4. Optimización

### E1 — Optuna

```bash
# Búsqueda rápida de hiperparámetros con solo 10 trials y configuración reducida (quick mode)
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 10 --quick
# Búsqueda completa con 50 trials usando todos los tickers del universo de E1
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 50
# Búsqueda con 30 trials restringida al ticker AAPL
python scripts/optimization/optimize_e1_hyperparameters.py --ticker AAPL --n_trials 30
# Búsqueda con 20 trials para los tickers AAPL, MSFT y JNJ (promedia el objetivo)
python scripts/optimization/optimize_e1_hyperparameters.py --tickers AAPL,MSFT,JNJ --n_trials 20
# Búsqueda per-ticker: optimiza hiperparámetros individualmente para cada ticker (30 trials cada uno)
python scripts/optimization/optimize_e1_hyperparameters.py --per_ticker --n_trials 30
# Reanuda o crea el study con nombre explícito "e1_hyperparameter_optimization" (20 trials adicionales)
python scripts/optimization/optimize_e1_hyperparameters.py --study_name e1_hyperparameter_optimization --n_trials 20
# Lanza la búsqueda con 50 trials y loguea los resultados en el servidor MLflow externo
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 50 --mlflow_uri http://localhost:5050
```

### E2 — Optuna

```bash
# Búsqueda rápida de hiperparámetros con solo 10 trials y configuración reducida (quick mode)
python scripts/optimization/optimize_e2_hyperparameters.py --n_trials 10 --quick
# Búsqueda completa con 50 trials usando todos los tickers del universo de E2
python scripts/optimization/optimize_e2_hyperparameters.py --n_trials 50 // NO USAR
# Búsqueda con 30 trials restringida al ticker NVDA
python scripts/optimization/optimize_e2_hyperparameters.py --ticker NVDA --n_trials 30
# Búsqueda con 20 trials para NVDA, GOOGL y AMZN
python scripts/optimization/optimize_e2_hyperparameters.py --tickers NVDA,GOOGL,AMZN --n_trials 20
# Búsqueda per-ticker: optimiza hiperparámetros individualmente para cada ticker (30 trials cada uno)
python scripts/optimization/optimize_e2_hyperparameters.py --per_ticker --n_trials 30
# Reanuda o crea el study con nombre explícito (20 trials adicionales)
python scripts/optimization/optimize_e2_hyperparameters.py --study_name e2_hyperparameter_optimization_timesplit --n_trials 20
# Lanza la búsqueda con 50 trials y loguea los resultados en el servidor MLflow externo
python scripts/optimization/optimize_e2_hyperparameters.py --n_trials 50 --mlflow_uri http://localhost:5050
```

### E3 — Optuna

```bash
# Búsqueda rápida de hiperparámetros con solo 10 trials y configuración reducida (quick mode, 2 tickers aleatorios)
python scripts/optimization/optimize_e3_hyperparameters.py --n_trials 10 --quick
# Búsqueda completa con 50 trials usando todos los tickers del universo de E3
python scripts/optimization/optimize_e3_hyperparameters.py --n_trials 50
# Búsqueda con 30 trials restringida al ticker SPY
python scripts/optimization/optimize_e3_hyperparameters.py --ticker SPY --n_trials 30
# Búsqueda con 20 trials para SPY, AAPL y NVDA (promedia el objetivo)
python scripts/optimization/optimize_e3_hyperparameters.py --tickers SPY,AAPL,NVDA --n_trials 20
# Búsqueda per-ticker: optimiza hiperparámetros individualmente para cada ticker (30 trials cada uno)
python scripts/optimization/optimize_e3_hyperparameters.py --per_ticker --n_trials 30
# Reanuda o crea el study con nombre explícito (20 trials adicionales)
python scripts/optimization/optimize_e3_hyperparameters.py --study_name e3_hyperparameter_optimization --n_trials 20
# Búsqueda con 30 trials usando time_split en lugar de walk_forward dentro de los trials
python scripts/optimization/optimize_e3_hyperparameters.py --n_trials 30 --validation_method time_split
# Búsqueda walk_forward con 4 folds en lugar del default de la config
python scripts/optimization/optimize_e3_hyperparameters.py --n_trials 30 --validation_method walk_forward --optuna_folds 4
# Búsqueda con 30 trials descargando datos frescos hasta hoy en cada trial
python scripts/optimization/optimize_e3_hyperparameters.py --n_trials 30 --use-latest-data
# Lanza la búsqueda con 50 trials y loguea los resultados en el servidor MLflow externo
python scripts/optimization/optimize_e3_hyperparameters.py --n_trials 50 --mlflow_uri http://localhost:5050
```

### Reanudar / limitar búsqueda

```bash
# Reanuda el study de E1 existente y lo detiene automáticamente tras 3600 segundos (1 hora)
python scripts/optimization/optimize_e1_hyperparameters.py --study_name e1_hyperparameter_optimization --timeout 3600
# Reanuda el study de E2 existente y lo detiene automáticamente tras 3600 segundos (1 hora)
python scripts/optimization/optimize_e2_hyperparameters.py --study_name e2_hyperparameter_optimization_timesplit --timeout 3600
# Reanuda el study de E3 existente y lo detiene automáticamente tras 3600 segundos (1 hora)
python scripts/optimization/optimize_e3_hyperparameters.py --study_name e3_hyperparameter_optimization --timeout 3600
```

### Ajustar rangos de búsqueda

```bash
# Búsqueda de E1 con 20 trials, acotando el espacio: tau_buy entre 0.03-0.08 y dropout entre 0.20-0.40
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 20 --tau_buy_min 0.03 --tau_buy_max 0.08 --dropout_min 0.20 --dropout_max 0.40
# Búsqueda de E2 con 20 trials, acotando el espacio: tau_buy entre 0.015-0.03 y dropout entre 0.10-0.25
python scripts/optimization/optimize_e2_hyperparameters.py --n_trials 20 --tau_buy_min 0.015 --tau_buy_max 0.03 --dropout_min 0.10 --dropout_max 0.25
# Búsqueda de E3 con 20 trials, acotando umbrales y arquitectura típica de intradía
python scripts/optimization/optimize_e3_hyperparameters.py --n_trials 20 --tau_buy_min 0.001 --tau_buy_max 0.003 --dropout_min 0.10 --dropout_max 0.30 --lstm_hidden_size_min 64 --lstm_hidden_size_max 128
# Búsqueda de E3 con 20 trials acotando ensemble (3-5 miembros) y consensus_tol
python scripts/optimization/optimize_e3_hyperparameters.py --n_trials 20 --ensemble_members_min 3 --ensemble_members_max 5 --consensus_tol_min 0.0005 --consensus_tol_max 0.0015
# Búsqueda de E3 con 20 trials y batch sizes restringidos a 128 y 256
python scripts/optimization/optimize_e3_hyperparameters.py --n_trials 20 --batch_sizes 128,256
```

### Consumir parámetros tuneados en training

```bash
# Indica al pipeline de E1 dónde leer los mejores hiperparámetros por ticker encontrados por Optuna
export E1_TUNED_PARAMS_PATH=reports/hyperparameter_optimization/e1_tuned_params_by_ticker.yaml
# Entrena E1 para AAPL y MSFT usando los hiperparámetros tuneados + datos frescos
python -m src.e1.train_pipeline --tickers AAPL,MSFT --refresh-data

# Indica al pipeline de E2 dónde leer los mejores hiperparámetros por ticker encontrados por Optuna
export E2_TUNED_PARAMS_PATH=reports/hyperparameter_optimization/e2_tuned_params_by_ticker.yaml
# Entrena E2 para NVDA y GOOGL usando los hiperparámetros tuneados + datos frescos
python -m src.e2.train_pipeline --tickers NVDA,GOOGL --refresh-data

# Path de tuneo per-ticker de E3 ya viene configurado en base.yaml (strategies.e3_intraday.tuned_params_path)
# Forzar override apuntando a un YAML alternativo:
export E3_TUNED_PARAMS_PATH=reports/hyperparameter_optimization/e3_tuned_params_by_ticker.yaml
# Entrena E3 para SPY y AAPL aplicando los hiperparámetros tuneados (datos ya descargados)
python -m src.e3.train_pipeline --tickers SPY,AAPL
# Entrena E3 para SPY y AAPL refrescando datos intradía hasta hoy
python -m src.e3.train_pipeline --tickers SPY,AAPL --use-latest-data
```

### Dashboard / inspección de Optuna

```bash
# Lanza el dashboard web de Optuna apuntando a la base de datos local de trials (puerto por defecto)
optuna-dashboard sqlite:///runs/optuna_trials/optuna_studies.db
# Script propio que lee la DB de Optuna y genera tablas y gráficos de análisis en reports/
python scripts/optimization/analyze_optuna_db.py
```

### Archivos útiles de optimización

```bash
# Muestra los mejores hiperparámetros globales encontrados para E1 (promedio sobre todos los tickers)
cat reports/hyperparameter_optimization/best_params_e1.yaml
# Muestra los mejores hiperparámetros globales encontrados para E2
cat reports/hyperparameter_optimization/best_params_e2.yaml
# Muestra los mejores hiperparámetros encontrados individualmente para cada ticker en E1
cat reports/hyperparameter_optimization/e1_tuned_params_by_ticker.yaml
# Muestra los mejores hiperparámetros encontrados individualmente para cada ticker en E2
cat reports/hyperparameter_optimization/e2_tuned_params_by_ticker.yaml
# Muestra los mejores hiperparámetros globales encontrados para E3
cat reports/hyperparameter_optimization/best_params_e3.yaml
# Muestra los mejores hiperparámetros encontrados individualmente para cada ticker en E3
cat reports/hyperparameter_optimization/e3_tuned_params_by_ticker.yaml
# Abre el directorio con los gráficos generados por Optuna (importance, pareto, etc.)
open reports/hyperparameter_optimization/figures/
# Muestra los últimos 20 trials registrados del CSV completo de E1 (para inspección rápida)
tail -n 20 reports/hyperparameter_optimization/e1_all_trials.csv
# Muestra los últimos 20 trials registrados del CSV completo de E2
tail -n 20 reports/hyperparameter_optimization/e2_all_trials.csv
# Muestra los últimos 20 trials registrados del CSV completo de E3
tail -n 20 reports/hyperparameter_optimization/e3_all_trials.csv
```

## 5. Flujo corto recomendado

### E1

```bash
# Paso 1: entrena e1_conservative para AAPL descargando datos frescos → genera un nuevo candidate en el registry
python -m src.e1.train_pipeline --tickers AAPL --refresh-data
# Paso 2: verifica en el dashboard que las métricas del candidate son razonables antes de promover
python -m src.dashboard.checker --strategy e1_conservative
# Paso 3: dry-run de promoción para ver si el candidate supera al champion (sin modificar el registry)
python -m scripts.evaluation.promote_candidate --strategy e1 --tickers AAPL --verbose
# Paso 4: ejecuta la promoción si el dry-run fue satisfactorio
python -m scripts.evaluation.promote_candidate --strategy e1 --tickers AAPL --execute
```

### E2

```bash
# Paso 1: entrena e2_moderate para NVDA y GOOGL descargando datos frescos → nuevos candidates
python -m src.e2.train_pipeline --tickers NVDA,GOOGL --refresh-data
# Paso 2: verifica métricas del candidate en el dashboard
python -m src.dashboard.checker --strategy e2_moderate
# Paso 3: dry-run de promoción para todos los candidates de E2
python -m scripts.evaluation.promote_candidate --strategy e2 --verbose
# Paso 4: ejecuta la promoción para todos los candidates de E2 que superen el umbral
python -m scripts.evaluation.promote_candidate --strategy e2 --execute
```

### E3

```bash
# Paso 1a: descarga datos intradía (barras 5 min) para SPY y AAPL
python -m src.e3.train_pipeline --mode download --tickers SPY,AAPL
# Paso 1b: entrena e3_intraday para SPY y AAPL con los datos descargados → nuevos candidates
python -m src.e3.train_pipeline --tickers SPY,AAPL
# Paso 2: verifica métricas del candidate en el dashboard
python -m src.dashboard.checker --strategy e3_intraday
# Paso 3: dry-run de promoción para todos los candidates de E3
python -m scripts.evaluation.promote_candidate --strategy e3 --verbose
# Paso 4: ejecuta la promoción para los candidates de E3 que superen el umbral
python -m scripts.evaluation.promote_candidate --strategy e3 --execute
```

## 6. Archivos de salida útiles

```bash
# Reporte global más reciente con champion/candidate/baseline de todas las estrategias y tickers
cat reports/dashboard/dashboard_report_latest.csv
# Reporte con una fila por ticker del champion actual de e1_conservative
cat reports/dashboard/ticker_report_e1_conservative_latest.csv
# Reporte histórico de todos los runs de e1_conservative para AAPL
cat reports/dashboard/history_report_e1_conservative_AAPL_latest.csv
# CSV de resumen de todos los runs de e1_conservative (un run por directorio con timestamp)
cat runs/e1_conservative/*/summary_all.csv
# CSV de resumen de todos los runs de e2_moderate
cat runs/e2_moderate/*/summary_all.csv
# CSV de resumen de todos los runs del baseline de E3
cat runs/e3_baseline/*/baseline_summary_all.csv
# Estado actual del registry: champions, candidates, baselines y retired de todas las estrategias
cat models/registry.json
# Últimas 20 decisiones de promoción con timestamp, estrategia, ticker y score (audit trail)
tail -n 20 models/promotion_log.jsonl
# Últimas 20 métricas registradas tras cada run (Sharpe, IC, DirectionalAcc, Calmar, etc.)
tail -n 20 models/metrics_log.jsonl
```

## 7. Evaluación y comparación de modelos (Conclusiones)

### Comparativos por estrategia (modelo vs baseline)

```bash
# Compara el champion E1 vs su baseline (auto-detecta los runs más recientes en runs/e1_*)
python -m scripts.evaluation.compare_e1_models
# Compara el champion E2 vs su baseline (auto-detecta los runs más recientes)
python -m scripts.evaluation.compare_e2_models
# Compara el champion E3 ensemble vs su baseline ridge (auto-detecta los runs más recientes)
python -m scripts.evaluation.compare_e3_models
# Comparación E1 forzando paths explícitos a un baseline-run y model-run específicos
python -m scripts.evaluation.compare_e1_models --baseline-run runs/e1_baseline/20251015_120000 --model-run runs/e1_conservative/20251020_153000
# Compara versiones históricas de E1 (e1_simple vs e1_conservative)
python -m scripts.evaluation.compare_e1_versions
```

### Comparativo cross-strategy

```bash
# Consolida métricas de baseline/champion/candidate desde registry.json a tablas chapter_4
python -m scripts.evaluation.consolidate_strategy_metrics
# Genera figuras cross-strategy (Sharpe boxplot, risk-return, IC vs Sharpe, dir-acc heatmap)
python -m scripts.evaluation.compare_strategies
# Análisis de estabilidad: evolución del champion en el tiempo + timeline de promociones
python -m scripts.evaluation.stability_analysis
# Valida los requisitos de la Iteración 1 contra config + registry + runs
python -m scripts.evaluation.requirements_validation
```

### Validación retrospectiva y guardrails

```bash
# Validación retrospectiva del champion de E1 (replay sobre histórico)
python -m scripts.evaluation.e1_retrospective_validation
# Aplica guardrails Phase 1 sobre los summaries más recientes de E1
python -m scripts.evaluation.e1_simple_guardrails
```

### Figuras de conclusiones 

```bash
# Genera las 8 figuras de nivel de conclusión leyendo predictions.csv y backtest.csv de cada champion
# (no re-entrena modelos — solo lee los CSVs ya existentes en runs/)
python -m scripts.evaluation.plot_conclusions

# Figuras generadas en reports/conclusiones/figuras/:
#   fig_conclusions_roc.png              — Curvas ROC + AUC por ticker, 1 panel por estrategia (E1/E2/E3)
#   fig_conclusions_confusion.png        — Matriz de confusión binaria (Baja/Sube), pooled por estrategia
#   fig_conclusions_underwater.png       — Gráfico underwater de drawdown, overlay de tickers + media en rojo
#   fig_conclusions_rolling_ic.png       — IC Spearman rolling (ventana 30/60/100 según estrategia)
#   fig_conclusions_quantile_returns.png — Retorno real medio por quintil de predicción (Q1 a Q5)
#   fig_conclusions_monthly_heatmap_e1.png — Heatmap de retornos mensuales E1 (promedio 10 tickers)
#   fig_conclusions_monthly_heatmap_e2.png — Heatmap de retornos mensuales E2 (promedio 11 tickers)
#   fig_conclusions_monthly_heatmap_e3.png — Heatmap de retornos mensuales E3 intraday (4 tickers)

# Abre el directorio de figuras de conclusiones en el Finder
open reports/conclusiones/figuras/
```

## 8. Tests

```bash
# Corre toda la suite de tests
pytest tests/
# Suite completa con verbosidad y output de prints
pytest tests/ -v -s
# Tests del registry (escritura atómica, race conditions de tempfile + os.replace)
pytest tests/test_registry_atomicity.py -v
# Tests de paths del lifecycle (resolución de rutas runs/<variant>/<ts>/<TICKER>/)
pytest tests/test_lifecycle_paths.py -v
# Tests de promoción filtrada por estrategia (composite score + threshold de 5%)
pytest tests/test_promotion_per_strategy.py -v
# Tests de los guardrails Phase 1 (NaN/Inf, Sharpe<0, IC peor que baseline)
pytest tests/test_guardrails.py -v
# Tests de splits walk-forward (sin overlap, embargo >= horizon, orden temporal)
pytest tests/test_walkforward_splits.py -v
# Tests de Z-score: estadísticas calculadas solo en train (no data snooping)
pytest tests/test_feature_scaling.py -v
# Tests de costos diarios en backtest (commissions, slippage, spread)
pytest tests/test_backtest_costs.py -v
# Corre un test específico dentro de una clase
pytest tests/test_walkforward_splits.py::TestEmbargo -v
# Corre tests con coverage report (requiere pytest-cov)
pytest tests/ --cov=src --cov-report=term-missing
# Corre solo los tests cuyo nombre matchee un patrón
pytest tests/ -k "registry or guardrails" -v
```

## 9. Notas rápidas

- `src.dashboard.checker --view history` exige `--strategy` y `--ticker`.
- `scripts.evaluation.promote_candidate` usa `--strategy e1` por default si no se especifica `--strategy`.
- En dashboard, las claves válidas de estrategia son las de `src/config/dashboard_thresholds.yaml`.
- En lifecycle, el filtro `--strategy` usa prefijos de registry: `e1`, `e2`, `e3`, `e4`.
