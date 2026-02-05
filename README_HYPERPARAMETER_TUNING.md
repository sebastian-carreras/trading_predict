# Optimización de Hiperparámetros E1 con Optuna + MLflow

Sistema completo de búsqueda automática de hiperparámetros óptimos para la estrategia E1 Conservative.

## 📋 Tabla de Contenidos

- [Descripción](#descripción)
- [Configuración Inicial](#configuración-inicial)
- [Hiperparámetros Optimizados](#hiperparámetros-optimizados)
- [Métrica Objetivo](#métrica-objetivo)
- [Uso](#uso)
- [Orquestación con Airflow](#orquestación-con-airflow)
- [Interpretación de Resultados](#interpretación-de-resultados)
- [Visualizaciones](#visualizaciones)
- [MLflow Tracking](#mlflow-tracking)

---

## Descripción

El script `optimize_e1_hyperparameters.py` implementa **búsqueda bayesiana** de hiperparámetros usando:

- **Optuna**: Framework de optimización automática con Tree-structured Parzen Estimator (TPE)
- **MLflow**: Tracking de experimentos con backend SQLite (local) o servidor remoto
- **Validación robusta**: Entrenamiento en todos los tickers de E1 para asegurar generalización

### Ventajas sobre Grid Search

| Método | Grid Search | Optuna (Bayesian) |
|--------|-------------|-------------------|
| **Eficiencia** | Prueba todas las combinaciones | Aprende de trials previos |
| **Trials necesarios** | ~1000 para 5 parámetros | ~50 para convergencia |
| **Adaptabilidad** | Fija | Explora zonas prometedoras |
| **Early stopping** | No | Sí (pruning) |

**Ejemplo**: Con 5 hiperparámetros y 10 valores cada uno:
- Grid Search: 10^5 = **100,000 trials** 
- Optuna: ~**50 trials** para encontrar óptimo 

---

## Configuración Inicial

### Requisitos

```bash
# Instalar dependencias
pip install optuna optuna-dashboard kaleido plotly boto3
```

### Modos de Ejecución

El script soporta dos modos de tracking MLflow:

#### 1. Modo Local (Recomendado para desarrollo)

```bash
# Usa SQLite local - NO requiere Docker
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 50
```

**Características**:
- No requiere servicios externos
- Tracking en `runs/mlflow_local/mlflow.db` (SQLite)
- Artifacts en `runs/mlflow_local/<experiment_id>/`
- Sin warnings de deprecación
- Más rápido (sin overhead de red)

#### 2. Modo Remoto (Servidor MLflow en Docker)

```bash
# Usa servidor MLflow remoto
python scripts/optimization/optimize_e1_hyperparameters.py \
  --n_trials 50 \
  --mlflow_uri http://localhost:5050
```

**Características**:
- 🔧 Requiere MLflow server corriendo (Docker)
- UI más completa en http://localhost:5050
- 🌐 Compartir resultados en equipo
- ☁ Backend S3/MinIO para artifacts

**Iniciar servidor MLflow (si usas modo remoto)**:
```bash
docker compose --profile mlflow up -d
```

---

## Hiperparámetros Optimizados

### 1. Trading Thresholds (Críticos para P&L)

```python
tau_buy:  0.02 → 0.10  # Umbral para señal de COMPRA
tau_sell: -0.02 → 0.02  # Umbral para señal de VENTA
```

**Impacto**:
- `tau_buy` muy bajo → muchos trades, comisiones altas
- `tau_buy` muy alto → pocas señales, oportunidades perdidas
- `tau_sell` negativo → vende en predicción de caída

**Ejemplo**:
```
tau_buy = 0.04 (4%)
→ Compra solo si predice ganancia > 4% en 90 días
→ Conservador pero con alta convicción

tau_sell = -0.01 (-1%)
→ Vende si predice caída > 1%
→ Salida rápida ante señales bajistas
```

### 2. Arquitectura GRU

```python
gru_units_1: 32 → 128   # Neuronas en primera capa
gru_units_2: 16 → 64    # Neuronas en segunda capa
```

**Impacto**:
- Más neuronas → mayor capacidad, riesgo de overfitting
- Menos neuronas → más rápido, riesgo de underfitting

**Rango típico**:
- Modelos simples: [32, 16]
- Modelos complejos: [128, 64]

### 3. Regularización

```python
dropout: 0.2 → 0.5  # Proporción de neuronas desactivadas
```

**Impacto**:
- Dropout bajo → memoriza entrenamiento (overfitting)
- Dropout alto → generaliza mejor pero aprende más lento

### 4. Entrenamiento

```python
learning_rate: 1e-4 → 1e-2  # Tasa de aprendizaje (log scale)
batch_size: [32, 64, 128]   # Tamaño de batch
```

**Impacto learning_rate**:
- Alto (1e-2) → aprende rápido pero inestable
- Bajo (1e-4) → aprende lento pero estable

**Impacto batch_size**:
- 32 → más actualizaciones, más ruidoso
- 128 → menos actualizaciones, más estable

---

## Métrica Objetivo

El optimizador maximiza una **métrica compuesta**:

```python
objective = 0.5 * IC_mean + 0.5 * Sharpe_mean + trade_penalty
```

### Componentes

**1. IC (Information Coefficient)** - Calidad de predicción
- Rango: -1 a +1
- IC > 0.05 = predicción útil
- IC > 0.20 = predicción excelente

**2. Sharpe Ratio** - Performance de trading
- Rango: -∞ a +∞
- Sharpe > 1.0 = estrategia buena
- Sharpe > 2.0 = estrategia excelente

**3. Trade Penalty** - Evitar thresholds irreales
- Penaliza si <50% de tickers generan trades
- Evita tau_buy muy alto (ej: 10%) que no opera nunca

### ¿Por qué esta métrica?

| Solo IC | Solo Sharpe | IC + Sharpe (USADO) |
|---------|-------------|---------------------|
| Buena predicción | Buen trading | Balance óptimo |
| Pero puede no tradear | Pero predicción pobre | Predice bien Y tradea bien |
|  |  |  |

---

## Uso

### Comandos Principales

#### 1. Optimización Estándar (50 trials, modo local)

```bash
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 50
```

**Tiempo estimado**: ~3 horas (depende de hardware y número de tickers)

**Output**:
- `reports/hyperparameter_optimization/best_params_e1.yaml`
- `reports/hyperparameter_optimization/e1_all_trials.csv`
- `reports/hyperparameter_optimization/e1_optimization_summary.txt`
- `reports/hyperparameter_optimization/figures/*.png`
- MLflow tracking: `runs/mlflow_local/mlflow.db` (SQLite)
- Optuna database: `optuna_studies.db`

#### 2. Modo Rápido (Prueba con 3 tickers)

```bash
# Solo 3 tickers aleatorios, 10 trials
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 10 --quick
```

**Tiempo estimado**: ~20 minutos

**Uso**: Validar que funciona antes de optimización completa

#### 3. Optimizar Ticker Específico

```bash
# Optimizar solo para AAPL (útil para análisis individual)
python scripts/optimization/optimize_e1_hyperparameters.py --ticker AAPL --n_trials 30
```

**Tiempo estimado**: ~30 minutos (1 ticker)

**Uso**: Tunear estrategia para un activo específico

#### 4. Continuar Optimización Existente

```bash
# Agregar 20 trials más a estudio existente
python scripts/optimization/optimize_e1_hyperparameters.py \
  --study_name e1_hyperparameter_optimization \
  --n_trials 20
```

**Nota**: Optuna guarda progreso en `optuna_studies.db`, puedes interrumpir y continuar después

#### 5. Con Timeout (Limitar Tiempo)

```bash
# Detener después de 2 horas (7200 segundos)
python scripts/optimization/optimize_e1_hyperparameters.py \
  --n_trials 100 \
  --timeout 7200
```

#### 6. Usar Servidor MLflow Remoto

```bash
# Conectar a servidor MLflow en Docker
python scripts/optimization/optimize_e1_hyperparameters.py \
  --n_trials 50 \
  --mlflow_uri http://localhost:5050
```

**Requiere**: `docker compose --profile mlflow up -d`

#### 7. Ajustar Rango de Búsqueda desde CLI

```bash
python scripts/optimization/optimize_e1_hyperparameters.py \
  --n_trials 40 \
  --tau_buy_min 0.03 \
  --tau_buy_max 0.07 \
  --dropout_max 0.40 \
  --batch_sizes 32,48,64 \
  --tickers AAPL,MSFT,GOOGL
```

**Uso**: Probar ventanas específicas cuando ya hay intuiciones sobre thresholds o arquitectura

### Todas las Opciones

```bash
python scripts/optimization/optimize_e1_hyperparameters.py --help
```

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--config` | `src/config/base.yaml` | Archivo de configuración |
| `--n_trials` | `50` | Número de trials a ejecutar |
| `--study_name` | `e1_hyperparameter_optimization` | Nombre del estudio Optuna |
| `--ticker` | `None` | Optimizar solo un ticker |
| `--tickers` | `None` | Lista de tickers separados por coma (overridea universo) |
| `--quick` | `False` | Modo rápido (3 tickers) |
| `--timeout` | `None` | Timeout en segundos |
| `--mlflow_uri` | `local` | URI de MLflow (`local` o `http://...`) |
| `--output_dir` | `reports/hyperparameter_optimization` | Directorio de salida |
| `--batch_sizes` | `None` | Opciones de batch (coma-separado) |

**Overrides del espacio de búsqueda**

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--tau_buy_min` / `--tau_buy_max` | `None` | Ajustar rango de compra |
| `--tau_sell_min` / `--tau_sell_max` | `None` | Ajustar rango de venta |
| `--dropout_min` / `--dropout_max` | `None` | Limitar dropout |
| `--learning_rate_min` / `--learning_rate_max` | `None` | Cambiar rango de learning rate |
| `--gru_units_1_min` / `--gru_units_1_max` | `None` | Rango de neuronas capa 1 |
| `--gru_units_2_min` / `--gru_units_2_max` | `None` | Rango de neuronas capa 2 |

---

## Orquestación con Airflow

La DAG [dockerfiles/airflow/dags/E1/e1_optuna_tuning.py](dockerfiles/airflow/dags/E1/e1_optuna_tuning.py) permite lanzar la optimización desde la UI de Airflow sin tocar consola.

- **Activación**: copia el archivo al volumen de DAGs del scheduler (`docker compose --profile airflow up -d` ya lo incluye). El proyecto se monta en `/opt/airflow` por defecto; ajusta la variable `TRADING_PREDICT_ROOT` si usas otra ruta.
- **Parámetros disponibles** (`Params` en la UI): config_path, n_trials, study_name, timeout_seconds, mlflow_uri, output_dir, tickers, single_ticker, quick_mode, batch_sizes y todos los overrides (`tau_*`, `dropout_*`, `learning_rate_*`, `gru_units_*`).
- **Comando reproducible**: la task `run_optuna_tuning` publica el comando exacto en XCom (`cli_command`) para replicar la ejecución desde terminal.
- **Uso recomendado**: seleccionar tickers desde la UI, limitar el rango de búsqueda cuando haya hipótesis concretas y abastecer `extra_cli_args` para flags avanzados.

---

## Interpretación de Resultados

### 1. Archivo `best_params_e1.yaml`

```yaml
optimization:
  study_name: e1_hyperparameter_optimization
  n_trials: 50
  best_value: 0.4523
  best_trial: 37

best_params:
  tau_buy: 0.04
  tau_sell: -0.01
  gru_units_1: 64
  gru_units_2: 32
  dropout: 0.35
  learning_rate: 0.0015
  batch_size: 64
```

**Interpretación**:
- Trial #37 fue el mejor de 50
- Objective value = 0.4523 (IC + Sharpe combinados)
- `tau_buy = 0.04` → estrategia conservadora (4% threshold)
- `dropout = 0.35` → regularización moderada
- `learning_rate = 0.0015` → aprendizaje medio

**Acción**: Actualizar `src/config/base.yaml` con estos parámetros

### 2. CSV `e1_all_trials.csv`

Columnas importantes:
- `number`: ID del trial
- `value`: Métrica objetivo
- `params_tau_buy`, `params_tau_sell`, etc.: Parámetros probados
- `state`: COMPLETE, PRUNED, FAIL
- `datetime_start`, `datetime_complete`: Timestamps
- `duration`: Tiempo de ejecución

**Análisis útil**:

```python
import pandas as pd

df = pd.read_csv('reports/hyperparameter_optimization/e1_all_trials.csv')

# Top 10 trials
print(df.nlargest(10, 'value'))

# Correlación entre parámetros y objetivo
params = ['params_tau_buy', 'params_dropout', 'params_learning_rate']
print(df[params + ['value']].corr()['value'])

# Rango de parámetros en top 10%
top_10pct = df.nlargest(int(len(df)*0.1), 'value')
for param in params:
    print(f"{param}: {top_10pct[param].min():.4f} - {top_10pct[param].max():.4f}")
```

### 3. MLflow Tracking

**Modo Local (SQLite)**:
```bash
# Ver base de datos directamente
sqlite3 runs/mlflow_local/mlflow.db

# Listar experimentos
MLFLOW_TRACKING_URI=sqlite:///runs/mlflow_local/mlflow.db \
  mlflow experiments search

# Iniciar UI local
mlflow ui --backend-store-uri sqlite:///runs/mlflow_local/mlflow.db --port 5001
# Abrir: http://localhost:5001
```

**Modo Remoto (Servidor Docker)**:
```bash
# Acceder directamente
open http://localhost:5050
```

**Métricas registradas por trial**:
- `ic_mean`: IC promedio en todos los tickers
- `ic_median`: Mediana de IC
- `ic_std`: Desviación estándar de IC
- `sharpe_mean`: Sharpe ratio promedio
- `sharpe_median`: Mediana de Sharpe
- `sharpe_std`: Desviación estándar de Sharpe
- `ic_positive_pct`: % de tickers con IC > 0.05
- `sharpe_positive_pct`: % de tickers con Sharpe > 0
- `tickers_with_trades`: Número de tickers que generan señales
- `avg_num_trades`: Promedio de trades por ticker
- `objective_value`: Métrica final optimizada

**Artifacts guardados**:
- `trial_results.csv`: Resultados detallados por ticker en cada trial

---

## MLflow Tracking

### Estructura de Datos

```
runs/mlflow_local/
├── mlflow.db                        # Base de datos SQLite (metadata)
└── <experiment_id>/                 # Directorio del experimento
    ├── meta.yaml                    # Configuración del experimento
    └── <run_id>/                    # Directorio de cada run/trial
        ├── meta.yaml                # Metadata del run
        ├── metrics/                 # Métricas (IC, Sharpe, etc.)
        ├── params/                  # Parámetros (tau_buy, dropout, etc.)
        ├── tags/                    # Tags del run
        └── artifacts/               # Archivos (trial_results.csv, etc.)
```

### Comandos Útiles

**Buscar runs**:
```bash
# Exportar variable de entorno
export MLFLOW_TRACKING_URI=sqlite:///runs/mlflow_local/mlflow.db

# Buscar runs del experimento
mlflow runs search --experiment-name E1_Hyperparameter_Optimization

# Listar todos los experimentos
mlflow experiments search
```

**Iniciar UI**:
```bash
# Modo local
mlflow ui --backend-store-uri sqlite:///runs/mlflow_local/mlflow.db --port 5001

# Modo remoto (ya corriendo en Docker)
open http://localhost:5050
```

**Comparar runs en la UI**:
1. Ir a experimento "E1_Hyperparameter_Optimization"
2. Seleccionar múltiples runs (checkbox)
3. Click en "Compare"
4. Ver Parallel Coordinates, Scatter Matrix, etc.

### Visualizaciones en MLflow UI

**Charts disponibles**:
- **Parallel Coordinates**: Ver relación entre todos los parámetros
- **Scatter Matrix**: Correlaciones entre parámetros y métricas
- **Line Charts**: Evolución temporal (si hay métricas por epoch)
- **Bar Charts**: Comparar métricas entre runs

**Filtros útiles**:
```
# Runs con Sharpe > 1.0
metrics.sharpe_mean > 1.0

# Runs con IC alto y Sharpe positivo
metrics.ic_mean > 0.15 and metrics.sharpe_mean > 0.5

# Parámetros específicos
params.tau_buy >= 0.04 and params.tau_buy <= 0.06

# Top 10% de runs
metrics.objective_value >= [percentile(objective_value, 90)]
```

---

## Visualizaciones

El script genera 3 gráficos automáticamente en `reports/hyperparameter_optimization/figures/`:

### 1. `optimization_history.png`

![Optimization History Example](https://via.placeholder.com/800x400?text=Optimization+History)

**Interpretación**:
- **Eje Y**: Valor objetivo
- **Eje X**: Número de trial
- **Azul**: Valor de cada trial
- **Rojo**: Mejor valor acumulado

**Qué buscar**:
- Línea roja estabiliza → convergencia
- Línea roja sigue subiendo → necesita más trials

### 2. `param_importances.png`

![Parameter Importances Example](https://via.placeholder.com/800x400?text=Parameter+Importances)

**Interpretación**: plotly
```

### Error: "No module named 'boto3'"

```bash
# Solo si usas MLflow remoto con S3/MinIO
pip install boto3
```

### Error: MLflow connection refused (modo remoto)

```bash
# Verificar que MLflow está corriendo
docker compose ps mlflow

# Si no está, iniciar
docker compose --profile mlflow up -d

# Verificar acceso
curl http://localhost:5050/health
```

### Warning: "FutureWarning: filesystem tracking backend"

 **Ya resuelto**: El script ahora usa SQLite por defecto (`sqlite:///runs/mlflow_local/mlflow.db`)

Si ves este warning, verifica que estés usando la versión actualizada del script.

### Warning: "Malformed experiment 'artifacts'"

```bash
# Limpiar directorio MLflow local y reiniciar
rm -rf runs/mlflow_local/* setup
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 10 --quick
```

**Resultado**: Valores aproximados en ~20 minutos
**Objetivo**: Verificar que todo funciona correctamente

### Paso 2: Optimización Completa

```bash
# 50 trials con todos los tickers (modo local)
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 50
```

**Resultado**: Parámetros óptimos en ~3 horas
**Persiste en**: `optuna_studies.db` + `runs/mlflow_local/mlflow.db`

### Paso 3: Analizar Resultados

```bash
# Ver gráficos generados
open reports/hyperparameter_optimization/figures/optimization_history.png
open reports/hyperparameter_optimization/figures/param_importances.png

# Ver mejores parámetros
cat reports/hyperparameter_optimization/best_params_e1.yaml

# Explorar en MLflow UI
mlflow ui --backend-store-uri sqlite:///runs/mlflow_local/mlflow.db --port 5001
# Abrir: http://localhost:5001
```

### Paso 4: Refinar si es Necesario

```bash
# Si no converge, agregar más trials al mismo estudio
python scripts/optimization/optimize_e1_hyperparameters.py \
  --study_name e1_hyperparameter_optimization \
  --n_trials 30
# Total trials: 50 + 30 = 80
```

### Paso 5: Aplicar Mejores Parámetros

Editar `src/config/base.yaml`:

```yaml
strategies:
  e1_conservative:
    thresholds:
      tau_buy: 0.04    # ← Del optimization
      tau_sell: -0.01  # ← Del optimization
    
    model:
      gru_units: [64, 32]  # ← Del optimization
      dropout: 0.35        # ← Del optimization
      learning_rate: 0.0015
      batch_size: 64
      early_stopping_patience: 10
```

### Paso 6: Validar Mejora

```bash
# Re-entrenar modelo con nuevos parámetros
python src/train_e1_pipeline.py

# Comparar métricas en MLflow
# Experimento "E1_Conservative" con parámetros antiguos vs nuevos
```

### Paso 7: Documentar Resultados (Tesis)

Para tu tesis, documenta:

1. **Espacio de búsqueda**: Rangos explorados
   ```yaml
   tau_buy: [0.02, 0.10]
   dropout: [0.2, 0.5]
   # etc.
   ```

2. **Convergencia**: Gráfico `optimization_history.png`
   - Mostrar que converge en ~50 trials

3. **Importancia de parámetros**: Gráfico `param_importances.png`
   - "tau_buy es el parámetro más crítico (45% importancia)"

4. **Mejora obtenida**:
   ```
   Parámetros default:
   - IC: 0.12, Sharpe: 0.45
   
   Parámetros optimizados:
   - IC: 0.18 (+50%), Sharpe: 0.78 (+73%)
   ```

5. **Robustez**: Análisis de top 10% trials
   Optuna Examples](https://github.com/optuna/optuna-examples)
- [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html)
- [Hyperparameter Optimization in ML](https://arxiv.org/abs/1502.02127)
- [Tree-structured Parzen Estimator](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)

---

## Próximos Pasos

Después de optimización de hiperparámetros, siguiente fase:

1.  **Aplicar parámetros óptimos** → Actualizar `base.yaml`
2.  **Walk-Forward Validation** → Validar robustez temporal
3. 🔍 **Feature Importance** → Identificar features más relevantes
4. 💼 **Portfolio Optimization** → Allocation multi-ticker óptimo
5.  **Ensemble Methods** → Combinar múltiples modelos
6.  **Risk Management** → Optimizar position sizing

Ver `README_E1.md` para roadmap completo del proyecto.
# Iniciar dashboard
optuna-dashboard sqlite:///optuna_studies.db

# Abrir: http://localhost:8080
```

**Ventajas del Dashboard**:
- Visualizaciones interactivas
- 🔄 Actualización en tiempo real (ver progreso mientras corre)
- Múltiples estudios en una vista
- Análisis de convergencia detallado
- 📋 Comparación entre estudios

---

## Tips de Optimización

### Para tu Tesis

1. **Documentar búsqueda**: Guardar `e1_all_trials.csv` y gráficos
2. **Reportar convergencia**: Mostrar que 50 trials es suficiente
3. **Analizar importancia**: Justificar qué parámetros tunear manualmente
4. **Validar robustez**: Top 10% trials tienen parámetros similares
5. **Comparar con baseline**: Mejora vs parámetros default

### Para Producción

1. **Re-optimizar periódicamente**: Cada 6 meses o cuando cambie el mercado
2. **Usar walk-forward**: Optimizar en ventana móvil
3. **Validar out-of-sample**: No usar mismos datos para optimizar y validar
4. **Considerar costos**: Incluir comisiones en Sharpe ratio
5. **Multi-objetivo**: Considerar también max drawdown, win rate

### Mejores Prácticas

 **Hacer**:
- Empezar con `--quick` para validar
- Usar seeds (el script ya usa `seed=42`)
- Guardar versión del código con git
- Documentar cambios en parámetros

 **Evitar**:
- Optimizar con muy pocos datos (<1 año)
- Usar mismos datos para entrenar y evaluar
- Cambiar parámetros manualmente sin re-optimizar
- Sobre-optimizar (>100 trials puede causar overfitting al proceso)
```bash
# Si ves "database is locked"
# Asegúrate de que no hay otra instancia corriendo
ps aux | grep optimize_e1

# Si es necesario, eliminar lock
rm optuna_studies.db-shm optuna_studies.db-wal
```

### Visualizaciones no se generan

```bash
# Instalar kaleido (requerido para exportar gráficos)
pip install kaleido

# Si sigue fallando, actualizar plotly
pip install --upgrade plotly kaleido
```

### Paso 1: Optimización Rápida (Exploración)

```bash
# 10 trials con 3 tickers para validar
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 10 --quick
```

**Resultado**: Valores aproximados en ~20 minutos

### Paso 2: Optimización Completa

```bash
# 50 trials con todos los tickers
python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 50
```

**Resultado**: Parámetros óptimos en ~3 horas

### Paso 3: Analizar Resultados

```bash
# Ver gráficos
open reports/hyperparameter_optimization/figures/

# Ver mejores parámetros
cat reports/hyperparameter_optimization/best_params_e1.yaml

# Explorar en MLflow
open http://localhost:5050
```

### Paso 4: Aplicar Mejores Parámetros

Editar `src/config/base.yaml`:

```yaml
strategies:
  e1_conservative:
    thresholds:
      tau_buy: 0.04  # Del optimization
      tau_sell: -0.01
    
    model:
      gru_units: [64, 32]  # Del optimization
      dropout: 0.35
      learning_rate: 0.0015
      batch_size: 64
```

### Paso 5: Validar Mejora

```bash
# Re-entrenar con nuevos parámetros
docker compose exec airflow-scheduler airflow dags trigger e1_conservative_pipeline

# Comparar métricas en MLflow
# Experimento anterior vs nuevo
```

---

## Troubleshooting

### Error: "No module named 'optuna'"

```bash
pip install optuna optuna-dashboard kaleido
```

### Error: MLflow connection refused

```bash
# Verificar que MLflow está corriendo
docker compose ps mlflow

# Si no está, iniciar
docker compose up -d mlflow
```

### Optimización muy lenta

**Opciones**:
1. Usar `--quick` para menos tickers
2. Reducir `--n_trials`
3. Usar `--ticker AAPL` para un solo ticker
4. Usar `--timeout` para limitar tiempo

### No converge (línea roja sigue subiendo)

**Solución**: Aumentar `--n_trials`

Típicamente:
- 20 trials: exploración inicial
- 50 trials: suficiente para mayoría
- 100 trials: si espacio de búsqueda muy grande

---

## Referencias

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html)
- [Hyperparameter Optimization in ML](https://arxiv.org/abs/1502.02127)

---

## Próximos Pasos

Después de optimización:

1.  **Aplicar parámetros** en `base.yaml`
2.  **Validar con Walk-Forward** (implementar siguiente)
3. 🔍 **Feature Importance** (analizar qué features usar)
4. 💼 **Portfolio Optimization** (multi-ticker allocation)
