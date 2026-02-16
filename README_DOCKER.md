# Integración Docker: Airflow + MLflow + FastAPI + PostgreSQL

## 📋 Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                     Trading Predict Stack                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐    ┌──────────┐    ┌─────────────┐        │
│  │   Airflow   │───▶│  MLflow  │───▶│   FastAPI   │        │
│  │  Scheduler  │    │ Tracking │    │     API     │        │
│  │  Webserver  │    │  Server  │    │   (8800)    │        │
│  │   (8080)    │    │  (5000)  │    └─────────────┘        │
│  └─────────────┘    └──────────┘           │                │
│        │                  │                 │                │
│        │           ┌──────┴──────┐         │                │
│        │           │             │         │                │
│        ▼           ▼             ▼         ▼                │
│  ┌──────────────────────────────────────────────┐          │
│  │          PostgreSQL Database                 │          │
│  │  - Airflow metadata                          │          │
│  │  - MLflow experiments & runs                 │          │
│  │            (5432)                             │          │
│  └──────────────────────────────────────────────┘          │
│        │                                                     │
│        ▼                                                     │
│  ┌──────────────────────────────────────────────┐          │
│  │          MinIO (S3-compatible)                │          │
│  │  - MLflow artifacts (modelos, métricas)      │          │
│  │  - Datasets                                   │          │
│  │            (9000, UI: 9001)                   │          │
│  └──────────────────────────────────────────────┘          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Inicio Rápido

### 1. Configurar variables de entorno

```bash
# Copiar template y editar
cp .env.example .env
```

### 2. Levantar servicios

```bash
# Opción A: Solo MLflow + PostgreSQL
docker-compose --profile mlflow up -d

# Opción B: Solo Airflow
docker-compose --profile airflow up -d

# Opción C: Stack completo (recomendado)
docker-compose --profile all up -d
```

### 2.b Modo transparente MLflow (offline→online)

Para que los entrenamientos ejecutados sin Docker aparezcan luego en la UI de MLflow:

```bash
bash scripts/mlflow/up_transparent_mlflow.sh
```

Este script:
- alinea la versión de MLflow del contenedor con tu entorno local,
- reconstruye el servicio `mlflow`,
- y lo levanta apuntando a `runs/mlflow_local` (mismo store de ejecuciones offline).

### 3. Acceder a interfaces

- **Airflow UI**: http://localhost:8080
  - Usuario: `admin` (configurable en .env)
  - Password: `admin123`

- **MLflow UI**: http://localhost:5050
  - Tracking de experimentos
  - Registro de modelos

- **FastAPI Docs**: http://localhost:8800/docs
  - Swagger UI interactivo
  - Endpoints de predicción

- **MinIO UI**: http://localhost:9001
  - Usuario: `minio_admin`
  - Password: `minio_secret_key_123`

## Flujo de Trabajo

### Pipeline E1 (Conservadora - GRU)

**Schedule**: Lunes 2 AM (semanal)

```bash
# Activar DAG manualmente en Airflow UI
# O trigger desde CLI:
docker exec -it airflow_webserver airflow dags trigger e1_conservative_pipeline
```

**Pasos**:
1. Descarga datos diarios (Yahoo Finance o IOL)
2. Calcula 15 features técnicos
3. Entrena modelos GRU por ticker
4. Registra experimentos en MLflow
5. Guarda artefactos en MinIO
6. Notifica a FastAPI (modelos disponibles)

**Outputs**:
- `runs/e1_conservative/<timestamp>/` - Predicciones y métricas
- MLflow experiments: `E1_Conservative_Strategy`
- MinIO bucket: `s3://mlflow/`

### Pipeline E3 (Intradía - LSTM Ensemble)

**Schedule**: Diario 1 AM

```bash
docker exec -it airflow_webserver airflow dags trigger train_e3_pipeline
```

**Pasos**:
1. Descarga datos 5-min (últimos 60 días)
2. Entrena ensemble de 3 LSTM
3. Ejecuta backtest con costos 20 bps
4. Registra en MLflow
5. Actualiza modelos en FastAPI

**Outputs**:
- `runs/e3_intraday/<timestamp>/` - Backtests y métricas
- MLflow experiments: `E3_Intraday_Strategy`

## 🔧 Desarrollo Local

### Ejecutar pipelines sin Docker (debugging)

```bash
# Activar ambiente Python
conda activate ia_ceia_18co

# E1 manual
python -m src.train_e1_pipeline --tickers AAPL

# E3 manual
python -m src.train_e3_pipeline --mode run --tickers SPY

# Con tracking MLflow
export MLFLOW_TRACKING_URI=http://localhost:5050
python -m src.train_e1_pipeline
```

### Logs de Airflow

```bash
# Ver logs de scheduler
docker logs -f airflow_scheduler

# Ver logs de un DAG específico
docker exec -it airflow_webserver airflow tasks test e1_conservative_pipeline download_daily_data 2026-01-06
```

## 📡 API Endpoints (FastAPI)

### Health check
```bash
curl http://localhost:8800/health
```

### Registrar modelos (desde Airflow)
```bash
curl -X POST http://localhost:8800/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "e1_conservative",
    "run_dir": "/opt/airflow/runs/e1_conservative/20260106_020530",
    "timestamp": "2026-01-06T02:05:30"
  }'
```

### Predicción E1
```bash
curl -X POST http://localhost:8800/predict/e1/AAPL \
  -H "Content-Type: application/json" \
  -d '{"use_latest_data": true}'
```

### Ver estado de modelos
```bash
curl http://localhost:8800/models/status
```

## 🗄 Base de Datos PostgreSQL

### Conectar desde terminal

```bash
# Entrar al container
docker exec -it postgres psql -U airflow

# Ver bases de datos
\l

# Conectar a MLflow DB
\c mlflow_db

# Ver tablas de MLflow
\dt

# Ver experimentos
SELECT experiment_id, name FROM experiments;
```

## 📦 MinIO (S3)

### Ver artifacts de MLflow

```bash
# Listar buckets
docker exec -it minio mc ls s3/mlflow

# Ver artifacts de un experimento
docker exec -it minio mc ls s3/mlflow/0/<run_id>/artifacts/
```

## 🛑 Detener servicios

```bash
# Detener todos
docker-compose --profile all down

# Detener y eliminar volúmenes (¡cuidado!)
docker-compose --profile all down -v
```

## Monitoreo

### Ver recursos Docker

```bash
# Uso de recursos
docker stats

# Logs combinados
docker-compose --profile all logs -f

# Solo un servicio
docker-compose logs -f mlflow
```

## 🔐 Seguridad

**Para producción**:
1. Cambiar todas las contraseñas en `.env`
2. Usar secretos externos (no .env commiteado)
3. Configurar SSL/TLS para endpoints públicos
4. Restringir acceso a red (firewall)
5. Usar autenticación en FastAPI (OAuth2, API keys)

## Estructura de Volúmenes

```
docker volumes:
├── db_data/              # PostgreSQL data
├── minio_data/           # MinIO buckets
└── airflow/
    ├── dags/             # DAGs de Airflow (subcarpetas por estrategia)
    │   ├── E1/
    │   ├── E2/
    │   ├── E3/
    │   └── E4/
    ├── logs/             # Logs de ejecución
    ├── plugins/          # Plugins custom
    └── config/           # Airflow config
```

## 🐛 Troubleshooting

### Airflow no inicia
```bash
# Reiniciar init
docker-compose --profile all down
docker volume rm trading_predict_db_data
docker-compose --profile all up airflow-init
docker-compose --profile all up -d
```

### MLflow no conecta a PostgreSQL
```bash
# Verificar que DB existe
docker exec -it postgres psql -U airflow -c "CREATE DATABASE mlflow_db;"
```

### MinIO buckets no se crean
```bash
# Crear manualmente
docker exec -it minio mc mb s3/mlflow
docker exec -it minio mc mb s3/data
```

---

**Proyecto**: Trading Predict  
**Stack**: Airflow 2.8.1 + MLflow + FastAPI + PostgreSQL 13 + MinIO  
**Autor**: Sebastian Carreras - FIUBA AI Posgrado
