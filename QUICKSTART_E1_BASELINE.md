# Quick Start: Baseline E1

Guía rápida para ejecutar y comparar el baseline de Regresión Lineal contra el modelo GRU.

## � Tabla de Contenidos

- [Ejecución Local (Manual)](#-ejecución-rápida-local-3-pasos)
- [Ejecución con Docker + Airflow (Automatizado)](#-ejecución-con-airflow-automatizado)
- [Comparación de Resultados](#-output-esperado)

---

## 🚀 Ejecución Rápida (Local - 3 pasos)

### Paso 1: Entrenar ambos modelos

```bash
# 1. Entrenar GRU (modelo principal)
python -m src.train_e1_pipeline --tickers AAPL

# 2. Entrenar Baseline (Regresión Lineal)
python -m src.train_e1_baseline --tickers AAPL
```

### Paso 2: Comparar resultados

```bash
# Genera reporte de comparación
python scripts/compare_e1_models.py
```

### Paso 3: Revisar resultados

```bash
# Ver reporte en consola (ya se muestra automáticamente en Paso 2)
# O abrir archivos generados:
cat reports/e1_model_comparison.md
```

## 📊 Output Esperado

```
COMPARACIÓN: Regresión Lineal Baseline vs GRU
====================================================================================================

📊 MÉTRICAS ML (Offline)
----------------------------------------------------------------------------------------------------
MAE                  | Base:     0.0342 | GRU:     0.0298 | Δ:    -0.0044 |  -12.87% ✓
RMSE                 | Base:     0.0521 | GRU:     0.0467 | Δ:    -0.0054 |  -10.36% ✓
Dir. Accuracy        | Base:     0.5423 | GRU:     0.5891 | Δ:    +0.0468 |   +8.63% ✓
IC (Spearman)        | Base:     0.0312 | GRU:     0.0589 | Δ:    +0.0277 |  +88.78% ✓

💰 MÉTRICAS TRADING (Online - Backtest)
----------------------------------------------------------------------------------------------------
Sharpe Ratio         | Base:     0.6734 | GRU:     0.9123 | Δ:    +0.2389 |  +35.47% ✓
CAGR                 | Base:     0.0823 | GRU:     0.1156 | Δ:    +0.0333 |  +40.46% ✓

RESUMEN
====================================================================================================
GRU superior en: 9 métricas (81.8%)
```

## 🔍 Interpretación

### ✅ GRU es mejor si:
- IC (Spearman) significativamente mayor (>+50%)
- Sharpe Ratio > 0.9 vs baseline < 0.7
- Directional Accuracy > 58% vs baseline ~ 52-54%

**Conclusión**: La arquitectura recurrente captura patrones temporales valiosos

### ⚠️ Resultados similares:
- Diferencias < 10% en métricas principales
- Ambos con IC < 0.05

**Conclusión**: Considerar simplificar modelo o mejorar features

### ❌ Baseline es mejor:
- Baseline supera en múltiples métricas

**Conclusión**: Revisar overfitting, data leakage, o hiperparámetros del GRU

## 📁 Estructura de Archivos

```
runs/
├── e1_conservative/20260115_120000/     # GRU
│   └── AAPL/
│       ├── AAPL_predictions.csv
│       ├── AAPL_summary.csv
│       └── AAPL_backtest.csv
│
└── e1_baseline/20260115_130000/         # Baseline
    └── AAPL/
        ├── AAPL_baseline_predictions.csv
        ├── AAPL_baseline_summary.csv
        └── AAPL_baseline_backtest.csv

reports/
├── e1_model_comparison.csv              # Tabla comparativa
└── e1_model_comparison.md               # Reporte Markdown
```

## 🎯 Métricas Clave

| Métrica | Objetivo | Descripción |
|---------|----------|-------------|
| **IC** | > 0.05 | Correlación Spearman (significativo en finanzas) |
| **Dir. Accuracy** | > 58% | % predicciones con dirección correcta |
| **Sharpe** | > 0.9 | Retorno ajustado por riesgo |
| **MAE** | < 0.03 | Error absoluto promedio |

## 💡 Tips

### Múltiples tickers
```bash
# Entrenar con múltiples tickers para comparación robusta
python -m src.train_e1_pipeline --tickers AAPL GOOGL MSFT
python -m src.train_e1_baseline --tickers AAPL GOOGL MSFT
python scripts/compare_e1_models.py
```

### Especificar runs
```bash
# Comparar runs específicos
python scripts/compare_e1_models.py \
  --gru-run runs/e1_conservative/20260115_120000 \
  --baseline-run runs/e1_baseline/20260115_130000
```

### Solo baseline con comparación automática
```bash
# El baseline puede auto-comparar con el último run de GRU
python -m src.train_e1_baseline --tickers AAPL --compare-with-gru
```

## 📚 Documentación Completa

- [README_E1_BASELINE.md](README_E1_BASELINE.md) - Documentación detallada del baseline
- [README_E1.md](README_E1.md) - Documentación del modelo GRU
- [README.md](README.md) - Visión general del proyecto

---

## 🐳 Ejecución con Airflow (Automatizado)

### Prerequisitos
```bash
# Levantar stack completo (Airflow + MLflow)
docker-compose --profile all up -d

# Verificar que Airflow esté corriendo
docker ps | grep airflow
```

### Opción 1: Desde Airflow UI

1. **Acceder a Airflow**: http://localhost:8080
   - User: `airflow`
   - Password: `airflow`

2. **Ejecutar GRU primero**:
   - Buscar DAG: `e1_conservative_pipeline`
   - Click "Play" (▶) → "Trigger DAG w/ config"
   - Configurar:
   ```json
   {
     "tickers": "AAPL,MSFT"
   }
   ```
   - Click "Trigger"
   - ⏱ Esperar ~10-15 min

3. **Ejecutar Baseline con comparación**:
   - Buscar DAG: `e1_baseline_linear_regression`
   - Click "Play" (▶) → "Trigger DAG w/ config"
   - Configurar:
   ```json
   {
     "tickers": "AAPL,MSFT",
     "auto_compare_with_gru": "True"
   }
   ```
   - Click "Trigger"
   - ⏱ Esperar ~8-12 min

4. **Ver resultados en MLflow**: http://localhost:5000
   - Experimento: `E1_Baseline_LinearRegression`
   - Comparar con: `E1_Conservative_Strategy`

### Opción 2: Desde CLI

```bash
# 1. Ejecutar GRU
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_conservative_pipeline \
  --conf '{"tickers": "AAPL,MSFT"}'

# 2. Ejecutar Baseline (espera que GRU termine)
docker compose exec -T airflow-scheduler airflow dags trigger \
  e1_baseline_linear_regression \
  --conf '{"tickers": "AAPL,MSFT", "auto_compare_with_gru": "True"}'

# 3. Ver resultados
ls -lht runs/e1_baseline/ | head -n 2
cat runs/e1_baseline/<timestamp>/comparison_vs_gru.csv
```

### Schedules Automáticos

Los DAGs se ejecutan automáticamente:
- **GRU**: Lunes 2:00 AM
- **Baseline**: Lunes 3:00 AM (con comparación automática)

### Ver Logs

```bash
# Logs de Airflow
docker compose logs -f airflow-scheduler

# Logs de tarea específica
# Desde Airflow UI → DAG → Task → Logs
```

---

## 📚 Documentación Completa

- [README_E1_BASELINE.md](README_E1_BASELINE.md) - Documentación detallada del baseline
- [README_E1.md](README_E1.md) - Documentación del modelo GRU
- [docs/AIRFLOW_E1_BASELINE_DAG.md](docs/AIRFLOW_E1_BASELINE_DAG.md) - Documentación del DAG
- [README_AIRFLOW_USAGE.md](README_AIRFLOW_USAGE.md) - Guía de uso de Airflow
- [README_DOCKER.md](README_DOCKER.md) - Setup Docker completo
- [README.md](README.md) - Visión general del proyecto
