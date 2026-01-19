# Índice de Documentación E1

## 📖 Documentación Principal

### E1 Conservative (Producción)
- **[README_E1.md](README_E1.md)** - Documentación completa de E1 Conservative
  - Arquitectura, decision score por perfiles, walk-forward validation
  - Uso CLI y Airflow
  - Configuración y personalización

### E1 Simple (Desarrollo)
- **[README_E1_SIMPLE.md](README_E1_SIMPLE.md)** - Documentación completa de E1 Simple
  - Pipeline simplificado, 80% más rápido
  - Decision score 5 métricas, time split
  - Quick start, uso CLI y Airflow

---

## 📊 Baseline y Optimización

### E1 Baseline (Comparación)
- **[README_E1_BASELINE.md](README_E1_BASELINE.md)** - Regresión lineal baseline
- **[QUICKSTART_E1_BASELINE.md](QUICKSTART_E1_BASELINE.md)** - Guía rápida
- **[SUMMARY_E1_BASELINE.md](SUMMARY_E1_BASELINE.md)** - Resumen ejecutivo
- **[docs/AIRFLOW_E1_BASELINE_DAG.md](docs/AIRFLOW_E1_BASELINE_DAG.md)** - DAG de Airflow

### Optimización
- **[README_E1_OPTIMIZATION.md](README_E1_OPTIMIZATION.md)** - Hyperparameter tuning con Optuna

### Comparación
- **[COMPARISON_E1.md](COMPARISON_E1.md)** - E1 Simple vs E1 Conservative

---

## 🚀 Quick Start por Perfil

**Nuevo usuario → desarrollo rápido:**
1. [README_E1_SIMPLE.md](README_E1_SIMPLE.md) - Quick start

**Investigador → comparación rigurosa:**
1. [README_E1_BASELINE.md](README_E1_BASELINE.md) - Baseline
2. [README_E1.md](README_E1.md) - E1 Conservative
3. [COMPARISON_E1.md](COMPARISON_E1.md) - Comparativa

**DevOps → deploy producción:**
1. [README_E1.md](README_E1.md) - E1 Conservative
2. [README_AIRFLOW_USAGE.md](README_AIRFLOW_USAGE.md) - DAGs
3. [README_E1_OPTIMIZATION.md](README_E1_OPTIMIZATION.md) - Tuning

---

## 🔄 Cambios Recientes (Enero 2026)

- ✅ Consolidación de docs: 5 READMEs E1 Simple → 1
- ✅ Sanitización de métricas: NaN/Inf → valores válidos
- ✅ Simplificación MLflow: Sin decision_component_* logging
- ✅ Actualización con últimos cambios en pipelines y DAGs

---

**Última actualización:** Enero 18, 2026
