# Consultas a runs/optuna_trials/optuna_studies.db - Guía Completa

## ❌ Errores Comunes

### Error: "no such column: study_id"

**Problema**: La tabla `trial_values` NO tiene la columna `study_id`

```sql
-- ❌ INCORRECTO
SELECT * FROM trial_values WHERE study_id = 1;
-- Error: no such column: study_id
```

**Solución**: Hacer JOIN correctamente a través de `trials`

```sql
-- ✅ CORRECTO
SELECT tv.*
FROM trial_values tv
JOIN trials t ON tv.trial_id = t.trial_id
WHERE t.study_id = 1;
```

---

## 📊 Estructura de Tablas

### Relaciones entre tablas:

```
studies (study_id, study_name)
    ↓ (one-to-many)
trials (trial_id, study_id, number, state, datetime_start, datetime_complete)
    ↓ (one-to-many)
trial_values (trial_value_id, trial_id, objective, value)
trial_params (trial_param_id, trial_id, param_name, param_value)
```

### Diagrama completo:

```
┌──────────────────────────────────────────────────────────┐
│ STUDIES (study_id, study_name)                           │
├──────────────────────────────────────────────────────────┤
│  1 │ e2_optimization_LOMA.BA                              │
│  2 │ e2_optimization_BMA.BA                               │
│  ... (14 estudios totales)                                │
└──────┬─────────────────────────────────────────────────────┘
       │ (one-to-many)
       ↓
┌──────────────────────────────────────────────────────────┐
│ TRIALS (trial_id, study_id, number, state, ...)         │
├──────────────────────────────────────────────────────────┤
│  100 │ 1 │ 0  │ COMPLETE │ ...                           │
│  101 │ 1 │ 1  │ COMPLETE │ ...                           │
│  ... (303 trials totales)                                 │
└──────┬──────────┬──────────────────────────────────────────┘
       │          │
       │          └─────────────┐
       │                        ↓
       │          ┌──────────────────────────────┐
       │          │ TRIAL_PARAMS                 │
       │          ├──────────────────────────────┤
       │          │ trial_id │ param_name       │
       │          │ 100 │ tau_buy: 0.025 │
       │          │ 100 │ dropout: 0.3   │
       │          └──────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────────────────────────┐
│ TRIAL_VALUES (trial_value_id, trial_id, objective, value)│
├──────────────────────────────────────────────────────────┤
│  1 │ 100 │ 0 │ 4.352459 │                               │
│  2 │ 101 │ 0 │ 2.287823 │                               │
│  ... (303 valores totales)                                │
└──────────────────────────────────────────────────────────┘
```

---

## ✅ Consultas Correctas

### 1. Listar todos los estudios con estadísticas

```sql
SELECT
    s.study_name,
    COUNT(DISTINCT t.trial_id) as n_trials,
    ROUND(AVG(tv.value), 4) as avg_objective,
    ROUND(MAX(tv.value), 4) as best_objective,
    COUNT(CASE WHEN t.state = 'COMPLETE' THEN 1 END) as completed
FROM studies s
LEFT JOIN trials t ON s.study_id = t.study_id
LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
GROUP BY s.study_name
ORDER BY best_objective DESC;
```

### 2. Ver mejores parámetros de un estudio

```sql
SELECT
    tp.param_name,
    tp.param_value,
    tv.value as objective
FROM studies s
JOIN trials t ON s.study_id = t.study_id
JOIN trial_values tv ON t.trial_id = tv.trial_id
JOIN trial_params tp ON t.trial_id = tp.trial_id
WHERE s.study_name = 'e2_optimization_LOMA.BA'
ORDER BY tv.value DESC
LIMIT 1;
```

### 3. Comparar últimos 10 trials de un estudio

```sql
SELECT
    t.number,
    tv.value as objective,
    t.state,
    ROUND(
        CAST((julianday(t.datetime_complete) - julianday(t.datetime_start)) * 24 * 60 AS FLOAT),
        2
    ) as duration_minutes
FROM studies s
JOIN trials t ON s.study_id = t.study_id
JOIN trial_values tv ON t.trial_id = tv.trial_id
WHERE s.study_name = 'e2_optimization_LOMA.BA'
ORDER BY t.number DESC
LIMIT 10;
```

### 4. Encontrar qué parámetro tiene mayor variación

```sql
SELECT
    tp.param_name,
    COUNT(*) as occurrences,
    ROUND(AVG(CAST(tp.param_value AS FLOAT)), 4) as avg_value,
    ROUND(MAX(CAST(tp.param_value AS FLOAT)), 4) as max_value,
    ROUND(MIN(CAST(tp.param_value AS FLOAT)), 4) as min_value,
    ROUND(
        MAX(CAST(tp.param_value AS FLOAT)) - MIN(CAST(tp.param_value AS FLOAT)),
        4
    ) as range
FROM studies s
JOIN trials t ON s.study_id = t.study_id
JOIN trial_params tp ON t.trial_id = tp.trial_id
WHERE s.study_name = 'e2_optimization_LOMA.BA'
GROUP BY tp.param_name
ORDER BY range DESC;
```

### 5. Exportar todos los trials a CSV

```sql
SELECT
    t.number,
    tv.value as objective,
    t.state,
    t.datetime_start,
    t.datetime_complete
FROM studies s
JOIN trials t ON s.study_id = t.study_id
JOIN trial_values tv ON t.trial_id = tv.trial_id
WHERE s.study_name = 'e2_optimization_LOMA.BA'
ORDER BY t.number;
```

### 6. Ver trials fallidos

```sql
SELECT
    t.trial_id,
    t.number,
    t.state,
    t.datetime_start
FROM studies s
JOIN trials t ON s.study_id = t.study_id
WHERE s.study_name = 'e2_optimization_LOMA.BA'
    AND t.state != 'COMPLETE'
ORDER BY t.datetime_start DESC;
```

### 7. Mejores valores de tau_buy

```sql
SELECT
    tp.param_value as tau_buy,
    tv.value as objective,
    COUNT(*) as frequency
FROM trials t
JOIN trial_params tp ON t.trial_id = tp.trial_id
JOIN trial_values tv ON t.trial_id = tv.trial_id
WHERE tp.param_name = 'tau_buy'
    AND t.study_id = (SELECT study_id FROM studies WHERE study_name = 'e2_optimization_LOMA.BA')
GROUP BY tp.param_value
ORDER BY objective DESC
LIMIT 10;
```

### 8. Correlación entre dropout y objetivo

```sql
SELECT
    tp.param_value as dropout,
    ROUND(AVG(CAST(tv.value AS FLOAT)), 4) as avg_objective,
    COUNT(*) as count
FROM trials t
JOIN trial_params tp ON t.trial_id = tp.trial_id
JOIN trial_values tv ON t.trial_id = tv.trial_id
WHERE tp.param_name = 'dropout'
    AND t.study_id = (SELECT study_id FROM studies WHERE study_name = 'e2_optimization_LOMA.BA')
GROUP BY tp.param_value
ORDER BY tp.param_value;
```

---

## 🐍 Usar desde Python

### Script interactivo (recomendado)

```bash
python scripts/analyze_optuna_db.py
```

### O desde Python directamente:

```python
from scripts.analyze_optuna_db import OptunaDBAnalyzer

analyzer = OptunaDBAnalyzer()

# Ver todos los estudios
studies = analyzer.get_all_studies()
print(studies)

# Ver mejores parámetros de un estudio
params = analyzer.get_best_params('e2_optimization_LOMA.BA')
print(params)

# Ver histórico de trials
history = analyzer.get_trial_history('e2_optimization_LOMA.BA', limit=10)
print(history)

# Exportar a CSV
analyzer.export_to_csv('e2_optimization_LOMA.BA', 'results.csv')
```

---

## 📌 Trucos Útiles

### Limpiar estudios antiguos

```bash
# Eliminar un estudio específico
sqlite3 runs/optuna_trials/optuna_studies.db \
  "DELETE FROM studies WHERE study_name LIKE 'e1_test_%';"

# Eliminar todos los E1
sqlite3 runs/optuna_trials/optuna_studies.db \
  "DELETE FROM studies WHERE study_name LIKE 'e1_%';"
```

### Crear backup de la DB

```bash
cp runs/optuna_trials/optuna_studies.db runs/optuna_trials/optuna_studies.db.backup
```

### Ver información de una única tabla

```bash
# Todos los estudios
sqlite3 runs/optuna_trials/optuna_studies.db "SELECT * FROM studies;"

# Todos los trials de un estudio
sqlite3 runs/optuna_trials/optuna_studies.db \
  "SELECT * FROM trials WHERE study_id = 1 LIMIT 5;"

# Parámetros del trial 100
sqlite3 runs/optuna_trials/optuna_studies.db \
  "SELECT * FROM trial_params WHERE trial_id = 100;"
```

---

## 🎯 Resumen de Estudios Actuales

### E2 Optimization (Por ticker) - Los mejores:

| Estudio | Best Value | Avg | Trials |
|---------|-----------|-----|--------|
| **LOMA.BA** | 4.35 | 2.45 | 30 ✓ |
| **BMA.BA** | 1.95 | 0.54 | 30 ✓ |
| **NVDA** | 1.82 | 0.27 | 32 ✓ |
| **EDN.BA** | 1.69 | -0.48 | 30 ✓ |
| **AMZN** | 1.34 | 0.49 | 30 ✓ |

### E1 y Otros:

| Estudio | Best Value | Status |
|---------|-----------|--------|
| e1_hyperparameter_optimization | -10.5 | ❌ Malo |
| e2_optimization_TGSU2.BA | -12.0 | ❌ Malo |
| e2_hyperparameter_optimization | 0.93 | ⚠️ Pocos trials |

---

## 🔗 Referencias

- [Optuna SQL Storage](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.RDBStorage.html)
- [SQLite PRAGMA](https://www.sqlite.org/pragma.html)
- Script de análisis: [scripts/analyze_optuna_db.py](scripts/analyze_optuna_db.py)
