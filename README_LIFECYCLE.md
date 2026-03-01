# Model Lifecycle System

Sistema de gestión del ciclo de vida de modelos: registro, validación, comparación, promoción y retiro.

## Arquitectura general

```
Training Pipeline ──► Guardrails ──► Registry (candidate) ──► Promotion CLI ──► Champion / Retired
                      (validación)    (registro)               (comparación)
```

Cada ticker se evalúa de forma independiente. El estado de todos los modelos se persiste en un único archivo JSON con escrituras atómicas.

## Etapas del modelo

| Etapa | Descripción |
|-------|-------------|
| **Baseline** | Referencia fija (ej: Linear Regression). Nunca se promueve. |
| **Candidate** | Modelo recién entrenado, pendiente de evaluación. |
| **Champion** | Modelo activo en uso. Uno por ticker por estrategia. |
| **Retired** | Campeones anteriores archivados (máx. 5 por ticker). |

## Archivos clave

| Archivo | Propósito |
|---------|-----------|
| `src/lifecycle/registry.py` | CRUD central del estado de modelos (`ModelRegistry`) |
| `src/lifecycle/guardrails.py` | Validación técnica Phase-1 |
| `src/lifecycle/promotion.py` | Scoring compuesto y lógica de comparación |
| `src/lifecycle/loader.py` | Carga de modelos champion/baseline desde disco |
| `scripts/evaluation/promote_candidate.py` | CLI de evaluación y promoción |
| `src/config/base.yaml` | Configuración (sección `lifecycle`) |
| `models/registry.json` | Estado actual de todos los modelos |
| `models/promotion_log.jsonl` | Log de auditoría de decisiones de promoción |
| `models/metrics_log.jsonl` | Historial de métricas (para calibración futura Phase-2) |

## Flujo de trabajo

### 1. Entrenamiento

El pipeline de entrenamiento (ej: `python -m src.e1.train_pipeline`) automáticamente:
1. Entrena el modelo y guarda artefactos en `runs/<strategy>/<timestamp>/<TICKER>/`
2. Ejecuta guardrails de validación técnica
3. Registra el modelo como **candidate** en el registry

### 2. Guardrails (validación técnica)

Phase-1 rechaza modelos técnicamente rotos:
- Archivo del modelo (`.pth`) existe
- Sin NaN/Inf en predicciones
- Sin NaN/Inf en backtest
- Sharpe ratio > 0
- No peor que baseline en IC (si hay baseline disponible)

### 3. Evaluación y promoción

Compara el candidate contra el champion actual usando un **score compuesto ponderado**:

```
score = Σ (weight_i × metric_i_normalizado)
```

El candidate se promueve si supera al champion por un margen mínimo (default: 5%).

**Pesos por defecto (globales):**

| Métrica | Peso |
|---------|------|
| `bt_sharpe` | 0.35 |
| `ml_ic` | 0.25 |
| `ml_directional_accuracy` | 0.20 |
| `bt_calmar` | 0.20 |

Los pesos se pueden personalizar por estrategia en `base.yaml` (sección `lifecycle.promotion.per_strategy`).

## Comandos CLI

### Evaluar candidatos (dry-run)

```bash
python -m scripts.evaluation.promote_candidate
```

Muestra qué candidatos serían promovidos **sin modificar nada**.

### Ejecutar promoción

```bash
python -m scripts.evaluation.promote_candidate --execute
```

### Filtrar por tickers específicos

```bash
python -m scripts.evaluation.promote_candidate --tickers AAPL,MSFT --execute
```

### Forzar promoción (ignora scoring)

```bash
python -m scripts.evaluation.promote_candidate --tickers YPFD.BA --force --execute
```

### Ver detalle por métrica

```bash
python -m scripts.evaluation.promote_candidate --verbose
```

### Ver log de auditoría

```bash
python -m scripts.evaluation.promote_candidate --show-log 20
```

## Configuración

La configuración del lifecycle está en `src/config/base.yaml` bajo la sección `lifecycle`:

```yaml
lifecycle:
  enabled: true
  registry_path: "models/registry.json"
  metrics_log_path: "models/metrics_log.jsonl"

  guardrails:
    phase: 1
    reject_if:
      nan_in_predictions: true
      nan_in_backtest: true
      sharpe_below_zero: true
      worse_than_baseline: true

  promotion:
    auto_promote: false          # Dry-run por defecto
    first_champion_strategy: "promote"  # Si no hay champion, promover automáticamente
    min_improvement: 0.05        # Mejora mínima del 5%
    require_positive_sharpe: true

    scoring_weights:
      bt_sharpe: 0.35
      ml_ic: 0.25
      ml_directional_accuracy: 0.20
      bt_calmar: 0.20

    per_strategy:
      e1:
        scoring_weights:
          bt_sharpe: 0.35
          ml_ic: 0.25
          ml_directional_accuracy: 0.20
          bt_calmar: 0.20
      e2:
        scoring_weights:
          bt_sharpe: 0.30
          ml_ic: 0.20
          ml_directional_accuracy: 0.30
          bt_calmar: 0.20

  retirement:
    keep_last_n: 5
```

### Resolución jerárquica de config por estrategia

1. Match exacto: `per_strategy.e1_conservative`
2. Fallback por prefijo: `per_strategy.e1` (para `e1_conservative`)
3. Fallback global: config de nivel superior

## Estructura del registry

`models/registry.json` almacena el estado de todos los modelos:

```json
{
  "version": 1,
  "last_updated": "2026-03-01T00:44:11+00:00",
  "strategies": {
    "e1": {
      "tickers": {
        "YPFD.BA": {
          "baseline":  { "variant": "e1_baseline", "run_dir": "...", "metrics": {...} },
          "champion":  { "variant": "e1_conservative", "run_dir": "...", "metrics": {...} },
          "candidate": { "variant": "e1_conservative", "run_dir": "...", "metrics": {...} },
          "retired":   [{ "variant": "e1_simple", "reason": "superseded_by_...", "metrics": {...} }]
        }
      }
    }
  }
}
```

## Skills de Claude Code

Además del CLI, hay skills disponibles para uso interactivo:

| Skill | Uso | Descripción |
|-------|-----|-------------|
| `/model-status` | `/model-status` | Ver estado de todos los modelos |
| `/compare-models` | `/compare-models e1 AAPL` | Comparar baseline vs champion vs candidate |
| `/promote-model` | `/promote-model e1 AAPL` | Promover candidate a champion (interactivo) |
| `/retire-model` | `/retire-model e1 AAPL` | Retirar champion actual |
| `/new-experiment` | `/new-experiment e1 attention_gru` | Crear scaffold para nueva variante |
