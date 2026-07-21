# Model lifecycle system

Manages the full model lifecycle: registration, validation, comparison, promotion and retirement.

## Overview

```
Training Pipeline ──► Guardrails ──► Registry (candidate) ──► Promotion CLI ──► Champion / Retired
                      (validation)   (registration)           (comparison)
```

Every ticker is evaluated independently. The state of all models is persisted in a single JSON
file, written atomically.

## Model stages

| Stage | Description |
|---|---|
| **Baseline** | Fixed reference (e.g. Linear Regression). Never promoted. |
| **Candidate** | Freshly trained model, awaiting evaluation. |
| **Champion** | The model in use. One per ticker per strategy. |
| **Retired** | Archived former champions (max. 5 per ticker). |

## Key files

| File | Purpose |
|---|---|
| `src/lifecycle/registry.py` | Central CRUD over model state (`ModelRegistry`) |
| `src/lifecycle/guardrails.py` | Phase-1 technical validation |
| `src/lifecycle/promotion.py` | Composite scoring and comparison logic |
| `src/lifecycle/loader.py` | Loads champion/baseline models from disk |
| `scripts/evaluation/promote_candidate.py` | Evaluation and promotion CLI |
| `src/config/base.yaml` | Configuration (`lifecycle` section) |
| `models/registry.json` | Current state of every model |
| `models/promotion_log.jsonl` | Audit log of promotion decisions |
| `models/metrics_log.jsonl` | Metric history (for future Phase-2 calibration) |

## Workflow

### 1. Training

The training pipeline (e.g. `python -m src.e1.train_pipeline`) automatically:

1. Trains the model and writes artifacts to `runs/<strategy>/<timestamp>/<TICKER>/`
2. Runs the technical validation guardrails
3. Registers the model as a **candidate** in the registry

### 2. Guardrails (technical validation)

Phase-1 rejects models that are technically broken, before any metric comparison happens:

- The model file (`.pth`) exists
- No NaN/Inf in predictions
- No NaN/Inf in the backtest
- Sharpe ratio > 0
- Not worse than the baseline on IC (when a baseline is available)

### 3. Evaluation and promotion

The candidate is compared against the current champion using a **weighted composite score**:

```
score = Σ (weight_i × normalized_metric_i)
```

The candidate is promoted only if it beats the champion by a minimum margin (default 5%).

**Default global weights:**

| Metric | Weight |
|---|---|
| `bt_sharpe` | 0.35 |
| `ml_ic` | 0.25 |
| `ml_directional_accuracy` | 0.20 |
| `bt_calmar` | 0.20 |

Weights can be overridden per strategy in `base.yaml`, under
`lifecycle.promotion.per_strategy`.

## CLI

```bash
# Evaluate candidates (dry run — changes nothing)
python -m scripts.evaluation.promote_candidate

# Execute promotions
python -m scripts.evaluation.promote_candidate --execute

# Restrict to specific tickers
python -m scripts.evaluation.promote_candidate --tickers AAPL,MSFT --execute

# Force promotion, bypassing scoring
python -m scripts.evaluation.promote_candidate --tickers YPFD.BA --force --execute

# Per-metric detail
python -m scripts.evaluation.promote_candidate --verbose

# Inspect the audit log
python -m scripts.evaluation.promote_candidate --show-log 20
```

## Configuration

Lifecycle configuration lives in `src/config/base.yaml` under `lifecycle`:

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
    auto_promote: false                 # dry run by default
    first_champion_strategy: "promote"  # with no champion yet, promote automatically
    min_improvement: 0.05               # 5% minimum improvement
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

### Hierarchical config resolution

1. Exact match: `per_strategy.e1_conservative`
2. Prefix fallback: `per_strategy.e1` (covers `e1_conservative`)
3. Global fallback: the top-level config

## Registry structure

`models/registry.json` holds the state of every model:

```json
{
  "version": 1,
  "last_updated": "2026-03-01T00:44:11+00:00",
  "strategies": {
    "e1": {
      "tickers": {
        "YPFD.BA": {
          "baseline":  { "variant": "e1_baseline", "run_dir": "...", "metrics": {} },
          "champion":  { "variant": "e1_conservative", "run_dir": "...", "metrics": {} },
          "candidate": { "variant": "e1_conservative", "run_dir": "...", "metrics": {} },
          "retired":   [{ "variant": "e1_simple", "reason": "superseded_by_...", "metrics": {} }]
        }
      }
    }
  }
}
```

> Never edit `registry.json` by hand. All writes go through `ModelRegistry`, which uses
> `tempfile` + `os.replace` so a crash can't leave the file half-written.

## Claude Code skills

Beyond the CLI, these skills are available for interactive use:

| Skill | Example | Description |
|---|---|---|
| `/model-status` | `/model-status` | Show the state of every model |
| `/compare-models` | `/compare-models e1 AAPL` | Compare baseline vs champion vs candidate |
| `/promote-model` | `/promote-model e1 AAPL` | Promote candidate to champion (interactive) |
| `/retire-model` | `/retire-model e1 AAPL` | Retire the current champion |
| `/new-experiment` | `/new-experiment e1 attention_gru` | Scaffold a new variant |
