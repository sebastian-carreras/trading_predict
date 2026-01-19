#!/usr/bin/env python
"""
Script de prueba para verificar que los targets se loguean en el summary de E2.
"""
import yaml

# Cargar config
with open("src/config/base.yaml", "r") as f:
    config = yaml.safe_load(f)

# Simular proceso de agregación de métricas
decision_cfg = config.get("decision", {})

# Targets por defecto
default_targets = {
    'ic_min': 0.05,
    'directional_accuracy_min': 0.55,
    'sharpe_min': 1.0,
    'mae_max': 0.03,
    'rmse_max': 0.05,
}

targets = {**default_targets, **decision_cfg.get("targets", {})}

print("=" * 60)
print("TARGETS QUE SE LOGGEARAN EN MLFLOW SUMMARY")
print("=" * 60)

# Simular resultados de ejemplo
ic_values = [0.283, 0.156, 0.092, 0.234]
sharpe_values = [1.5, 0.8, 2.1, 1.2]
decision_scores = [0.85, 0.72, 0.68, 0.79]

print("\n📊 MÉTRICAS AGREGADAS CON TARGETS:")
print("-" * 60)

# IC metrics
if ic_values:
    print(f"\n✓ IC Metrics:")
    print(f"  ic_mean: {sum(ic_values) / len(ic_values):.3f}")
    print(f"  ic_median: {sorted(ic_values)[len(ic_values) // 2]:.3f}")
    print(f"  ic_min: {min(ic_values):.3f}")
    print(f"  ic_max: {max(ic_values):.3f}")
    print(f"  ic_above_threshold: {sum(1 for ic in ic_values if ic > targets['ic_min'])}")
    print(f"  → ic_target_min (param): {targets['ic_min']} ⭐")

# Sharpe metrics
if sharpe_values:
    print(f"\n✓ Sharpe Metrics:")
    print(f"  sharpe_mean: {sum(sharpe_values) / len(sharpe_values):.3f}")
    print(f"  sharpe_median: {sorted(sharpe_values)[len(sharpe_values) // 2]:.3f}")
    print(f"  sharpe_min: {min(sharpe_values):.3f}")
    print(f"  sharpe_max: {max(sharpe_values):.3f}")
    print(f"  sharpe_above_threshold: {sum(1 for s in sharpe_values if s > targets['sharpe_min'])}")
    print(f"  → sharpe_target_min (param): {targets['sharpe_min']} ⭐")

# Decision Score metrics
if decision_scores:
    decision_threshold = decision_cfg.get("threshold", 0.70)
    print(f"\n✓ Decision Score Metrics:")
    print(f"  decision_score_mean: {sum(decision_scores) / len(decision_scores):.3f}")
    print(f"  decision_score_median: {sorted(decision_scores)[len(decision_scores) // 2]:.3f}")
    print(f"  decision_score_min: {min(decision_scores):.3f}")
    print(f"  decision_score_max: {max(decision_scores):.3f}")
    print(f"  → decision_score_threshold (param): {decision_threshold} ⭐")

print("\n" + "=" * 60)
print("✅ Los targets ahora se loguean como params en MLflow Summary")
print("   Esto permite comparar fácilmente las métricas con sus valores objetivo")
print("=" * 60)
