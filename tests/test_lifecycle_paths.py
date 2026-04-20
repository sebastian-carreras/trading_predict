from __future__ import annotations

from pathlib import Path

from src.utils import resolve_lifecycle_paths


def test_resolve_lifecycle_paths_uses_config_relative_to_root() -> None:
    config = {
        "lifecycle": {
            "registry_path": "reports/verify/registry.json",
            "metrics_log_path": "reports/verify/metrics.jsonl",
        }
    }

    root = Path("/tmp/trading_predict")
    registry_path, metrics_log_path = resolve_lifecycle_paths(config, root=root)

    assert registry_path == root / "reports/verify/registry.json"
    assert metrics_log_path == root / "reports/verify/metrics.jsonl"


def test_resolve_lifecycle_paths_keeps_absolute_paths() -> None:
    config = {
        "lifecycle": {
            "registry_path": "/var/tmp/custom_registry.json",
            "metrics_log_path": "/var/tmp/custom_metrics.jsonl",
        }
    }

    registry_path, metrics_log_path = resolve_lifecycle_paths(config, root=Path("/tmp/ignored"))

    assert registry_path == Path("/var/tmp/custom_registry.json")
    assert metrics_log_path == Path("/var/tmp/custom_metrics.jsonl")


def test_resolve_lifecycle_paths_falls_back_to_defaults() -> None:
    root = Path("/tmp/trading_predict")

    registry_path, metrics_log_path = resolve_lifecycle_paths({}, root=root)

    assert registry_path == root / "models/registry.json"
    assert metrics_log_path == root / "models/metrics_log.jsonl"