from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def project_root() -> Path:
    # trading_predict/src/utils.py -> trading_predict/
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'pyyaml'. Install with: pip install pyyaml"
        ) from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root object in {path}")
    return data


def get_nested(mapping: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cursor: Any = mapping
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


@dataclass(frozen=True)
class IntradayDataset:
    X: "Any"  # np.ndarray
    y: "Any"  # np.ndarray
    feature_names: list[str]
    timestamps: "Any"  # pd.DatetimeIndex
