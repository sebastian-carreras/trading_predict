from __future__ import annotations

from dataclasses import dataclass
import json
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


def log_timing_event(
    *,
    strategy: str,
    phase: str,
    duration_seconds: float,
    started_at: str,
    ended_at: str,
    ticker: str | None = None,
    run_dir: Path | None = None,
    extra: dict[str, Any] | None = None,
    log_path: Path | None = None,
) -> Path:
    """Append a timing event in JSONL format under reports/timing.

    Args:
        strategy: Strategy key (e.g., "e1_conservative").
        phase: Phase name (e.g., "train", "predict").
        duration_seconds: Elapsed time in seconds.
        started_at: ISO timestamp (UTC) at start.
        ended_at: ISO timestamp (UTC) at end.
        ticker: Optional ticker or pair identifier.
        run_dir: Optional run output directory.
        extra: Optional extra fields.
        log_path: Optional explicit log path (overrides default).

    Returns:
        Path to the JSONL log file.
    """
    root = project_root()
    if log_path is None:
        log_dir = root / "reports" / "timing"
        ensure_dir(log_dir)
        log_path = log_dir / "timing_log.jsonl"
    else:
        ensure_dir(log_path.parent)

    event: dict[str, Any] = {
        "strategy": strategy,
        "phase": phase,
        "duration_seconds": float(duration_seconds),
        "started_at": started_at,
        "ended_at": ended_at,
    }
    if ticker:
        event["ticker"] = ticker
    if run_dir is not None:
        event["run_dir"] = str(run_dir)
    if extra:
        event.update(extra)

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")

    try:
        import mlflow  # type: ignore

        if mlflow.active_run() is not None:
            metric_name = f"timing_{phase}_seconds"
            component = extra.get("component") if extra else None
            if component:
                metric_name = f"timing_{phase}_{component}_seconds"

            step = None
            if extra:
                if "fold" in extra:
                    step = int(extra["fold"])
                elif "member" in extra:
                    step = int(extra["member"])

            if step is not None:
                mlflow.log_metric(metric_name, float(duration_seconds), step=step)
            else:
                mlflow.log_metric(metric_name, float(duration_seconds))
    except Exception:
        pass

    return log_path


@dataclass(frozen=True)
class IntradayDataset:
    X: "Any"  # np.ndarray
    y: "Any"  # np.ndarray
    feature_names: list[str]
    timestamps: "Any"  # pd.DatetimeIndex
