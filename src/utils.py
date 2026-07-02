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


def get_universe_tickers(config: dict[str, Any]) -> list[str]:
    """Universo de descarga/limpieza, derivado de la config.

    Fuente única de verdad: ``universe.tickers_by_strategy`` (unión de todas las
    estrategias) más ``universe.extra_download_tickers`` (activos reservados que
    aún no pertenecen a una estrategia activa, p.ej. futuros E4). El orden es
    estable y sin duplicados.

    Retrocompatible: si todavía existe la lista manual ``universe.tickers`` (o
    algún fork la reintroduce), sus tickers también se incluyen.
    """
    universe_cfg = config.get("universe", {}) if isinstance(config, dict) else {}
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(items: Any) -> None:
        for ticker in items or []:
            if isinstance(ticker, str):
                ticker = ticker.strip()
            if ticker and ticker not in seen:
                seen.add(ticker)
                ordered.append(ticker)

    by_strategy = universe_cfg.get("tickers_by_strategy", {}) or {}
    if isinstance(by_strategy, dict):
        for tickers in by_strategy.values():
            _add(tickers)

    _add(universe_cfg.get("extra_download_tickers", []))
    _add(universe_cfg.get("tickers", []))  # retrocompat / override manual

    return ordered


def get_training_window(
    config: dict[str, Any],
    granularity: str = "daily",
) -> tuple[Any, Any]:
    """Return (start_ts, end_ts) as UTC pd.Timestamps from config['data']['training_window'][granularity].

    Returns (None, None) if the section is missing — callers should treat that as "no filter".
    """
    import pandas as pd  # local import: utils.py avoids a global pandas dep

    window_cfg = get_nested(config, ["data", "training_window", granularity], default=None)
    if not isinstance(window_cfg, dict):
        return None, None

    start_raw = window_cfg.get("start")
    end_raw = window_cfg.get("end")

    start_ts = pd.Timestamp(start_raw, tz="UTC") if start_raw else None
    end_ts = pd.Timestamp(end_raw, tz="UTC") if end_raw else None
    return start_ts, end_ts


def apply_training_window(
    df: "Any",
    config: dict[str, Any],
    *,
    granularity: str = "daily",
    use_latest: bool = False,
    ticker: str | None = None,
):
    """Filter a DataFrame indexed by UTC timestamp to the configured training window.

    When use_latest is True, end is replaced by today (UTC midnight) so the caller
    trains on the newest data available. start is preserved either way.

    Logs a single line `[ticker] Training window: start -> end (N rows)` so baseline
    and champion runs can be verified to use the same slice.
    """
    import pandas as pd

    start_ts, end_ts = get_training_window(config, granularity=granularity)

    if use_latest:
        end_ts = pd.Timestamp.utcnow().normalize()

    if df is None or len(df) == 0:
        return df

    filtered = df
    if start_ts is not None:
        filtered = filtered[filtered.index >= start_ts]
    if end_ts is not None:
        filtered = filtered[filtered.index <= end_ts]

    prefix = f"[{ticker}] " if ticker else ""
    start_disp = start_ts.date() if start_ts is not None else "min"
    end_disp = end_ts.date() if end_ts is not None else "max"
    print(
        f"{prefix}Training window: {start_disp} -> {end_disp} "
        f"({len(filtered)} rows, granularity={granularity}, use_latest={use_latest})"
    )
    return filtered


def last_csv_timestamp(path: Path) -> "Any":
    """Return the latest `timestamp` (UTC pd.Timestamp) in an OHLCV CSV, or None.

    Used by the incremental download path to know from which date to fetch new
    bars. Returns None if the file is missing, empty, or has no parseable
    `timestamp` column — callers treat None as "no prior data" (bootstrap).
    """
    import pandas as pd  # local import: utils.py avoids a global pandas dep

    if path is None or not Path(path).exists():
        return None

    try:
        df = pd.read_csv(path, usecols=["timestamp"])
    except Exception:
        return None

    if df.empty or "timestamp" not in df.columns:
        return None

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return None
    return ts.max()


def resolve_lifecycle_paths(
    config: dict[str, Any],
    *,
    root: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve lifecycle registry and metrics log paths from config.

    Relative paths are resolved against the project root. If config values are
    missing, the historical defaults under models/ are used.
    """
    resolved_root = root or project_root()
    lifecycle_cfg = config.get("lifecycle", {}) if isinstance(config, dict) else {}

    registry_raw = lifecycle_cfg.get("registry_path", "models/registry.json")
    metrics_raw = lifecycle_cfg.get("metrics_log_path", "models/metrics_log.jsonl")

    registry_path = Path(registry_raw)
    metrics_log_path = Path(metrics_raw)

    if not registry_path.is_absolute():
        registry_path = resolved_root / registry_path
    if not metrics_log_path.is_absolute():
        metrics_log_path = resolved_root / metrics_log_path

    return registry_path, metrics_log_path


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
