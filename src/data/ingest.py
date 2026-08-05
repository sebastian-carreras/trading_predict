"""
Orquestación de descarga de datos para el entrenamiento (E1/E2/E3).

Fuente única de la lógica de descarga que consumen tanto los pipelines de CLI
(`src/e{1,2,3}/train_pipeline.py`) como los DAGs de Airflow. Centraliza:

  1. La interpretación del bloque ``data.download`` de ``base.yaml``
     (`resolve_download_settings`).
  2. La orquestación descarga → limpieza según granularidad
     (`refresh_data_for_training`).

Comportamiento por defecto (config `data.download.enabled: true`): al entrenar se
descargan datos nuevos de forma incremental (solo las fechas faltantes) y se
acumulan en el CSV. Se puede desactivar por corrida con ``--skip-download`` en la
CLI, o de forma global con ``data.download.enabled: false``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..utils import get_nested, project_root


# Defaults usados cuando falta (parcial o totalmente) el bloque data.download.
_DEFAULTS = {
    "enabled": True,
    "incremental": True,
    "overlap_days": 5,
    "daily_period": "10y",
    "intraday_period": "60d",
}


def resolve_download_settings(config: dict[str, Any], granularity: str = "daily") -> dict[str, Any]:
    """Resuelve la configuración de descarga para una granularidad.

    Único punto que interpreta ``data.download`` de ``base.yaml``. Devuelve un dict
    con: ``enabled`` (bool), ``incremental`` (bool), ``overlap_days`` (int) y
    ``period`` (str, el de bootstrap según granularidad).
    """
    if granularity not in ("daily", "intraday"):
        raise ValueError(f"granularity inválida: {granularity!r} (usar 'daily' o 'intraday')")

    cfg = get_nested(config, ["data", "download"], default={}) or {}

    period_key = "daily_period" if granularity == "daily" else "intraday_period"
    return {
        "enabled": bool(cfg.get("enabled", _DEFAULTS["enabled"])),
        "incremental": bool(cfg.get("incremental", _DEFAULTS["incremental"])),
        "overlap_days": int(cfg.get("overlap_days", _DEFAULTS["overlap_days"])),
        "period": str(cfg.get(period_key, _DEFAULTS[period_key])),
    }


def refresh_data_for_training(
    config: dict[str, Any],
    tickers: Iterable[str],
    *,
    granularity: str = "daily",
    skip_download: bool = False,
    root: Path | None = None,
    raw_dir: Path | None = None,
    clean_dir: Path | None = None,
) -> bool:
    """Descarga (y limpia, si aplica) datos antes de entrenar.

    Args:
        config: Config cargada de ``base.yaml``.
        tickers: Símbolos a refrescar.
        granularity: ``"daily"`` (E1/E2/E4) o ``"intraday"`` (E3).
        skip_download: Si True, no descarga (usa datos existentes). Override por
            corrida equivalente a ``--skip-download`` en la CLI.
        root: Raíz del proyecto (default: ``project_root()``).
        raw_dir: Override del directorio raw (los DAGs lo usan para staging).
        clean_dir: Override del directorio clean (solo diario).

    Returns:
        True si se descargó, False si se omitió (skip o ``enabled: false``).
    """
    tickers = [t for t in tickers if t]
    settings = resolve_download_settings(config, granularity=granularity)

    # Features exógenas (macro / cross-asset): refresco no fatal del cache global,
    # solo si data.exog.enabled y no se pidió --skip-download (que significa "usar los
    # datos/cachés existentes"). Si el cache no existe, load_exog_for lo construye al
    # vuelo la primera vez. Ver src/data/macro.py.
    if granularity == "daily" and not skip_download:
        try:
            from .macro import exog_enabled, refresh_exog

            if exog_enabled(config):
                print("\nRefrescando features exógenas (macro/cross-asset)...")
                if refresh_exog(config, root=root):
                    print("✓ Cache exógeno actualizado\n")
        except Exception as exc:
            print(f"⚠️  Exógenas (refresco omitido): {exc}\n")

    if skip_download or not settings["enabled"]:
        motivo = "--skip-download" if skip_download else "data.download.enabled=false"
        print(f"\nUsando datos existentes (descarga omitida: {motivo})\n")
        return False

    resolved_root = root or project_root()
    modo = "incremental" if settings["incremental"] else f"período completo {settings['period']}"

    if granularity == "daily":
        raw = raw_dir or (resolved_root / "data" / "raw" / "daily")
        clean = clean_dir or (resolved_root / "data" / "clean")

        print(f"\nPaso 1/2: Descargando datos diarios ({modo})...")
        try:
            from .download_daily import download_daily_ohlcv

            written = download_daily_ohlcv(
                tickers,
                out_dir=raw,
                period=settings["period"],
                skip_existing=False,
                incremental=settings["incremental"],
                overlap_days=settings["overlap_days"],
            )
            print(f"✓ Descargados/actualizados {len(written)} archivos\n")
        except Exception as exc:
            print(f"⚠️  Descarga: {exc}\n")

        print("Paso 2/2: Limpiando datos...")
        try:
            from .clean_daily import process_daily_data_with_cleaning

            reports = process_daily_data_with_cleaning(
                raw_dir=raw,
                clean_dir=clean,
                strategy="forward_fill",
                min_days=252,
                remove_zero_volume=True,
                verbose=False,
                tickers=list(dict.fromkeys(tickers)),
            )
            cleaned = sum(1 for r in reports.values() if r.get("status") == "cleaned")
            print(f"✓ Limpiados: {cleaned}\n")
        except Exception as exc:
            print(f"⚠️  Limpieza: {exc}\n")

    else:  # intraday (E3): descarga sin limpieza en esta etapa
        raw = raw_dir or (resolved_root / "data" / "raw" / "intraday")

        print(f"\nDescargando datos intradía 5-min ({modo})...")
        try:
            from ..e3.intraday_data import download_ohlcv_5m

            written = download_ohlcv_5m(
                tickers,
                out_dir=raw,
                period=settings["period"],
                interval="5m",
                accumulate=True,
                incremental=settings["incremental"],
                overlap_days=settings["overlap_days"],
            )
            print(f"✓ Descargados/actualizados {len(written)} archivos\n")
        except Exception as exc:
            print(f"⚠️  Descarga: {exc}\n")

    return True
