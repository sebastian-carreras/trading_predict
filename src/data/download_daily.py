"""
Descarga de datos OHLCV diarios para estrategias E1/E2/E4.

Según especificación:
- OHLCV ajustado (splits/dividendos)
- Calendario de mercado (solo días de trading)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..utils import ensure_dir, get_universe_tickers, last_csv_timestamp, load_yaml, project_root


def _is_argentino(ticker: str) -> bool:
    return (
        ticker.endswith(".BA") or
        ticker.startswith("AL") or
        ticker.startswith("GD") or
        ticker.startswith("AE") or
        ticker.startswith("AY") or
        ticker.startswith("TV") or
        ticker.startswith("T2")
    )


def download_daily_ohlcv(
    tickers: Iterable[str],
    out_dir: Path,
    period: str = "10y",
    auto_adjust: bool = True,
    skip_existing: bool = True,
    use_iol_fallback: bool = True,
    incremental: bool = False,
    overlap_days: int = 5,
) -> list[Path]:
    """Descarga OHLCV diario usando yfinance con fallback a IOL API.

    Si el CSV ya existe y skip_existing=False, los nuevos datos se acumulan
    (merge + deduplicación por timestamp) en lugar de sobreescribir.

    Args:
        tickers: Lista de símbolos a descargar
        out_dir: Directorio de salida (data/raw/)
        period: Período histórico (10y = 10 años). En modo incremental se usa solo
            como bootstrap cuando el CSV del ticker aún no existe.
        auto_adjust: Si True, ajusta por splits/dividendos
        skip_existing: Si True, omite tickers cuyo CSV ya existe
        use_iol_fallback: Si True, intenta IOL API cuando Yahoo Finance falla (tickers .BA)
        incremental: Si True y el CSV existe, descarga solo desde la última fecha
            registrada (menos ``overlap_days``) en lugar del período completo.
            El resultado se acumula igual (merge + dedup por timestamp).
        overlap_days: Días de solapamiento re-solicitados en modo incremental
            (robustez ante barras tardías / correcciones).

    Returns:
        Lista de archivos CSV creados/actualizados
    """
    from datetime import timedelta as _timedelta
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'yfinance'. Install with: pip install yfinance"
        ) from exc

    ensure_dir(out_dir)
    written: list[Path] = []
    skipped: list[str] = []
    failed: list[str] = []

    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()  # Cargar variables desde .env
    except Exception:
        pass  # python-dotenv no instalado o fallo al cargar

    # Verificar si IOL está disponible (credenciales configuradas)
    iol_available = use_iol_fallback and os.getenv("IOL_USERNAME") and os.getenv("IOL_PASSWORD")
    if use_iol_fallback and not iol_available:
        print("⚠️  IOL fallback deshabilitado: falta IOL_USERNAME o IOL_PASSWORD en variables de entorno")

    for ticker in tickers:
        out_path = out_dir / f"{ticker}_daily.csv"
        
        if skip_existing and out_path.exists():
            print(f"⏭️  {ticker} ya existe - omitido (usa --skip-download para reutilizar sin descargar)")
            skipped.append(ticker)
            continue

        # Descarga incremental: si el CSV existe, bajar solo desde la última fecha
        # (menos overlap) en vez del período completo. Bootstrap con `period` si no existe.
        fetch_start = None
        if incremental:
            last_ts = last_csv_timestamp(out_path)
            if last_ts is not None:
                last_date = last_ts.tz_convert("UTC").normalize()
                today = pd.Timestamp.now(tz="UTC").normalize()
                if last_date >= today:
                    # Ya tenemos datos hasta hoy: no puede haber una barra más nueva
                    # que Yahoo Finance pueda devolver, así que evitamos el request.
                    print(f"⏭️  {ticker} ya está al día (última fecha {last_date.date()}) - sin descarga")
                    skipped.append(ticker)
                    continue
                fetch_start = (last_ts - _timedelta(days=overlap_days)).strftime("%Y-%m-%d")

        # Intentar Yahoo Finance primero
        origen = f"desde {fetch_start} (incremental)" if fetch_start else f"período {period}"
        print(f"Descargando {ticker} desde Yahoo Finance ({origen})...")
        success = False

        try:
            if fetch_start:
                df = yf.download(
                    tickers=ticker,
                    start=fetch_start,
                    auto_adjust=auto_adjust,
                    progress=False,
                    threads=True,
                )
            else:
                df = yf.download(
                    tickers=ticker,
                    period=period,
                    auto_adjust=auto_adjust,
                    progress=False,
                    threads=True,
                )

            if df is not None and not df.empty:
                # Aplanar columnas MultiIndex si existe
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Resetear índice y renombrar columnas
                df = df.reset_index()
                if "Date" in df.columns:
                    df = df.rename(columns={"Date": "timestamp"})
                elif "Datetime" in df.columns:
                    df = df.rename(columns={"Datetime": "timestamp"})

                # Estandarizar nombres de columnas
                rename_map = {
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",  # Versiones antiguas de yfinance
                    "Volume": "volume",
                }
                df = df.rename(columns=rename_map)
                
                # Si no existe adj_close (yfinance nuevo), usar close
                if "close" in df.columns and "adj_close" not in df.columns:
                    df["adj_close"] = df["close"]

                # Timestamp como UTC-aware
                df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601', utc=True)
                df = df.sort_values("timestamp")

                # Validación básica (solo en bootstrap: en incremental el slice es
                # deliberadamente corto y este chequeo no aplica al CSV acumulado).
                if not fetch_start and len(df) < 252:
                    print(f"  ⚠️  {ticker} tiene menos de 1 año de datos ({len(df)} días)")

                chosen_df = df
                chosen_source = "YFinance"

                if _is_argentino(ticker) and iol_available:
                    try:
                        from .iol_api import IOLClient

                        years = 10
                        if period.endswith("y"):
                            try:
                                years = int(period[:-1])
                            except ValueError:
                                pass

                        from datetime import datetime, timedelta

                        end_date = datetime.now().strftime("%Y-%m-%d")
                        # En modo incremental acotamos IOL a la misma ventana que yfinance
                        # (evita re-bajar toda la historia y ganarle siempre por longitud).
                        start_date = fetch_start or (
                            datetime.now() - timedelta(days=years * 365)
                        ).strftime("%Y-%m-%d")

                        client = IOLClient()
                        iol_df = client.get_historical_data(ticker, start_date, end_date)

                        if not iol_df.empty and len(iol_df) > len(chosen_df):
                            chosen_df = iol_df
                            chosen_source = "IOL"
                    except Exception as e:
                        print(f"  ⚠️  IOL comparación falló para {ticker}: {e}")

                if out_path.exists():
                    existing = pd.read_csv(out_path)
                    # format='ISO8601' tolera timestamps mixtos (con/sin microsegundos):
                    # los datos de IOL traen segundos fraccionarios y los de yfinance no.
                    # errors='coerce' + dropna: una fila corrupta (append partido que deja
                    # un número en la columna timestamp) se vuelve NaT y se descarta, en
                    # vez de abortar el merge y hacer fallar toda la descarga.
                    existing["timestamp"] = pd.to_datetime(
                        existing["timestamp"], format="ISO8601", utc=True, errors="coerce"
                    )
                    n_bad = int(existing["timestamp"].isna().sum())
                    if n_bad:
                        print(f"  🗑️  {ticker}: descartadas {n_bad} filas corruptas del CSV existente")
                        existing = existing.dropna(subset=["timestamp"]).reset_index(drop=True)
                    before = len(existing)
                    chosen_df = (
                        pd.concat([existing, chosen_df], ignore_index=True)
                        .drop_duplicates(subset="timestamp")
                        .sort_values("timestamp")
                        .reset_index(drop=True)
                    )
                    added = len(chosen_df) - before
                    print(f"  ↕️  Acumulando: {before} existentes + {added} nuevos = {len(chosen_df)} días")

                chosen_df.to_csv(out_path, index=False)
                written.append(out_path)
                print(f"  ✓ {chosen_source}: {len(chosen_df)} días guardados en {out_path.name}")
                success = True
            else:
                print(f"  ⚠️  YFinance: Sin datos para {ticker}")
        
        except Exception as e:
            print(f"  ⚠️  YFinance falló para {ticker}: {e}")
        
        # Si Yahoo Finance falló, intentar IOL para activos argentinos
        # Incluye: .BA (acciones), bonos soberanos (AL*, GD*, AE*), etc.
        is_argentino = _is_argentino(ticker)
        
        if not success and is_argentino and iol_available:
            print(f"  🔄 Intentando fallback con IOL API...")
            try:
                from .iol_api import download_iol_daily
                
                # Calcular años desde period (ej: "10y" -> 10)
                years = 10
                if period.endswith("y"):
                    try:
                        years = int(period[:-1])
                    except ValueError:
                        pass
                
                iol_path = download_iol_daily(ticker, out_dir, years=years)
                if iol_path:
                    written.append(iol_path)
                    success = True
            
            except Exception as e:
                print(f"  ⚠️  IOL también falló para {ticker}: {e}")
        
        if not success:
            failed.append(ticker)

    if skipped:
        print(f"\n⏭️  {len(skipped)} tickers omitidos (ya existían): {', '.join(skipped[:5])}")
        if len(skipped) > 5:
            print(f"   ... y {len(skipped) - 5} más")
    
    if failed:
        print(f"\n❌ {len(failed)} tickers fallaron: {', '.join(failed[:10])}")
        if len(failed) > 10:
            print(f"   ... y {len(failed) - 10} más")
    
    return written


def main() -> None:
    """Entrypoint: descarga datos para universo definido en base.yaml.
    
    Uso:
        python -m src.data.download_daily              # Solo descarga nuevos (skip_existing=True)
        python -m src.data.download_daily --force      # Fuerza reDescarga de todos
    """
    import sys
    
    root = project_root()
    config = load_yaml(root / "src/config/base.yaml")

    # Universo de descarga: unión de tickers_by_strategy + extra_download_tickers
    tickers = get_universe_tickers(config)

    if not tickers:
        raise ValueError(
            "No hay tickers para descargar: revisá universe.tickers_by_strategy "
            "y universe.extra_download_tickers en base.yaml"
        )

    # Control de argumentos
    force_download = "--force" in sys.argv or "-f" in sys.argv
    skip_existing = not force_download
    
    out_dir = root / "data" / "raw" / "daily"
    mode_str = "forzada" if force_download else "incremental (skip existentes)"
    print(f"Descarga {mode_str} de {len(tickers)} tickers a {out_dir}\n")

    written = download_daily_ohlcv(
        tickers,
        out_dir=out_dir,
        period="10y",
        skip_existing=skip_existing,
    )

    print(f"\n✓ Descargados {len(written)} archivos nuevos a {out_dir}")


if __name__ == "__main__":
    main()
