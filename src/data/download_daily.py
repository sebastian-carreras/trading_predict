"""
Descarga de datos OHLCV diarios para estrategias E1/E2/E4.

Según especificación:
- OHLCV ajustado (splits/dividendos)
- Benchmark SPY
- Calendario de mercado (solo días de trading)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..utils import ensure_dir, load_yaml, project_root


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
    min_days_fresh: int = 7,
    use_iol_fallback: bool = True,
) -> list[Path]:
    """Descarga OHLCV diario usando yfinance con fallback a IOL API.

    Args:
        tickers: Lista de símbolos a descargar
        out_dir: Directorio de salida (data/raw/)
        period: Período histórico (10y = 10 años)
        auto_adjust: Si True, ajusta por splits/dividendos
        skip_existing: Si True, no descarga tickers que ya existen
        min_days_fresh: Días mínimos para considerar archivo "fresco" (skip si reciente)
        use_iol_fallback: Si True, intenta IOL API cuando Yahoo Finance falla (tickers .BA)

    Returns:
        Lista de archivos CSV creados
    """
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
        
        # Control: evitar descargar si ya existe y es reciente
        if skip_existing and out_path.exists():
            from datetime import datetime, timezone
            
            file_age_days = (
                datetime.now(timezone.utc) - 
                datetime.fromtimestamp(out_path.stat().st_mtime, tz=timezone.utc)
            ).days
            
            if file_age_days < min_days_fresh:
                print(f"⏭️  {ticker} ya existe ({file_age_days} días) - omitido")
                skipped.append(ticker)
                continue
            else:
                print(f"♻️  {ticker} existe pero antiguo ({file_age_days} días) - reDescargando...")
        
        # Intentar Yahoo Finance primero
        print(f"Descargando {ticker} desde Yahoo Finance...")
        success = False
        
        try:
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

                # Validación básica
                if len(df) < 252:
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
                        start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

                        client = IOLClient()
                        iol_df = client.get_historical_data(ticker, start_date, end_date)

                        if not iol_df.empty and len(iol_df) > len(chosen_df):
                            chosen_df = iol_df
                            chosen_source = "IOL"
                    except Exception as e:
                        print(f"  ⚠️  IOL comparación falló para {ticker}: {e}")

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

    # Universo global (sin benchmark forzado)
    tickers = list(config.get("universe", {}).get("tickers", []))

    if not tickers:
        raise ValueError("No tickers found in base.yaml universe.tickers")

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
        min_days_fresh=7,  # Considerar "fresco" si < 7 días
    )

    print(f"\n✓ Descargados {len(written)} archivos nuevos a {out_dir}")


if __name__ == "__main__":
    main()
