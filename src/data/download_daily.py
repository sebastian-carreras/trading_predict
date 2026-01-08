"""
Descarga de datos OHLCV diarios para estrategias E1/E2/E4.

Según especificación:
- OHLCV ajustado (splits/dividendos)
- Benchmark SPY
- Calendario de mercado (solo días de trading)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..utils import ensure_dir, load_yaml, project_root


def download_daily_ohlcv(
    tickers: Iterable[str],
    out_dir: Path,
    period: str = "10y",
    auto_adjust: bool = True,
    skip_existing: bool = True,
    min_days_fresh: int = 7,
) -> list[Path]:
    """Descarga OHLCV diario usando yfinance.

    Args:
        tickers: Lista de símbolos a descargar
        out_dir: Directorio de salida (data/raw/)
        period: Período histórico (10y = 10 años)
        auto_adjust: Si True, ajusta por splits/dividendos
        skip_existing: Si True, no descarga tickers que ya existen
        min_days_fresh: Días mínimos para considerar archivo "fresco" (skip si reciente)

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
        
        print(f"Descargando {ticker}...")
        df = yf.download(
            tickers=ticker,
            period=period,
            auto_adjust=auto_adjust,
            progress=False,
            threads=True,
        )

        if df is None or df.empty:
            print(f"  ⚠️  Sin datos para {ticker}")
            continue

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
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        df = df.rename(columns=rename_map)

        # Timestamp como UTC-aware
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp")

        # Validación básica
        if len(df) < 252:
            print(f"  ⚠️  {ticker} tiene menos de 1 año de datos ({len(df)} días)")

        df.to_csv(out_path, index=False)
        written.append(out_path)
        print(f"  ✓ {len(df)} días guardados en {out_path.name}")

    if skipped:
        print(f"\n⏭️  {len(skipped)} tickers omitidos (ya existían): {', '.join(skipped[:5])}")
        if len(skipped) > 5:
            print(f"   ... y {len(skipped) - 5} más")
    
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

    # Universo global (E1 + E2 + benchmark)
    tickers = list(config.get("universe", {}).get("tickers", []))
    benchmark = config.get("universe", {}).get("benchmark", "SPY")

    if benchmark and benchmark not in tickers:
        tickers.append(benchmark)

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
