"""
Descarga y carga de barras OHLCV de 5-min para la estrategia E3.

Prioridad de fuentes por tipo de ticker:
  Ticker argentino (.BA):  IOL  → yfinance
  Ticker US (SPY, AAPL…): Alpaca → yfinance

IOL actualmente solo provee datos diarios (no 5-min). La integración está preparada
para cuando/si IOL agregue un endpoint intradiario.

Alpaca Markets (alpaca-py) es la fuente primaria para acciones US: da 2+ años de
historia de 5-min de forma gratuita.

La función principal `download_ohlcv_5m()` acumula datos en el CSV existente
en lugar de sobreescribir. Cada ejecución agrega solo las barras nuevas.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..utils import ensure_dir

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Columnas estándar que guardamos en el CSV (orden fijo).
# `adj_close` es opcional en algunas fuentes; cuando no existe, se deriva de `close`.
_OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "adj_close", "volume"]


# ---------------------------------------------------------------------------
# Helpers privados: una función por fuente
# ---------------------------------------------------------------------------

def _fetch_alpaca(ticker: str) -> pd.DataFrame | None:
    """Descarga barras de 5-min desde Alpaca Markets (histórico completo ~2 años).

    Requiere variables de entorno:
        ALPACA_API_KEY    — API key de Alpaca
        ALPACA_SECRET_KEY — Secret key de Alpaca

    Retorna DataFrame con columnas estándar OHLCV indexado por timestamp UTC,
    o None si las credenciales no están disponibles o la descarga falla.
    """
    api_key = os.getenv("ALPACA_API_KEY", "").strip()
    secret_key = os.getenv("ALPACA_SECRET_KEY", "").strip()

    if not api_key or not secret_key:
        return None

    try:
        from alpaca.data.enums import Adjustment                       # type: ignore
        from alpaca.data.historical import StockHistoricalDataClient  # type: ignore
        from alpaca.data.requests import StockBarsRequest             # type: ignore
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit    # type: ignore
    except ImportError:
        print(f"  ⚠️  alpaca-py no instalado. Instalar con: pip install alpaca-py")
        return None

    try:
        # Fecha de inicio: 2 años atrás para maximizar historia disponible
        start = datetime(datetime.now().year - 2, 1, 1, tzinfo=timezone.utc)

        client = StockHistoricalDataClient(api_key, secret_key)
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start,
            adjustment=Adjustment.SPLIT,
        )
        bars = client.get_stock_bars(request)
        raw_df = bars.df

        if raw_df is None or raw_df.empty:
            return None

        # Alpaca devuelve MultiIndex (symbol, timestamp) para múltiples tickers.
        # Para un solo ticker puede devolver índice simple con nombre 'timestamp'.
        if isinstance(raw_df.index, pd.MultiIndex):
            raw_df = raw_df.xs(ticker, level="symbol")

        # El índice puede llamarse 'timestamp' o ser el DatetimeIndex directamente
        df = raw_df.reset_index()
        if "timestamp" not in df.columns:
            # renombrar la primera columna de índice al nombre estándar
            df = df.rename(columns={df.columns[0]: "timestamp"})

        # Normalizar nombres de columnas (Alpaca ya devuelve lower case)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Alpaca con Adjustment.SPLIT ya entrega precios ajustados por split.
        # Guardamos también `adj_close` para unificar esquema entre fuentes.
        if "adj_close" not in df.columns and "close" in df.columns:
            df["adj_close"] = df["close"]

        # Seleccionar solo columnas OHLCV estándar
        rename = {"open": "open", "high": "high", "low": "low",
                  "close": "close", "volume": "volume"}
        df = df.rename(columns=rename)
        available = [c for c in _OHLCV_COLS if c in df.columns]
        return df[available]

    except Exception as exc:
        print(f"  ⚠️  Alpaca falló para {ticker}: {exc}")
        return None


def _fetch_iol_5m(ticker: str) -> pd.DataFrame | None:
    """Placeholder: descarga barras de 5-min desde IOL.

    IOL actualmente solo expone datos diarios en su API pública (seriehistorica).
    Esta función retorna None + warning hasta que IOL agregue un endpoint intradiario.

    Cuando IOL soporte 5-min, implementar aquí usando IOLClient de src/data/iol_api.py.
    """
    print(
        f"  ⚠️  IOL no soporta datos intradiarios (5-min) en su API pública actual. "
        f"Usando fallback para {ticker}."
    )
    return None


def _fetch_yfinance(ticker: str, period: str = "60d", interval: str = "5m") -> pd.DataFrame | None:
    """Descarga barras de 5-min desde yfinance (límite ~60 días de historia).

    Retorna DataFrame con columnas estándar OHLCV, o None si falla.
    """
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        print("  ⚠️  yfinance no instalado. Instalar con: pip install yfinance")
        return None

    try:
        raw = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw.reset_index()
        # yfinance puede devolver columna "Datetime" o "Date"
        for col in ("Datetime", "Date"):
            if col in raw.columns:
                raw = raw.rename(columns={col: "timestamp"})
                break
        if "timestamp" not in raw.columns:
            raw = raw.rename(columns={raw.columns[0]: "timestamp"})

        rename_map = {
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
        }
        raw = raw.rename(columns=rename_map)

        # En intraday puede no venir `Adj Close`; mantenemos esquema consistente.
        if "adj_close" not in raw.columns and "close" in raw.columns:
            raw["adj_close"] = raw["close"]

        # Estandar interno: `close` representa el cierre ajustado cuando está disponible.
        if "adj_close" in raw.columns:
            raw["close"] = raw["adj_close"]

        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)

        available = [c for c in _OHLCV_COLS if c in raw.columns]
        return raw[available]

    except Exception as exc:
        print(f"  ⚠️  yfinance falló para {ticker}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Función pública principal
# ---------------------------------------------------------------------------

def download_ohlcv_5m(
    tickers: Iterable[str],
    out_dir: Path,
    period: str = "60d",
    interval: str = "5m",
    accumulate: bool = True,
) -> list[Path]:
    """Descarga barras OHLCV de 5-min y acumula en CSV existente.

    Prioridad de fuentes:
      - Ticker argentino (.BA): IOL → yfinance
      - Ticker US:              Alpaca → yfinance

    Args:
        tickers:    Símbolos a descargar.
        out_dir:    Directorio de salida (data/raw/intraday/).
        period:     Período para yfinance si Alpaca no está disponible (default "60d").
        interval:   Intervalo de barras, solo aplica a yfinance (default "5m").
        accumulate: Si True (default), mergea con datos existentes en lugar de
                    sobreescribir. Permite construir un historial largo ejecutando
                    periódicamente.

    Returns:
        Lista de paths CSV escritos/actualizados.
    """
    from ..data.download_daily import _is_argentino  # reutilizar detección de mercado

    ensure_dir(out_dir)
    written: list[Path] = []

    for ticker in tickers:
        print(f"\nDescargando {ticker} (5-min)...")

        # Elegir cadena de fuentes según mercado
        if _is_argentino(ticker):
            sources = [
                ("IOL",     lambda t=ticker: _fetch_iol_5m(t)),
                ("yfinance", lambda t=ticker: _fetch_yfinance(t, period, interval)),
            ]
        else:
            sources = [
                ("Alpaca",   lambda t=ticker: _fetch_alpaca(t)),
                ("yfinance", lambda t=ticker: _fetch_yfinance(t, period, interval)),
            ]

        df_new: pd.DataFrame | None = None
        source_used = "ninguna"

        for source_name, fetch_fn in sources:
            df_new = fetch_fn()
            if df_new is not None and not df_new.empty:
                source_used = source_name
                break

        if df_new is None or df_new.empty:
            print(f"  ✗ Sin datos para {ticker} en ninguna fuente.")
            continue

        out_path = out_dir / f"{ticker}_5m.csv"

        if accumulate and out_path.exists():
            df_existing = pd.read_csv(out_path)
            df_existing["timestamp"] = pd.to_datetime(
                df_existing["timestamp"], format="ISO8601", utc=True
            )
            n_before = len(df_existing)

            df_merged = (
                pd.concat([df_existing, df_new], ignore_index=True)
                .drop_duplicates(subset="timestamp")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            n_added = len(df_merged) - n_before
            df_merged.to_csv(out_path, index=False)
            written.append(out_path)
            print(
                f"  ✓ {ticker} ({source_used}): {len(df_merged)} barras totales "
                f"(+{n_added} nuevas, {n_before} existentes)"
            )
        else:
            df_new.to_csv(out_path, index=False)
            written.append(out_path)
            print(f"  ✓ {ticker} ({source_used}): {len(df_new)} barras descargadas")

    return written


def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    """Carga un CSV de barras OHLCV con timestamp UTC como índice."""
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing 'timestamp' column in {path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")

    # Compatibilidad retroactiva: CSVs viejos no tienen `adj_close`.
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    # Estandar interno: usar siempre `close` ajustado si está disponible.
    if "adj_close" in df.columns:
        df["close"] = df["adj_close"]

    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")

    return df
