from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..utils import ensure_dir


def download_ohlcv_5m(
    tickers: Iterable[str],
    out_dir: Path,
    period: str = "60d",
    interval: str = "5m",
) -> list[Path]:
    """Download intraday OHLCV bars using yfinance.

    Notes:
    - Yahoo Finance typically limits 5m history (~60 days).
    - Output is one CSV per ticker, with a UTC timestamp column.
    """
    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'yfinance'. Install with: pip install yfinance"
        ) from exc

    ensure_dir(out_dir)

    written: list[Path] = []
    for ticker in tickers:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        if df is None or df.empty:
            continue

        df = df.reset_index().rename(columns={"Datetime": "timestamp", "Date": "timestamp"})
        if "timestamp" not in df.columns:
            # yfinance can return an index without name
            df = df.rename(columns={df.columns[0]: "timestamp"})

        # Standardize column names
        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        df = df.rename(columns=rename_map)

        # Ensure timestamp is UTC-naive ISO for portability
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        out_path = out_dir / f"{ticker}_5m.csv"
        df.to_csv(out_path, index=False)
        written.append(out_path)

    return written


def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing 'timestamp' column in {path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")

    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")

    return df
