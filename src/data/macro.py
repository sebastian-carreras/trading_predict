"""
Features exógenas (macro / cross-asset) para E2 — Fase 1: Bloques A + B + C + D.

Hoy los modelos entrenan solo con features derivadas del propio OHLCV del ticker.
Este módulo agrega el CONTEXTO que domina a 10-90 días: apetito por riesgo
(cross-asset), fuerza relativa sectorial, macro fundamental y commodities.

Bloques (ver docs/FEATURES.md):
  A · Cross-asset:  vix_ts, vix_pctl_252, hy_oas_z, curve_10y2y, dxy_mom_20, spy_mom_20/60
  B · Sector:       rel_strength_sector, sector_rotation, beta_60 (derivadas en build_features)
  C · Macro FRED:   breakeven_10y, jobless_claims_z, nfci (+ rate_shock_10y)
  D · Commodities:  oil_mom_20, copper_mom_20, gold_mom_20, soy_mom_20

Diseño (acorde al CLAUDE.md: simple y robusto, sin dependencias nuevas):
  - FRED se baja por su endpoint CSV público (fredgraph.csv) con requests+pandas.
    No requiere API key. Para PIT estricto (vintages) se puede migrar a ALFRED.
  - El resto (VIX, ETFs, futuros) por yfinance, igual que download_daily.py.
  - Point-in-time (PIT): cada serie se sella con su fecha *known-as-of* = fecha de
    referencia + offset de publicación fijo por serie (``_PUB_LAG``), y luego se hace
    ffill a un calendario diario. Usar la fecha de referencia = filtrar futuro.
  - Estacionariedad: deltas / momentum / z-scores / percentiles, nunca niveles crudos
    (salvo series que ya son diferencias/estandarizadas: curve_10y2y, breakeven_10y, nfci).
  - Fetch NO fatal: si una serie falla, se omite su columna con warning (no rompe el
    entrenamiento). Detrás del flag ``data.exog.enabled`` (default false) → cero impacto
    en el pipeline diario del champion hasta que la ablación pruebe su valor.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils import ensure_dir, get_nested, load_yaml, project_root

# ── Registro de series FRED: id → offset de publicación en días (PIT) ──────────
# Diarias market-based (breakeven, tasas, curva, OAS) se publican al día hábil
# siguiente (lag 1). Semanales (claims, NFCI) llegan con ~1 semana de rezago.
_PUB_LAG: dict[str, int] = {
    "T10YIE": 1,          # breakeven inflación 10Y (diaria)
    "DGS10": 1,           # UST 10Y (diaria)
    "T10Y2Y": 1,          # pendiente curva 10Y-2Y (diaria)
    "ICSA": 6,            # initial jobless claims (semanal)
    "NFCI": 8,            # National Financial Conditions Index (semanal)
}
# NOTA: el spread de crédito NO se baja de FRED. La serie ICE BofA HY OAS
# (BAMLH0A0HYM2) tiene la historia pública RESTRINGIDA en FRED (por licencia): solo
# devuelve ~3 años, lo que dejaba hy_oas_z NaN antes de 2023 y recortaba el train set
# de los modelos exógenos. Se reemplazó por un proxy HYG/LQD (yfinance, historia completa
# 2007+) — ver _MARKET y _compute_transforms.

# ── Series de mercado (yfinance): símbolo → clave interna ─────────────────────
_MARKET: dict[str, str] = {
    "^VIX": "vix",
    "^VIX3M": "vix3m",
    "DX-Y.NYB": "dxy",
    "SPY": "SPY",
    "HYG": "HYG",   # high-yield corp (proxy de crédito HY)
    "LQD": "LQD",   # investment-grade corp (proxy de crédito IG)
    "CL=F": "oil",
    "HG=F": "copper",
    "GC=F": "gold",
    "ZS=F": "soy",
    "XLK": "XLK", "XLF": "XLF", "XLV": "XLV", "XLE": "XLE", "XLY": "XLY",
    "XLP": "XLP", "XLI": "XLI", "XLC": "XLC", "XLU": "XLU", "XLB": "XLB", "XLRE": "XLRE",
}

_SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLY", "XLP", "XLI", "XLC", "XLU", "XLB", "XLRE"]

# Columnas-feature globales (passthrough) que consume build_features. Las columnas
# helper (sector_ret20/spy_ret20/spy_ret1d) NO son features: alimentan al Bloque B.
GLOBAL_FEATURE_COLS = [
    # A
    "vix_ts", "vix_pctl_252", "hy_oas_z", "curve_10y2y", "dxy_mom_20", "spy_mom_20", "spy_mom_60",
    # C
    "rate_shock_10y", "breakeven_10y", "jobless_claims_z", "nfci",
    # D
    "oil_mom_20", "copper_mom_20", "gold_mom_20", "soy_mom_20",
]

# Nombres que build_features tratará como INPUTS del Bloque B (no como features).
EXOG_HELPER_COLS = ("sector_ret20", "spy_ret20", "spy_ret1d")

# Features derivadas per-ticker del Bloque B (calculadas en build_features, no globales).
EXOG_DERIVED_COLS = ("rel_strength_sector", "sector_rotation", "beta_60")

# Conjunto completo de nombres de features exógenas — lo usa reevaluation.py para
# detectar si un modelo (por sus feature_names guardados) necesita el exog al recomputar.
EXOG_FEATURE_COLS = tuple(GLOBAL_FEATURE_COLS) + EXOG_DERIVED_COLS

_CACHE_REL = Path("data") / "macro" / "exog_global.csv"


# ──────────────────────────────────────────────────────────────────────────────
# Fetch de series crudas
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_fred_csv(series_id: str, *, start: str = "1990-01-01", verbose: bool = True, retries: int = 3) -> pd.Series | None:
    """Descarga una serie de FRED por su endpoint CSV público (sin API key).

    ``cosd`` (Change Observation Start Date) fuerza el rango COMPLETO: sin él,
    ``fredgraph.csv`` trunca algunas series a ~3 años (bug observado con
    BAMLH0A0HYM2, que dejaba hy_oas_z NaN antes de 2023 y recortaba el train set
    de los modelos exógenos). Reintentos con backoff por la flakiness de FRED.
    """
    import time as _time

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; trading_predict/1.0)"}
    df = None
    for attempt in range(retries):
        try:
            import requests  # dependencia transitiva de yfinance

            resp = requests.get(url, timeout=60, headers=headers)
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text), na_values=".")
            break
        except Exception as exc:
            if attempt < retries - 1:
                _time.sleep(5 * (attempt + 1))  # backoff
                continue
            try:
                df = pd.read_csv(url, na_values=".")  # último intento: pandas directo
            except Exception as exc2:
                if verbose:
                    print(f"  ⚠️  FRED {series_id}: {exc2}")
                return None

    if df is None or df.shape[1] < 2:
        return None
    date_col = df.columns[0]
    val_col = series_id if series_id in df.columns else df.columns[-1]
    s = pd.Series(
        pd.to_numeric(df[val_col], errors="coerce").to_numpy(),
        index=pd.to_datetime(df[date_col], errors="coerce"),
        name=series_id,
    )
    s = s[~s.index.isna()].dropna().sort_index()
    return s if len(s) else None


def _fetch_yf_close(yf_ticker: str, *, start: str = "2014-01-01", verbose: bool = True) -> pd.Series | None:
    """Descarga el cierre ajustado de un símbolo con yfinance (índice fecha naive)."""
    try:
        import yfinance as yf

        df = yf.download(yf_ticker, start=start, auto_adjust=True, progress=False, threads=False)
    except Exception as exc:
        if verbose:
            print(f"  ⚠️  yfinance {yf_ticker}: {exc}")
        return None
    if df is None or df.empty:
        if verbose:
            print(f"  ⚠️  yfinance {yf_ticker}: sin datos")
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        return None
    s = df["Close"].copy()
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    s.index = idx.normalize()
    s = s[~s.index.duplicated(keep="last")].dropna().sort_index()
    s.name = yf_ticker
    return s if len(s) else None


def _pit_shift(s: pd.Series, lag_days: int) -> pd.Series:
    """Sella la serie con su fecha known-as-of desplazando el índice por el lag."""
    if lag_days and lag_days > 0:
        s = s.copy()
        s.index = s.index + pd.Timedelta(days=int(lag_days))
    return s


def _filter_low_coverage(
    raw: dict[str, pd.Series],
    idx: pd.DatetimeIndex,
    *,
    min_coverage: float = 0.9,
    verbose: bool = True,
) -> dict[str, pd.Series]:
    """Descarta series con cobertura histórica insuficiente sobre el rango esperado.

    GUARDRAIL contra series truncadas (p.ej. las ICE BofA en FRED, con historia pública
    restringida): una serie que arranca mucho después del inicio del rango produciría una
    columna casi-toda-NaN que, tras el ``dropna`` de ``make_sequences``, RECORTA en silencio
    el set de entrenamiento de los modelos exógenos (bug real de 2026-07). Acá se detecta y
    se DESCARTA la serie (con warning), para que su feature simplemente no se calcule y aguas
    abajo se saltee — en vez de contaminar el entrenamiento. Es el equivalente exógeno del
    ``min_days`` de ``clean_daily.py`` para el OHLCV.

    Cobertura = fracción del rango esperado (``idx``) en o después de la primera observación
    de la serie (tras ffill solo quedan NaN de "gap inicial", así que esto la mide directo).
    """
    kept: dict[str, pd.Series] = {}
    dropped: list[str] = []
    for key, s in raw.items():
        cov = float((idx >= s.index.min()).mean()) if len(s) else 0.0
        if cov < min_coverage:
            dropped.append(f"{key} ({cov:.0%}, desde {s.index.min().date() if len(s) else 'N/A'})")
        else:
            kept[key] = s
    if dropped and verbose:
        print(f"  ⚠️  COBERTURA INSUFICIENTE (<{min_coverage:.0%}) — series DESCARTADAS: {', '.join(dropped)}")
        print("      (su feature exógena no se calculará; revisá la fuente antes de confiar en ella)")
    return kept


# ──────────────────────────────────────────────────────────────────────────────
# Construcción del panel exógeno global (con transforms estacionarios)
# ──────────────────────────────────────────────────────────────────────────────

def build_global_exog(
    *,
    start: str = "2014-01-01",
    root: Path | None = None,
    write: bool = True,
    verbose: bool = True,
    min_coverage: float = 0.9,
) -> pd.DataFrame:
    """Descarga, alinea (PIT) y transforma todas las series exógenas globales.

    Devuelve un DataFrame indexado por día hábil con las columnas-feature globales
    (``GLOBAL_FEATURE_COLS``) más las helper del Bloque B (``{ETF}_ret20``, ``SPY_ret20``,
    ``SPY_ret1d``). Si ``write``, cachea a ``data/macro/exog_global.csv``.
    """
    root = root or project_root()
    raw: dict[str, pd.Series] = {}

    if verbose:
        print("Descargando series FRED...")
    for sid, lag in _PUB_LAG.items():
        s = _fetch_fred_csv(sid, verbose=verbose)
        if s is not None:
            raw[sid] = _pit_shift(s, lag)

    if verbose:
        print("Descargando series de mercado (yfinance)...")
    for yft, key in _MARKET.items():
        s = _fetch_yf_close(yft, start=start, verbose=verbose)
        if s is not None:
            raw[key] = _pit_shift(s, 0)  # cierre conocido el mismo día

    if not raw:
        raise RuntimeError("No se pudo descargar ninguna serie exógena (¿sin internet?).")

    # Calendario diario hábil común + reindex con ffill (respeta known-as-of).
    all_start = max(pd.Timestamp(start), min(s.index.min() for s in raw.values()))
    all_end = max(s.index.max() for s in raw.values())
    idx = pd.bdate_range(all_start, all_end)

    # Guardrail de cobertura: descartar series truncadas ANTES de armar el panel, para
    # que no generen columnas casi-todo-NaN que recorten el train set en silencio.
    raw = _filter_low_coverage(raw, idx, min_coverage=min_coverage, verbose=verbose)
    if not raw:
        raise RuntimeError("Todas las series exógenas se descartaron por baja cobertura histórica.")

    panel = pd.DataFrame(index=idx)
    for key, s in raw.items():
        panel[key] = s.reindex(idx, method="ffill")

    out = _compute_transforms(panel)

    if write:
        out_path = ensure_dir(root / _CACHE_REL.parent) / _CACHE_REL.name
        out.to_csv(out_path)
        if verbose:
            print(f"✓ Exógenas globales: {out.shape[1]} columnas × {out.shape[0]} días → {out_path}")
    return out


def _compute_transforms(panel: pd.DataFrame) -> pd.DataFrame:
    """Transforma los niveles crudos en features estacionarias. Ver docs/FEATURES.md."""
    p = panel
    out = pd.DataFrame(index=p.index)

    # ── Bloque A — cross-asset / apetito por riesgo ──
    if {"vix", "vix3m"}.issubset(p.columns):
        out["vix_ts"] = p["vix3m"] / (p["vix"] + 1e-12) - 1.0            # contango/backwardation
    if "vix" in p.columns:
        out["vix_pctl_252"] = p["vix"].rolling(252, min_periods=60).rank(pct=True)
    if {"HYG", "LQD"}.issubset(p.columns):
        # Proxy de spread de crédito HY: log(LQD/HYG) sube cuando el high-yield rinde
        # peor que el investment-grade (estrés de crédito) — mismo signo que el OAS.
        # z-score 120d para estacionariedad (reemplaza el ICE BofA OAS, restringido en FRED).
        ratio = np.log(p["LQD"] / (p["HYG"] + 1e-12))
        out["hy_oas_z"] = (ratio - ratio.rolling(120, min_periods=40).mean()) / (
            ratio.rolling(120, min_periods=40).std() + 1e-12
        )
    if "T10Y2Y" in p.columns:
        out["curve_10y2y"] = p["T10Y2Y"]                                 # ya es una diferencia
    if "dxy" in p.columns:
        out["dxy_mom_20"] = np.log(p["dxy"]).diff(20)
    if "SPY" in p.columns:
        lspy = np.log(p["SPY"])
        out["spy_mom_20"] = lspy.diff(20)
        out["spy_mom_60"] = lspy.diff(60)
        out["SPY_ret20"] = lspy.diff(20)   # helper Bloque B
        out["SPY_ret1d"] = lspy.diff()     # helper Bloque B (beta_60)

    # ── Bloque C — macro fundamental ──
    if "DGS10" in p.columns:
        out["rate_shock_10y"] = p["DGS10"].diff(20)
    if "T10YIE" in p.columns:
        out["breakeven_10y"] = p["T10YIE"]                              # inflación esperada (nivel)
    if "ICSA" in p.columns:
        ic = p["ICSA"]
        out["jobless_claims_z"] = (ic - ic.rolling(252, min_periods=60).mean()) / (
            ic.rolling(252, min_periods=60).std() + 1e-12
        )
    if "NFCI" in p.columns:
        out["nfci"] = p["NFCI"]                                         # ya estandarizado (~0)

    # ── Bloque D — commodities ──
    for key, col in [("oil", "oil_mom_20"), ("copper", "copper_mom_20"), ("gold", "gold_mom_20"), ("soy", "soy_mom_20")]:
        if key in p.columns:
            out[col] = np.log(p[key]).diff(20)

    # ── Bloque B — helper: retorno 20d de cada ETF sectorial ──
    for etf in _SECTOR_ETFS:
        if etf in p.columns:
            out[f"{etf}_ret20"] = np.log(p[etf]).diff(20)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Carga / uso desde el pipeline
# ──────────────────────────────────────────────────────────────────────────────

def load_global_exog(*, root: Path | None = None, build_if_missing: bool = True) -> pd.DataFrame:
    """Lee el panel exógeno global cacheado (o lo construye si falta)."""
    root = root or project_root()
    path = root / _CACHE_REL
    if path.exists():
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce")
        return df[~df.index.isna()].sort_index()
    if build_if_missing:
        return build_global_exog(root=root, write=True)
    raise FileNotFoundError(f"No existe el cache exógeno: {path} (correr src.data.macro o refresh_exog)")


def _sector_etf_for(ticker: str, root: Path | None = None) -> str:
    """Devuelve el ETF sectorial de un ticker según src/config/sectors.yaml."""
    root = root or project_root()
    try:
        cfg = load_yaml(root / "src" / "config" / "sectors.yaml")
    except Exception:
        return "SPY"
    return str((cfg.get("map") or {}).get(ticker, cfg.get("default", "SPY")))


def load_exog_for(
    ticker: str,
    config: dict[str, Any] | None = None,
    *,
    root: Path | None = None,
    build_if_missing: bool = True,
) -> pd.DataFrame:
    """Arma el frame exógeno para un ticker: features globales + helpers de sector.

    Las columnas globales (A/C/D) son idénticas para todos los tickers; ``sector_ret20``
    se selecciona según el ETF sectorial del ticker. build_features derivará el Bloque B
    (rel_strength_sector, sector_rotation, beta_60) a partir de las helpers.

    ``build_if_missing=False`` evita construir el cache por red (para reevaluación /
    serving: si el cache no existe, se propaga FileNotFoundError y el llamador degrada).
    """
    g = load_global_exog(root=root, build_if_missing=build_if_missing)
    etf = _sector_etf_for(ticker, root=root)

    cols = [c for c in GLOBAL_FEATURE_COLS if c in g.columns]
    frame = g[cols].copy()

    spy20 = g["SPY_ret20"] if "SPY_ret20" in g.columns else pd.Series(index=g.index, dtype="float64")
    frame["sector_ret20"] = g[f"{etf}_ret20"] if f"{etf}_ret20" in g.columns else spy20
    frame["spy_ret20"] = spy20
    frame["spy_ret1d"] = g["SPY_ret1d"] if "SPY_ret1d" in g.columns else pd.Series(index=g.index, dtype="float64")
    return frame


def align_exog(exog: pd.DataFrame, index: "pd.Index") -> pd.DataFrame:
    """Alinea el frame exógeno al índice de un ticker por fecha, con ffill (PIT-safe).

    Reindexa por fecha normalizada (UTC) tomando el último valor conocido ≤ fecha, lo que
    respeta el sellado known-as-of hecho en build_global_exog y tolera calendarios distintos
    (p.ej. tickers .BA con feriados propios). Compartida por compute_e1_features y
    compute_e2_features (import perezoso).
    """
    ex = exog.copy()
    ex.index = pd.to_datetime(ex.index, utc=True).normalize()
    ex = ex[~ex.index.duplicated(keep="last")].sort_index()

    target = pd.DatetimeIndex(pd.to_datetime(index, utc=True)).normalize()
    aligned = ex.reindex(target, method="ffill")
    aligned.index = index  # restaurar el índice original del ticker
    return aligned


def exog_enabled(config: dict[str, Any]) -> bool:
    """True si data.exog.enabled está activo en la config."""
    return bool(get_nested(config, ["data", "exog", "enabled"], default=False))


def exog_active_for_ticker(config: dict[str, Any], ticker: str) -> bool:
    """True si las exógenas deben usarse para ESTE ticker.

    Regla (allowlist per-ticker): requiere data.exog.enabled Y, si data.exog.tickers
    está definido y no vacío, que el ticker esté en esa lista. Si la lista está vacía o
    ausente → aplica a todos los tickers (modo ablación con --exog).

    Esto materializa el resultado de la ablación: activar solo el subconjunto que mejora
    de forma robusta (ver reports/ablation/ablation_report.md).
    """
    if not exog_enabled(config):
        return False
    allow = get_nested(config, ["data", "exog", "tickers"], default=None) or []
    if not allow:
        return True
    return ticker in set(allow)


def refresh_exog(config: dict[str, Any], *, force: bool = False, root: Path | None = None, verbose: bool = True) -> bool:
    """Reconstruye el cache exógeno (usado por ingest.refresh_data_for_training).

    No fatal: devuelve True si construyó, False si se omitió o falló.
    """
    if not force and not exog_enabled(config):
        return False
    start = str(get_nested(config, ["data", "exog", "history_start"], default="2014-01-01"))
    try:
        build_global_exog(start=start, root=root, write=True, verbose=verbose)
        return True
    except Exception as exc:
        if verbose:
            print(f"⚠️  Exógenas: {exc}")
        return False


def main() -> None:
    """Entrypoint manual: reconstruye el cache exógeno global.

    Uso: ``PYTHONPATH=. python -m src.data.macro``
    """
    build_global_exog(write=True, verbose=True)


if __name__ == "__main__":
    main()
