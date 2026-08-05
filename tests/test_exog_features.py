"""
Tests de las features exógenas (Fase 1: Bloques A/B/C/D).

Cubren lo crítico SIN tocar la red:
  - Compatibilidad hacia atrás: compute_e2_features(df) sin exog == comportamiento histórico.
  - PIT / no-leakage del alineado (_align_exog): una fila nunca usa un valor exógeno
    cuya fecha known-as-of sea posterior.
  - Alineado robusto a calendarios distintos (ffill, sin NaN interiores tras warmup).
  - Derivación del Bloque B (rel_strength_sector / sector_rotation / beta_60).
  - Transforms de macro._compute_transforms sobre un panel sintético.
  - Mapeo ticker→ETF de sectors.yaml.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.e2.build_features import _align_exog, compute_e2_features
from src.data import macro


# ── Fixtures sintéticos (sin red) ─────────────────────────────────────────────

def _make_ohlcv(n: int = 400, start: str = "2019-01-01", seed: int = 7) -> pd.DataFrame:
    """OHLCV diario sintético con índice tz-aware UTC (como load_ohlcv_csv)."""
    idx = pd.bdate_range(start, periods=n, tz="UTC")
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    )


def _make_exog(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Frame exógeno sintético con una feature global + las 3 helpers del Bloque B."""
    dates = pd.DatetimeIndex(index).tz_convert(None).normalize().unique().sort_values()
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "vix_ts": rng.normal(0, 0.1, len(dates)),
            "sector_ret20": rng.normal(0, 0.03, len(dates)),
            "spy_ret20": rng.normal(0, 0.02, len(dates)),
            "spy_ret1d": rng.normal(0, 0.01, len(dates)),
        },
        index=dates,
    )


# ── Compatibilidad hacia atrás ────────────────────────────────────────────────

def test_backward_compat_sin_exog():
    """Sin exog, compute_e2_features devuelve solo el catálogo base (sin columnas exógenas)."""
    df = _make_ohlcv()
    base = compute_e2_features(df)  # exog por defecto = None
    for exog_col in ("vix_ts", "rel_strength_sector", "sector_rotation", "beta_60"):
        assert exog_col not in base.columns
    # El catálogo base histórico sigue intacto (features activas de E2).
    for col in ("ret_1d", "ret_20d", "vol_20d", "atr_14", "rsi_14", "adx_14", "skew_ret_20d"):
        assert col in base.columns


def test_exog_none_igual_que_omitido():
    df = _make_ohlcv()
    a = compute_e2_features(df)
    b = compute_e2_features(df, exog=None)
    pd.testing.assert_frame_equal(a, b)


# ── PIT / no-leakage del alineado ─────────────────────────────────────────────

def test_align_exog_no_leakage_ffill():
    """El valor de una fecha known-as-of NO puede aparecer antes de esa fecha."""
    target = pd.bdate_range("2020-01-06", "2020-01-31", tz="UTC")
    exog = pd.DataFrame(
        {"x": [1.0, 2.0]},
        index=pd.to_datetime(["2020-01-10", "2020-01-20"]),
    )
    aligned = _align_exog(exog, target)

    # Antes del primer known-as-of: NaN (no hay dato disponible aún).
    assert aligned.loc[aligned.index < "2020-01-10", "x"].isna().all()
    # El 2020-01-15 solo puede ver el valor de 01-10 (=1.0), NUNCA el de 01-20 (=2.0).
    assert aligned.loc["2020-01-15", "x"] == 1.0
    # Desde 01-20 en adelante: 2.0.
    assert aligned.loc["2020-01-20", "x"] == 2.0
    assert aligned.loc["2020-01-21", "x"] == 2.0
    # El índice devuelto es exactamente el del ticker.
    assert aligned.index.equals(target)


def test_align_exog_calendarios_distintos():
    """Un ticker con feriados propios (faltan días) igual queda ffill-eado sin huecos interiores."""
    df = _make_ohlcv(n=300)
    # Simular calendario .BA: sacar algunos días hábiles.
    ticker_idx = df.index[::1]
    ticker_idx = ticker_idx.delete([50, 51, 120])
    exog = _make_exog(df.index)
    aligned = _align_exog(exog, ticker_idx)
    # Tras el primer dato exógeno no debe haber NaN interiores.
    first_valid = aligned["vix_ts"].first_valid_index()
    assert aligned.loc[first_valid:, "vix_ts"].notna().all()


# ── Derivación del Bloque B ───────────────────────────────────────────────────

def test_bloque_b_derivacion():
    df = _make_ohlcv()
    exog = _make_exog(df.index)
    feats = compute_e2_features(df, exog=exog)

    for col in ("vix_ts", "rel_strength_sector", "sector_rotation", "beta_60"):
        assert col in feats.columns
    # Las helpers NO deben quedar como features.
    for helper in macro.EXOG_HELPER_COLS:
        assert helper not in feats.columns

    aligned = _align_exog(exog, df.index)
    # rel_strength_sector = ret_20d − sector_ret20 ; sector_rotation = sector_ret20 − spy_ret20
    exp_rel = feats["ret_20d"] - aligned["sector_ret20"]
    exp_rot = aligned["sector_ret20"] - aligned["spy_ret20"]
    pd.testing.assert_series_equal(feats["rel_strength_sector"], exp_rel, check_names=False)
    pd.testing.assert_series_equal(feats["sector_rotation"], exp_rot, check_names=False)


# ── E1: mismo cableado exógeno que E2 (off por default, para exploración futura) ──

def test_e1_backward_compat_y_exog():
    """compute_e1_features: sin exog = base intacto; con exog = agrega A/C/D + Bloque B."""
    from src.e1.build_features import compute_e1_features

    df = _make_ohlcv()
    base = compute_e1_features(df)  # sin exog → comportamiento histórico
    for exog_col in ("vix_ts", "rel_strength_sector", "sector_rotation", "beta_60"):
        assert exog_col not in base.columns
    for col in ("ret_1w", "ret_4w", "ret_13w", "atr_14", "adx_14"):  # base E1 intacto
        assert col in base.columns

    exog = _make_exog(df.index)
    feats = compute_e1_features(df, exog=exog)
    for col in ("vix_ts", "rel_strength_sector", "sector_rotation", "beta_60"):
        assert col in feats.columns
    for helper in macro.EXOG_HELPER_COLS:  # las helpers no quedan como features
        assert helper not in feats.columns


# ── Transforms de macro (panel sintético, sin red) ────────────────────────────

def test_compute_transforms_columnas_y_formula():
    idx = pd.bdate_range("2015-01-01", periods=400)
    rng = np.random.default_rng(1)
    panel = pd.DataFrame(
        {
            "vix": np.abs(rng.normal(18, 4, len(idx))) + 5,
            "vix3m": np.abs(rng.normal(20, 4, len(idx))) + 5,
            # Crédito HY: proxy HYG/LQD (reemplazó al ICE BofA OAS de FRED, restringido).
            "HYG": 80 * np.exp(np.cumsum(rng.normal(0, 0.004, len(idx)))),
            "LQD": 110 * np.exp(np.cumsum(rng.normal(0, 0.003, len(idx)))),
            "T10Y2Y": rng.normal(0.5, 0.3, len(idx)),
            "dxy": 95 + np.cumsum(rng.normal(0, 0.2, len(idx))),
            "SPY": 300 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))),
            "T10YIE": rng.normal(2.2, 0.2, len(idx)),
            "ICSA": np.abs(rng.normal(250_000, 20_000, len(idx))),
            "NFCI": rng.normal(-0.2, 0.1, len(idx)),
            "XLK": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))),
        },
        index=idx,
    )
    out = macro._compute_transforms(panel)

    for col in ("vix_ts", "vix_pctl_252", "hy_oas_z", "curve_10y2y", "dxy_mom_20",
                "spy_mom_20", "spy_mom_60", "breakeven_10y", "jobless_claims_z", "nfci"):
        assert col in out.columns
    # Helpers Bloque B
    assert "SPY_ret20" in out.columns and "SPY_ret1d" in out.columns and "XLK_ret20" in out.columns
    # vix_ts = vix3m/vix − 1
    exp_vix_ts = panel["vix3m"] / (panel["vix"] + 1e-12) - 1.0
    pd.testing.assert_series_equal(out["vix_ts"], exp_vix_ts, check_names=False)
    # curve/breakeven/nfci son passthrough (nivel)
    pd.testing.assert_series_equal(out["curve_10y2y"], panel["T10Y2Y"], check_names=False)
    # vix_pctl_252 acotado a [0, 1]
    v = out["vix_pctl_252"].dropna()
    assert ((v >= 0) & (v <= 1)).all()


def test_filter_low_coverage_descarta_series_truncadas():
    """Guardrail: una serie que arranca tarde (baja cobertura) se descarta; las completas quedan."""
    idx = pd.bdate_range("2014-01-01", "2026-01-01")
    full = pd.Series(1.0, index=pd.bdate_range("2014-01-01", "2026-01-01"))     # cobertura ~100%
    truncada = pd.Series(1.0, index=pd.bdate_range("2023-07-01", "2026-01-01"))  # ~20% (el bug ICE BofA)
    kept = macro._filter_low_coverage(
        {"completa": full, "truncada": truncada}, idx, min_coverage=0.9, verbose=False
    )
    assert "completa" in kept
    assert "truncada" not in kept  # descartada → su feature no se calcula (no recorta el train)


def test_pit_shift_desplaza_hacia_adelante():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    shifted = macro._pit_shift(s, 5)
    assert (shifted.index == pd.to_datetime(["2020-01-06", "2020-01-07"])).all()
    # lag 0 no cambia el índice
    assert macro._pit_shift(s, 0).index.equals(s.index)


# ── Mapeo sectorial ───────────────────────────────────────────────────────────

def test_sector_etf_mapping():
    assert macro._sector_etf_for("NVDA") == "XLK"
    assert macro._sector_etf_for("GS") == "XLF"
    assert macro._sector_etf_for("WMT") == "XLP"
    # Ticker desconocido → default SPY
    assert macro._sector_etf_for("TICKER_INEXISTENTE_XYZ") == "SPY"


# ── Activación per-ticker (allowlist, resultado de la ablación) ───────────────

def test_exog_active_for_ticker_allowlist():
    cfg = {"data": {"exog": {"enabled": True, "tickers": ["NVDA", "GS"]}}}
    assert macro.exog_active_for_ticker(cfg, "NVDA")
    assert macro.exog_active_for_ticker(cfg, "GS")
    assert not macro.exog_active_for_ticker(cfg, "AAPL")           # fuera del allowlist
    # enabled=False → siempre False, aunque esté en la lista
    assert not macro.exog_active_for_ticker({"data": {"exog": {"enabled": False, "tickers": ["NVDA"]}}}, "NVDA")
    # allowlist vacía/ausente → todos (modo ablación)
    assert macro.exog_active_for_ticker({"data": {"exog": {"enabled": True}}}, "CUALQUIERA")
    assert macro.exog_active_for_ticker({"data": {"exog": {"enabled": True, "tickers": []}}}, "CUALQUIERA")


def test_reevaluation_exog_detection_no_toca_cache():
    """_exog_for devuelve None (sin cargar cache) para modelos base, E1, o ticker None."""
    from src.lifecycle.reevaluation import _exog_for

    # feature_names base (sin exógenas) → None, sin tocar el cache — para E1 y E2
    assert _exog_for("e2_moderate", "NVDA", ["ret_1d", "ret_20d", "rsi_14"]) is None
    assert _exog_for("e1_conservative", "AAPL", ["ret_1w", "sma_50", "adx_14"]) is None
    # ticker desconocido → None
    assert _exog_for("e2_moderate", None, ["vix_ts"]) is None
    assert _exog_for("e1_conservative", None, ["vix_ts"]) is None
    # estrategia no soportada (e3) → None
    assert _exog_for("e3_intraday", "SPY", ["vix_ts"]) is None
