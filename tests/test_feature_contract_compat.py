"""Regression test: current feature-computation code must still be able to
reconstruct the frozen feature contract of every e1/e2 model artifact, and
`base.yaml`'s active-feature allowlists must stay a subset of that catalog.

This is exactly the drift class that silently degraded several champions
(promoted ~feb-2026) to a permanent "stored" comparison mode in
src.lifecycle.promotion.evaluate_and_promote: a historical feature column was
deleted from compute_e{1,2}_features without checking whether a live model
still declared it in feature_names. See the "never delete, only deactivate"
convention in src/e1/build_features.py and src/e2/build_features.py, and
strategies.e{1,2}_*.features.active in src/config/base.yaml.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.e1.build_features import compute_e1_features
from src.e2.build_features import compute_e2_features

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "models" / "registry.json"
BASE_CONFIG_PATH = ROOT / "src" / "config" / "base.yaml"

COMPUTE_FUNCS = {
    "e1": compute_e1_features,
    "e2": compute_e2_features,
}


def _synthetic_ohlcv(n: int = 260, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=n)))
    high = close * (1 + np.abs(rng.normal(0, 0.005, size=n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, size=n)))
    open_ = close * (1 + rng.normal(0, 0.002, size=n))
    volume = rng.integers(1_000, 100_000, size=n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _catalog_columns(prefix: str, ohlcv: pd.DataFrame) -> set[str]:
    """Columnas que compute_e{1,2}_features PUEDE producir hoy (catálogo completo).

    Para E2 el catálogo incluye las features exógenas (Bloques A/B/C/D): igual que
    la reproducción en producción (``reevaluation._exog_for`` pasa el exog), acá se
    pasa un frame exógeno sintético para que las columnas macro sean producibles.
    El test solo compara NOMBRES de columna, así que valores sintéticos alcanzan.
    """
    compute_features = COMPUTE_FUNCS[prefix]
    if prefix in ("e1", "e2"):
        from src.data.macro import EXOG_HELPER_COLS, GLOBAL_FEATURE_COLS

        cols = list(GLOBAL_FEATURE_COLS) + list(EXOG_HELPER_COLS)
        rng = np.random.default_rng(1)
        exog = pd.DataFrame(
            {c: rng.normal(0, 1, len(ohlcv)) for c in cols}, index=ohlcv.index
        )
        return set(compute_features(ohlcv, exog=exog).columns)
    return set(compute_features(ohlcv).columns)


def _iter_model_entries(prefix: str):
    """Yield (ticker, stage, run_dir) for every model artifact reference of a strategy."""
    if not REGISTRY_PATH.exists():
        return
    reg = json.loads(REGISTRY_PATH.read_text())
    for strat, sdata in reg.get("strategies", {}).items():
        if strat != prefix:
            continue
        for ticker, entry in sdata.get("tickers", {}).items():
            for stage in ("champion", "candidate", "baseline"):
                m = entry.get(stage)
                if m:
                    yield ticker, stage, m.get("run_dir")
            for i, m in enumerate(entry.get("retired", []) or []):
                yield ticker, f"retired[{i}]", m.get("run_dir")


def _model_feature_names(run_dir: str | None, ticker: str) -> list[str] | None:
    if not run_dir:
        return None
    import torch

    candidates = list((ROOT / run_dir).glob(f"{ticker}*_model.pth"))
    if not candidates:
        return None
    payload = torch.load(candidates[0], map_location="cpu", weights_only=False)
    return list(payload.get("feature_names", []))


@pytest.mark.parametrize("prefix", ["e1", "e2"])
def test_current_feature_code_covers_all_registered_models(prefix):
    """Every model artifact tracked in models/registry.json must still be
    reconstructable by the current compute_e{1,2}_features — i.e. no feature
    it was trained on may have been silently deleted since."""
    pytest.importorskip("torch")
    current_columns = _catalog_columns(prefix, _synthetic_ohlcv())

    missing: dict[str, list[str]] = {}
    checked = 0
    for ticker, stage, run_dir in _iter_model_entries(prefix):
        feat_names = _model_feature_names(run_dir, ticker)
        if feat_names is None:
            continue
        checked += 1
        gap = [f for f in feat_names if f not in current_columns]
        if gap:
            missing[f"{ticker}/{stage}"] = gap

    if checked == 0:
        pytest.skip(f"no {prefix} model artifacts on disk (models/runs are gitignored)")

    assert not missing, (
        f"compute_{prefix}_features no longer produces columns required by "
        f"{len(missing)} registered {prefix} model(s): {missing}. A feature was "
        "likely deleted instead of deactivated (see build_features.py header) — "
        "restore the column or the affected model permanently falls back to "
        "'stored' comparison mode in promotion.evaluate_and_promote."
    )


@pytest.mark.parametrize(
    "prefix,strategy_key",
    [("e1", "e1_conservative"), ("e2", "e2_moderate")],
)
def test_active_features_config_is_valid_subset_of_catalog(prefix, strategy_key):
    """strategies.<strategy_key>.features.active in base.yaml must be non-empty
    and every name in it must actually be computable today — catches a typo or
    stale feature name in config, independent of any local model artifacts."""
    current_columns = _catalog_columns(prefix, _synthetic_ohlcv())

    cfg = yaml.safe_load(BASE_CONFIG_PATH.read_text())
    active = cfg["strategies"][strategy_key]["features"]["active"]

    assert active, f"strategies.{strategy_key}.features.active is empty in base.yaml"
    unknown = [f for f in active if f not in current_columns]
    assert not unknown, (
        f"strategies.{strategy_key}.features.active in base.yaml references "
        f"unknown feature(s) {unknown} — not produced by compute_{prefix}_features"
    )
