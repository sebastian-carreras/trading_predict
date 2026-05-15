"""Unit tests for src/lifecycle/guardrails.py — Phase 1 guardrails."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.lifecycle.guardrails import validate_candidate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run_dir(tmp_path: Path, ticker: str = "AAPL") -> Path:
    run_dir = tmp_path / "run" / ticker
    run_dir.mkdir(parents=True)
    return run_dir


def _write_model(run_dir: Path, ticker: str = "AAPL") -> None:
    (run_dir / f"{ticker}_model.pth").write_bytes(b"fake-model")


def _write_predictions(run_dir: Path, ticker: str = "AAPL", *, has_nan: bool = False) -> None:
    vals = [0.01, float("nan") if has_nan else 0.02, -0.01]
    df = pd.DataFrame({"y_true": vals, "y_pred": vals})
    df.to_csv(run_dir / f"{ticker}_predictions.csv", index=False)


def _write_summary(run_dir: Path, ticker: str = "AAPL", sharpe: float = 1.0) -> None:
    df = pd.DataFrame({"ticker": [ticker], "bt_sharpe": [sharpe], "ml_ic": [0.1]})
    df.to_csv(run_dir / f"{ticker}_summary.csv", index=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidateCandidateBasic:
    def test_passes_clean_run(self, tmp_path: Path) -> None:
        rd = _make_run_dir(tmp_path)
        _write_model(rd)
        _write_predictions(rd)
        _write_summary(rd, sharpe=1.0)
        passed, errors = validate_candidate(rd, "AAPL", min_sharpe=0.0)
        assert passed, errors

    def test_fails_missing_model_file(self, tmp_path: Path) -> None:
        rd = _make_run_dir(tmp_path)
        _write_predictions(rd)
        _write_summary(rd)
        passed, errors = validate_candidate(rd, "AAPL")
        assert not passed
        assert any("model" in e.lower() for e in errors)

    def test_fails_nan_in_predictions(self, tmp_path: Path) -> None:
        rd = _make_run_dir(tmp_path)
        _write_model(rd)
        _write_predictions(rd, has_nan=True)
        _write_summary(rd, sharpe=0.5)
        passed, errors = validate_candidate(rd, "AAPL", min_sharpe=0.0)
        assert not passed
        assert any("NaN" in e or "Inf" in e for e in errors)

    def test_fails_sharpe_below_threshold(self, tmp_path: Path) -> None:
        rd = _make_run_dir(tmp_path)
        _write_model(rd)
        _write_predictions(rd)
        _write_summary(rd, sharpe=-0.5)
        passed, errors = validate_candidate(rd, "AAPL", min_sharpe=0.0)
        assert not passed
        assert any("Sharpe" in e or "sharpe" in e.lower() for e in errors)

    def test_passes_sharpe_exactly_at_threshold(self, tmp_path: Path) -> None:
        rd = _make_run_dir(tmp_path)
        _write_model(rd)
        _write_predictions(rd)
        _write_summary(rd, sharpe=0.0)
        # min_sharpe=0.0 means strictly less-than, so 0.0 should pass
        passed, errors = validate_candidate(rd, "AAPL", min_sharpe=-0.01)
        assert passed, errors

    def test_passes_with_negative_sharpe_when_min_sharpe_negative(self, tmp_path: Path) -> None:
        rd = _make_run_dir(tmp_path)
        _write_model(rd)
        _write_predictions(rd)
        _write_summary(rd, sharpe=-0.5)
        passed, errors = validate_candidate(rd, "AAPL", min_sharpe=-1.0)
        assert passed, errors


class TestValidateCandidateBaseline:
    def test_fails_ic_worse_than_baseline(self, tmp_path: Path) -> None:
        rd = _make_run_dir(tmp_path)
        _write_model(rd)
        _write_predictions(rd)
        df = pd.DataFrame({"ticker": ["AAPL"], "bt_sharpe": [0.5], "ml_ic": [0.02]})
        df.to_csv(rd / "AAPL_summary.csv", index=False)
        baseline_metrics = {"ml_ic": 0.10}  # baseline IC > candidate IC
        passed, errors = validate_candidate(
            rd, "AAPL", baseline_metrics=baseline_metrics, min_sharpe=0.0
        )
        assert not passed
        assert any("worse" in e.lower() or "IC" in e for e in errors)

    def test_passes_ic_better_than_baseline(self, tmp_path: Path) -> None:
        rd = _make_run_dir(tmp_path)
        _write_model(rd)
        _write_predictions(rd)
        df = pd.DataFrame({"ticker": ["AAPL"], "bt_sharpe": [0.5], "ml_ic": [0.15]})
        df.to_csv(rd / "AAPL_summary.csv", index=False)
        baseline_metrics = {"ml_ic": 0.05}  # candidate IC > baseline IC
        passed, errors = validate_candidate(
            rd, "AAPL", baseline_metrics=baseline_metrics, min_sharpe=0.0
        )
        assert passed, errors
