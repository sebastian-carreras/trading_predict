"""Tests for atomic writes in ModelRegistry._save().

Verifies:
  1. Uses tempfile + os.replace (atomic rename pattern)
  2. Original file not corrupted if write fails mid-way
  3. Data persisted correctly after save
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.lifecycle.registry import ModelRegistry


SAMPLE_METRICS = {
    "ml_mae": 0.05,
    "ml_rmse": 0.08,
    "ml_ic": 0.12,
    "ml_directional_accuracy": 0.58,
    "bt_sharpe": 1.2,
    "bt_cagr": 0.25,
    "bt_max_drawdown": 0.15,
    "bt_calmar": 1.67,
    "bt_profit_factor": 1.4,
    "bt_hit_rate": 0.55,
}


class TestRegistryAtomicWrite:
    def test_os_replace_called_on_save(self, tmp_path: Path) -> None:
        reg_path = tmp_path / "registry.json"
        reg = ModelRegistry(reg_path)

        with patch("os.replace") as mock_replace:
            reg.register_candidate(
                strategy="e1",
                ticker="AAPL",
                run_dir="runs/e1/test/AAPL",
                metrics=SAMPLE_METRICS,
                variant="e1_conservative",
            )
            mock_replace.assert_called_once()
            # First arg should be the temp file path, second should be registry path
            _, dest = mock_replace.call_args.args
            assert dest == str(reg_path)

    def test_data_persisted_after_register_candidate(self, tmp_path: Path) -> None:
        reg_path = tmp_path / "registry.json"
        reg = ModelRegistry(reg_path)
        reg.register_candidate(
            strategy="e1",
            ticker="MSFT",
            run_dir="runs/e1/test/MSFT",
            metrics=SAMPLE_METRICS,
            variant="e1_conservative",
        )
        # Re-load from disk
        reg2 = ModelRegistry(reg_path)
        tickers = reg2.data["strategies"]["e1"]["tickers"]
        assert "MSFT" in tickers
        assert tickers["MSFT"]["candidate"]["variant"] == "e1_conservative"

    def test_data_persisted_after_register_baseline(self, tmp_path: Path) -> None:
        reg_path = tmp_path / "registry.json"
        reg = ModelRegistry(reg_path)
        reg.register_baseline(
            strategy="e1",
            ticker="AAPL",
            run_dir="runs/e1_baseline/test/AAPL",
            metrics=SAMPLE_METRICS,
            variant="e1_baseline",
        )
        reg2 = ModelRegistry(reg_path)
        entry = reg2.data["strategies"]["e1"]["tickers"]["AAPL"]["baseline"]
        assert entry["variant"] == "e1_baseline"
        assert abs(entry["metrics"]["ml_ic"] - 0.12) < 1e-6

    def test_original_file_not_corrupted_on_write_failure(self, tmp_path: Path) -> None:
        """If write fails, the original registry.json must survive intact."""
        reg_path = tmp_path / "registry.json"
        reg = ModelRegistry(reg_path)
        # Write a known initial state
        reg.register_baseline(
            strategy="e2",
            ticker="NVDA",
            run_dir="runs/e2_baseline/test/NVDA",
            metrics=SAMPLE_METRICS,
            variant="e2_baseline",
        )
        original_content = reg_path.read_text()

        # Now simulate a failure during the second write
        def fail_write(*args, **kwargs):
            raise OSError("Simulated disk failure")

        with patch("os.replace", side_effect=fail_write):
            with pytest.raises(OSError):
                reg.register_candidate(
                    strategy="e2",
                    ticker="NVDA",
                    run_dir="runs/e2/test/NVDA",
                    metrics=SAMPLE_METRICS,
                    variant="e2_moderate",
                )

        # Original file must still be intact (not written)
        assert reg_path.read_text() == original_content

    def test_no_leftover_temp_files_on_success(self, tmp_path: Path) -> None:
        """After a successful save, no .tmp files should remain in the directory."""
        reg_path = tmp_path / "registry.json"
        reg = ModelRegistry(reg_path)
        reg.register_candidate(
            strategy="e1",
            ticker="AAPL",
            run_dir="runs/e1/test/AAPL",
            metrics=SAMPLE_METRICS,
            variant="e1_conservative",
        )
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0, f"Leftover temp files: {tmp_files}"

    def test_multiple_sequential_saves_consistent(self, tmp_path: Path) -> None:
        """Multiple saves must all commit and accumulate correctly."""
        reg_path = tmp_path / "registry.json"
        reg = ModelRegistry(reg_path)
        tickers = ["AAPL", "MSFT", "GOOGL"]
        for ticker in tickers:
            reg.register_candidate(
                strategy="e1",
                ticker=ticker,
                run_dir=f"runs/e1/test/{ticker}",
                metrics=SAMPLE_METRICS,
                variant="e1_conservative",
            )
        reg2 = ModelRegistry(reg_path)
        stored = list(reg2.data["strategies"]["e1"]["tickers"].keys())
        for ticker in tickers:
            assert ticker in stored, f"{ticker} not found after sequential saves"
