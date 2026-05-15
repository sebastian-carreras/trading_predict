"""Tests for per-strategy promotion config resolution.

Validates that get_strategy_promotion_config() correctly resolves
scoring_weights with the hierarchical lookup:
  per_strategy.{strategy} → per_strategy.{prefix} → global fallback
"""
from __future__ import annotations

import pytest

from src.lifecycle.promotion import (
    _DEFAULT_PROMOTION_CONFIG,
    compute_score,
    get_strategy_promotion_config,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def full_config() -> dict:
    """Simulates lifecycle.promotion from base.yaml with per_strategy."""
    return {
        "auto_promote": False,
        "first_champion_strategy": "promote",
        "min_improvement": 0.05,
        "require_positive_sharpe": True,
        "scoring_weights": {
            "bt_sharpe": 0.35,
            "ml_ic": 0.25,
            "ml_directional_accuracy": 0.20,
            "bt_calmar": 0.20,
        },
        "per_strategy": {
            "e1": {
                "scoring_weights": {
                    "bt_sharpe": 0.35,
                    "ml_ic": 0.25,
                    "ml_directional_accuracy": 0.20,
                    "bt_calmar": 0.20,
                },
            },
            "e3": {
                "scoring_weights": {
                    "bt_sharpe": 0.20,
                    "bt_profit_factor": 0.30,
                    "ml_directional_accuracy": 0.30,
                    "bt_win_rate": 0.20,
                },
                "min_improvement": 0.10,
                "require_positive_sharpe": False,
            },
        },
    }


# ---------------------------------------------------------------
# get_strategy_promotion_config
# ---------------------------------------------------------------

class TestGetStrategyPromotionConfig:
    """Test hierarchical config resolution."""

    def test_exact_match(self, full_config: dict) -> None:
        """per_strategy.e1 is found by exact key."""
        cfg = get_strategy_promotion_config(full_config, "e1")
        assert cfg["scoring_weights"]["bt_sharpe"] == 0.35
        assert cfg["scoring_weights"]["bt_calmar"] == 0.20
        # Global fields inherited
        assert cfg["min_improvement"] == 0.05
        assert cfg["require_positive_sharpe"] is True

    def test_prefix_fallback(self, full_config: dict) -> None:
        """e1_conservative falls back to per_strategy.e1."""
        cfg = get_strategy_promotion_config(full_config, "e1_conservative")
        assert cfg["scoring_weights"]["bt_sharpe"] == 0.35
        assert cfg["scoring_weights"]["bt_calmar"] == 0.20

    def test_global_fallback_unknown_strategy(self, full_config: dict) -> None:
        """Unknown strategy falls back to global scoring_weights."""
        cfg = get_strategy_promotion_config(full_config, "e99")
        assert cfg["scoring_weights"]["bt_sharpe"] == 0.35
        assert cfg["scoring_weights"]["ml_ic"] == 0.25

    def test_strategy_overrides_global_fields(self, full_config: dict) -> None:
        """e3 overrides min_improvement and require_positive_sharpe."""
        cfg = get_strategy_promotion_config(full_config, "e3")
        assert cfg["min_improvement"] == 0.10
        assert cfg["require_positive_sharpe"] is False
        # Different scoring weights
        assert cfg["scoring_weights"]["bt_profit_factor"] == 0.30
        assert "bt_calmar" not in cfg["scoring_weights"]

    def test_prefix_fallback_e3_intraday(self, full_config: dict) -> None:
        """e3_intraday falls back to per_strategy.e3."""
        cfg = get_strategy_promotion_config(full_config, "e3_intraday")
        assert cfg["scoring_weights"]["bt_profit_factor"] == 0.30
        assert cfg["require_positive_sharpe"] is False

    def test_none_config_uses_defaults(self) -> None:
        """None config returns _DEFAULT_PROMOTION_CONFIG values."""
        cfg = get_strategy_promotion_config(None, "e1")
        assert cfg["scoring_weights"] == _DEFAULT_PROMOTION_CONFIG["scoring_weights"]
        assert cfg["min_improvement"] == 0.05

    def test_empty_per_strategy_uses_global(self) -> None:
        """Empty per_strategy dict falls back to global."""
        config = {
            "scoring_weights": {"bt_sharpe": 0.50, "ml_ic": 0.50},
            "per_strategy": {},
        }
        cfg = get_strategy_promotion_config(config, "e1")
        assert cfg["scoring_weights"]["bt_sharpe"] == 0.50

    def test_no_per_strategy_key_uses_global(self) -> None:
        """Config without per_strategy falls back to global."""
        config = {
            "scoring_weights": {"bt_sharpe": 0.60, "ml_ic": 0.40},
        }
        cfg = get_strategy_promotion_config(config, "e1")
        assert cfg["scoring_weights"]["bt_sharpe"] == 0.60

    def test_per_strategy_does_not_leak_into_result(self, full_config: dict) -> None:
        """Resolved config should not contain per_strategy key."""
        cfg = get_strategy_promotion_config(full_config, "e1")
        assert "per_strategy" not in cfg

    def test_original_config_not_mutated(self, full_config: dict) -> None:
        """The input config dict must not be modified."""
        original_keys = set(full_config.keys())
        get_strategy_promotion_config(full_config, "e1")
        assert set(full_config.keys()) == original_keys
        assert "per_strategy" in full_config  # not popped from original


# ---------------------------------------------------------------
# compute_score with different weights
# ---------------------------------------------------------------

class TestComputeScorePerStrategy:
    """Verify that different weight dicts produce different scores."""

    @pytest.fixture
    def sample_metrics(self) -> dict:
        return {
            "bt_sharpe": 0.80,
            "ml_ic": 0.45,
            "ml_directional_accuracy": 0.60,
            "bt_calmar": 0.50,
            "bt_profit_factor": 1.8,
            "bt_win_rate": 0.55,
        }

    def test_e1_weights(self, sample_metrics: dict) -> None:
        """E1 weights produce expected score."""
        weights = {"bt_sharpe": 0.35, "ml_ic": 0.25, "ml_directional_accuracy": 0.20, "bt_calmar": 0.20}
        score, detail = compute_score(sample_metrics, weights)
        assert score > 0
        assert "bt_sharpe" in detail
        assert "bt_calmar" in detail

    def test_e3_weights_different_score(self, sample_metrics: dict) -> None:
        """E3 weights use different metrics and produce different score."""
        e1_weights = {"bt_sharpe": 0.35, "ml_ic": 0.25, "ml_directional_accuracy": 0.20, "bt_calmar": 0.20}
        e3_weights = {"bt_sharpe": 0.20, "bt_profit_factor": 0.30, "ml_directional_accuracy": 0.30, "bt_win_rate": 0.20}

        score_e1, _ = compute_score(sample_metrics, e1_weights)
        score_e3, _ = compute_score(sample_metrics, e3_weights)

        # Different weights → different scores (unless metrics happen to align)
        assert score_e1 != score_e3

    def test_e3_weights_include_profit_factor(self, sample_metrics: dict) -> None:
        """E3 score includes profit_factor contribution."""
        e3_weights = {"bt_profit_factor": 1.0}
        score, detail = compute_score(sample_metrics, e3_weights)
        assert "bt_profit_factor" in detail
        assert score == pytest.approx(1.8, rel=1e-4)
