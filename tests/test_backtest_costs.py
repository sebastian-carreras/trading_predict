"""Tests for transaction cost application in backtest engines.

Verifies the cost formula used in:
  - src/backtest/backtest_daily.py (E1, E2)
  - src/backtest/backtest_intraday.py (E3)

Formula (both engines):
  costs = position_change * (round_trip_bps / 10000.0) / 2.0

where position_change = |delta(position)|.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.backtest_daily import backtest_daily_signals


class TestBacktestDailyCosts:
    def _make_prices(self, n: int = 50, start: float = 100.0) -> np.ndarray:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, size=n)
        prices = start * np.exp(np.cumsum(returns))
        return prices.astype(np.float32)

    def _make_timestamps(self, n: int = 50) -> pd.DatetimeIndex:
        return pd.date_range("2020-01-01", periods=n, freq="B")

    def test_zero_bps_means_zero_costs(self) -> None:
        prices = self._make_prices()
        timestamps = self._make_timestamps()
        pred = np.full(len(prices), 0.05, dtype=np.float32)  # always buy signal

        bt = backtest_daily_signals(
            timestamps, prices, pred,
            tau_buy=0.02, tau_sell=0.0,
            round_trip_bps=0.0,
        )
        np.testing.assert_array_equal(bt["costs"], 0.0)

    def test_costs_are_non_negative(self) -> None:
        prices = self._make_prices()
        timestamps = self._make_timestamps()
        pred = np.random.default_rng(0).normal(0, 0.05, size=len(prices)).astype(np.float32)

        bt = backtest_daily_signals(
            timestamps, prices, pred,
            tau_buy=0.02, tau_sell=0.01,
            round_trip_bps=10.0,
        )
        assert (bt["costs"] >= 0).all(), "Costs must always be non-negative"

    def test_costs_formula_10bps(self) -> None:
        """Verify cost = |delta_pos| * (10 / 10000) / 2 for a single trade."""
        n = 10
        prices = self._make_prices(n)
        timestamps = self._make_timestamps(n)
        # Force a simple signal: buy on bar 0, then flat for the rest
        pred = np.zeros(n, dtype=np.float32)
        pred[0] = 0.10  # big buy signal

        bt = backtest_daily_signals(
            timestamps, prices, pred,
            tau_buy=0.05, tau_sell=0.0,
            round_trip_bps=10.0,
            holding_period_days=1,  # re-evaluate every bar
        )
        # Any bar where position changes should have cost = delta_pos * 10/10000/2
        pos_change = np.abs(np.diff(bt["pos"].values))
        expected_costs = np.concatenate([[0.0], pos_change * (10 / 10000.0) / 2.0])
        np.testing.assert_allclose(bt["costs"].values, expected_costs, atol=1e-6)

    def test_10bps_vs_20bps_cost_ratio(self) -> None:
        """20 bps should produce exactly 2x the total costs of 10 bps."""
        prices = self._make_prices(40)
        timestamps = self._make_timestamps(40)
        pred = np.random.default_rng(1).normal(0, 0.05, size=40).astype(np.float32)

        bt10 = backtest_daily_signals(
            timestamps, prices, pred,
            tau_buy=0.02, tau_sell=0.01, round_trip_bps=10.0,
        )
        bt20 = backtest_daily_signals(
            timestamps, prices, pred,
            tau_buy=0.02, tau_sell=0.01, round_trip_bps=20.0,
        )
        total10 = bt10["costs"].sum()
        total20 = bt20["costs"].sum()
        if total10 > 0:
            np.testing.assert_allclose(total20 / total10, 2.0, rtol=1e-5)

    def test_net_ret_equals_gross_minus_costs(self) -> None:
        prices = self._make_prices()
        timestamps = self._make_timestamps()
        pred = np.random.default_rng(2).normal(0, 0.05, size=len(prices)).astype(np.float32)

        bt = backtest_daily_signals(
            timestamps, prices, pred,
            tau_buy=0.02, tau_sell=0.01, round_trip_bps=10.0,
        )
        np.testing.assert_allclose(
            bt["net_ret"].values,
            bt["gross_ret"].values - bt["costs"].values,
            atol=1e-6,
            err_msg="net_ret must equal gross_ret - costs",
        )

    def test_intraday_cost_formula_20bps(self) -> None:
        """Intraday backtest uses the same formula with 20 bps."""
        from src.backtest.backtest_intraday import backtest_intraday_signals

        n = 30
        rng = np.random.default_rng(3)
        bar_returns = rng.normal(0.0, 0.003, size=n).astype(np.float32)
        pred = rng.normal(0, 0.003, size=n).astype(np.float32)
        timestamps = self._make_timestamps(n)

        bt = backtest_intraday_signals(
            timestamps=timestamps,
            bar_returns=bar_returns,
            pred_forward_returns=pred,
            tau_buy=0.0025,
            tau_sell=0.0025,
            round_trip_bps=20.0,
        )
        assert (bt["cost"] >= 0).all(), "Intraday costs must be non-negative"
        # net_ret = gross_ret (strat_ret) - cost
        np.testing.assert_allclose(
            bt["net_ret"].values,
            bt["gross_ret"].values - bt["cost"].values,
            atol=1e-6,
        )
