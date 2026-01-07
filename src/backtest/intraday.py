from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_intraday_signals(
    timestamps: pd.DatetimeIndex,
    bar_returns: np.ndarray,
    pred_forward_returns: np.ndarray,
    *,
    tau_buy: float,
    tau_sell: float,
    round_trip_bps: float,
    execution_delay_bars: int = 1,
    allow_short: bool = True,
    max_position: float = 1.0,
) -> pd.DataFrame:
    """Simple intraday backtest.

    - Uses predictions of forward returns to set position (long/flat/short).
    - Position is applied with an execution delay.
    - PnL is computed bar-by-bar using 1-bar log returns.
    - Costs applied on position changes (approximate round-trip bps).

    This is intentionally minimal (thesis-friendly baseline), not a full OMS simulator.
    """

    n = len(timestamps)
    if not (len(bar_returns) == len(pred_forward_returns) == n):
        raise ValueError("Inputs must have same length")

    desired = np.zeros(n, dtype=np.float32)
    desired[pred_forward_returns >= tau_buy] = max_position
    if allow_short:
        desired[pred_forward_returns <= -tau_sell] = -max_position

    # Execution delay
    pos = np.zeros(n, dtype=np.float32)
    if execution_delay_bars <= 0:
        pos[:] = desired
    else:
        pos[execution_delay_bars:] = desired[:-execution_delay_bars]

    # Strategy returns: position * bar return
    strat_ret = pos[:-1] * bar_returns[1:]
    strat_ret = np.concatenate([[0.0], strat_ret]).astype(np.float32)

    # Transaction costs: apply on position changes
    # round_trip_bps: cost for entering+exiting; approximate per-change cost as half-round-trip
    cost_per_change = (round_trip_bps / 1e4) / 2.0
    turnover = np.abs(np.diff(pos, prepend=0.0))
    costs = turnover * cost_per_change

    net_ret = strat_ret - costs

    equity = np.exp(np.cumsum(net_ret))

    out = pd.DataFrame(
        {
            "pos": pos,
            "pred": pred_forward_returns,
            "bar_ret": bar_returns,
            "gross_ret": strat_ret,
            "cost": costs,
            "net_ret": net_ret,
            "equity": equity,
            "turnover": turnover,
        },
        index=timestamps,
    )

    return out


def compute_profit_factor(net_returns: np.ndarray) -> float:
    gains = net_returns[net_returns > 0].sum()
    losses = -net_returns[net_returns < 0].sum()
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def compute_max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = 1.0 - equity / (peak + 1e-12)
    return float(np.max(dd))
