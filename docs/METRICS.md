# Evaluation metrics

Every strategy is evaluated on two levels: how well the model **predicts** (offline ML metrics)
and how the resulting strategy **performs when traded** (online trading metrics). A model can
look good on the first set and still lose money on the second — which is exactly why both are
reported.

All backtests are **net of transaction costs** (10 bps round-trip for daily strategies, 20 bps
for intraday).

---

## ML metrics (offline)

### MAE / RMSE — prediction error
MAE is the mean absolute error and is robust to outliers. RMSE squares the errors, so it
penalizes large misses much harder. Reporting both shows whether the error is spread evenly or
driven by a few bad days.

### Directional accuracy — did we get the sign right?
The share of predictions where the model got the direction of the move right (up or down),
regardless of magnitude. Above 50% means there is predictive signal. This matters more than
MAE for a trading system: being off by 2% but right about the direction is tradeable; being
precise about magnitude while wrong about the sign is not.

### Information Coefficient (IC) — ranking quality
The Spearman correlation between predicted and realized returns. It measures whether the model
ranks assets correctly, which is what matters when choosing *which* asset to trade.
**IC > 0.05** is generally considered meaningful in finance. **IC < 0** signals overfitting or
no predictive ability.

---

## Trading metrics (online)

### CAGR — compound annual growth rate
The annualized compound return of the strategy. Over windows shorter than a year it stays
mathematically valid but gets noisy, because annualizing amplifies short-run returns. Read it
alongside max drawdown, Sharpe/Sortino, Calmar and hit rate — never alone.

### Sharpe ratio — return per unit of risk
The standard risk-adjusted return measure, comparing excess return against the strategy's
volatility. Design targets for this project: E1 ≥ 0.9, E2 ≥ 0.8, E4 ≥ 1.0.

### Sortino ratio — return per unit of *downside* risk
Like Sharpe, but only penalizes downside volatility. Closer to how an investor actually
experiences risk, since upside volatility is not a problem anyone wants solved.

### Profit factor / hit rate — trade quality
Profit factor is gross profits divided by gross losses (minimum target 1.2–1.4 for the E3
intraday strategy). Hit rate is the share of winning trades, ideally above 50%. A strategy can
have a low hit rate and still be profitable if the winners are much larger than the losers.

### Max drawdown / Calmar — loss control
Max drawdown is the worst peak-to-trough loss over the period — the maximum financial pain an
investor would have sat through. It matters more than average risk for two reasons that averages
hide: it drives **risk of ruin**, and it drives the chance you **abandon the strategy right
before it recovers**.

Calmar divides CAGR by the absolute max drawdown: return per unit of worst-case loss. Think of
max drawdown as peak psychological stress and CAGR as the average economic benefit — Calmar is
the ratio between them. Low drawdown is essential for the conservative (E1) and moderate (E2)
strategies, whose whole premise is capital preservation.

### Turnover — how often we rebalance
Total value traded divided by capital under management. Critical because high turnover multiplies
transaction costs and can erase a model's entire predicted edge. This is why every backtest here
runs net of costs: a strategy that only works before fees does not work.

---

## Baseline

Every strategy is measured against a fixed **Linear Regression baseline** that is never promoted.
It acts as a sanity floor: if a deep-learning model can't beat a linear model on the same
features, the extra complexity isn't earning its keep. Current champion-vs-baseline numbers are
in the [README](../README.md).
