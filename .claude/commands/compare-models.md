# Compare Models

Compare candidate vs champion (or any two models) for a given strategy and ticker.

## Arguments
$ARGUMENTS: <strategy> <ticker>
Example: e1 AAPL

## Instructions

1. Read `models/registry.json`
2. For the given strategy/ticker, load metrics for:
   - Baseline (if exists)
   - Champion (if exists)
   - Candidate (if exists)
3. If summary CSVs exist in the run directories, read them for additional detail
4. Create a comparison table with these metrics:
   - **ML Metrics**: MAE, RMSE, IC (Information Coefficient), Directional Accuracy
   - **Trading Metrics**: Sharpe, Sortino, CAGR, Max Drawdown, Calmar, Profit Factor, Hit Rate, Num Trades
5. For each metric, show:
   - Baseline value | Champion value | Candidate value
   - Improvement % from baseline to champion
   - Improvement % from champion to candidate (if candidate exists)
6. Provide a recommendation:
   - PROMOTE: if candidate beats champion on Sharpe AND doesn't degrade IC by more than 20%
   - HOLD: if candidate is comparable but not clearly better
   - REJECT: if candidate is worse than champion on primary metrics
7. If backtest CSV files exist, show equity curve comparison summary (total return, max drawdown periods)
