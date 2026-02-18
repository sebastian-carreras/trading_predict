# Model Status

Show the current lifecycle status of all models across strategies.

## Arguments
$ARGUMENTS (optional): strategy name (e1, e2, e3, e4) or "all". Defaults to "all".

## Instructions

1. Read the file `models/registry.json`
2. For each strategy in the registry:
   a. For each ticker, show which stages have models (baseline, candidate, champion)
3. Display a markdown table with columns:
   - Strategy | Ticker | Baseline | Champion | Candidate | Champion Sharpe | Champion IC | Last Updated
   - Use checkmarks for stages that have models, show key metrics for champions
4. Show any tickers that lack a champion (highlight as needing attention)
5. Show the retired models count per ticker
6. If any candidates are pending promotion, note them
7. Compare the champion metrics against the baseline metrics if both exist, showing improvement percentage

Format the output as a clear, readable summary the user can scan quickly.
