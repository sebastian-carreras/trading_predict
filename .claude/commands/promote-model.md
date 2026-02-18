# Promote Model

Promote a candidate model to champion for a given strategy and ticker.

## Arguments
$ARGUMENTS should be: <strategy> <ticker>
Example: e1 AAPL

## Instructions

1. Read `models/registry.json`
2. Find the candidate for the given strategy/ticker
3. If no candidate exists, inform the user and stop
4. Show a comparison table between the candidate and the current champion (if one exists):
   - ML metrics: MAE, RMSE, IC, Directional Accuracy
   - Trading metrics: Sharpe, Sortino, CAGR, Max Drawdown, Calmar, Profit Factor, Hit Rate
   - Calculate improvement percentages
5. Also compare against the baseline if it exists
6. Ask the user to confirm the promotion
7. If confirmed, run this Python script to execute the promotion:

```python
import sys
sys.path.insert(0, ".")
from src.lifecycle.registry import ModelRegistry
registry = ModelRegistry("models/registry.json")
result = registry.promote_to_champion("<strategy>", "<ticker>", reason="manual_promotion")
if result:
    print(f"Promoted to champion: {result['variant']}")
else:
    print("No candidate to promote")
```

8. Show the updated registry state for that ticker after promotion
