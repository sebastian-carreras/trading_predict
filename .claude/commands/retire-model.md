# Retire Model

Archive a model from champion stage to retired.

## Arguments
$ARGUMENTS: <strategy> <ticker> [reason]
Example: e1 AAPL "superseded by new architecture"

If reason is omitted, default to "manual_retirement".

## Instructions

1. Read `models/registry.json`
2. Find the champion for the given strategy/ticker
3. If no champion exists, inform the user and stop
4. Show the champion's current metrics and ask for confirmation
5. Warn the user if this will leave the ticker without a champion
6. If confirmed, run this Python script:

```python
import sys
sys.path.insert(0, ".")
from src.lifecycle.registry import ModelRegistry
registry = ModelRegistry("models/registry.json")
success = registry.retire_champion("<strategy>", "<ticker>", reason="<reason>")
print("Retired successfully" if success else "Failed to retire")
```

7. Show the updated state and list of retired models for that ticker
8. Check if `lifecycle.retirement.keep_last_n` in `src/config/base.yaml` is exceeded and warn if there are too many retired models
