# Claude Code Automation Recommendations

## Codebase Profile
- **Type**: Python ML/AI
- **Key Libraries**: PyTorch, MLflow, scikit-learn, pandas, yfinance, alpaca-py
- **Infrastructure**: Docker (Airflow + MLflow + FastAPI), pytest
- **Existing Skills**: model-status, promote-model, compare-models, retire-model, new-experiment

---

## MCP Servers

### 1. context7
**Why**: You use PyTorch, MLflow, scikit-learn, yfinance, and alpaca-py daily. context7 pulls live, version-accurate docs into context — no more guessing API signatures for `mlflow.log_metric` or `alpaca_trade_api` calls.

**Install**: `claude mcp add context7 -- npx -y @upstash/context7-mcp`

### 2. GitHub MCP
**Why**: You have `.github/` already. With this you can create/view issues, search PRs, and check CI run status directly from Claude — useful for tracking experiment results and academic milestones.

**Install**: `claude mcp add github -- npx -y @modelcontextprotocol/server-github` (requires `GITHUB_TOKEN` env var)

---

## Hooks

### 1. Block direct edits to `registry.json`
**Why**: The CLAUDE.md explicitly says "never write registry.json directly — use ModelRegistry methods." A PreToolUse hook enforces this automatically, even if Claude forgets mid-task.

**Where**: `.claude/settings.json`
```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{
        "type": "command",
        "command": "if echo \"$CLAUDE_TOOL_INPUT\" | python3 -c \"import json,sys; d=json.load(sys.stdin); exit(0 if 'registry.json' not in d.get('file_path','') else 1)\"; then exit 0; else echo 'Direct edits to registry.json are forbidden. Use ModelRegistry methods.'; exit 2; fi"
      }]
    }]
  }
}
```

### 2. Run pytest on test file edits
**Why**: You have `tests/` with lifecycle and promotion tests. Auto-running them after edits to `src/lifecycle/` catches regressions immediately — critical since the registry is the single source of truth.

**Where**: `.claude/settings.json`
```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{
        "type": "command",
        "command": "if echo \"$CLAUDE_TOOL_INPUT\" | python3 -c \"import json,sys; d=json.load(sys.stdin); p=d.get('file_path',''); exit(0 if ('src/lifecycle' in p or 'tests/' in p) else 1)\"; then conda run -n ia_ceia_18co pytest tests/ -x -q 2>&1 | tail -5; fi"
      }]
    }]
  }
}
```

---

## Skills

### 1. `run-backtest`
**Why**: You have a full `src/backtest/` module and scripts under `scripts/evaluation/`, but no skill to trigger a quick backtest for a given strategy+ticker. This fills the gap between training and promoting.

**Create**: `.claude/commands/run-backtest.md`
```markdown
# Run Backtest

Run a backtest for a given strategy and ticker using the latest candidate model.

## Arguments
$ARGUMENTS: <strategy> <ticker>
Example: e1 AAPL

## Instructions
1. Locate the latest candidate run in `runs/<strategy>/` for the given ticker
2. Load its predictions.csv and metrics.json
3. Run the backtest module: `python -m src.backtest.backtest --strategy <strategy> --ticker <ticker>`
4. Print the key metrics: Sharpe, Calmar, IC, Directional Accuracy
5. Compare against the current champion metrics in `models/registry.json`
6. Suggest whether to promote or discard the candidate
```

### 2. `registry-audit`
**Why**: As tickers and strategies accumulate, the registry can gather stale candidates or runs with missing files. This skill does a full health check — no equivalent exists today.

**Create**: `.claude/commands/registry-audit.md`
```markdown
# Registry Audit

Audit the health of the model registry for all strategies and tickers.

## Instructions
1. Read `models/registry.json`
2. For each entry (strategy × ticker), check:
   - Champion: does its `run_dir` exist on disk? Does `model.pth` exist?
   - Candidate: same checks. If stale (>30 days old), flag it
   - Baseline: does its reference metrics exist?
3. Check `models/metrics_log.jsonl` for the last 5 promotions
4. Print a markdown table: Ticker | Strategy | Champion OK | Candidate Age | Issues
5. List any actions recommended (retire stale candidates, retrain missing models)
```

---

## Subagents

### 1. `metrics-reviewer`
**Why**: Your promotion logic uses a composite score (Sharpe + IC + DirectionalAcc + Calmar). When comparing multiple tickers after a bulk training run, a specialized subagent can review all metrics in parallel and flag anomalies — currently you'd have to run `/compare-models` per ticker manually.

**Create**: `.claude/agents/metrics-reviewer.md`
```markdown
---
name: metrics-reviewer
description: Reviews training metrics across multiple tickers for a strategy and flags statistical anomalies or promotion candidates. Use after bulk training runs.
---

You are a quantitative analyst reviewing model metrics for a trading prediction system.

When asked to review metrics for a strategy:
1. Read `models/registry.json` for all tickers in that strategy
2. Load each candidate's metrics.json from its run_dir
3. Compute the composite score: 0.35×Sharpe + 0.25×IC + 0.20×DirectionalAcc + 0.20×Calmar
4. Flag any model where Sharpe < 0, IC < 0, or composite score fails the 5% improvement threshold
5. Rank all tickers by composite score improvement over their champion
6. Output a summary table and a prioritized promotion list
```
