# Tests

Unit and light integration tests for the trading prediction project (final work of the AI
Specialization at FIUBA).

These are not generic "framework" tests. Each one verifies a concrete methodological guarantee of
the pipeline — no data leakage, atomic registry writes, consistent model-promotion logic, correct
backtest cost computation. Collectively they are the evidence that the practices described in
[`README.md`](../README.md) — walk-forward with embargo, leakage-free scaling, champion/candidate
promotion — are actually implemented and not just documented.

---

## Running them

```bash
# Full suite (what CI runs — see .github/workflows/ci.yml)
python -m pytest -v

# A single file
pytest tests/test_registry_atomicity.py -v

# A single test
pytest tests/test_walkforward_splits.py::TestEmbargo::test_embargo_gap_e1_90_days -v
```

CI runs this same suite on Python 3.10 and 3.12, on every push to `main` and every PR.

**No external dependencies.** The tests never download data or call external APIs (yfinance, IOL,
remote MLflow). They use synthetic data generated with `numpy.random.default_rng` (fixed,
reproducible seeds) and pytest's `tmp_path` for per-test isolated I/O.

**Conditional tests.** Two tests in `test_reevaluation.py`
(`test_predict_latest_on_real_champion_matches_predict_series` and
`test_predict_series_reproduces_stored_walkforward`) require `torch` plus at least one real
champion in `models/registry.json` with its artifacts (`*_model.pth`,
`*_walkforward_predictions.csv`, a cleaned CSV in `data/clean/`). If any condition is missing they
`pytest.skip` rather than fail — they don't depend on the repo being in a particular training
state. `test_feature_contract_compat.py::test_current_feature_code_covers_all_registered_models`
is conditional for the same reason; its counterpart
`test_active_features_config_is_valid_subset_of_catalog` is not — it always runs in CI and depends
only on `base.yaml`.

---

## Data-leakage prevention

### `test_walkforward_splits.py`
Verifies the properties of the walk-forward split (`sklearn.model_selection.TimeSeriesSplit` with
`gap`) used in `src/e1/train_pipeline.py` and its E2/E3 analogues:

- Train and test are always disjoint, and train precedes test in time.
- The embargo (`gap`) between the end of train and the start of test is ≥ the strategy's
  `horizon_days` (90 for E1, 20 for E2) — this is what stops the prediction horizon leaking
  across folds.
- Each successive fold has an equal or larger training set (a real walk-forward, not shuffled k-fold).
- If `test_size <= embargo`, the pipeline bumps `test_size = embargo + 1` (the same guard as
  `train_pipeline.py`).

### `test_feature_scaling.py`
Replicates the inline z-score scaling logic of `src/e{1,2,3}/train_pipeline.py` and verifies the
central invariant: **mean and standard deviation are computed on the training fold only**, never
on val/test. Includes cases where the val/test distributions are deliberately shifted (so leakage
would show up if present), non-mutation of the input arrays, and numerical stability when a
feature has std = 0.

---

## Backtesting

### `test_backtest_costs.py`
Verifies the transaction-cost formula shared by `src/backtest/backtest_daily.py` (E1/E2) and
`src/backtest/backtest_intraday.py` (E3):

```
costs = |Δposition| * (round_trip_bps / 10000) / 2
```

It checks zero cost at `round_trip_bps=0`, costs never negative, exact proportionality (20 bps is
twice the cost of 10 bps), `net_ret = gross_ret - costs`, and that the intraday engine uses the
same formula.

---

## Model lifecycle (`src/lifecycle/`)

### `test_registry_atomicity.py`
Verifies that `ModelRegistry._save()` writes atomically (`tempfile` + `os.replace`, not a direct
write over `registry.json`): a write that fails halfway leaves the original file intact, no orphan
`.tmp` files remain after a successful save, and data persists correctly across successive
`ModelRegistry` instances.

### `test_lifecycle_paths.py`
Verifies `resolve_lifecycle_paths()` (`src/utils.py`): relative config paths resolve against
`root`, absolute paths are respected as-is, and missing config falls back to the defaults
(`models/registry.json`, `models/metrics_log.jsonl`).

### `test_guardrails.py`
Verifies the Phase 1 guardrails (`src/lifecycle/guardrails.py::validate_candidate`) that block a
candidate's promotion before metrics are even compared: missing model, NaN/Inf in predictions,
Sharpe below the minimum, and IC worse than the baseline.

### `test_promotion_per_strategy.py`
Verifies the hierarchical resolution of promotion config
(`get_strategy_promotion_config()` in `src/lifecycle/promotion.py`):
`per_strategy.{strategy}` → `per_strategy.{prefix}` (e.g. `e1_conservative` → `e1`) → global
fallback. Covers that each strategy can carry its own `scoring_weights` (E3 weights
`profit_factor` and `win_rate` instead of `calmar`), that the original config isn't mutated or
lost, and that `compute_score()` yields different scores under different weights.

### `test_promotion_common_window.py`
The most business-critical test in the suite. It verifies `evaluate_and_promote()` and
`compare_on_common_window()` — the **fair** champion-vs-candidate comparison over a common
out-of-sample window, rather than each model on its own window, which would artificially favor
whichever has more recent metrics. Torch inference is mocked (predictions are injected directly)
so it runs without a GPU or a real model. It covers all four comparison modes: `fair_window` (a
valid common window), `identity_skip` (candidate identical to champion, comparison skipped),
`insufficient_evidence` (OOS window too small, champion retained), and `stored` (falling back to
saved metrics when the champion has no `train_data_end`).

### `test_reevaluation.py`
Support tests for the fair re-backtest: `recompute_metrics_on_window()` returns the keys the
composite score consumes (`bt_sharpe`, `bt_calmar`, `ml_ic`, `ml_directional_accuracy`) with
finite, coherent values; `load_walkforward_predictions()` round-trips correctly against CSV. The
two conditional tests verify something stronger — that `predict_series()` reconstructs
substantially the same `y_pred` (`atol=1e-3`) as was saved during the original training, using the
frozen model and the same scalers. That's the proof that post-hoc inference introduces no
reproducibility drift.

---

## Drift monitoring

### `test_drift.py`
Verifies `src/lifecycle/drift.py` (PSI plus a Kolmogorov-Smirnov test, to detect feature drift
between the reference and current sets): PSI ≈ 0 for identical distributions, PSI above threshold
under a large shift, PSI monotonic in the size of the shift, edge cases handled (constant
reference, empty inputs, NaN/Inf ignored rather than propagated), reports considering only the
numeric columns common to both frames, and JSONL logging (`log_drift_report`) writing one entry
per call and appending correctly.

---

## Suite conventions

- One test file per `src/` module or concern, named `test_<module_or_concern>.py`.
- Synthetic data via `numpy.random.default_rng(seed)` — never real market data, never network calls.
- Disk I/O (registry, CSVs, logs) always under pytest's `tmp_path` fixture, never against the real
  `models/registry.json`.
- Mocks only where standing up the real object is expensive (torch inference in
  `test_promotion_common_window.py`). The rest of the business logic — guardrails, promotion,
  drift — is exercised for real.
