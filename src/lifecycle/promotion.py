"""Champion/Challenger promotion logic.

Compares a candidate model against the current champion using a configurable
composite score.  If the candidate's score exceeds the champion's by at least
``min_improvement`` (code default 5 %; overridden by ``base.yaml``), it is
promoted to champion.

The composite score is a weighted average of the **raw** metric values
(weights renormalised to sum to 1):

    score = Σ  (w_i / Σw) · metric_i

Lower-is-better metrics (``ml_mae``, ``ml_rmse``, ``bt_max_drawdown``) are
inverted as ``1 / (1 + |metric|)`` before weighting so that a higher score is
always better.  Metrics are **not** rescaled to a common range, so the score
is dominated by large-magnitude metrics (typically ``bt_sharpe`` /
``bt_calmar``) — see the caveat below.

The relative comparison ("normalisation") happens at the *score* level, not
per metric:

    improvement = (candidate_score - champion_score) / champion_score

Design decisions
----------------
* **Per-ticker promotion** — each ticker is evaluated independently.
* **Bootstrap case** — if no champion exists for a ticker, the candidate is
  promoted automatically (configurable via ``first_champion_strategy``).
* **Safety net** — the candidate must have ``bt_sharpe > 0``
  (``require_positive_sharpe``).

Caveat
------
Because metrics are combined on their raw scales, the composite score is noisy
run-to-run (empirically ±30-65% for the same model config, driven by shifting
walk-forward windows).  ``min_improvement`` is therefore a hysteresis knob to
avoid tie-churn, **not** a statistical significance gate; the real safeguard is
manual review (``auto_promote=false`` + dry-run default in ``promote_candidate``).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .registry import ModelRegistry

# Default path for the append-only promotion audit log.
_DEFAULT_LOG_PATH = Path(__file__).resolve().parents[2] / "models" / "promotion_log.jsonl"


# ---------------------------------------------------------------
# Default configuration (overridden by base.yaml at runtime)
# ---------------------------------------------------------------
_DEFAULT_PROMOTION_CONFIG: dict[str, Any] = {
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
    "per_strategy": {},
}


# ---------------------------------------------------------------
# Result data-class
# ---------------------------------------------------------------
@dataclass
class PromotionDecision:
    """Result of a candidate-vs-champion comparison."""

    ticker: str
    should_promote: bool
    candidate_score: float
    champion_score: float
    improvement_pct: float
    reason: str
    detail: dict[str, float] = field(default_factory=dict)
    promoted: bool = False  # True only after registry write
    candidate_variant: str = ""  # e.g. "e1_conservative", "e1_simple"
    champion_variant: str = ""  # variant of current champion


# ---------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------
def _append_promotion_log(
    decision: "PromotionDecision",
    strategy: str,
    log_path: Path | None = None,
) -> None:
    """Append one promotion decision to the JSONL audit log.

    The file is created if it does not exist.  Each line is a valid JSON
    object so the file can be streamed / ``grep``-ped without parsing the
    whole file.
    """
    path = log_path or _DEFAULT_LOG_PATH
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "strategy": strategy,
        "ticker": decision.ticker,
        "candidate_variant": decision.candidate_variant,
        "champion_variant": decision.champion_variant,
        "should_promote": decision.should_promote,
        "promoted": decision.promoted,
        "candidate_score": decision.candidate_score,
        "champion_score": decision.champion_score,
        "improvement_pct": decision.improvement_pct,
        "reason": decision.reason,
        "detail": decision.detail,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------
# Per-strategy config resolution
# ---------------------------------------------------------------

def get_strategy_promotion_config(
    config: dict[str, Any] | None,
    strategy: str,
) -> dict[str, Any]:
    """Resolve promotion config for a specific strategy.

    Lookup order (first match wins):
    1. ``per_strategy.{strategy}`` — exact match (e.g. ``e1_conservative``)
    2. ``per_strategy.{prefix}`` — prefix before first ``_`` (e.g. ``e1``)
    3. Global promotion config (top-level keys)

    Per-strategy entries are *merged* over the global defaults so you only
    need to override the keys that differ.  For example, a strategy can
    override ``scoring_weights`` while inheriting ``min_improvement``.

    Parameters
    ----------
    config : dict | None
        The ``lifecycle.promotion`` sub-dict from ``base.yaml``.
    strategy : str
        Strategy key, e.g. ``"e1"``, ``"e1_conservative"``, ``"e3"``.

    Returns
    -------
    dict[str, Any]
        Fully-resolved promotion config (flat, no ``per_strategy`` key).
    """
    cfg = {**_DEFAULT_PROMOTION_CONFIG, **(config or {})}
    per_strategy = cfg.pop("per_strategy", {})

    # 1. Exact match
    overrides = per_strategy.get(strategy)

    # 2. Prefix fallback (e.g. "e1_conservative" → "e1")
    if overrides is None and "_" in strategy:
        prefix = strategy.split("_", 1)[0]
        overrides = per_strategy.get(prefix)

    if overrides:
        cfg = {**cfg, **overrides}

    return cfg


# ---------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------

# Metrics where *higher* is better.  Any metric added to the scoring
# weights dict is assumed higher-is-better unless listed in _LOWER_IS_BETTER.
_LOWER_IS_BETTER: frozenset[str] = frozenset({
    "ml_mae",
    "ml_rmse",
    "bt_max_drawdown",
})


def compute_score(
    metrics: dict[str, Any],
    weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Compute a weighted composite score.

    Returns ``(score, detail)`` where *detail* maps each metric key to
    its weighted contribution.
    """
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0, {}

    detail: dict[str, float] = {}
    score = 0.0
    for key, w in weights.items():
        raw = metrics.get(key)
        if raw is None or not isinstance(raw, (int, float)):
            continue
        val = float(raw)
        if math.isnan(val) or math.isinf(val):
            continue

        # For lower-is-better metrics, invert so "higher score = better".
        if key in _LOWER_IS_BETTER:
            # Use 1/(1+val) so score stays in (0, 1] and is monotonically
            # decreasing in val (val >= 0 by construction for MAE/RMSE/DD).
            val = 1.0 / (1.0 + abs(val))

        contribution = (w / total_weight) * val
        detail[key] = contribution
        score += contribution

    return score, detail


# ---------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------
def compare_candidate_vs_champion(
    candidate_metrics: dict[str, Any],
    champion_metrics: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> PromotionDecision:
    """Compare a candidate against the current champion.

    Parameters
    ----------
    candidate_metrics, champion_metrics
        Metric dictionaries as stored in ``registry.json``.
    config
        The ``lifecycle.promotion`` sub-dict.  Falls back to
        ``_DEFAULT_PROMOTION_CONFIG`` for missing keys.
    """
    cfg = {**_DEFAULT_PROMOTION_CONFIG, **(config or {})}
    weights: dict[str, float] = cfg.get("scoring_weights", _DEFAULT_PROMOTION_CONFIG["scoring_weights"])
    min_improvement: float = float(cfg.get("min_improvement", 0.05))
    require_positive_sharpe: bool = bool(cfg.get("require_positive_sharpe", True))

    # --- Safety net ---
    cand_sharpe = candidate_metrics.get("bt_sharpe")
    if require_positive_sharpe and (
        cand_sharpe is None or float(cand_sharpe) <= 0
    ):
        return PromotionDecision(
            ticker="",
            should_promote=False,
            candidate_score=0.0,
            champion_score=0.0,
            improvement_pct=0.0,
            reason=f"candidate bt_sharpe={cand_sharpe} <= 0 (safety net)",
        )

    cand_score, cand_detail = compute_score(candidate_metrics, weights)
    champ_score, champ_detail = compute_score(champion_metrics, weights)

    if champ_score > 0:
        improvement = (cand_score - champ_score) / champ_score
    elif cand_score > champ_score:
        improvement = 1.0  # any improvement over a zero/negative champion
    else:
        improvement = 0.0

    should = improvement >= min_improvement

    if should:
        reason = (
            f"candidate score {cand_score:.4f} beats champion {champ_score:.4f} "
            f"by {improvement:.1%} (>= {min_improvement:.0%} threshold)"
        )
    else:
        reason = (
            f"candidate score {cand_score:.4f} vs champion {champ_score:.4f}: "
            f"improvement {improvement:.1%} < {min_improvement:.0%} threshold"
        )

    return PromotionDecision(
        ticker="",
        should_promote=should,
        candidate_score=round(cand_score, 6),
        champion_score=round(champ_score, 6),
        improvement_pct=round(improvement, 6),
        reason=reason,
        detail={
            **{f"candidate_{k}": v for k, v in cand_detail.items()},
            **{f"champion_{k}": v for k, v in champ_detail.items()},
        },
    )


# ---------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------
def evaluate_and_promote(
    registry: ModelRegistry,
    strategy: str,
    ticker: str,
    config: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
    force: bool = False,
    log_path: Path | None = None,
) -> PromotionDecision:
    """Evaluate whether the candidate should replace the champion.

    Parameters
    ----------
    registry : ModelRegistry
        Registry instance (will be written to if promotion occurs).
    strategy, ticker : str
        Strategy and ticker to evaluate.
    config : dict | None
        The ``lifecycle.promotion`` sub-dict from ``base.yaml``.
    dry_run : bool
        If *True*, never writes to registry — only returns the decision.
    force : bool
        If *True*, promote regardless of score comparison.

    Returns
    -------
    PromotionDecision
    """
    cfg = get_strategy_promotion_config(config, strategy)

    candidate_entry = registry.get_candidate(strategy, ticker)
    if candidate_entry is None:
        decision = PromotionDecision(
            ticker=ticker,
            should_promote=False,
            candidate_score=0.0,
            champion_score=0.0,
            improvement_pct=0.0,
            reason="no candidate registered",
        )
        _append_promotion_log(decision, strategy, log_path)
        return decision

    cand_variant = candidate_entry.get("variant", "")
    champion_entry = registry.get_champion(strategy, ticker)
    champ_variant = champion_entry.get("variant", "") if champion_entry else ""

    # --- Bootstrap: no champion yet ---
    if champion_entry is None:
        first_strategy = cfg.get("first_champion_strategy", "promote")
        if first_strategy == "promote" or force:
            decision = PromotionDecision(
                ticker=ticker,
                should_promote=True,
                candidate_score=0.0,
                champion_score=0.0,
                improvement_pct=0.0,
                reason="no existing champion — bootstrap promotion",
                candidate_variant=cand_variant,
            )
        else:
            decision = PromotionDecision(
                ticker=ticker,
                should_promote=False,
                candidate_score=0.0,
                champion_score=0.0,
                improvement_pct=0.0,
                reason="no existing champion and first_champion_strategy != 'promote'",
                candidate_variant=cand_variant,
            )
        if decision.should_promote and not dry_run:
            registry.promote_to_champion(strategy, ticker, reason="bootstrap_promotion")
            decision.promoted = True
        _append_promotion_log(decision, strategy, log_path)
        return decision

    # --- Normal comparison ---
    cand_metrics = candidate_entry.get("metrics", {})
    champ_metrics = champion_entry.get("metrics", {})

    decision = compare_candidate_vs_champion(cand_metrics, champ_metrics, cfg)
    decision.ticker = ticker
    decision.candidate_variant = cand_variant
    decision.champion_variant = champ_variant

    if force:
        decision.should_promote = True
        decision.reason = f"forced promotion (original: {decision.reason})"

    if decision.should_promote and not dry_run:
        registry.promote_to_champion(
            strategy,
            ticker,
            reason=f"auto_promotion|improvement={decision.improvement_pct:.4f}",
        )
        decision.promoted = True

    _append_promotion_log(decision, strategy, log_path)
    return decision


def evaluate_and_promote_all(
    registry: ModelRegistry,
    strategy: str,
    tickers: list[str],
    config: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
    force: bool = False,
    log_path: Path | None = None,
) -> dict[str, PromotionDecision]:
    """Run promotion evaluation for multiple tickers.

    Returns a dict mapping ``ticker → PromotionDecision``.
    """
    results: dict[str, PromotionDecision] = {}
    for ticker in tickers:
        decision = evaluate_and_promote(
            registry, strategy, ticker, config,
            dry_run=dry_run, force=force, log_path=log_path,
        )
        results[ticker] = decision
    return results


# ---------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------
def print_promotion_summary(
    decisions: dict[str, PromotionDecision],
    strategy: str = "",
) -> None:
    """Print a human-readable summary table of promotion decisions."""
    if not decisions:
        print("No promotion decisions to display.")
        return

    header = (
        f"{'Ticker':<12} {'Champ variant':<18} {'Cand variant':<18} "
        f"{'Champion':>10} {'Candidate':>10} {'Improv%':>8} {'Decision':<10} Reason"
    )
    print("-" * len(header))
    if strategy:
        print(f"Strategy: {strategy}")
    print(header)
    print("-" * len(header))

    promoted_count = 0
    for ticker, d in decisions.items():
        status = "PROMOTE" if d.should_promote else "KEEP"
        if d.promoted:
            status += " ✓"
            promoted_count += 1
        champ_var = d.champion_variant or "-"
        cand_var = d.candidate_variant or "-"
        print(
            f"{ticker:<12} {champ_var:<18} {cand_var:<18} "
            f"{d.champion_score:>10.4f} {d.candidate_score:>10.4f} "
            f"{d.improvement_pct:>7.1%} {status:<10} {d.reason}"
        )

    print("-" * len(header))
    total = len(decisions)
    print(f"Total: {promoted_count}/{total} promoted, {total - promoted_count}/{total} kept")
