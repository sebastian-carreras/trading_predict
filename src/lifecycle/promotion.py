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
    # Fair comparison: re-backtest champion vs candidate on a common
    # out-of-sample window (dates after the champion's train_data_end that
    # neither model trained on) instead of comparing stale stored metrics.
    "fair_window": True,
    # Minimum number of common OOS samples required to trust a fair comparison.
    # Below this, KEEP the champion (not enough fresh evidence).
    "min_eval_samples": 30,
    "scoring_weights": {
        "bt_sharpe": 0.35,
        "ml_ic": 0.25,
        "ml_directional_accuracy": 0.20,
        "bt_calmar": 0.20,
    },
    "per_strategy": {},
    # Regla v2 (compuertas + métrica primaria). Ver compute_score_v2. En MODO SOMBRA:
    # se calcula y se registra, pero la decisión que se ejecuta sigue siendo la v1.
    "v2": {
        "enabled": True,        # calcular la v2 (sombra)
        "active": False,        # que la v2 DECIDA (paso 4 del plan; requiere piso de ruido)
        "primary_metric": "bt_sharpe_excess",
        "gates": {
            "bt_sharpe": 0.0,
            "ml_ic": 0.0,
            "ml_dir_acc_edge": 0.0,
        },
        # Mejora absoluta mínima en la métrica primaria. 0.0 hasta que
        # scripts/evaluation/noise_floor.py mida la dispersión semilla-a-semilla.
        "min_improvement_abs": 0.0,
    },
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
    # Comparison provenance (set by the fair-window path).
    comparison_mode: str = "stored"  # stored|fair_window|bootstrap|identity_skip|insufficient_evidence
    eval_window_start: str | None = None
    eval_window_end: str | None = None
    eval_n_samples: int | None = None
    # --- Regla v2 en MODO SOMBRA: se calcula y se registra, no decide nada ---
    score_v2_candidate: float | None = None
    score_v2_champion: float | None = None
    would_promote_v2: bool | None = None
    v2_gates_passed: bool | None = None
    reason_v2: str = ""


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
        "comparison_mode": decision.comparison_mode,
        "eval_window_start": decision.eval_window_start,
        "eval_window_end": decision.eval_window_end,
        "eval_n_samples": decision.eval_n_samples,
        # Sombra v2: permite comparar ambas reglas sobre el mismo histórico sin
        # que la v2 haya decidido nada. Ver scripts/evaluation/backfill_score_v2.py.
        "score_v2_candidate": decision.score_v2_candidate,
        "score_v2_champion": decision.score_v2_champion,
        "would_promote_v2": decision.would_promote_v2,
        "v2_gates_passed": decision.v2_gates_passed,
        "reason_v2": decision.reason_v2,
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
# Scoring v2 — compuertas duras + una métrica primaria (modo sombra)
# ---------------------------------------------------------------
#
# Por qué v2 existe
# -----------------
# ``compute_score`` suma métricas en escala CRUDA, así que el peso que decide no es
# el peso escrito sino ``w_i · std(metric_i)``. Medido sobre el registry:
#
#   E1: bt_calmar nominal 0.20 → efectivo 0.443 | ml_directional_accuracy 0.20 → 0.057
#   E2: ml_directional_accuracy nominal 0.30 → efectivo 0.082 | bt_sharpe 0.30 → 0.458
#
# Es decir, bt_sharpe + bt_calmar deciden el 85% (E1) / 82% (E2), y ambas derivan de la
# MISMA curva de equity. Además ``ml_directional_accuracy`` cruda premia la deriva del
# activo, no la habilidad (ver src/metrics/skill.py).
#
# v2 elimina los pesos: compuertas binarias + UNA métrica primaria neutral a la deriva.


def get_v2_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Resuelve el sub-config ``v2`` sobre los defaults."""
    base = dict(_DEFAULT_PROMOTION_CONFIG["v2"])
    override = (cfg or {}).get("v2") or {}
    merged = {**base, **override}
    merged["gates"] = {**base["gates"], **(override.get("gates") or {})}
    return merged


def evaluate_gates(
    metrics: dict[str, Any],
    gates: dict[str, float],
) -> tuple[bool, list[str]]:
    """Compuertas duras: cada métrica debe superar ESTRICTAMENTE su umbral.

    Returns ``(passed, failures)`` donde *failures* describe cada compuerta fallada.
    Una métrica ausente o no finita cuenta como fallo: no se promueve a ciegas.
    """
    failures: list[str] = []
    for key, threshold in gates.items():
        raw = metrics.get(key)
        if raw is None or not isinstance(raw, (int, float)):
            failures.append(f"{key}=ausente")
            continue
        val = float(raw)
        if math.isnan(val) or math.isinf(val):
            failures.append(f"{key}={raw}")
            continue
        if val <= float(threshold):
            failures.append(f"{key}={val:.4f}<={threshold}")
    return (not failures), failures


def compute_score_v2(
    metrics: dict[str, Any],
    cfg: dict[str, Any] | None = None,
) -> tuple[float, bool, list[str]]:
    """Score v2: el valor de la métrica primaria, más el resultado de las compuertas.

    Returns ``(score, gates_passed, gate_failures)``. El score es ``nan`` si la métrica
    primaria falta — que es lo que pasa con los modelos entrenados antes de este cambio.
    """
    v2 = get_v2_config(cfg)
    raw = metrics.get(v2["primary_metric"])
    if raw is None or not isinstance(raw, (int, float)):
        score = float("nan")
    else:
        score = float(raw)
        if math.isnan(score) or math.isinf(score):
            score = float("nan")

    passed, failures = evaluate_gates(metrics, v2["gates"])
    return score, passed, failures


def compare_v2(
    candidate_metrics: dict[str, Any],
    champion_metrics: dict[str, Any],
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decisión v2 (sombra): compuertas + mejora ABSOLUTA en la métrica primaria.

    A diferencia de la v1, la mejora es absoluta y no relativa: ``(cand-champ)/champ``
    sobre una escala arbitraria no tiene interpretación, y se dispara cuando el
    denominador es chico.
    """
    v2 = get_v2_config(cfg)
    cand_score, cand_ok, cand_fail = compute_score_v2(candidate_metrics, cfg)
    champ_score, _champ_ok, _ = compute_score_v2(champion_metrics, cfg)

    min_abs = float(v2.get("min_improvement_abs", 0.0))
    metric = v2["primary_metric"]

    if math.isnan(cand_score):
        return {
            "candidate_score": None, "champion_score": None,
            "gates_passed": cand_ok, "gate_failures": cand_fail,
            "improvement_abs": None, "would_promote": False,
            "reason": f"v2 no evaluable: falta {metric} en el candidato",
        }

    if not cand_ok:
        return {
            "candidate_score": round(cand_score, 6),
            "champion_score": None if math.isnan(champ_score) else round(champ_score, 6),
            "gates_passed": False, "gate_failures": cand_fail,
            "improvement_abs": None, "would_promote": False,
            "reason": f"v2 compuertas falladas: {', '.join(cand_fail)}",
        }

    # Sin champion evaluable en v2 (modelo viejo, sin la métrica) no hay contra qué comparar.
    if math.isnan(champ_score):
        return {
            "candidate_score": round(cand_score, 6), "champion_score": None,
            "gates_passed": True, "gate_failures": [],
            "improvement_abs": None, "would_promote": False,
            "reason": f"v2 sin comparación: falta {metric} en el champion (entrenado pre-v2)",
        }

    delta = cand_score - champ_score
    # Estrictamente mayor: un empate (Δ=0) NUNCA promueve. Con `min_improvement_abs: 0.0`
    # un `>=` promovería en cada empate y generaría churn — pasó con EDN.BA en el backfill,
    # donde candidato y champion tenían ambos sharpe_excess exactamente 0.
    would = delta > 0 and delta >= min_abs
    return {
        "candidate_score": round(cand_score, 6),
        "champion_score": round(champ_score, 6),
        "gates_passed": True, "gate_failures": [],
        "improvement_abs": round(delta, 6),
        "would_promote": bool(would),
        "reason": (
            f"v2 {metric}: candidato {cand_score:+.4f} vs champion {champ_score:+.4f} "
            f"(Δ={delta:+.4f}, umbral {min_abs:+.4f}) → "
            f"{'PROMOVER' if would else 'MANTENER'}"
        ),
    }


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

    # --- Sombra v2: se calcula siempre, no altera ninguna decisión ---
    v2_shadow: dict[str, Any] = {}
    if get_v2_config(cfg).get("enabled", True):
        try:
            v2_shadow = compare_v2(candidate_metrics, champion_metrics, cfg)
        except Exception as exc:  # la sombra nunca puede romper la promoción real
            v2_shadow = {"reason": f"v2 falló: {exc}"}

    def _with_v2(decision: PromotionDecision) -> PromotionDecision:
        decision.score_v2_candidate = v2_shadow.get("candidate_score")
        decision.score_v2_champion = v2_shadow.get("champion_score")
        decision.would_promote_v2 = v2_shadow.get("would_promote")
        decision.v2_gates_passed = v2_shadow.get("gates_passed")
        decision.reason_v2 = v2_shadow.get("reason", "")
        return decision

    # --- Safety net ---
    cand_sharpe = candidate_metrics.get("bt_sharpe")
    if require_positive_sharpe and (
        cand_sharpe is None or float(cand_sharpe) <= 0
    ):
        return _with_v2(PromotionDecision(
            ticker="",
            should_promote=False,
            candidate_score=0.0,
            champion_score=0.0,
            improvement_pct=0.0,
            reason=f"candidate bt_sharpe={cand_sharpe} <= 0 (safety net)",
        ))

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

    return _with_v2(PromotionDecision(
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
    ))


# ---------------------------------------------------------------
# Fair comparison on a common out-of-sample window
# ---------------------------------------------------------------
def _common_oos_window(
    candidate_wf: "Any",  # pd.DataFrame indexed by timestamp with y_true/y_pred
    champion_preds: "Any",  # pd.Series indexed by timestamp
    champion_train_end: str,
) -> "Any":  # pd.DatetimeIndex
    """Dates OOS for BOTH models: after the champion cutoff, predictable by
    both, and with a realised target (present in the candidate WF file)."""
    import pandas as pd

    cutoff = pd.Timestamp(champion_train_end).normalize()
    idx = candidate_wf.index
    idx_naive = idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx
    mask = idx_naive.normalize() > cutoff
    w = candidate_wf.index[mask]
    # Intersect with dates the champion can actually predict.
    return w.intersection(champion_preds.index)


def compare_on_common_window(
    registry: ModelRegistry,
    strategy: str,
    ticker: str,
    cfg: dict[str, Any],
    full_config: dict[str, Any],
    ohlcv: "Any",  # pd.DataFrame (today's clean data, ts-indexed with 'close')
) -> tuple[PromotionDecision, dict[str, float] | None, dict[str, float] | None]:
    """Fair champion-vs-candidate comparison on a common OOS window.

    Returns ``(decision, metrics_cand_W, metrics_champ_W)``. The metric dicts
    are ``None`` when the fair path could not run (decision explains why, and
    ``comparison_mode`` says which fallback applies). Never writes to the
    registry — the caller decides based on ``decision.should_promote``.
    """
    # Lazy import: reevaluation pulls in torch via the model loader, and we want
    # promotion.py to stay importable in torch-free environments (unit tests).
    from . import reevaluation

    candidate = registry.get_candidate(strategy, ticker)
    champion = registry.get_champion(strategy, ticker)
    cand_variant = candidate.get("variant", "") if candidate else ""
    champ_variant = champion.get("variant", "") if champion else ""
    champ_train_end = (champion or {}).get("train_data_end")

    min_eval_samples = int(cfg.get("min_eval_samples", 30))

    def _keep(reason: str, mode: str) -> tuple[PromotionDecision, None, None]:
        return (
            PromotionDecision(
                ticker=ticker, should_promote=False,
                candidate_score=0.0, champion_score=0.0, improvement_pct=0.0,
                reason=reason, comparison_mode=mode,
                candidate_variant=cand_variant, champion_variant=champ_variant,
            ),
            None, None,
        )

    # Identity guard: never compare a model against itself.
    if (
        candidate is not None
        and champion is not None
        and cand_variant == champ_variant
        and candidate.get("train_data_end") is not None
        and candidate.get("train_data_end") == champ_train_end
    ):
        return _keep(
            "candidate identical to champion (same variant + train_data_end) — skipped",
            "identity_skip",
        )

    if champ_train_end is None:
        # Champion predates train_data_end tracking → can't build a fair window.
        return _keep(
            "champion has no train_data_end; fair window unavailable (retrain to enable)",
            "stored",
        )

    champion_preds = reevaluation.champion_predictions(strategy, ticker, registry, ohlcv)
    if champion_preds is None or len(champion_preds) == 0:
        return _keep("champion model could not produce predictions", "stored")

    candidate_wf = reevaluation.load_walkforward_predictions(
        candidate.get("run_dir", ""), ticker,
    )
    if candidate_wf is None or candidate_wf.empty:
        return _keep("candidate walk-forward predictions not found", "stored")

    window = _common_oos_window(candidate_wf, champion_preds, champ_train_end)
    n = len(window)
    w_start = str(window.min().date()) if n else None
    w_end = str(window.max().date()) if n else None

    if n < min_eval_samples:
        dec, _, _ = _keep(
            f"insufficient fresh out-of-sample evidence: {n} < min_eval_samples={min_eval_samples}",
            "insufficient_evidence",
        )
        dec.eval_window_start, dec.eval_window_end, dec.eval_n_samples = w_start, w_end, n
        return dec, None, None

    # Recompute metrics for BOTH models on identical dates.
    bt_params_cand = reevaluation.backtest_params_from_config(full_config, cand_variant)
    bt_params_champ = reevaluation.backtest_params_from_config(full_config, champ_variant)
    y_true = candidate_wf.loc[window, "y_true"].to_numpy()
    cand_pred = candidate_wf.loc[window, "y_pred"].to_numpy()
    champ_pred = champion_preds.loc[window].to_numpy()

    metrics_cand_W = reevaluation.recompute_metrics_on_window(
        ohlcv, window, cand_pred, y_true, bt_params_cand,
    )
    metrics_champ_W = reevaluation.recompute_metrics_on_window(
        ohlcv, window, champ_pred, y_true, bt_params_champ,
    )

    decision = compare_candidate_vs_champion(metrics_cand_W, metrics_champ_W, cfg)
    decision.ticker = ticker
    decision.candidate_variant = cand_variant
    decision.champion_variant = champ_variant
    decision.comparison_mode = "fair_window"
    decision.eval_window_start = w_start
    decision.eval_window_end = w_end
    decision.eval_n_samples = n
    decision.reason = f"[fair OOS window {w_start}..{w_end}, n={n}] {decision.reason}"
    return decision, metrics_cand_W, metrics_champ_W


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
    fair_window: bool | None = None,
    ohlcv_loader: "Any" = None,
    full_config: dict[str, Any] | None = None,
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
    fair_window : bool | None
        Use the common out-of-sample re-backtest instead of stored-vs-stored
        metrics. ``None`` resolves from ``config['fair_window']`` (default True).
        Falls back to the stored comparison if prerequisites are missing
        (no ``ohlcv_loader``, champion without ``train_data_end``, etc.).
    ohlcv_loader : callable(ticker) -> DataFrame | None
        Loads today's clean OHLCV (ts-indexed, ``close`` column) for the fair
        window. Required for the fair path.
    full_config : dict | None
        The full ``base.yaml`` dict (for backtest thresholds/costs). Required
        for the fair path.

    Returns
    -------
    PromotionDecision
    """
    cfg = get_strategy_promotion_config(config, strategy)
    use_fair = cfg.get("fair_window", True) if fair_window is None else fair_window

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

    # --- Comparison ---
    # metrics_champ_W / metrics_cand_W are the recomputed OOS metrics when the
    # fair path runs; used to refresh the champion's recent_metrics afterwards.
    metrics_cand_W: dict[str, float] | None = None
    metrics_champ_W: dict[str, float] | None = None

    if use_fair and ohlcv_loader is not None and full_config is not None:
        try:
            ohlcv = ohlcv_loader(ticker)
            if ohlcv is not None and len(ohlcv):
                decision, metrics_cand_W, metrics_champ_W = compare_on_common_window(
                    registry, strategy, ticker, cfg, full_config, ohlcv,
                )
                # "stored" mode here means the fair path could not run (e.g. a
                # champion predating train_data_end tracking) — do the real
                # stored comparison rather than a forced KEEP.
                if decision.comparison_mode == "stored":
                    decision = _stored_comparison(
                        candidate_entry, champion_entry, cfg, ticker,
                        cand_variant, champ_variant, note=decision.reason,
                    )
                    metrics_cand_W = metrics_champ_W = None
            else:
                decision = _stored_comparison(
                    candidate_entry, champion_entry, cfg, ticker,
                    cand_variant, champ_variant,
                    note="ohlcv unavailable for fair window",
                )
        except Exception as exc:  # never let re-backtest failure block the DAG
            if isinstance(exc, KeyError):
                print(
                    f"[promotion] WARNING feature-drift fallback for {strategy}/{ticker}: "
                    f"{exc} — champion's frozen feature_names no longer producible by "
                    f"compute_{strategy}_features; falling back to stored comparison. "
                    "See tests/test_feature_contract_compat.py."
                )
            decision = _stored_comparison(
                candidate_entry, champion_entry, cfg, ticker,
                cand_variant, champ_variant,
                note=f"fair window failed ({exc})",
            )
    else:
        decision = _stored_comparison(
            candidate_entry, champion_entry, cfg, ticker,
            cand_variant, champ_variant,
        )

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
        # Newly promoted champion (former candidate): stamp its fresh OOS metrics.
        if metrics_cand_W is not None:
            registry.set_recent_metrics(
                strategy, ticker, metrics_cand_W,
                window_start=decision.eval_window_start,
                window_end=decision.eval_window_end,
                n_samples=decision.eval_n_samples,
            )
    elif not dry_run and metrics_champ_W is not None:
        # Kept champion: refresh its recent (current-quality) metrics.
        registry.set_recent_metrics(
            strategy, ticker, metrics_champ_W,
            window_start=decision.eval_window_start,
            window_end=decision.eval_window_end,
            n_samples=decision.eval_n_samples,
        )

    _append_promotion_log(decision, strategy, log_path)
    return decision


def _stored_comparison(
    candidate_entry: dict[str, Any],
    champion_entry: dict[str, Any],
    cfg: dict[str, Any],
    ticker: str,
    cand_variant: str,
    champ_variant: str,
    *,
    note: str = "",
) -> PromotionDecision:
    """Legacy stored-vs-stored comparison (fallback when the fair path can't run)."""
    decision = compare_candidate_vs_champion(
        candidate_entry.get("metrics", {}),
        champion_entry.get("metrics", {}),
        cfg,
    )
    decision.ticker = ticker
    decision.candidate_variant = cand_variant
    decision.champion_variant = champ_variant
    decision.comparison_mode = "stored"
    if note:
        decision.reason = f"[stored fallback: {note}] {decision.reason}"
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
