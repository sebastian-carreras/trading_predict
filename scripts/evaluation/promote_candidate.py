#!/usr/bin/env python3
"""CLI tool to evaluate and promote candidates to champion.

Usage examples
--------------
# Dry-run (default): see decisions without writing to registry
python -m scripts.evaluation.promote_candidate

# Promote candidates that beat the current champion
python -m scripts.evaluation.promote_candidate --execute

# Specific tickers only
python -m scripts.evaluation.promote_candidate --tickers AAPL,MSFT --execute

# Force-promote regardless of score comparison
python -m scripts.evaluation.promote_candidate --tickers YPFD.BA --force --execute

# Show verbose detail per metric
python -m scripts.evaluation.promote_candidate --verbose

# Show the last 20 entries from the promotion audit log
python -m scripts.evaluation.promote_candidate --show-log 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Resolve project root (assumes this script lives in scripts/evaluation/)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.lifecycle.registry import ModelRegistry  # noqa: E402
from src.lifecycle.promotion import (  # noqa: E402
    _DEFAULT_LOG_PATH,
    evaluate_and_promote_all,
    get_strategy_promotion_config,
    print_promotion_summary,
)


def _show_log(log_path: Path, last_n: int) -> None:
    """Print the last *last_n* entries from the promotion audit log."""
    if not log_path.exists():
        print(f"No promotion log found at: {log_path}")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    entries = lines[-last_n:] if last_n > 0 else lines
    if not entries:
        print("Promotion log is empty.")
        return

    print(f"\n{'='*60}")
    print(f"Promotion Audit Log  (last {len(entries)} of {len(lines)} entries)")
    print(f"Log file: {log_path}")
    print(f"{'='*60}")
    header = f"{'Timestamp':<22} {'Strategy':<6} {'Ticker':<12} {'Decision':<9} {'Improv%':>7}  Reason"
    print(header)
    print("-" * len(header))
    for raw in entries:
        try:
            e = json.loads(raw)
        except json.JSONDecodeError:
            continue
        decision = "PROMOTED" if e.get("promoted") else ("WOULD" if e.get("should_promote") else "KEPT")
        improv = e.get("improvement_pct", 0.0)
        print(
            f"{e.get('timestamp',''):<22} "
            f"{e.get('strategy',''):<6} "
            f"{e.get('ticker',''):<12} "
            f"{decision:<9} "
            f"{improv:>7.1%}  "
            f"{e.get('reason','')}"
        )
    print(f"{'='*60}\n")


def _load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    try:
        import yaml
    except ImportError:
        # Fallback: simple YAML-ish loading
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and promote model candidates to champion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="e1",
        help="Strategy key in registry (default: e1)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated tickers. If empty, evaluates all tickers with pending candidates.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/base.yaml",
        help="Path to config YAML (default: src/config/base.yaml)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually write promotions to registry. Without this flag, operates in dry-run mode.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force-promote all candidates regardless of score comparison.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed per-metric breakdown for each ticker.",
    )
    parser.add_argument(
        "--show-log",
        metavar="N",
        type=int,
        nargs="?",
        const=20,
        default=None,
        help="Print the last N entries from the promotion audit log (default N=20) and exit.",
    )
    args = parser.parse_args()

    # --show-log: display audit log and exit immediately
    if args.show_log is not None:
        log_path = _PROJECT_ROOT / "models" / "promotion_log.jsonl"
        _show_log(log_path, args.show_log)
        sys.exit(0)

    dry_run = not args.execute

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _PROJECT_ROOT / config_path
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)

    config = _load_config(config_path)
    promo_cfg = config.get("lifecycle", {}).get("promotion", {})
    registry_path = _PROJECT_ROOT / config.get("lifecycle", {}).get(
        "registry_path", "models/registry.json"
    )

    if not registry_path.exists():
        print(f"❌ Registry not found: {registry_path}")
        sys.exit(1)

    registry = ModelRegistry(registry_path)

    # Determine tickers
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        # Find all tickers with a pending candidate
        all_candidates = registry.list_all(strategy=args.strategy, stage="candidate")
        tickers = [c["ticker"] for c in all_candidates]

    if not tickers:
        print(f"No pending candidates found for strategy '{args.strategy}'.")
        sys.exit(0)

    # Header
    resolved_cfg = get_strategy_promotion_config(promo_cfg, args.strategy)
    mode = "DRY-RUN" if dry_run else "EXECUTE"
    force_label = " (FORCE)" if args.force else ""
    print(f"\n{'='*60}")
    print(f"Model Promotion — {mode}{force_label}")
    print(f"Strategy: {args.strategy} | Tickers: {len(tickers)}")
    print(f"Min improvement: {resolved_cfg.get('min_improvement', 0.05):.0%}")
    weights = resolved_cfg.get("scoring_weights", {})
    if weights:
        print(f"Scoring weights: {weights}")
    print(f"{'='*60}\n")

    # Run evaluation
    log_path = _PROJECT_ROOT / "models" / "promotion_log.jsonl"
    decisions = evaluate_and_promote_all(
        registry,
        strategy=args.strategy,
        tickers=tickers,
        config=promo_cfg,
        dry_run=dry_run,
        force=args.force,
        log_path=log_path,
    )

    # Print summary table
    print_promotion_summary(decisions, strategy=args.strategy)

    # Verbose detail
    if args.verbose:
        print(f"\n{'='*60}")
        print("Detailed breakdown per ticker:")
        print(f"{'='*60}")
        for ticker, d in decisions.items():
            print(f"\n  {ticker}:")
            print(f"    Champion variant: {d.champion_variant or '-'}")
            print(f"    Candidate variant: {d.candidate_variant or '-'}")
            print(f"    Candidate score: {d.candidate_score:.6f}")
            print(f"    Champion score:  {d.champion_score:.6f}")
            print(f"    Improvement:     {d.improvement_pct:.2%}")
            if d.detail:
                print("    Per-metric contributions:")
                for k, v in sorted(d.detail.items()):
                    print(f"      {k}: {v:.6f}")

    # Summary line
    promoted = sum(1 for d in decisions.values() if d.promoted)
    if dry_run and any(d.should_promote for d in decisions.values()):
        would_promote = sum(1 for d in decisions.values() if d.should_promote)
        print(f"\n💡 {would_promote} ticker(s) would be promoted. Run with --execute to apply.")
    elif promoted > 0:
        print(f"\n✓ {promoted} ticker(s) promoted to champion.")


if __name__ == "__main__":
    main()
