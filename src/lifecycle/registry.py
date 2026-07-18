"""Centralized model registry backed by a single JSON file.

All lifecycle state lives in ``models/registry.json``.  Writes are atomic
(write to a temp file, then ``os.replace``) so the file is never left in a
partially-written state.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_EMPTY_REGISTRY: dict[str, Any] = {
    "version": 1,
    "last_updated": None,
    "strategies": {},
}


class ModelRegistry:
    """CRUD operations over ``models/registry.json``."""

    def __init__(self, registry_path: str | Path) -> None:
        self.path = Path(registry_path)
        self.data: dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_candidate(
        self,
        strategy: str,
        ticker: str,
        run_dir: str,
        metrics: dict[str, Any],
        variant: str,
        *,
        mlflow_run_id: str | None = None,
        feature_names: list[str] | None = None,
        hyperparams: dict[str, Any] | None = None,
        train_data_end: str | None = None,
    ) -> None:
        """Register a freshly-trained model as *candidate*.

        ``train_data_end`` is the last date of the training data window (ISO
        string). It is the cutoff used by the fair-comparison logic to build a
        common out-of-sample window vs. the champion, so it must follow the
        model through promotion (``promote_to_champion`` copies the whole dict).
        """
        tickers = self._ensure_strategy_tickers(strategy)
        entry = tickers.setdefault(ticker, _empty_ticker())
        candidate_entry: dict[str, Any] = {
            "variant": variant,
            "run_dir": str(run_dir),
            "registered_at": _now_iso(),
            "mlflow_run_id": mlflow_run_id,
            "metrics": _clean_metrics(metrics),
        }
        if train_data_end is not None:
            candidate_entry["train_data_end"] = str(train_data_end)
        if feature_names is not None:
            candidate_entry["features"] = list(feature_names)
        if hyperparams is not None:
            candidate_entry["hyperparams"] = hyperparams
        entry["candidate"] = candidate_entry
        self._save()

    def register_baseline(
        self,
        strategy: str,
        ticker: str,
        run_dir: str,
        metrics: dict[str, Any],
        variant: str,
        *,
        feature_names: list[str] | None = None,
    ) -> None:
        """Register or overwrite the fixed *baseline* for a ticker."""
        tickers = self._ensure_strategy_tickers(strategy)
        entry = tickers.setdefault(ticker, _empty_ticker())
        baseline_entry: dict[str, Any] = {
            "variant": variant,
            "run_dir": str(run_dir),
            "registered_at": _now_iso(),
            "metrics": _clean_metrics(metrics),
        }
        if feature_names is not None:
            baseline_entry["features"] = list(feature_names)
        entry["baseline"] = baseline_entry
        self._save()

    def promote_to_champion(
        self,
        strategy: str,
        ticker: str,
        *,
        reason: str = "manual_promotion",
    ) -> dict[str, Any] | None:
        """Promote current *candidate* to *champion*.

        The previous champion (if any) is moved to the *retired* list.
        Returns the newly promoted champion entry, or ``None`` if there is
        no candidate.
        """
        tickers = self._ensure_strategy_tickers(strategy)
        entry = tickers.get(ticker)
        if entry is None or entry.get("candidate") is None:
            return None

        # Retire current champion
        current_champion = entry.get("champion")
        if current_champion is not None:
            current_champion["retired_at"] = _now_iso()
            current_champion["reason"] = reason
            entry.setdefault("retired", []).insert(0, current_champion)
            # Enforce keep_last_n
            max_retired = 5
            entry["retired"] = entry["retired"][:max_retired]

        # Promote candidate -> champion
        candidate = entry.pop("candidate")
        candidate["promoted_at"] = _now_iso()
        entry["champion"] = candidate
        entry["candidate"] = None

        self._save()
        return candidate

    def get_champion(self, strategy: str, ticker: str) -> dict[str, Any] | None:
        """Return the champion entry for a strategy/ticker, or ``None``."""
        tickers = self._get_strategy_tickers(strategy)
        if tickers is None:
            return None
        entry = tickers.get(ticker)
        if entry is None:
            return None
        return entry.get("champion")

    def get_baseline(self, strategy: str, ticker: str) -> dict[str, Any] | None:
        """Return the baseline entry for a strategy/ticker, or ``None``."""
        tickers = self._get_strategy_tickers(strategy)
        if tickers is None:
            return None
        entry = tickers.get(ticker)
        if entry is None:
            return None
        return entry.get("baseline")

    def get_candidate(self, strategy: str, ticker: str) -> dict[str, Any] | None:
        """Return the current candidate entry, or ``None``."""
        tickers = self._get_strategy_tickers(strategy)
        if tickers is None:
            return None
        entry = tickers.get(ticker)
        if entry is None:
            return None
        return entry.get("candidate")

    def get_model_features(
        self,
        strategy: str,
        ticker: str,
        stage: str = "champion",
    ) -> list[str] | None:
        """Return the feature names for a given model stage, or ``None`` if not tracked."""
        tickers = self._get_strategy_tickers(strategy)
        if tickers is None:
            return None
        entry = tickers.get(ticker)
        if entry is None:
            return None
        if stage == "retired":
            retired = entry.get("retired", [])
            return retired[0].get("features") if retired else None
        model = entry.get(stage)
        if model is None:
            return None
        return model.get("features")

    def retire_champion(
        self,
        strategy: str,
        ticker: str,
        reason: str = "manual_retirement",
    ) -> bool:
        """Move current champion to retired. Returns True if retired."""
        tickers = self._get_strategy_tickers(strategy)
        if tickers is None:
            return False
        entry = tickers.get(ticker)
        if entry is None or entry.get("champion") is None:
            return False

        champion = entry.pop("champion")
        champion["retired_at"] = _now_iso()
        champion["reason"] = reason
        entry.setdefault("retired", []).insert(0, champion)
        entry["champion"] = None
        self._save()
        return True

    def restore_champion_from_retired(
        self,
        strategy: str,
        ticker: str,
        *,
        index: int = 0,
        reason: str = "rollback_restore",
    ) -> dict[str, Any] | None:
        """Restore a retired model back to champion (undo a bad promotion).

        Pops ``retired[index]`` and installs it as champion. The current
        champion (if any) is pushed to the front of the retired list, stamped
        with ``retired_at``/``reason``. The restored model's own retirement
        bookkeeping (``retired_at``/``reason``) is cleared.

        Returns the restored entry, or ``None`` if there is no retired model at
        ``index`` (or no such strategy/ticker).
        """
        tickers = self._get_strategy_tickers(strategy)
        if tickers is None:
            return None
        entry = tickers.get(ticker)
        if entry is None:
            return None
        retired = entry.get("retired", [])
        if not retired or index >= len(retired):
            return None

        restored = retired.pop(index)
        restored.pop("retired_at", None)
        restored.pop("reason", None)

        current = entry.get("champion")
        if current is not None:
            current["retired_at"] = _now_iso()
            current["reason"] = reason
            retired.insert(0, current)

        entry["retired"] = retired[:5]  # enforce keep_last_n
        entry["champion"] = restored
        self._save()
        return restored

    def set_recent_metrics(
        self,
        strategy: str,
        ticker: str,
        metrics: dict[str, Any],
        *,
        asof: str | None = None,
        window_start: str | None = None,
        window_end: str | None = None,
        n_samples: int | None = None,
    ) -> bool:
        """Attach freshly re-backtested metrics to the current champion.

        These are the champion's metrics recomputed on a recent out-of-sample
        window (see ``lifecycle.reevaluation``). The original training-time
        ``metrics`` are left untouched; consumers (e.g. the leaderboard) can
        prefer ``recent_metrics`` when present to reflect *current* quality.

        Returns True if a champion existed and was updated.
        """
        tickers = self._get_strategy_tickers(strategy)
        if tickers is None:
            return False
        entry = tickers.get(ticker)
        if entry is None or entry.get("champion") is None:
            return False
        champion = entry["champion"]
        champion["recent_metrics"] = _clean_metrics(metrics)
        champion["recent_metrics_asof"] = asof or _now_iso()
        if window_start is not None:
            champion["recent_metrics_window_start"] = str(window_start)
        if window_end is not None:
            champion["recent_metrics_window_end"] = str(window_end)
        if n_samples is not None:
            champion["recent_metrics_n_samples"] = int(n_samples)
        self._save()
        return True

    def set_train_data_end(
        self,
        strategy: str,
        ticker: str,
        train_data_end: str,
    ) -> bool:
        """Backfill ``train_data_end`` on the current champion.

        For champions registered before this field was tracked. Enables the
        fair-window promotion path (``promotion.compare_on_common_window``),
        which otherwise falls back to the legacy stored-vs-stored comparison
        when ``train_data_end`` is missing.

        Returns True if a champion existed and was updated.
        """
        tickers = self._get_strategy_tickers(strategy)
        if tickers is None:
            return False
        entry = tickers.get(ticker)
        if entry is None or entry.get("champion") is None:
            return False
        entry["champion"]["train_data_end"] = str(train_data_end)
        self._save()
        return True

    def list_all(
        self,
        strategy: str | None = None,
        stage: str | None = None,
    ) -> list[dict[str, Any]]:
        """List all models, optionally filtered by strategy and/or stage."""
        results: list[dict[str, Any]] = []
        strategies = self.data.get("strategies", {})

        for strat_name, strat_data in strategies.items():
            if strategy is not None and strat_name != strategy:
                continue
            tickers = strat_data.get("tickers", {})
            for ticker_name, entry in tickers.items():
                for s in ("baseline", "candidate", "champion"):
                    if stage is not None and s != stage:
                        continue
                    model = entry.get(s)
                    if model is not None:
                        results.append({
                            "strategy": strat_name,
                            "ticker": ticker_name,
                            "stage": s,
                            **model,
                        })
                if stage is None or stage == "retired":
                    for retired_model in entry.get("retired", []):
                        results.append({
                            "strategy": strat_name,
                            "ticker": ticker_name,
                            "stage": "retired",
                            **retired_model,
                        })
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return json.loads(json.dumps(_EMPTY_REGISTRY))
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self) -> None:
        self.data["last_updated"] = _now_iso()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file in same directory, then rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.path.parent),
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False, default=str)
                f.write("\n")
            os.replace(tmp_path, str(self.path))
        except BaseException:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_strategy_tickers(self, strategy: str) -> dict[str, Any]:
        strategies = self.data.setdefault("strategies", {})
        strat = strategies.setdefault(strategy, {})
        return strat.setdefault("tickers", {})

    def _get_strategy_tickers(self, strategy: str) -> dict[str, Any] | None:
        return self.data.get("strategies", {}).get(strategy, {}).get("tickers")


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _empty_ticker() -> dict[str, Any]:
    return {
        "baseline": None,
        "candidate": None,
        "champion": None,
        "retired": [],
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep only numeric metrics, round floats to 6 decimals."""
    cleaned: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            try:
                if not (value != value):  # skip NaN
                    cleaned[key] = round(float(value), 6)
            except (TypeError, ValueError):
                pass
    return cleaned
