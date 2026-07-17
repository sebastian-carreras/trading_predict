"""Load champion models from the lifecycle registry.

Replaces ad-hoc model loading with a registry-aware loader that
always loads the current champion for a given strategy/ticker.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .registry import ModelRegistry


class ModelLoader:
    """Load models using the lifecycle registry as source of truth."""

    def __init__(self, registry: ModelRegistry, project_root: Path | None = None) -> None:
        self.registry = registry
        if project_root is None:
            from ..utils import project_root as _project_root
            project_root = _project_root()
        self.root = project_root

    def load_champion(
        self,
        strategy: str,
        ticker: str,
    ) -> dict[str, Any] | None:
        """Load the champion model payload for a strategy/ticker.

        Returns the full torch payload dict (state_dict, model_kwargs, etc.)
        or None if no champion exists.
        """
        return self._load_stage(strategy, ticker, "champion")

    def load_baseline(
        self,
        strategy: str,
        ticker: str,
    ) -> dict[str, Any] | None:
        """Load the baseline model payload for a strategy/ticker."""
        return self._load_stage(strategy, ticker, "baseline")

    def get_champion_path(
        self,
        strategy: str,
        ticker: str,
    ) -> Path | None:
        """Get the filesystem path to the champion model file."""
        return self._get_model_path(strategy, ticker, "champion")

    def _load_stage(
        self,
        strategy: str,
        ticker: str,
        stage: str,
    ) -> dict[str, Any] | None:
        model_path = self._get_model_path(strategy, ticker, stage)
        if model_path is None or not model_path.exists():
            return None
        import torch  # Lazy: entornos sin torch (ej. FastAPI) solo leen predicciones guardadas.

        return torch.load(model_path, map_location="cpu", weights_only=False)

    def _get_model_path(
        self,
        strategy: str,
        ticker: str,
        stage: str,
    ) -> Path | None:
        if stage == "champion":
            entry = self.registry.get_champion(strategy, ticker)
        elif stage == "baseline":
            entry = self.registry.get_baseline(strategy, ticker)
        elif stage == "candidate":
            entry = self.registry.get_candidate(strategy, ticker)
        else:
            return None

        if entry is None:
            return None

        run_dir = Path(entry["run_dir"])
        if not run_dir.is_absolute():
            run_dir = self.root / run_dir

        # Find model file in run_dir
        model_files = list(run_dir.glob(f"{ticker}*_model.pth")) + list(
            run_dir.glob(f"{ticker}*model.pth")
        )
        if model_files:
            return model_files[0]
        return None
