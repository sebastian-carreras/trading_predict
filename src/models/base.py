"""Model contract + serialization mixin for the ``trading_predict`` registry.

Every model the training pipeline / lifecycle consumes satisfies the
``BaseRegressor`` contract below: a duck-typed ``fit``/``predict`` plus a payload
round-trip (``to_payload`` / ``from_payload``) so a trained model can be frozen
to disk and reconstructed **byte-for-byte** for fair-window promotion and
serving (see ``lifecycle.reevaluation._rebuild_model``).

Two serialization paths
------------------------
* **Torch models** mix in :class:`TorchRegressorMixin`. The learned weights
  travel as a ``state_dict`` inside the ``.pth`` payload — the exact format the
  existing GRU/LSTM models already used, so old artifacts keep loading.

* **Non-torch models** (e.g. a future sklearn / XGBoost adapter) override
  ``to_payload`` / ``from_payload`` themselves. They have no ``state_dict``; the
  idiomatic pattern is to serialize the fitted estimator to bytes and stash them
  in the payload, e.g.::

      import cloudpickle

      def to_payload(self) -> dict:
          return {"estimator_bytes": cloudpickle.dumps(self.estimator)}

      @classmethod
      def from_payload(cls, payload: dict) -> "TabularSklearnRegressor":
          obj = cls(**payload["model_kwargs"])
          obj.estimator = cloudpickle.loads(payload["estimator_bytes"])
          return obj

  This module intentionally ships **no** such adapter yet (scaffolding only) —
  it documents the seam so the day a classic-ML model is added, the lifecycle
  (registry, promotion, leaderboard, serving) needs no further change.

Note on config → kwargs
------------------------
``kwargs_from_config`` documents how a model reads its hyperparameters from a
``model:`` block in ``base.yaml``. The E1/E2 pipelines do **not** call it for the
existing models: they build ``model_kwargs`` from values already resolved
(including per-ticker Optuna overrides), so re-reading raw config there would
silently drop tuning. It exists as the contract for future, purely
config-driven models.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BaseRegressor(Protocol):
    """Structural contract for a model usable by the pipeline and lifecycle.

    A class need not inherit from this Protocol — duck typing is enough — but
    conforming to it is what lets the registry freeze and rebuild the model.
    """

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs: Any,
    ) -> Any:
        """Train on ``(X_train, y_train)``, early-stopping on the val split."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return a flat ``(n_samples,)`` array of predicted returns."""
        ...

    def to_payload(self) -> dict[str, Any]:
        """Model-specific serialization fragment (weights/estimator bytes).

        Merged by the pipeline with metadata (``model_class``, ``model_kwargs``,
        scalers, ``feature_names``, …) into the full ``.pth`` payload.
        """
        ...

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BaseRegressor":
        """Reconstruct a ready-to-``predict`` model from a full payload dict."""
        ...

    @classmethod
    def kwargs_from_config(
        cls, model_cfg: dict[str, Any], *, input_size: int, seed: int
    ) -> dict[str, Any]:
        """Build constructor kwargs from a ``model:`` config block."""
        ...


class TorchRegressorMixin:
    """Default torch serialization: weights as a ``state_dict``.

    Assumes the concrete class exposes a ``self.model`` ``torch.nn.Module`` and a
    constructor whose kwargs (``payload["model_kwargs"]``) fully rebuild that
    module's architecture. This reproduces the historical GRU/LSTM behavior, so
    ``.pth`` files written before this refactor reconstruct unchanged.
    """

    # Concrete torch models set these attributes in __init__.
    model: Any

    def to_payload(self) -> dict[str, Any]:
        """Serialization fragment: the module's ``state_dict`` (CPU tensors)."""
        return {
            "state_dict": {
                k: v.detach().cpu() for k, v in self.model.state_dict().items()
            }
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TorchRegressorMixin":
        """Rebuild the architecture from ``model_kwargs`` and load the weights."""
        model = cls(**payload["model_kwargs"])
        model.model.load_state_dict(payload["state_dict"])
        model.model.eval()
        return model
