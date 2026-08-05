"""Round-trip guard: rebuild a just-saved model and check it still predicts the same.

Cheap defense against the "trains and registers, but cannot be reconstructed"
failure mode — a payload whose ``model_kwargs`` don't fully rebuild the class, a
missing ``state_dict`` key, or a ``model_class`` string that isn't registered.
Without this, such a model surfaces its failure **late and silently**, at
fair-window promotion or serving. Enabled at train time via
``VERIFY_MODEL_ROUNDTRIP=1`` so it fails loudly where it's cheap to fix.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np


def roundtrip_enabled() -> bool:
    """True when ``VERIFY_MODEL_ROUNDTRIP`` is set to a truthy value."""
    return os.getenv("VERIFY_MODEL_ROUNDTRIP", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _rebuild(payload: dict[str, Any]):
    from .registry import get_model_class

    cls = get_model_class(payload["model_class"])  # raises if unregistered
    if hasattr(cls, "from_payload"):
        return cls.from_payload(payload)
    # Fallback for classes without the torch mixin (older/foreign contracts).
    model = cls(**payload["model_kwargs"])
    model.model.load_state_dict(payload["state_dict"])
    model.model.eval()
    return model


def verify_model_roundtrip(
    model: Any,
    payload: dict[str, Any],
    X_sample: np.ndarray,
    *,
    atol: float = 1e-5,
) -> None:
    """Rebuild ``model`` from ``payload`` and assert it predicts like the live model.

    Compares the live (in-memory) model's predictions on ``X_sample`` against the
    predictions of the model reconstructed purely from the on-disk ``payload``.
    Raises ``RuntimeError`` on mismatch or reconstruction failure. No-op on an
    empty sample.
    """
    Xs = np.asarray(X_sample)
    if Xs.shape[0] == 0:
        return

    rebuilt = _rebuild(payload)
    live = np.asarray(model.predict(Xs), dtype=np.float64).ravel()
    got = np.asarray(rebuilt.predict(Xs), dtype=np.float64).ravel()

    n = min(len(live), len(got))
    if n == 0 or not np.allclose(got[:n], live[:n], atol=atol):
        diff = float(np.max(np.abs(got[:n] - live[:n]))) if n else float("nan")
        raise RuntimeError(
            f"Model round-trip verification failed for model_class="
            f"{payload.get('model_class')!r}: predictions from the reconstructed "
            f"model differ from the live model (max abs diff={diff:.3g}, "
            f"atol={atol}). The saved payload cannot be faithfully rebuilt, which "
            "would break fair-window promotion and serving. Set "
            "VERIFY_MODEL_ROUNDTRIP=0 to bypass."
        )
