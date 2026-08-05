"""Model abstraction layer: declarative registry + serialization contract.

Importing this package stays lightweight (no torch): only the registry helpers
are re-exported. Model classes and the ``base`` contract are imported lazily by
``get_model_class`` on first lookup. See ``registry`` and ``base`` for details.
"""
from __future__ import annotations

from .registry import (
    MODEL_REGISTRY,
    get_model_class,
    register_model,
    registered_names,
)

__all__ = [
    "MODEL_REGISTRY",
    "get_model_class",
    "register_model",
    "registered_names",
]
