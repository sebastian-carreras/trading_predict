"""Declarative model registry: ``name -> class`` with lazy imports.

Single source of truth for instantiating and reconstructing models by their
string name (the ``model_class`` stored in every ``<TICKER>_model.pth`` payload).

Why a registry
--------------
Historically the mapping lived in two hard-coded places: the per-strategy
``import`` in each ``train_pipeline`` and the string→class dict in
``lifecycle.reevaluation._strategy_kit``. Adding a model meant editing both, and
forgetting the second one failed **late and silently** (at fair-window promotion
or serving, not at train time). With this registry, a model registers itself
once (via ``@register_model``) and is resolvable everywhere.

Lazy by design
--------------
Importing this module does **not** import torch or any heavy model module.
Model modules are imported on first lookup through ``_LAZY_TARGETS`` (which runs
their ``@register_model`` decorator), so lightweight consumers — e.g. the
FastAPI serving layer that only reads stored predictions — can import the
registry without pulling torch.
"""
from __future__ import annotations

import importlib
from typing import Callable

# name -> class, populated by @register_model when the owning module is imported.
MODEL_REGISTRY: dict[str, type] = {}

# name -> dotted module path whose import registers ``name``. Lets
# ``get_model_class`` trigger the import on demand (and thus run the decorator)
# without this module importing the heavy model code eagerly.
_LAZY_TARGETS: dict[str, str] = {
    "GRURegressor": "src.e1.gru",
    "LSTMRegressor": "src.e2.lstm",
}


def register_model(name: str) -> Callable[[type], type]:
    """Class decorator that registers ``cls`` under ``name``.

    Re-registering the same class under the same name is a no-op (import can run
    more than once); registering a *different* class under a taken name raises,
    to catch accidental name collisions early.
    """
    def _decorator(cls: type) -> type:
        existing = MODEL_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Model name {name!r} already registered to {existing!r}; "
                f"cannot re-register to {cls!r}."
            )
        MODEL_REGISTRY[name] = cls
        return cls

    return _decorator


def get_model_class(name: str) -> type:
    """Return the model class registered under ``name``.

    Triggers a lazy import of the owning module when the name is known but not
    yet imported. Raises ``ValueError`` (listing the known names) when the name
    cannot be resolved — a loud, early failure instead of a silent fallback.
    """
    cls = MODEL_REGISTRY.get(name)
    if cls is not None:
        return cls

    target = _LAZY_TARGETS.get(name)
    if target is not None:
        importlib.import_module(target)  # runs @register_model at import time
        cls = MODEL_REGISTRY.get(name)
        if cls is not None:
            return cls

    raise ValueError(
        f"Unknown model_class={name!r}. Registered models: {registered_names()}. "
        "New models must be decorated with @register_model and (for lazy "
        "resolution) added to src.models.registry._LAZY_TARGETS."
    )


def registered_names() -> list[str]:
    """All resolvable model names (imported + lazily importable)."""
    return sorted(set(MODEL_REGISTRY) | set(_LAZY_TARGETS))
