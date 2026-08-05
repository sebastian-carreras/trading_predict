"""Tests for the declarative model registry (src.models.registry).

Covers: register/lookup, lazy resolution of the existing models, loud failure on
unknown names, and the invariant that importing the registry does NOT import
torch (so the FastAPI serving layer can import it without the heavy dep).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.models.registry import (
    MODEL_REGISTRY,
    get_model_class,
    register_model,
    registered_names,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_registered_names_include_existing_models():
    names = registered_names()
    assert "GRURegressor" in names
    assert "LSTMRegressor" in names


def test_get_model_class_resolves_all_registered_names():
    # Every resolvable name must actually import + return a class.
    for name in registered_names():
        cls = get_model_class(name)
        assert isinstance(cls, type)
        # Contract: reconstructable from a payload.
        assert hasattr(cls, "from_payload")


def test_get_model_class_lazy_populates_registry():
    # After a lookup, the class is cached in MODEL_REGISTRY.
    get_model_class("GRURegressor")
    assert MODEL_REGISTRY.get("GRURegressor") is not None


def test_unknown_model_class_raises_valueerror():
    with pytest.raises(ValueError) as exc:
        get_model_class("NoSuchModel_xyz")
    # Error is actionable: lists the known names.
    assert "Registered models" in str(exc.value)


def test_register_model_rejects_name_collision():
    @register_model("dummy_collision_model")
    class _A:  # noqa: N801
        pass

    # Same name, different class → collision guard fires.
    with pytest.raises(ValueError):
        @register_model("dummy_collision_model")
        class _B:  # noqa: N801
            pass

    # Re-registering the SAME class under the SAME name is a no-op.
    register_model("dummy_collision_model")(_A)
    MODEL_REGISTRY.pop("dummy_collision_model", None)


def test_registry_import_is_torch_free():
    """Importing the registry must not pull torch (serving-layer invariant)."""
    code = "import sys, src.models.registry; print('torch' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", out.stdout
