"""Tests for the model payload round-trip (serialization contract).

A trained model must reconstruct byte-for-byte from its ``.pth`` payload — this
is what fair-window promotion and serving rely on. We verify:
* ``to_payload`` + ``from_payload`` reproduce predictions exactly (GRU and LSTM).
* ``verify_model_roundtrip`` passes for a faithful payload and RAISES for a
  corrupted one (guarding the "trains but can't be rebuilt" failure mode).
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.registry import get_model_class  # noqa: E402
from src.models.verify import verify_model_roundtrip  # noqa: E402


def _tiny_payload(model_class: str, kwargs_key: str):
    """Build a small model + a full payload dict (metadata + weights)."""
    cls = get_model_class(model_class)
    model_kwargs = {
        "input_size": 4,
        "hidden_sizes": [8, 4],
        "dropout": 0.1,
        "dense_units": 4,
        "seed": 7,
    }
    model = cls(**model_kwargs)
    payload = {
        "model_class": model_class,
        "model_kwargs": model_kwargs,
        **model.to_payload(),  # {"state_dict": ...}
    }
    return model, payload


@pytest.mark.parametrize("model_class", ["GRURegressor", "LSTMRegressor"])
def test_payload_roundtrip_reproduces_predictions(model_class):
    model, payload = _tiny_payload(model_class, model_class)
    X = np.random.default_rng(0).standard_normal((16, 5, 4)).astype(np.float32)

    live = model.predict(X)
    rebuilt = get_model_class(model_class).from_payload(payload)
    got = rebuilt.predict(X)

    assert got.shape == live.shape == (16,)
    assert np.allclose(got, live, atol=1e-6), np.max(np.abs(got - live))


def test_verify_model_roundtrip_passes_on_good_payload():
    model, payload = _tiny_payload("GRURegressor", "GRURegressor")
    X = np.random.default_rng(1).standard_normal((8, 5, 4)).astype(np.float32)
    # Should not raise.
    verify_model_roundtrip(model, payload, X)


def test_verify_model_roundtrip_raises_on_corrupted_payload():
    model, payload = _tiny_payload("GRURegressor", "GRURegressor")
    X = np.random.default_rng(2).standard_normal((8, 5, 4)).astype(np.float32)

    # Corrupt the architecture recipe so the rebuilt model differs from the live one.
    bad = dict(payload)
    bad["model_kwargs"] = dict(payload["model_kwargs"], hidden_sizes=[16, 8])

    with pytest.raises((RuntimeError, Exception)):
        verify_model_roundtrip(model, bad, X)
