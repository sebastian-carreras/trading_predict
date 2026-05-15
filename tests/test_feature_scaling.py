"""Tests for z-score feature scaling: fit on train only, apply to val/test.

The z-score logic is inline in each pipeline (not a separate class), so we
replicate the exact pattern used there and verify the invariant: mean/std come
exclusively from the training fold.
"""

from __future__ import annotations

import numpy as np
import pytest


def zscore_like_pipeline(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replica of the z-score logic used in src/e{1,2,3}/train_pipeline.py."""
    X_tr2d = X_train.reshape(-1, X_train.shape[-1])
    mean_X = X_tr2d.mean(axis=0)
    std_X = X_tr2d.std(axis=0) + 1e-12

    X_train_s = ((X_train - mean_X) / std_X).astype(np.float32)
    X_val_s = ((X_val - mean_X) / std_X).astype(np.float32)
    X_test_s = ((X_test - mean_X) / std_X).astype(np.float32)
    return X_train_s, X_val_s, X_test_s, mean_X, std_X


class TestZScoreNoLeakage:
    def test_mean_std_from_train_only(self) -> None:
        """Stats computed from train fold must ignore val/test distribution."""
        rng = np.random.default_rng(42)
        n_train, n_val, n_test, seq_len, n_feat = 100, 20, 20, 10, 5

        X_train = rng.normal(loc=0, scale=1, size=(n_train, seq_len, n_feat)).astype(np.float32)
        # Val/test have a very different distribution (large shift)
        X_val = rng.normal(loc=100, scale=1, size=(n_val, seq_len, n_feat)).astype(np.float32)
        X_test = rng.normal(loc=200, scale=1, size=(n_test, seq_len, n_feat)).astype(np.float32)

        _, _, _, mean_X, std_X = zscore_like_pipeline(X_train, X_val, X_test)

        # mean_X must be close to 0 (train distribution), not 100 or 200
        assert np.all(np.abs(mean_X) < 1.0), f"mean_X={mean_X} — looks like val/test leaked in"
        # std_X must be close to 1
        assert np.all(np.abs(std_X - 1.0) < 0.3), f"std_X={std_X}"

    def test_scaled_train_is_approximately_standard_normal(self) -> None:
        """After scaling with train stats, train set has mean≈0, std≈1."""
        rng = np.random.default_rng(0)
        X_train = rng.normal(loc=5, scale=3, size=(200, 15, 8)).astype(np.float32)
        X_val = rng.normal(loc=5, scale=3, size=(40, 15, 8)).astype(np.float32)
        X_test = rng.normal(loc=5, scale=3, size=(40, 15, 8)).astype(np.float32)

        X_train_s, _, _, _, _ = zscore_like_pipeline(X_train, X_val, X_test)

        flat = X_train_s.reshape(-1, X_train_s.shape[-1])
        assert np.allclose(flat.mean(axis=0), 0, atol=0.05), "Scaled train mean should be ~0"
        assert np.allclose(flat.std(axis=0), 1, atol=0.1), "Scaled train std should be ~1"

    def test_val_test_shifted_when_different_distribution(self) -> None:
        """Val/test scaled with train stats should not be centered if distributions differ."""
        rng = np.random.default_rng(7)
        X_train = rng.normal(loc=0, scale=1, size=(100, 10, 4)).astype(np.float32)
        X_val = rng.normal(loc=10, scale=1, size=(20, 10, 4)).astype(np.float32)
        X_test = rng.normal(loc=10, scale=1, size=(20, 10, 4)).astype(np.float32)

        _, X_val_s, X_test_s, _, _ = zscore_like_pipeline(X_train, X_val, X_test)

        # Scaled val should have mean far from 0 (train centered but val is shifted)
        val_mean = X_val_s.reshape(-1, X_val_s.shape[-1]).mean(axis=0)
        assert np.all(np.abs(val_mean) > 5), (
            "Val should be far from 0 when its distribution differs from train"
        )

    def test_no_mutation_of_input_arrays(self) -> None:
        """Scaling must not modify the original arrays in place."""
        rng = np.random.default_rng(1)
        X_train = rng.standard_normal((50, 5, 3)).astype(np.float32)
        X_val = rng.standard_normal((10, 5, 3)).astype(np.float32)
        X_test = rng.standard_normal((10, 5, 3)).astype(np.float32)

        orig_train = X_train.copy()
        orig_val = X_val.copy()
        zscore_like_pipeline(X_train, X_val, X_test)

        np.testing.assert_array_equal(X_train, orig_train)
        np.testing.assert_array_equal(X_val, orig_val)

    def test_std_epsilon_prevents_division_by_zero(self) -> None:
        """Constant features (std=0) must not produce NaN after scaling."""
        X_train = np.ones((50, 5, 3), dtype=np.float32)
        X_val = np.ones((10, 5, 3), dtype=np.float32)
        X_test = np.ones((10, 5, 3), dtype=np.float32)

        X_train_s, X_val_s, X_test_s, _, _ = zscore_like_pipeline(X_train, X_val, X_test)

        assert np.isfinite(X_train_s).all(), "NaN/Inf from zero-std feature"
        assert np.isfinite(X_val_s).all()
        assert np.isfinite(X_test_s).all()
