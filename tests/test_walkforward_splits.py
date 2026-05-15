"""Tests for walk-forward split properties used in E1/E2/E3 training.

Uses sklearn.model_selection.TimeSeriesSplit with gap (embargo), which is the
exact mechanism used in src/e1/train_pipeline.py:543 and analogous pipelines.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import TimeSeriesSplit


def make_splits(
    n_samples: int,
    folds: int,
    embargo_days: int,
    test_size: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Replicate the split creation from train_pipeline.py."""
    if test_size is None:
        test_size = max(1, n_samples // (folds + 1))
    gap_samples = max(0, embargo_days)
    if test_size <= gap_samples:
        test_size = gap_samples + 1
    splitter = TimeSeriesSplit(n_splits=folds, test_size=test_size, gap=gap_samples)
    indices = np.arange(n_samples)
    return list(splitter.split(indices))


class TestWalkForwardSplitCount:
    def test_correct_number_of_folds_e1(self) -> None:
        splits = make_splits(n_samples=1000, folds=5, embargo_days=90)
        assert len(splits) == 5

    def test_correct_number_of_folds_e2(self) -> None:
        splits = make_splits(n_samples=500, folds=5, embargo_days=20)
        assert len(splits) == 5

    def test_minimum_two_folds_required_by_sklearn(self) -> None:
        """TimeSeriesSplit requires n_splits >= 2; enforce this at the API level."""
        with pytest.raises(ValueError, match="n_splits"):
            make_splits(n_samples=300, folds=1, embargo_days=30)


class TestNoTrainTestOverlap:
    def test_train_test_disjoint_e1(self) -> None:
        splits = make_splits(n_samples=1000, folds=5, embargo_days=90)
        for train_idx, test_idx in splits:
            assert len(np.intersect1d(train_idx, test_idx)) == 0, (
                "Train and test sets must be disjoint"
            )

    def test_train_test_disjoint_e2(self) -> None:
        splits = make_splits(n_samples=500, folds=5, embargo_days=20)
        for train_idx, test_idx in splits:
            assert len(np.intersect1d(train_idx, test_idx)) == 0


class TestTemporalOrder:
    def test_train_always_before_test(self) -> None:
        """All train indices must be strictly earlier than all test indices."""
        splits = make_splits(n_samples=1000, folds=5, embargo_days=90)
        for train_idx, test_idx in splits:
            assert train_idx.max() < test_idx.min(), (
                "Train must end before test begins (temporal ordering)"
            )

    def test_successive_folds_grow_train_size(self) -> None:
        """Each fold's train set must be larger than the previous fold's."""
        splits = make_splits(n_samples=1000, folds=5, embargo_days=90)
        train_sizes = [len(tr) for tr, _ in splits]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1], (
                "Walk-forward: each fold must have at least as many train samples as the previous"
            )


class TestEmbargo:
    def test_embargo_gap_e1_90_days(self) -> None:
        """With gap=90, the gap between last train and first test must be >= 90."""
        embargo = 90
        splits = make_splits(n_samples=1500, folds=5, embargo_days=embargo)
        for train_idx, test_idx in splits:
            actual_gap = test_idx.min() - train_idx.max() - 1
            assert actual_gap >= embargo, (
                f"Embargo gap {actual_gap} < expected {embargo}"
            )

    def test_embargo_gap_e2_20_days(self) -> None:
        embargo = 20
        splits = make_splits(n_samples=600, folds=5, embargo_days=embargo)
        for train_idx, test_idx in splits:
            actual_gap = test_idx.min() - train_idx.max() - 1
            assert actual_gap >= embargo, (
                f"Embargo gap {actual_gap} < expected {embargo}"
            )

    def test_zero_embargo_still_disjoint(self) -> None:
        splits = make_splits(n_samples=300, folds=3, embargo_days=0)
        for train_idx, test_idx in splits:
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_test_size_adjusted_when_embargo_large(self) -> None:
        """When test_size <= embargo, pipeline forces test_size = embargo + 1."""
        # This mirrors the guard in train_pipeline.py:539-540
        embargo = 90
        small_test = 50  # < embargo
        adjusted_test = embargo + 1  # expected adjustment
        splits = make_splits(n_samples=1000, folds=3, embargo_days=embargo, test_size=small_test)
        # All test sets should have size == adjusted_test
        for _, test_idx in splits:
            assert len(test_idx) == adjusted_test, (
                f"Test size should be {adjusted_test} when original ({small_test}) <= embargo ({embargo})"
            )
