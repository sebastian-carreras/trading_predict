"""Tests para el monitoreo de drift (PSI + KS) en src/lifecycle/drift.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.lifecycle.drift import (
    PSI_MODERATE,
    PSI_STABLE,
    drift_report,
    feature_drift,
    log_drift_report,
    population_stability_index,
    summarize,
)


class TestPSI:
    def test_identical_distribution_is_near_zero(self) -> None:
        """Misma distribución → PSI ~ 0 (por debajo del umbral estable)."""
        rng = np.random.default_rng(0)
        ref = rng.normal(size=5000)
        cur = rng.normal(size=5000)
        assert population_stability_index(ref, cur) < PSI_STABLE

    def test_large_shift_is_significant(self) -> None:
        """Shift grande de media → PSI por encima del umbral significativo."""
        rng = np.random.default_rng(1)
        ref = rng.normal(loc=0, scale=1, size=5000)
        cur = rng.normal(loc=3, scale=1, size=5000)
        assert population_stability_index(ref, cur) >= PSI_MODERATE

    def test_psi_monotonic_in_shift(self) -> None:
        """Mayor desplazamiento → mayor PSI."""
        rng = np.random.default_rng(2)
        ref = rng.normal(size=5000)
        small = population_stability_index(ref, rng.normal(loc=0.5, size=5000))
        big = population_stability_index(ref, rng.normal(loc=2.0, size=5000))
        assert big > small

    def test_constant_reference_does_not_crash(self) -> None:
        """Referencia constante → sin división por cero / NaN, PSI finito."""
        ref = np.ones(1000)
        cur = np.ones(1000)
        psi = population_stability_index(ref, cur)
        assert np.isfinite(psi)
        assert psi == 0.0

    def test_empty_input_returns_zero(self) -> None:
        assert population_stability_index([], [1.0, 2.0]) == 0.0
        assert population_stability_index([1.0, 2.0], []) == 0.0

    def test_nan_inf_are_ignored(self) -> None:
        """NaN/Inf no deben propagarse al PSI."""
        rng = np.random.default_rng(3)
        ref = rng.normal(size=2000)
        cur = np.concatenate([rng.normal(size=2000), [np.nan, np.inf, -np.inf]])
        psi = population_stability_index(ref, cur)
        assert np.isfinite(psi)
        assert psi < PSI_STABLE


class TestFeatureDrift:
    def test_no_drift_flagged_stable(self) -> None:
        rng = np.random.default_rng(10)
        fd = feature_drift("ret_1w", rng.normal(size=4000), rng.normal(size=4000))
        assert fd.severity == "estable"
        assert not fd.drifted
        assert fd.ks_pvalue > 0.05

    def test_shift_flagged_and_ks_significant(self) -> None:
        rng = np.random.default_rng(11)
        fd = feature_drift(
            "atr_14",
            rng.normal(loc=0, scale=1, size=4000),
            rng.normal(loc=2.5, scale=1, size=4000),
        )
        assert fd.severity == "significativo"
        assert fd.drifted
        assert fd.ks_pvalue < 0.05

    def test_counts_recorded(self) -> None:
        fd = feature_drift("x", np.zeros(30), np.ones(20))
        assert fd.n_ref == 30
        assert fd.n_cur == 20


class TestDriftReport:
    def _frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(20)
        n = 3000
        ref = pd.DataFrame(
            {
                "stable_feat": rng.normal(size=n),
                "drifting_feat": rng.normal(loc=0, size=n),
            }
        )
        cur = pd.DataFrame(
            {
                "stable_feat": rng.normal(size=n),
                "drifting_feat": rng.normal(loc=3, size=n),  # drift fuerte
            }
        )
        return ref, cur

    def test_report_sorted_by_psi_desc(self) -> None:
        ref, cur = self._frames()
        report = drift_report(ref, cur)
        assert [r.feature for r in report][0] == "drifting_feat"
        assert report[0].psi >= report[-1].psi

    def test_only_common_numeric_columns(self) -> None:
        ref, cur = self._frames()
        ref = ref.assign(only_ref=1.0, label=["a"] * len(ref))
        report = drift_report(ref, cur)
        feats = {r.feature for r in report}
        assert feats == {"stable_feat", "drifting_feat"}  # excluye only_ref y label

    def test_summarize_counts(self) -> None:
        ref, cur = self._frames()
        summary = summarize(drift_report(ref, cur))
        assert summary["n_features"] == 2
        assert summary["worst_feature"] == "drifting_feat"
        assert "drifting_feat" in summary["drifted_features"]
        assert summary["n_drifted"] >= 1


class TestLogging:
    def test_log_writes_jsonl_entry(self, tmp_path) -> None:
        import json

        rng = np.random.default_rng(30)
        ref = pd.DataFrame({"f": rng.normal(size=1000)})
        cur = pd.DataFrame({"f": rng.normal(loc=2, size=1000)})
        report = drift_report(ref, cur)

        log_file = tmp_path / "drift_log.jsonl"
        returned = log_drift_report(
            report, "e1", "AAPL", reference_tag="e1_conservative_eda_v1", log_path=log_file
        )
        assert returned == log_file

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["strategy"] == "e1"
        assert entry["ticker"] == "AAPL"
        assert entry["reference_tag"] == "e1_conservative_eda_v1"
        assert entry["worst_feature"] == "f"
        assert len(entry["features"]) == 1

    def test_log_appends(self, tmp_path) -> None:
        ref = pd.DataFrame({"f": np.zeros(100)})
        cur = pd.DataFrame({"f": np.zeros(100)})
        report = drift_report(ref, cur)
        log_file = tmp_path / "drift_log.jsonl"
        log_drift_report(report, "e1", "AAPL", reference_tag="v1", log_path=log_file)
        log_drift_report(report, "e1", "MSFT", reference_tag="v1", log_path=log_file)
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
