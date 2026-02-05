from pathlib import Path

from scripts.evaluation.e1_simple_guardrails import validate_summary


ROOT = Path(__file__).resolve().parents[1]


def test_guardrails_good_passes():
    summary_path = ROOT / "tests" / "fixtures" / "e1_simple_good" / "summary_all.csv"
    errors = validate_summary(
        summary_path=summary_path,
        project_root=ROOT,
        run_dir=None,
        sharpe_min=1.0,
        max_drawdown_max=0.20,
        cagr_min=0.10,
    )
    assert errors == []


def test_guardrails_bad_fails():
    summary_path = ROOT / "tests" / "fixtures" / "e1_simple_bad" / "summary_all.csv"
    errors = validate_summary(
        summary_path=summary_path,
        project_root=ROOT,
        run_dir=None,
        sharpe_min=1.0,
        max_drawdown_max=0.20,
        cagr_min=0.10,
    )
    assert errors
