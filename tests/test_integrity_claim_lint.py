"""Tests for G6 -- the forbidden-claim linter."""

from __future__ import annotations

import pytest

from vesp.uq.integrity.claim_lint import lint_report, scan_text


def test_clean_prose_passes():
    text = (
        "VESP-UQ predicts a local force-error covariance and screens trajectories for selective "
        "rerun. The scalar ranking matches altitude; the contribution is the calibrated covariance."
    )
    assert scan_text(text) == []


@pytest.mark.parametrize(
    "line",
    [
        "We report a validated operational orbit covariance for the full mission.",
        "The method achieves density recovery from the residual field.",
        "VESP-UQ outperforms all baselines on every band.",
        "This provides a guaranteed risk bound on captured error.",
        "An end-to-end trajectory correction is demonstrated.",
        "We show end-to-end ST-LRPS validation of the pipeline.",
        "A learned noise model is fit to the residuals.",
    ],
)
def test_each_forbidden_claim_is_flagged(line):
    violations = scan_text(line)
    assert len(violations) == 1, violations


def test_evidence_tag_excuses_a_line():
    line = "Validated operational orbit covariance. <!-- evidence: appendix_C_table_4 -->"
    assert scan_text(line) == []


def test_future_work_disclaimer_excuses_a_line():
    # the auto-generated claims table legitimately lists a forbidden claim as future work
    line = "| Validated operational 6x6 orbit/state covariance | future work | not attempted |"
    assert scan_text(line) == []


def test_code_fences_are_ignored():
    text = "Intro.\n\n```\n# validated operational orbit covariance (a variable name)\n```\nDone."
    assert scan_text(text) == []


def test_lint_report_aggregates_report_and_manuscript(tmp_path):
    report = tmp_path / "report.md"
    report.write_text("All numbers are sourced and calibration is per band.\n", encoding="utf-8")
    manuscript = tmp_path / "paper.tex"
    manuscript.write_text("We claim density recovery from sparse data.\n", encoding="utf-8")

    clean = lint_report(report)
    assert clean["ok"] is True

    flagged = lint_report(report, manuscript=manuscript)
    assert flagged["ok"] is False
    assert flagged["violations"][0]["claim"] == "density recovery"
    assert str(manuscript) == flagged["violations"][0]["source"]
