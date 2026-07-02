"""Regression tests for roadmap and evidence-pack documentation status."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_hardening_plan_marks_h1_through_h8_complete():
    plan = _read("docs/VESP_SYSTEM_HARDENING_PLAN.md")

    assert "Status: **H1-H8 complete" in plan
    completed = re.findall(r"^### H([1-8]) .* - \*\*DONE", plan, flags=re.MULTILINE)
    assert completed == [str(index) for index in range(1, 9)]
    assert "no active hardening item is scheduled" in plan


def test_roadmap_is_history_with_an_unscheduled_backlog():
    roadmap = _read("docs/VESP_UQ_NEXT_STEPS.md")

    assert roadmap.startswith("# VESP-UQ — Implementation Roadmap and Backlog")
    assert "implementation history for N1-N21" in roadmap
    assert "All scheduled implementation items N1-N21 are complete" in roadmap
    assert "N18 — Operational conformal calibration packaged with the model — " in roadmap
    assert "**DONE; validation target partially failed**" in roadmap
    assert "## Current research backlog (deliberately not scheduled)" in roadmap
    assert "long-run validation pending" not in roadmap


def test_readme_matches_the_current_surface_and_claim_boundary():
    readme = _read("README.md")

    assert "Eight pages" in readme
    assert "**Compare**" in readme
    assert "operational orbit/state covariance realism is not validated" in readme
    assert "**Conservative-field limitation.**" in readme
    assert "surrogate-interface agnostic" in readme
    assert "VESP_SYSTEM_HARDENING_PLAN.md" in readme
    assert "Seven pages" not in readme
    assert "surrogate-agnostic uncertainty / risk-calibration layer" not in readme

    iac_plan = _read("docs/VESP_UQ_IAC_PLAN.md")
    assert "committed evidence is SH-derived" in iac_plan
    assert "prioritization guidance, not a validated operational rerun loop" in iac_plan


def test_benchmark_readme_keeps_evidence_pack_risks_visible():
    readme = _read("benchmarks/README.md")

    assert "## Current open risks" in readme
    for required in (
        "--collect-only",
        "--allow-placeholder-figures",
        "--train-run",
        "status: missing_data",
    ):
        assert required in readme

    assert (ROOT / "benchmarks/vespuq_conformal_validation.md").is_file()


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")
