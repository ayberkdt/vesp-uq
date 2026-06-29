"""Tests for G1 -- the no-orphan-number auditor."""

from __future__ import annotations

import json
from pathlib import Path

from vesp.common.artifacts import compute_file_sha256
from vesp.uq.integrity.number_audit import (
    audit_latex_tables,
    audit_report_numbers,
    collect_csv_values,
    verify_csv_manifests,
)

# Source CSV: the only legitimate provenance for any number the report may state.
CSV = (
    "band,selector,spearman,capture,rerun_fraction\n"
    "L90,min_altitude,0.856,0.730,0.20\n"
    "L60,vespuq_supervisor,0.753,0.512,0.20\n"
)


def _write_source(dirpath: Path, *, tamper_manifest: bool = False, with_manifest: bool = True) -> Path:
    csv_path = dirpath / "benchmark_summary.csv"
    csv_path.write_text(CSV, encoding="utf-8")
    if with_manifest:
        sha = "0" * 64 if tamper_manifest else compute_file_sha256(csv_path)
        manifest = {"artifacts": {csv_path.name: {
            "path": str(csv_path), "sha256": sha, "bytes": csv_path.stat().st_size,
            "origin": "generated"}}}
        (dirpath / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return csv_path


def test_collect_csv_values_parses_numeric_cells(tmp_path):
    _write_source(tmp_path)
    values = collect_csv_values([tmp_path])
    for expected in (0.856, 0.730, 0.753, 0.512, 0.20):
        assert any(abs(v - expected) < 1e-9 for v in values)


def test_clean_report_is_ok(tmp_path):
    _write_source(tmp_path)
    report = tmp_path / "report.md"
    report.write_text(
        "# Benchmark summary\n\n"
        "On L90 altitude reaches Spearman 0.856 (capture 0.730 at the 20% budget), while on L60 the "
        "vespuq_supervisor reaches 0.753 with capture 0.512.\n",
        encoding="utf-8",
    )
    result = audit_report_numbers(report, [tmp_path])
    assert result["ok"] is True
    assert result["n_orphans"] == 0
    assert result["checked"] >= 5  # 0.856, 0.730, 20%, 0.753, 0.512
    assert result["manifests"]["ok"] is True


def test_planted_orphan_fails_and_is_listed(tmp_path):
    _write_source(tmp_path)
    report = tmp_path / "report.md"
    report.write_text(
        "# Results\n\nThe supervisor reaches Spearman 0.753, and a calibration z_std of 0.999.\n",
        encoding="utf-8",
    )
    result = audit_report_numbers(report, [tmp_path])
    assert result["ok"] is False
    orphan_values = [o["value"] for o in result["orphans"]]
    assert "0.999" in orphan_values
    assert "0.753" not in orphan_values  # the sourced number is not flagged


def test_identifiers_and_references_are_not_orphans(tmp_path):
    _write_source(tmp_path)
    report = tmp_path / "report.md"
    # L90 / p95 / picp_90 are identifiers; "Table 3.14" / "Phase-14" are references; 5 seeds is an
    # int -- none of these numbers exist in the CSV, yet none must be flagged as an orphan.
    report.write_text(
        "# Notes\n\n"
        "See Table 3.14 and Phase-14. The L90 band, p95 aggregation and picp_90 column over 5 seeds "
        "give Spearman 0.856.\n",
        encoding="utf-8",
    )
    result = audit_report_numbers(report, [tmp_path])
    assert result["ok"] is True, result["orphans"]


def test_code_fences_are_ignored(tmp_path):
    _write_source(tmp_path)
    report = tmp_path / "report.md"
    report.write_text(
        "# Run\n\nSpearman 0.856.\n\n```\npython run.py --threshold 0.4242\n```\n"
        "Inline `--scale 0.7373` is also ignored.\n",
        encoding="utf-8",
    )
    result = audit_report_numbers(report, [tmp_path])
    assert result["ok"] is True, result["orphans"]


def test_percentage_matches_fraction(tmp_path):
    _write_source(tmp_path)
    report = tmp_path / "report.md"
    report.write_text("# Budget\n\nEvaluated at the 20% rerun budget.\n", encoding="utf-8")
    result = audit_report_numbers(report, [tmp_path])
    assert result["ok"] is True  # 20% sourced by the 0.20 rerun_fraction cell


def test_unmanifested_csv_fails_verification(tmp_path):
    _write_source(tmp_path, with_manifest=False)
    man = verify_csv_manifests([tmp_path])
    assert man["ok"] is False
    assert man["unmanifested"]


def test_tampered_manifest_checksum_fails(tmp_path):
    _write_source(tmp_path, tamper_manifest=True)
    report = tmp_path / "report.md"
    report.write_text("# R\n\nSpearman 0.856.\n", encoding="utf-8")
    result = audit_report_numbers(report, [tmp_path])
    assert result["ok"] is False  # numbers sourced, but the CSV checksum does not match the manifest
    assert result["manifests"]["changed"]


def test_latex_table_orphan_detected(tmp_path):
    _write_source(tmp_path)
    tables = tmp_path / "latex"
    tables.mkdir()
    (tables / "table_clean.tex").write_text(
        r"\begin{tabular}{lr} L90 & 0.856 \\ L60 & 0.753 \end{tabular}", encoding="utf-8")
    (tables / "table_orphan.tex").write_text(
        r"\begin{tabular}{lr} fabricated & 0.4242 \end{tabular}", encoding="utf-8")
    result = audit_latex_tables(tables, [tmp_path])
    assert result["ok"] is False
    assert "table_orphan.tex" in result["tables"]
    assert "table_clean.tex" not in result["tables"]
