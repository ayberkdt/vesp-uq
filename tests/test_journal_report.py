"""Unit tests for the journal report generator (WP12)."""

from __future__ import annotations

from vesp.uq.journal_report import (
    build_claims,
    build_latex_tables,
    build_report,
    verdict_from_ranking,
    write_report,
)


def _ranking(band, sup_sp, alt_sp):
    return [
        {"band": band, "selector": "vespuq_supervisor", "spearman_mean": str(sup_sp),
         "capture_rate_mean": "0.4", "lift_over_random_mean": "2.0"},
        {"band": band, "selector": "min_altitude", "spearman_mean": str(alt_sp),
         "capture_rate_mean": "0.4", "lift_over_random_mean": "2.0"},
    ]


def _partial(band, val):
    return [{"band": band, "selector": "vespuq_supervisor",
             "partial_pearson_given_min_radius_mean": str(val), "spearman_mean": "0.4"}]


def test_verdict_beats_altitude():
    v = verdict_from_ranking(_ranking("L60", 0.50, 0.30), _partial("L60", 0.2))
    assert v["overall"] == "consistent"
    assert v["bands"]["L60"] == "beats_altitude"


def test_verdict_altitude_dominant():
    v = verdict_from_ranking(_ranking("L60", 0.25, 0.45), _partial("L60", 0.0))
    assert v["overall"] == "altitude_dominant"


def test_verdict_mixed_across_bands():
    rows = _ranking("L60", 0.50, 0.30) + _ranking("L90", 0.25, 0.45)
    part = _partial("L60", 0.2) + _partial("L90", 0.0)
    v = verdict_from_ranking(rows, part)
    assert v["overall"] == "mixed"


def test_verdict_pending_when_empty():
    assert verdict_from_ranking(None, None)["overall"] == "pending"


def test_claims_status_mapping():
    v = verdict_from_ranking(_ranking("L60", 0.50, 0.30), _partial("L60", 0.2))
    claims = build_claims({"calibration": [1], "reliability": [1], "drift": [1]}, v)
    by = {c["claim"]: c["status"] for c in claims}
    assert by["VESP-UQ adds force-error ranking value beyond altitude-only heuristics"] == "supported"
    # the long-horizon drift claim is a scope boundary, never "supported"
    drift = next(c for c in claims if "long-horizon position error" in c["claim"])
    assert drift["status"].startswith("not supported")


def test_latex_tables_escape_and_pending():
    tables = build_latex_tables({"ranking": _ranking("L60", 0.5, 0.3)})
    tex = tables["table_ranking_robustness.tex"]
    assert "\\toprule" in tex and "\\bottomrule" in tex
    assert "vespuq\\_supervisor" in tex  # underscores escaped
    # a missing study yields a pending comment, not a crash
    assert tables["table_expanded_baselines.tex"].startswith("%")


def test_build_report_sections_and_claims_dedup(tmp_path):
    bs = tmp_path / "benchmark_suite"
    bs.mkdir()
    (bs / "benchmark_summary.csv").write_text(
        "band,selector,spearman_mean,capture_rate_mean,lift_over_random_mean\n"
        "L60,vespuq_supervisor,0.50,0.4,2.0\nL60,min_altitude,0.30,0.4,2.0\n", encoding="utf-8")
    (bs / "partial_correlation_summary.csv").write_text(
        "band,selector,partial_pearson_given_min_radius_mean,spearman_mean\n"
        "L60,vespuq_supervisor,0.2,0.5\n", encoding="utf-8")
    result = build_report(tmp_path)
    md = result["report_md"]
    assert "## 1. Executive summary" in md
    assert "## 14. Claims that must remain future work" in md
    # the long-horizon claim must not appear in the supported section
    sec13 = md.split("## 13. Claims supported")[1].split("## 14.")[0]
    assert "ranks long-horizon position error" not in sec13


def test_write_report_emits_files(tmp_path):
    bs = tmp_path / "benchmark_suite"
    bs.mkdir()
    (bs / "benchmark_summary.csv").write_text(
        "band,selector,spearman_mean,capture_rate_mean,lift_over_random_mean\n"
        "L60,vespuq_supervisor,0.50,0.4,2.0\n", encoding="utf-8")
    out = tmp_path / "journal"
    write_report(tmp_path, out_dir=out)
    assert (out / "journal_validation_report.md").exists()
    assert (out / "latex_tables" / "table_ranking_robustness.tex").exists()
