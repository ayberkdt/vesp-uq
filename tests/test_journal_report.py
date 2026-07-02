"""Unit tests for the journal report generator (WP12)."""

from __future__ import annotations

from vesp.uq.journal_report import (
    build_claims,
    build_latex_tables,
    build_report,
    significance_verdict,
    verdict_from_ranking,
    write_report,
)


def _significance(band, delta, lo, hi, metric="spearman"):
    return [{"band": band, "candidate": "vespuq_supervisor", "comparator": "min_altitude",
             "metric": metric, "boot_delta": str(delta), "boot_ci_low": str(lo),
             "boot_ci_high": str(hi), "boot_p": "0.04"}]


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


def test_significance_verdict_detects_significant_win():
    v = significance_verdict(_significance("L60", 0.12, 0.03, 0.21))
    assert v["available"] and v["any_significant"]
    assert "spearman" in v["bands"]["L60"]["wins"]


def test_significance_verdict_indistinguishable_when_ci_brackets_zero():
    v = significance_verdict(_significance("L60", 0.05, -0.02, 0.12))
    assert v["available"] and not v["any_significant"]


def test_significance_verdict_unavailable_when_empty():
    assert significance_verdict(None)["available"] is False


def test_claims_include_significance_and_baseline():
    v = verdict_from_ranking(_ranking("L60", 0.50, 0.30), _partial("L60", 0.2))
    data = {"significance": _significance("L60", 0.12, 0.03, 0.21),
            "uq_baseline": [1], "decision": [1],
            "calibration": [{"band": "L60", "region": "low", "radial_z_std_mean": "1.1"}]}
    by = {c["claim"]: c["status"] for c in build_claims(data, v)}
    assert by["The supervisor's ranking edge over altitude is statistically significant"] == "supported"
    assert by["The predictive covariance is calibrated per component (radial vs tangential)"] == "supported"
    assert by["VESP-UQ is benchmarked head-to-head against a Gaussian-process UQ baseline"] == "supported"


def test_claims_significance_indistinguishable_status():
    v = verdict_from_ranking(_ranking("L60", 0.50, 0.30), _partial("L60", 0.2))
    data = {"significance": _significance("L60", 0.05, -0.02, 0.12)}
    by = {c["claim"]: c["status"] for c in build_claims(data, v)}
    assert by["The supervisor's ranking edge over altitude is statistically significant"].startswith(
        "not supported")


def test_latex_tables_for_new_studies():
    tables = build_latex_tables({"significance": _significance("L60", 0.12, 0.03, 0.21),
                                 "decision": [{"band": "L60", "selector": "vespuq_supervisor",
                                               "auroc_mean": "0.75", "auprc_mean": "0.5",
                                               "capture_auc_normalized_mean": "0.6",
                                               "oracle_regret_mean": "0.4"}]})
    assert "\\toprule" in tables["table_significance.tex"]
    assert "\\toprule" in tables["table_decision_quality.tex"]
    # missing baseline study still yields a pending comment
    assert tables["table_uq_baseline.tex"].startswith("%")


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

    # G1/G7: the journal dir now carries a provenance manifest covering the report + every table.
    from vesp.uq.io.run_artifacts import verify_manifest

    assert (out / "run_manifest.json").exists()
    report = verify_manifest(out)
    assert report["ok"], report
    assert "journal_validation_report.md" in report["verified"]
    assert any(name.endswith("table_ranking_robustness.tex") for name in report["verified"])
