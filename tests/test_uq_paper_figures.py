"""Tests for the WP-B/C/D paper-rigor figure rendering."""

from __future__ import annotations

import json

import pytest

from vesp.uq.figures import PAPER_FIGURE_STEMS, render_paper_figures

pytest.importorskip("matplotlib")


def _write_benchmark_csvs(bench):
    (bench / "rerun_budget_curves.csv").write_text(
        "\n".join([
            "band,selector,rerun_fraction,n_seeds,capture_rate_mean,capture_rate_std,"
            "precision_mean,precision_std,lift_over_random_mean,lift_over_random_std,"
            "force_error_ratio_flagged_to_accepted_mean,force_error_ratio_flagged_to_accepted_std,"
            "spearman_mean,spearman_std",
            "L60,vespuq_supervisor,0.1,5,0.50,0.02,0.3,0.01,3.0,0.1,1.5,0.1,0.75,0.02",
            "L60,vespuq_supervisor,0.2,5,0.67,0.02,0.34,0.01,3.3,0.1,1.6,0.1,0.75,0.02",
            "L60,min_altitude,0.1,5,0.45,0.03,0.3,0.01,3.0,0.1,1.5,0.1,0.72,0.02",
            "L60,min_altitude,0.2,5,0.60,0.03,0.3,0.01,3.0,0.1,1.5,0.1,0.72,0.02",
        ]),
        encoding="utf-8",
    )
    (bench / "calibration_summary.csv").write_text(
        "\n".join([
            "band,region,n_seeds,z_std_mean,z_std_std,radial_z_std_mean,radial_z_std_std,"
            "tangential_z_std_mean,tangential_z_std_std,calibration_error_90_mean,calibration_error_90_std",
            "L60,low,5,1.16,0.05,1.21,0.05,1.08,0.04,0.04,0.01",
            "L60,mid,5,0.78,0.04,1.08,0.05,0.70,0.04,0.05,0.01",
            "L60,high,5,0.35,0.03,1.85,0.06,0.30,0.03,0.10,0.02",
        ]),
        encoding="utf-8",
    )
    (bench / "significance_summary.csv").write_text(
        "\n".join([
            "band,candidate,comparator,metric,n_seeds,seed_mean_delta,seed_wilcoxon_p,"
            "boot_delta,boot_ci_low,boot_ci_high,boot_p,boot_significant",
            "L60,vespuq_supervisor,min_altitude,spearman,5,0.03,0.06,0.03,0.01,0.06,0.02,True",
            "L60,vespuq_supervisor,min_altitude,capture,5,0.01,0.4,0.01,-0.03,0.05,0.6,False",
            "L60,vespuq_supervisor,min_altitude,auroc,5,-0.02,0.3,-0.02,-0.05,0.01,0.2,False",
        ]),
        encoding="utf-8",
    )


def _write_baseline_csv(baseline):
    (baseline / "uq_baseline_comparison.csv").write_text(
        "\n".join([
            "band,region,model,n_seeds,z_std_mean,z_std_std,picp_90_mean,picp_90_std,"
            "radial_z_std_mean,radial_z_std_std",
            "L60,low,vespuq,5,1.16,0.05,0.86,0.02,1.21,0.05",
            "L60,low,gp,5,1.73,0.06,0.75,0.03,1.90,0.06",
            "L60,mid,vespuq,5,0.78,0.04,0.96,0.01,1.08,0.05",
            "L60,mid,gp,5,0.92,0.04,0.92,0.02,1.06,0.05",
        ]),
        encoding="utf-8",
    )


def test_render_paper_figures_writes_all_outputs(tmp_path):
    bench = tmp_path / "benchmark_suite"
    baseline = tmp_path / "uq_baseline_comparison"
    out = tmp_path / "paper_figures"
    bench.mkdir()
    baseline.mkdir()
    _write_benchmark_csvs(bench)
    _write_baseline_csv(baseline)

    manifest = render_paper_figures(benchmark_dir=bench, baseline_dir=baseline, out_dir=out)
    names = {fig["name"] for fig in manifest["figures"]}
    assert names == set(PAPER_FIGURE_STEMS)
    for fig in manifest["figures"]:
        assert fig["status"] == "ok", f"{fig['name']} not ok: {fig.get('message')}"
        assert (out / f"{fig['name']}.png").exists()
        assert (out / f"{fig['name']}.pdf").exists()
    assert (out / "paper_figures_manifest.json").exists()
    json.loads((out / "paper_figures_manifest.json").read_text())


def test_render_paper_figures_placeholders_when_missing(tmp_path):
    # No CSVs at all -> every figure is a placeholder, no crash, manifest still written.
    manifest = render_paper_figures(
        benchmark_dir=tmp_path / "nope", baseline_dir=tmp_path / "nope2",
        out_dir=tmp_path / "figs",
    )
    assert {f["name"] for f in manifest["figures"]} == set(PAPER_FIGURE_STEMS)
    for fig in manifest["figures"]:
        assert fig["status"] == "missing_data"
        assert (tmp_path / "figs" / f"{fig['name']}.png").exists()
