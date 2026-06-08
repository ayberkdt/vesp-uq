"""End-to-end smoke test for the VESP-UQ driver: artifacts + report structure."""

from __future__ import annotations

from pathlib import Path

from vesp.common.config import load_config
from vesp.uq.run import run

ROOT = Path(__file__).resolve().parents[1]


def test_run_smoke_writes_all_artifacts(tmp_path):
    cfg = load_config(ROOT / "configs" / "vespuq_smoke.yaml")
    cfg["output"]["output_dir"] = str(tmp_path)
    cfg["output"]["run_name"] = "smoke"
    report = run(cfg)

    run_dir = tmp_path / "smoke"
    for fname in (
        "vespuq_report.json",
        "vespuq_report.md",
        "calibration_by_band.csv",
        "trajectory_scores.csv",
        "flagged_trajectories.csv",
        "fit_summary.json",
    ):
        assert (run_dir / fname).exists(), f"missing artifact {fname}"

    # report structure
    assert "experiment_1_calibration" in report
    assert "experiment_3_screening" in report
    assert "runtime" in report
    assert "summary" in report

    screen = report["experiment_3_screening"]
    # regression: risk screening is evaluated at OUTPUT points only (no online-RHS assumption)
    assert "n_output_points_total" in screen
    assert "score_us_per_output_point" in report["runtime"]
    assert report["runtime"]["score_us_per_output_point"] > 0.0

    # markdown contains the IAC claim summary and the not-RHS disclaimer
    md = (run_dir / "vespuq_report.md").read_text(encoding="utf-8")
    assert "IAC claim summary" in md
    assert "not inside every integrator RHS call" in md


def test_trajectory_scores_csv_has_expected_columns(tmp_path):
    cfg = load_config(ROOT / "configs" / "vespuq_smoke.yaml")
    cfg["output"]["output_dir"] = str(tmp_path)
    cfg["output"]["run_name"] = "smoke2"
    run(cfg)
    header = (tmp_path / "smoke2" / "trajectory_scores.csv").read_text(encoding="utf-8").splitlines()[0]
    for col in ("trajectory_id", "risk_score", "max_sigma", "flagged_for_rerun", "true_error"):
        assert col in header
