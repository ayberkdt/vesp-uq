"""End-to-end smoke test for the VESP-UQ driver: artifacts + report structure."""

from __future__ import annotations

from pathlib import Path

from vesp.common.config import load_config
from vesp.uq.run import run

ROOT = Path(__file__).resolve().parents[1]


def test_run_smoke_writes_all_artifacts(tmp_path):
    cfg = load_config(ROOT / "configs" / "vespuq" / "vespuq_smoke.yaml")
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
        "sentinel_audit.csv",
        "fit_summary.json",
    ):
        assert (run_dir / fname).exists(), f"missing artifact {fname}"

    # report structure
    assert "experiment_1_calibration" in report
    assert "experiment_3_screening" in report
    assert "sentinel_audit" in report
    assert "runtime" in report
    assert "summary" in report
    assert report["sentinel_audit"]["enabled"] is True
    assert report["sentinel_audit"]["n_sentinel"] >= 0

    screen = report["experiment_3_screening"]
    # regression: risk screening is evaluated at OUTPUT points only (no online-RHS assumption)
    assert "n_output_points_total" in screen
    assert "score_us_per_output_point" in report["runtime"]
    assert report["runtime"]["score_us_per_output_point"] > 0.0

    # markdown contains the IAC claim summary and the not-RHS disclaimer
    md = (run_dir / "vespuq_report.md").read_text(encoding="utf-8")
    assert "IAC claim summary" in md
    assert "not inside every integrator RHS call" in md
    assert "Accepted-set sentinel audit" in md
    assert "Component-wise calibration" in md
    assert "Uncertainty decomposition" in md

    calibration_header = (run_dir / "calibration_by_band.csv").read_text(encoding="utf-8").splitlines()[0]
    for col in (
        "radial_z_std",
        "tangential_z_std",
        "radial_picp_90",
        "tangential_picp_90",
        "epistemic_to_pred_std_ratio",
        "approx_posthoc_remainder_std",
        "conformal_prediction_scale",
    ):
        assert col in calibration_header


def test_trajectory_scores_csv_has_expected_columns(tmp_path):
    cfg = load_config(ROOT / "configs" / "vespuq" / "vespuq_smoke.yaml")
    cfg["output"]["output_dir"] = str(tmp_path)
    cfg["output"]["run_name"] = "smoke2"
    run(cfg)
    header = (tmp_path / "smoke2" / "trajectory_scores.csv").read_text(encoding="utf-8").splitlines()[0]
    for col in (
        "trajectory_id",
        "risk_score",
        "max_sigma",
        "mean_calibrated_point_risk",
        "above_screening_threshold",
        "flagged_for_rerun",
        "true_error",
    ):
        assert col in header


def test_sentinel_audit_csv_has_expected_columns(tmp_path):
    cfg = load_config(ROOT / "configs" / "vespuq" / "vespuq_smoke.yaml")
    cfg["output"]["output_dir"] = str(tmp_path)
    cfg["output"]["run_name"] = "smoke_audit"
    run(cfg)
    header = (tmp_path / "smoke_audit" / "sentinel_audit.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == [
        "trajectory_id",
        "risk_score",
        "true_force_error",
        "is_high_force_error",
        "flagged",
    ]


def test_run_save_model_persists_loadable_plugin(tmp_path):
    import json

    from vesp.uq import VESPUQPlugin

    cfg = load_config(ROOT / "configs" / "vespuq" / "vespuq_smoke.yaml")
    cfg["output"]["output_dir"] = str(tmp_path)
    cfg["output"]["run_name"] = "smoke3"
    cfg["output"]["save_model"] = True
    run(cfg)

    model_path = tmp_path / "smoke3" / "vespuq_plugin.pt"
    assert model_path.exists(), "save_model: true must write vespuq_plugin.pt"

    manifest = json.loads((tmp_path / "smoke3" / "run_manifest.json").read_text(encoding="utf-8"))
    assert "vespuq_plugin_pt" in manifest["artifacts"], "manifest must checksum the saved plugin"

    loaded = VESPUQPlugin.load(model_path)
    pred = loaded.predict_uncertainty([[0.0, 0.0, 1.2], [0.0, 1.4, 0.0]])
    assert pred.sigma.shape == (2,)
    assert bool((pred.sigma > 0).all())


def test_run_save_model_records_operational_conformal_metadata(tmp_path):
    from vesp.uq import VESPUQPlugin

    cfg = load_config(ROOT / "configs" / "vespuq" / "vespuq_smoke.yaml")
    cfg["uq"]["conformal"]["apply"] = True
    cfg["output"]["output_dir"] = str(tmp_path)
    cfg["output"]["run_name"] = "smoke_conformal"
    cfg["output"]["save_model"] = True
    run(cfg)

    model_path = tmp_path / "smoke_conformal" / "vespuq_plugin.pt"
    loaded = VESPUQPlugin.load(model_path)
    conformal = loaded.user_metadata.get("conformal_prediction", {})
    assert conformal.get("enabled") is True
    assert conformal.get("global", {}).get("scale") is not None

    card = (tmp_path / "smoke_conformal" / "vespuq_plugin_card.md").read_text(encoding="utf-8")
    assert "operational conformal prediction scale" in card
