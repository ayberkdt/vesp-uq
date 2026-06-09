"""Tests for the VESP-UQ experiment pipeline: external CSV trajectory source + units metadata."""

from __future__ import annotations

import math

from vesp.uq.experiment import run_vespuq


def _base_config(tmp_path):
    return {
        "seed": 0,
        "device": "cpu",
        "dtype": "float64",
        "data": {"type": "synthetic", "n": 240, "noise_std": 1.0e-4, "train_fraction": 0.7},
        "model": {"type": "multishell", "shell_alphas": [0.75, 0.9], "n_sources_per_shell": [24, 32]},
        "kernel": {"eps": 0.0},
        "uq": {
            "risk": {"scoring": "expected_abs_p95", "low_altitude_radius": 1.15},
            "screening": {"rerun_fraction": 0.25},
        },
        "output": {"output_dir": str(tmp_path / "out"), "run_name": "t"},
    }


def _write_accel_csv(path, n_traj=4, n_pts=8):
    header = ["trajectory_id", "t", "x", "y", "z", "ax_sur", "ay_sur", "az_sur", "ax_ref", "ay_ref", "az_ref"]
    rows = []
    for tid in range(n_traj):
        r0 = 1.05 + 0.1 * tid
        for j in range(n_pts):
            ang = 2 * math.pi * j / n_pts
            x, y, z = r0 * math.cos(ang), r0 * math.sin(ang), 0.0
            sur = [0.01 * tid, 0.0, 0.0]
            ref = [0.01 * tid + 0.002 * (tid + 1), 0.001, 0.0]  # residual grows with tid
            rows.append([tid, float(j), x, y, z, *sur, *ref])
    lines = [",".join(header)] + [",".join(str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_run_vespuq_with_external_csv_trajectories(tmp_path):
    csv = _write_accel_csv(tmp_path / "traj.csv", n_traj=5, n_pts=8)
    cfg = _base_config(tmp_path)
    cfg["uq"]["screening"]["trajectory_source"] = "csv"
    cfg["uq"]["screening"]["trajectory_path"] = str(csv)

    report = run_vespuq(cfg)
    sc = report["experiment_3_screening"]
    assert sc["trajectory_source"] == "csv"
    assert sc["trajectory_path"] == str(csv)
    assert sc["external_trajectory_count"] == 5
    assert sc["external_output_points_total"] == 40
    assert sc["n_trajectories"] == 5
    # accel pairs present -> true force error comes from the residual directly, not the NN oracle
    assert sc["true_error_mode"] == "residual_csv"


def test_run_vespuq_units_metadata_present(tmp_path):
    cfg = _base_config(tmp_path)  # generated trajectories (default)
    report = run_vespuq(cfg)
    units = report["units"]
    for key in (
        "risk_score_units", "acceleration_metric_units", "position_units",
        "force_error_scale_note", "physical_R_body", "physical_R_body_units",
    ):
        assert key in units
    assert report["experiment_3_screening"]["trajectory_source"] == "generated"
    assert report["experiment_3_screening"]["external_trajectory_count"] is None


def test_external_positions_only_csv_uses_nn_oracle(tmp_path):
    # positions-only CSV -> no residual -> falls back to the nearest-neighbour oracle
    header = ["trajectory_id", "t", "x", "y", "z"]
    rows = []
    for tid in range(3):
        r0 = 1.1 + 0.1 * tid
        for j in range(6):
            ang = 2 * math.pi * j / 6
            rows.append([tid, float(j), r0 * math.cos(ang), r0 * math.sin(ang), 0.0])
    csv = tmp_path / "pos.csv"
    csv.write_text("\n".join([",".join(header)] + [",".join(str(v) for v in r) for r in rows]) + "\n", encoding="utf-8")

    cfg = _base_config(tmp_path)
    cfg["uq"]["screening"]["trajectory_source"] = "csv"
    cfg["uq"]["screening"]["trajectory_path"] = str(csv)
    report = run_vespuq(cfg)
    assert report["experiment_3_screening"]["true_error_mode"].startswith("nn_oracle")


def test_run_vespuq_csv_physical_acceleration_units_converted(tmp_path):
    # External CSV accelerations declared in m/s^2 are converted into model units via the
    # explicit body.acceleration_scale_m_s2 before scoring/fitting; the report records it.
    csv = _write_accel_csv(tmp_path / "traj.csv", n_traj=5, n_pts=8)
    cfg = _base_config(tmp_path)
    cfg["body"] = {"acceleration_units": "model_normalized_accel", "acceleration_scale_m_s2": 1.0e-6}
    cfg["uq"]["screening"]["trajectory_source"] = "csv"
    cfg["uq"]["screening"]["trajectory_path"] = str(csv)
    cfg["uq"]["screening"]["trajectory_acceleration_units"] = "m/s^2"

    report = run_vespuq(cfg)
    sc = report["experiment_3_screening"]
    units = sc["trajectory_units"]
    assert units["acceleration_converted_to_model"] is True
    assert abs(units["acceleration_scale_m_s2"] - 1.0e-6) < 1.0e-18
    assert sc["true_error_mode"] == "residual_csv"  # accel pairs still used for the true force error


def test_run_vespuq_csv_physical_units_without_scale_raises(tmp_path):
    csv = _write_accel_csv(tmp_path / "traj.csv", n_traj=4, n_pts=8)
    cfg = _base_config(tmp_path)  # no body.acceleration_scale_m_s2 -> not physical
    cfg["uq"]["screening"]["trajectory_source"] = "csv"
    cfg["uq"]["screening"]["trajectory_path"] = str(csv)
    cfg["uq"]["screening"]["trajectory_acceleration_units"] = "m/s^2"

    import pytest

    with pytest.raises(ValueError):
        run_vespuq(cfg)
