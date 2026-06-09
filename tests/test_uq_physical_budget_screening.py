"""Tests for physical acceleration-budget thresholding and screening."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vesp.core.sources import make_shell_sources
from vesp.uq import VESPUQPlugin, make_synthetic_uq_samples
from vesp.uq.data import split_uq_samples
from vesp.uq.physical_units import resolve_acceleration_scale
from vesp.uq.thresholds import resolve_physical_budget_threshold, resolve_threshold

import scripts.run_physical_budget_screening as pbs


def _fitted_plugin_and_held():
    samples = make_synthetic_uq_samples(n=400, noise_std=1.0e-4, seed=1)
    train, held = split_uq_samples(samples, train_fraction=0.7, seed=0)
    src = make_shell_sources([0.75, 0.9], [24, 32], dtype=torch.float64)
    plugin = VESPUQPlugin(src, reg_method="lcurve", noise_model="heteroscedastic", seed=0)
    plugin.fit_error(train.positions, train.error)
    return plugin, held


def _physical_config():
    return {
        "body": {"acceleration_units": "model_normalized_accel", "acceleration_scale_m_s2": 1.0e-6},
        "uq": {"physical_budget": {"enabled": True, "value": 1.0e-8, "units": "m/s^2"}},
    }


def test_physical_budget_rejects_relative_scoring():
    scale = resolve_acceleration_scale(_physical_config())
    physical_cfg = _physical_config()["uq"]["physical_budget"]
    with pytest.raises(ValueError):
        resolve_physical_budget_threshold(physical_cfg, scale, "supervisor_rel")
    with pytest.raises(ValueError):
        resolve_physical_budget_threshold(physical_cfg, scale, "supervisor_rel_p95")


def test_physical_budget_accepts_expected_abs_p95():
    scale = resolve_acceleration_scale(_physical_config())
    physical_cfg = _physical_config()["uq"]["physical_budget"]
    model_thr, phys_val, phys_units = resolve_physical_budget_threshold(
        physical_cfg, scale, "expected_abs_p95"
    )
    # 1e-8 m/s^2 / (1e-6 m/s^2 per model unit) = 1e-2 model units
    assert model_thr == pytest.approx(1.0e-2)
    assert phys_val == pytest.approx(1.0e-8)
    assert phys_units == "m/s^2"


def test_physical_budget_requires_scale():
    cfg = {"body": {}, "uq": {"physical_budget": {"enabled": True, "value": 1.0e-8, "units": "m/s^2"}}}
    scale = resolve_acceleration_scale(cfg)
    with pytest.raises(ValueError):
        resolve_physical_budget_threshold(cfg["uq"]["physical_budget"], scale, "expected_abs_p95")


def test_resolve_threshold_physical_budget_metadata_has_both_thresholds():
    plugin, held = _fitted_plugin_and_held()
    cfg = _physical_config()
    cfg["uq"]["screening"] = {"threshold_source": "physical_budget"}
    thr, meta = resolve_threshold(
        cfg["uq"]["screening"], plugin, held, "expected_abs_p95",
        dtype=torch.float64, seed=0, config=cfg,
    )
    assert meta["threshold_source"] == "physical_budget"
    assert meta["threshold_model_units"] == pytest.approx(1.0e-2)
    assert meta["threshold_physical_value"] == pytest.approx(1.0e-8)
    assert meta["threshold_physical_units"] == "m/s^2"
    assert meta["acceleration_scale_m_s2"] == pytest.approx(1.0e-6)
    assert thr == pytest.approx(meta["threshold_model_units"])


def test_resolve_threshold_physical_budget_requires_config():
    plugin, held = _fitted_plugin_and_held()
    with pytest.raises(ValueError):
        resolve_threshold(
            {"threshold_source": "physical_budget"}, plugin, held, "expected_abs_p95",
            dtype=torch.float64, seed=0, config=None,
        )


def test_existing_threshold_modes_still_work():
    plugin, held = _fitted_plugin_and_held()
    # manual mode unchanged
    thr, meta = resolve_threshold(
        {"threshold_source": "manual", "threshold": 0.05}, plugin, held, "supervisor_rel",
        dtype=torch.float64, seed=0,
    )
    assert thr == pytest.approx(0.05)
    assert meta["threshold_source"] == "manual"
    # fraction mode (no threshold) -> None
    thr2, meta2 = resolve_threshold({}, plugin, held, "supervisor_rel", dtype=torch.float64, seed=0)
    assert thr2 is None
    assert meta2["threshold_source"] is None
    # pointwise calibration still works for absolute scoring
    thr3, meta3 = resolve_threshold(
        {"threshold_source": "pointwise_calibration_quantile", "threshold_quantile": 0.9},
        plugin, held, "expected_abs_p95", dtype=torch.float64, seed=0,
    )
    assert thr3 is not None
    assert meta3["threshold_source"] == "pointwise_calibration_quantile"


def test_physical_budget_enabled_inferred_without_explicit_threshold_source():
    plugin, held = _fitted_plugin_and_held()
    cfg = _physical_config()  # physical_budget.enabled = True, no screening.threshold_source
    cfg["uq"]["screening"] = {}
    thr, meta = resolve_threshold(
        cfg["uq"]["screening"], plugin, held, "expected_abs_p95",
        dtype=torch.float64, seed=0, config=cfg,
    )
    assert meta["threshold_source"] == "physical_budget"
    assert thr == pytest.approx(1.0e-2)


def _tiny_screening_config():
    return {
        "seed": 0,
        "device": "cpu",
        "dtype": "float64",
        "body": {"acceleration_units": "model_normalized_accel", "acceleration_scale_m_s2": 1.0e-6},
        "data": {"type": "synthetic", "n": 240, "noise_std": 1.0e-4, "train_fraction": 0.7},
        "model": {"type": "multishell", "shell_alphas": [0.75, 0.9], "n_sources_per_shell": [24, 32]},
        "kernel": {"eps": 0.0},
        "uq": {
            "risk": {"scoring": "expected_abs_p95", "low_altitude_radius": 1.15},
            "screening": {"n_orbits": 12, "n_points": 16, "rerun_fraction": 0.25},
        },
        "output": {"output_dir": "out", "run_name": "t"},
    }


def test_script_runs_on_tiny_synthetic_config(tmp_path):
    cfg = _tiny_screening_config()
    cfg["_config_path"] = "tiny.yaml"
    args = pbs.argparse.Namespace(
        budget=1.0e-8, units="m/s^2", scoring="expected_abs_p95", max_rerun_fraction=None,
    )
    pbs._configure_physical_budget(cfg, args)
    out_dir = tmp_path / "physical_budget"
    result = pbs.run_and_write(cfg, out_dir=out_dir)

    assert (out_dir / "physical_budget_screening.json").exists()
    assert (out_dir / "physical_budget_screening.md").exists()
    assert (out_dir / "physical_budget_scores.csv").exists()

    data = json.loads((out_dir / "physical_budget_screening.json").read_text())
    assert data["threshold"]["physical_value"] == pytest.approx(1.0e-8)
    assert data["threshold"]["model_units"] == pytest.approx(1.0e-2)
    assert data["physical_conversion_available"] is True
    assert data["scoring"] == "expected_abs_p95"
    assert data["n_trajectories"] == result["n_trajectories"]
    # CSV has a header + one row per trajectory
    csv_lines = (out_dir / "physical_budget_scores.csv").read_text().strip().splitlines()
    assert len(csv_lines) == 1 + result["n_trajectories"]


def test_script_rejects_relative_scoring(tmp_path):
    cfg = _tiny_screening_config()
    args = pbs.argparse.Namespace(
        budget=1.0e-8, units="m/s^2", scoring="supervisor_rel", max_rerun_fraction=None,
    )
    with pytest.raises(SystemExit):
        pbs._configure_physical_budget(cfg, args)


def test_script_requires_enabled_or_budget(tmp_path):
    cfg = _tiny_screening_config()
    cfg["uq"]["physical_budget"] = {"enabled": False}
    args = pbs.argparse.Namespace(budget=None, units=None, scoring=None, max_rerun_fraction=None)
    with pytest.raises(SystemExit):
        pbs._configure_physical_budget(cfg, args)


def test_no_position_error_references_in_modules():
    import inspect

    import vesp.uq.thresholds as thr_mod

    for mod in (thr_mod, pbs):
        src = inspect.getsource(mod).lower()
        assert "position error" not in src
        assert "position-error" not in src


# ----------------------------------------------- P5.2: conformally-corrected physical budget

def _physical_config_with_conformal(mode="norm"):
    cfg = _physical_config()
    cfg["uq"]["physical_budget"]["conformal"] = {"enabled": True, "alpha": 0.10, "mode": mode}
    cfg["uq"]["screening"] = {"threshold_source": "physical_budget"}
    return cfg


def test_conformal_correction_tightens_threshold_and_reports_coverage():
    plugin, held = _fitted_plugin_and_held()
    cfg = _physical_config_with_conformal()
    thr, meta = resolve_threshold(
        cfg["uq"]["screening"], plugin, held, "expected_abs_p95",
        dtype=torch.float64, seed=0, config=cfg,
    )
    assert meta["conformal_enabled"] is True
    assert meta["conformal_scale"] is not None and meta["conformal_scale"] > 0.0
    # raw threshold preserved; corrected = raw / scale
    assert meta["threshold_model_units_raw"] == pytest.approx(1.0e-2)
    assert thr == pytest.approx(meta["threshold_model_units_raw"] / meta["conformal_scale"])
    assert thr == pytest.approx(meta["threshold_model_units"])
    assert meta["conformal_coverage_before"] is not None
    assert meta["conformal_coverage_after"] is not None


def test_conformal_disabled_leaves_threshold_unchanged():
    plugin, held = _fitted_plugin_and_held()
    cfg = _physical_config()
    cfg["uq"]["screening"] = {"threshold_source": "physical_budget"}
    thr, meta = resolve_threshold(
        cfg["uq"]["screening"], plugin, held, "expected_abs_p95",
        dtype=torch.float64, seed=0, config=cfg,
    )
    assert meta["conformal_enabled"] is False
    assert meta["threshold_model_units_raw"] is None
    assert thr == pytest.approx(1.0e-2)


def test_conformal_invalid_mode_raises():
    plugin, held = _fitted_plugin_and_held()
    cfg = _physical_config_with_conformal(mode="banana")
    with pytest.raises(ValueError):
        resolve_threshold(
            cfg["uq"]["screening"], plugin, held, "expected_abs_p95",
            dtype=torch.float64, seed=0, config=cfg,
        )


def test_script_conformal_flag_runs_and_reports(tmp_path):
    cfg = _tiny_screening_config()
    cfg["_config_path"] = "tiny.yaml"
    args = pbs.argparse.Namespace(
        budget=1.0e-8, units="m/s^2", scoring="expected_abs_p95", max_rerun_fraction=None,
        conformal=True,
    )
    pbs._configure_physical_budget(cfg, args)
    result = pbs.run_and_write(cfg, out_dir=tmp_path / "pb")
    data = json.loads((tmp_path / "pb" / "physical_budget_screening.json").read_text())
    assert data["conformal"]["enabled"] is True
    assert data["conformal"]["scale"] is not None
    assert data["conformal"]["threshold_model_units_raw"] == pytest.approx(1.0e-2)
