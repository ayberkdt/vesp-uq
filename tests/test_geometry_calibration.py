"""Unit + smoke tests for the geometry x calibration sweep (E8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vesp.common.config import load_config
from vesp.uq.geometry_calibration import (
    GEOMETRIES,
    _aggregate,
    _counts_for,
    _verdict,
    geometry_calib_run,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "vespuq" / "vespuq_smoke.yaml"


def test_counts_scale_to_base_total():
    spec = GEOMETRIES["five_shell"]
    counts = _counts_for(spec, 1280)
    assert len(counts) == 5
    assert abs(sum(counts) - 1280) <= 5
    # single-shell geometry takes the whole budget
    assert _counts_for(GEOMETRIES["single_surface"], 1280) == [1280]


def test_verdict_picks_closest_to_one():
    agg = {
        ("L60", "baseline"): {"low_z_std": {"mean": 0.60}, "low_picp_90": {"mean": 0.88},
                              "rel_accel_rmse": {"mean": 0.5}, "weight_mode": "surface_area"},
        ("L60", "surface_dense"): {"low_z_std": {"mean": 0.95}, "low_picp_90": {"mean": 0.90},
                                   "rel_accel_rmse": {"mean": 0.5}, "weight_mode": "surface_area"},
    }
    v = _verdict(agg)
    assert v["L60"]["best_geometry"] == "surface_dense"
    assert v["L60"]["geometry_improves_low_calibration"] is True  # 0.95 much closer to 1 than 0.60


def test_verdict_reports_no_improvement():
    agg = {
        ("L60", "baseline"): {"low_z_std": {"mean": 0.60}, "low_picp_90": {"mean": 0.88},
                              "rel_accel_rmse": {"mean": 0.5}, "weight_mode": "surface_area"},
        ("L60", "deep"): {"low_z_std": {"mean": 0.62}, "low_picp_90": {"mean": 0.88},
                          "rel_accel_rmse": {"mean": 0.4}, "weight_mode": "surface_area"},
    }
    v = _verdict(agg)
    assert v["L60"]["geometry_improves_low_calibration"] is False  # 0.62 vs 0.60: no material gain


@pytest.mark.skipif(not SMOKE_CONFIG.exists(), reason="smoke config missing")
def test_geometry_calib_run_smoke():
    cfg = load_config(str(SMOKE_CONFIG))
    cfg["_config_path"] = str(SMOKE_CONFIG)
    row = geometry_calib_run(cfg, seed=0, geometry="surface_dense")
    assert row["geometry"] == "surface_dense"
    assert "low_z_std" in row and "rel_accel_rmse" in row
    assert row["weight_mode"] == "surface_area"
    agg = _aggregate([row])
    assert (row["band"], "surface_dense") in agg
