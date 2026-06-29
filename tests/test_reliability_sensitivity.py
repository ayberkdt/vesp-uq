"""Unit + smoke tests for raw-vs-calibrated reliability (WP7) and sensitivity (WP8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vesp.common.config import load_config
from vesp.uq.calibration_reliability import NOMINAL_LEVELS, _half_width, reliability_run
from vesp.uq.sensitivity import _scaled_source_counts, source_geometry_sweep

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "vespuq" / "vespuq_smoke.yaml"


def test_half_width_known_values():
    assert abs(_half_width(0.90) - 1.6448536269514722) < 1e-6
    assert abs(_half_width(0.95) - 1.959963984540054) < 1e-6


def test_scaled_source_counts_preserves_proportions():
    counts = _scaled_source_counts([384, 512, 384], 640)
    assert sum(counts) == pytest.approx(640, abs=2)
    assert counts[1] > counts[0]  # middle shell stays the largest


@pytest.mark.skipif(not SMOKE_CONFIG.exists(), reason="smoke config missing")
def test_reliability_run_smoke():
    cfg = load_config(str(SMOKE_CONFIG))
    cfg["_config_path"] = str(SMOKE_CONFIG)
    run = reliability_run(cfg, seed=0)
    assert "all" in run["regions"]
    allr = run["regions"]["all"]
    for level in NOMINAL_LEVELS:
        p = int(round(level * 100))
        assert f"picp_{p}_raw" in allr
        assert f"picp_{p}_calibrated" in allr
    # calibrated z-std equals raw z-std divided by the conformal scale
    assert abs(allr["z_std_calibrated"] - allr["z_std_raw"] / run["scale"]) < 1e-9


@pytest.mark.skipif(not SMOKE_CONFIG.exists(), reason="smoke config missing")
def test_source_geometry_sweep_smoke():
    cfg = load_config(str(SMOKE_CONFIG))
    cfg["_config_path"] = str(SMOKE_CONFIG)
    rows = source_geometry_sweep(cfg, seed=0, n_sources_targets=[56, 112])
    assert len(rows) == 2
    assert all("rel_accel_rmse" in r and "effective_source_count" in r for r in rows)
    assert rows[0]["n_sources_total"] != rows[1]["n_sources_total"]
