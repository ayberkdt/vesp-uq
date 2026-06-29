"""Unit + smoke tests for the drift-boundary characterization (B)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from vesp.common.config import load_config
from vesp.uq.drift_boundary import HORIZON_PERIODS, _aggregate, _verdict, drift_boundary_run
from vesp.uq.trajectory_families import generate_family

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "vespuq" / "vespuq_smoke.yaml"


def test_family_now_carries_initial_states_and_period():
    fam = generate_family("low_alt_near_circular", n_orbits=12, n_points=40, seed=0)
    assert fam.initial_states is not None and fam.initial_states.shape == (12, 6)
    assert fam.period is not None and fam.period.shape == (12,)
    # Kepler third law for mu=1: T = 2 pi a^1.5, with a = 0.5(r_peri + r_apo)
    a0 = 0.5 * (float(fam.periapsis[0]) + float(fam.apoapsis[0]))
    assert abs(float(fam.period[0]) - 2 * math.pi * a0 ** 1.5) < 1e-6
    # initial position magnitude equals periapsis radius
    r0 = torch.linalg.norm(fam.initial_states[0, :3])
    assert abs(float(r0) - float(fam.periapsis[0])) < 1e-9


def test_verdict_boundary_logic():
    band = "L60"
    # intermediate-horizon peak: weak at 1, strong at 3-6, weak at 12
    agg = {
        (band, "fam", 1): {"spearman_forcerisk_vs_drift": {"mean": 0.05}},
        (band, "fam", 3): {"spearman_forcerisk_vs_drift": {"mean": 0.55}},
        (band, "fam", 6): {"spearman_forcerisk_vs_drift": {"mean": 0.48}},
        (band, "fam", 12): {"spearman_forcerisk_vs_drift": {"mean": 0.05}},
    }
    v = _verdict(agg)
    assert v[(band, "fam")]["predicts_drift"] is True
    assert v[(band, "fam")]["predicting_horizons"] == [3, 6]
    assert v[(band, "fam")]["peak_horizon_periods"] == 3  # max Spearman at horizon 3
    assert v[(band, "fam")]["predicts_drift_up_to_periods"] == 6


def test_verdict_no_prediction():
    agg = {("L60", "fam", k): {"spearman_forcerisk_vs_drift": {"mean": 0.05}} for k in HORIZON_PERIODS}
    v = _verdict(agg)
    assert v[("L60", "fam")]["predicts_drift"] is False


@pytest.mark.skipif(not SMOKE_CONFIG.exists(), reason="smoke config missing")
def test_drift_boundary_run_smoke():
    cfg = load_config(str(SMOKE_CONFIG))
    cfg["_config_path"] = str(SMOKE_CONFIG)
    run = drift_boundary_run(cfg, seed=0, families=["low_alt_near_circular"], n_orbits=6, n_points=40)
    fams = {r["family"] for r in run["rows"]}
    assert fams == {"low_alt_near_circular"}
    assert {r["horizon_periods"] for r in run["rows"]} == set(HORIZON_PERIODS)
    agg = _aggregate(run["rows"])
    assert all("spearman_forcerisk_vs_drift" in v for v in agg.values())
