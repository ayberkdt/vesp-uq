"""Unit + smoke tests for trajectory families (WP9), drift horizon (WP10), physical budget (WP11)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from vesp.common.config import load_config
from vesp.uq.drift_horizon import HORIZON_PERIODS, _aggregate, _build_initial_ensemble, _horizon_metrics
from vesp.uq.physical_budget_status import _to_model_threshold, physical_budget_status_run
from vesp.uq.physical_units import resolve_acceleration_scale
from vesp.uq.trajectory_families import FAMILIES, family_descriptor, generate_family

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "vespuq" / "vespuq_smoke.yaml"


# ----------------------------- WP9 trajectory families ----------------------------- #
def test_family_inclination_bands():
    polar = generate_family("polar", n_orbits=64, n_points=40, seed=0)
    assert float(polar.inclination_deg.min()) >= 80.0 - 1e-6
    assert float(polar.inclination_deg.max()) <= 100.0 + 1e-6
    eq = generate_family("equatorial", n_orbits=64, n_points=40, seed=0)
    assert float(eq.inclination_deg.max()) <= 15.0 + 1e-6


def test_family_ood_low_alt_below_support_edge():
    ood = generate_family("ood_low_alt", n_orbits=64, n_points=40, seed=0)
    min_radii = torch.stack([torch.linalg.norm(t, dim=-1).min() for t in ood.trajectories])
    assert float(min_radii.min()) < 1.03  # periapsis at/below the training-support edge


def test_family_descriptor_keys_and_descent_arc():
    fam = generate_family("descent_arc", n_orbits=16, n_points=50, seed=1)
    d = family_descriptor(fam)
    assert {"family", "n_trajectories", "min_radius_low", "inclination_deg_high"} <= set(d)
    assert all(t.shape == (50, 3) for t in fam.trajectories)
    assert set(FAMILIES) >= {"polar", "equatorial", "ood_low_alt", "descent_arc"}


# ----------------------------- WP10 drift horizon ----------------------------- #
def test_build_initial_ensemble_shapes():
    paths, states, period, a = _build_initial_ensemble(10, 30, seed=0, dtype=torch.float64)
    assert len(paths) == 10
    assert states.shape == (10, 6)
    assert period.shape == (10,)
    assert bool(torch.isfinite(states).all())
    # Kepler third law: period ~ 2 pi a^1.5 for mu=1
    assert abs(float(period[0]) - 2 * math.pi * float(a[0]) ** 1.5) < 1e-6


def test_horizon_metrics_and_aggregate():
    n = 50
    torch.manual_seed(0)
    fr = torch.rand(n)
    te = fr + 0.1 * torch.randn(n)
    run = {"band": "L60", "seed": 0, "n_orbits": n, "force_risk": fr, "true_error": te,
           "dispersion": {k: torch.rand(n) for k in HORIZON_PERIODS}}
    rows = _horizon_metrics(run)
    assert any(r["diagnostic"] == "force_error_ranking" for r in rows)
    assert sum(r["diagnostic"] == "drift_ranking" for r in rows) == len(HORIZON_PERIODS)
    agg = _aggregate(rows)
    assert ("L60", "force_error_ranking", 0) in agg


# ----------------------------- WP11 physical budget ----------------------------- #
def test_to_model_threshold_conversion():
    cfg = {"body": {"acceleration_units": "model_normalized_accel", "acceleration_scale_m_s2": 1.0e-6}}
    scale = resolve_acceleration_scale(cfg)
    # 1e-8 m/s^2 / (1e-6 m/s^2 per unit) = 1e-2 model units
    assert abs(_to_model_threshold(1.0e-8, scale) - 1.0e-2) < 1e-12


def test_physical_budget_not_activated_without_scale():
    cfg = {"data": {"path": "data/x_L60.csv"}, "body": {}}
    out = physical_budget_status_run(cfg, seed=0, tolerance_m_s2=1.0e-8)
    assert out["activated"] is False
    assert "no physical scaling metadata" in out["reason"]


@pytest.mark.skipif(not SMOKE_CONFIG.exists(), reason="smoke config missing")
def test_physical_budget_activated_smoke():
    cfg = load_config(str(SMOKE_CONFIG))
    cfg["_config_path"] = str(SMOKE_CONFIG)
    out = physical_budget_status_run(cfg, seed=0, tolerance_m_s2=1.0e-8)
    assert out["activated"] is True
    assert out["n_trajectories"] > 0
    assert out["n_flagged"] == out["true_positives"] + out["false_positives"]
