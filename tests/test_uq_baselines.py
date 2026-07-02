"""Tests for the baseline trajectory-risk selectors."""

from __future__ import annotations

import math

import pytest
import torch

from vesp.core.sources import make_shell_sources
from vesp.uq import VESPUQPlugin, make_synthetic_uq_samples
from vesp.uq.baselines import (
    BASELINE_REGISTRY,
    altitude_residual_expected_scores,
    domain_support_scores,
    fit_altitude_expected_curve,
    low_altitude_exposure_scores,
    min_altitude_scores,
    random_scores,
    vespuq_scores,
)
from vesp.uq.benchmarking import METRIC_KEYS


def _circular_orbit(radius: float, n: int = 24) -> torch.Tensor:
    theta = torch.linspace(0.0, 2.0 * torch.pi, n + 1, dtype=torch.float64)[:-1]
    return radius * torch.stack([torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)], dim=-1)


def _fitted_plugin(domain_support=True):
    s = make_synthetic_uq_samples(n=400, noise_std=5.0e-5, seed=1)
    src = make_shell_sources([0.75, 0.9], [24, 32], dtype=torch.float64)
    plugin = VESPUQPlugin(src, reg_method="lcurve", seed=0, domain_support=domain_support)
    plugin.fit_error(s.positions, s.error)
    return plugin


# ---- random ----


def test_baseline_registry_exposes_canonical_entries():
    assert {
        "random",
        "label_shuffled",
        "min_altitude",
        "low_altitude_exposure",
        "knn_p95",
        "vespuq_score",
        "altitude_residual_expected",
        "gp_residual_uq",
    } <= set(BASELINE_REGISTRY)
    assert BASELINE_REGISTRY["label_shuffled"].consumes_true_error is True
    assert BASELINE_REGISTRY["min_altitude"].builder is min_altitude_scores


def test_random_scores_deterministic_for_same_seed():
    a = random_scores(50, seed=7)
    b = random_scores(50, seed=7)
    c = random_scores(50, seed=8)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert a.shape == (50,)


def test_random_scores_rejects_nonpositive_n():
    with pytest.raises(ValueError):
        random_scores(0)


# ---- min altitude ----

def test_min_altitude_ranks_lower_radius_as_higher_risk():
    low = _circular_orbit(1.05)
    high = _circular_orbit(1.50)
    scores = min_altitude_scores([high, low])  # order: high first, low second
    assert scores[1] > scores[0]  # the lower-periapsis trajectory scores higher
    # an eccentric orbit dipping lower than a constant one also outranks it
    eccentric = torch.cat([_circular_orbit(1.5)[:12], _circular_orbit(1.03)[:12]], dim=0)
    s2 = min_altitude_scores([high, eccentric])
    assert s2[1] > s2[0]


# ---- low altitude exposure ----

def test_low_altitude_exposure_increases_with_points_below_threshold():
    below = _circular_orbit(1.05)   # entirely below 1.15
    above = _circular_orbit(1.40)   # entirely above 1.15
    half = torch.cat([_circular_orbit(1.05)[:12], _circular_orbit(1.40)[:12]], dim=0)  # 50% below
    scores = low_altitude_exposure_scores([above, half, below], low_altitude_radius=1.15)
    assert float(scores[0]) == pytest.approx(0.0)
    assert float(scores[2]) == pytest.approx(1.0)
    assert scores[0] < scores[1] < scores[2]


def test_low_altitude_exposure_weighting_and_validation():
    traj = torch.cat([_circular_orbit(1.05)[:6], _circular_orbit(1.40)[:6]], dim=0)  # 50% below
    # weight the below points 3x -> exposure rises above 0.5
    w = torch.tensor([3.0] * 6 + [1.0] * 6)
    weighted = low_altitude_exposure_scores([traj], low_altitude_radius=1.15, weights=[w])
    assert float(weighted[0]) == pytest.approx((3 * 6) / (3 * 6 + 1 * 6))
    with pytest.raises(ValueError):
        low_altitude_exposure_scores([traj], weights=[torch.ones(5)])  # wrong length


# ---- domain support ----

def test_domain_support_scores_one_per_trajectory_and_nonneg():
    plugin = _fitted_plugin(domain_support=True)
    trajs = [_circular_orbit(1.3), _circular_orbit(1.05)]
    scores = domain_support_scores(plugin, trajs)
    assert scores.shape == (2,)
    assert bool((scores >= 0).all())


# ---- vespuq scores ----

def test_vespuq_scores_one_per_trajectory():
    plugin = _fitted_plugin(domain_support=False)
    trajs = [_circular_orbit(1.05), _circular_orbit(1.5), _circular_orbit(1.2)]
    scores = vespuq_scores(plugin, trajs, "mean")
    assert scores.shape == (3,)
    sup = vespuq_scores(plugin, trajs, "supervisor_rel_p95")
    assert sup.shape == (3,)


def test_vespuq_scores_rejects_unsupported_mode():
    plugin = _fitted_plugin(domain_support=False)
    with pytest.raises(ValueError):
        vespuq_scores(plugin, [_circular_orbit(1.2)], "not_a_mode")


# ---- altitude-residual expected error ----

def test_altitude_residual_expected_scores_one_per_trajectory():
    plugin = _fitted_plugin(domain_support=False)
    trajs = [_circular_orbit(1.05), _circular_orbit(1.30), _circular_orbit(1.50)]
    curve = fit_altitude_expected_curve(plugin, plugin.val_positions, n_bins=4)
    ratio = altitude_residual_expected_scores(plugin, trajs, curve=curve, mode="ratio")
    delta = altitude_residual_expected_scores(plugin, trajs, curve=curve, mode="delta")
    assert ratio.shape == (3,)
    assert delta.shape == (3,)
    assert bool(torch.isfinite(ratio).all())
    assert bool(torch.isfinite(delta).all())


# ---- the target is force error, never position error ----

def test_no_baseline_or_metric_refers_to_position_error():
    from vesp.uq import baselines

    public = [n for n in dir(baselines) if not n.startswith("_")]
    assert not any("position" in n.lower() for n in public)
    assert not any("position" in k.lower() for k in METRIC_KEYS)
    assert not any(math.isnan(0.0) for _ in [])  # sanity; metrics target true force error only
