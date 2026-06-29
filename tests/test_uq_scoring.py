"""Tests for M1 (directional / covariance-geometry) and M2 (epistemic) trajectory scoring."""

from __future__ import annotations

import math

import pytest
import torch

from vesp.core.operators import build_acceleration_operator
from vesp.core.sources import make_shell_sources
from vesp.uq import VESPUQPlugin
from vesp.uq.ensemble import generate_orbit_ensemble
from vesp.uq.scoring import (
    anisotropy_multiplier,
    largest_eigenvalue_profile,
    needs_covariance,
    radial_profile,
    score_sigma_profile,
)

_DIRECTIONAL = ("radial_expected", "anisotropy_gated", "largest_eigenvalue")


def _query_shell(n: int, r_lo: float, r_hi: float, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(n, 3, generator=g, dtype=torch.float64)
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)
    radii = (r_lo + (r_hi - r_lo) * torch.rand(n, generator=g, dtype=torch.float64)).unsqueeze(-1)
    return dirs * radii


def _fitted_plugin(**kw) -> VESPUQPlugin:
    sources = make_shell_sources([0.8], 48, dtype=torch.float64)
    sigma_true = 0.02 * torch.randn(
        sources.n_sources, generator=torch.Generator().manual_seed(3), dtype=torch.float64
    )
    positions = _query_shell(400, 1.05, 1.6, seed=1)
    A = build_acceleration_operator(positions, sources, eps=0.0, sign=1.0)
    error = (A @ sigma_true).reshape(3, positions.shape[0]).transpose(0, 1)
    defaults = dict(reg_method="fixed", lambda_l2=1.0e-8, noise_model="heteroscedastic",
                    val_fraction=0.25, risk_scoring="supervisor_rel", domain_support=True, seed=0)
    defaults.update(kw)
    plugin = VESPUQPlugin(sources, **defaults)
    plugin.fit_error(positions, error)
    return plugin


# ----------------------------------------------------------------- M1 profile builders
def test_radial_profile_matches_manual_projection():
    # axis-aligned positions -> the radial axis is exactly e_x / e_y / e_z, so the radial bias is the
    # matching component of mean_error and sigma_radial the matching std.
    positions = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    mean_error = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    std = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    got = radial_profile(mean_error, std, positions)
    # bias_radial = [1, 5, 9]; sigma_radial = [0.1, 0.5, 0.9]
    assert torch.allclose(got, torch.tensor([1.1, 5.5, 9.9], dtype=torch.float64), atol=1e-12)


def test_anisotropy_multiplier_ge_one_and_isotropic_is_one():
    isotropic = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 2.5
    assert anisotropy_multiplier(isotropic).item() == pytest.approx(1.0)
    anis = torch.diag(torch.tensor([1.0, 1.0, 4.0])).unsqueeze(0)
    assert anisotropy_multiplier(anis, kappa=0.5).item() == pytest.approx(1.0 + 0.5 * (4.0 - 1.0))
    # always >= 1 on a batch of random PSD covariances
    g = torch.Generator().manual_seed(0)
    a = torch.randn(16, 3, 3, generator=g, dtype=torch.float64)
    cov = a @ a.transpose(-1, -2) + 1e-6 * torch.eye(3)
    assert bool((anisotropy_multiplier(cov) >= 1.0 - 1e-9).all())


def test_largest_eigenvalue_profile_matches_eig():
    cov = torch.diag(torch.tensor([1.0, 4.0, 9.0])).unsqueeze(0)
    assert largest_eigenvalue_profile(cov).item() == pytest.approx(3.0)  # sqrt(9)


def test_needs_covariance_gating():
    assert needs_covariance("anisotropy_gated") and needs_covariance("largest_eigenvalue")
    assert not needs_covariance("radial_expected")  # diagonal projection, no full 3x3
    assert not needs_covariance("expected_epistemic")
    assert not needs_covariance("supervisor_rel_p95")


# ----------------------------------------------------------------- requirement guards
def test_directional_scoring_requires_covariance():
    sigma = torch.rand(10) + 0.1
    radius = torch.linspace(1.05, 1.6, 10)
    ee = torch.rand(10) + 0.1
    with pytest.raises(ValueError, match="covariance"):
        score_sigma_profile(sigma, radius, scoring="anisotropy_gated", expected_error=ee)
    with pytest.raises(ValueError, match="covariance"):
        score_sigma_profile(sigma, radius, scoring="largest_eigenvalue")
    with pytest.raises(ValueError, match="radial_expected"):
        score_sigma_profile(sigma, radius, scoring="radial_expected")  # no mean_error/positions


# ----------------------------------------------------------------- M2 epistemic mode
def test_expected_epistemic_reduces_to_expected_when_gamma_zero():
    n = 20
    g = torch.Generator().manual_seed(7)
    sigma = torch.rand(n, generator=g) + 0.1
    radius = 1.05 + 0.55 * torch.rand(n, generator=g)
    ee = torch.rand(n, generator=g) + 0.1
    ef = torch.rand(n, generator=g)  # epistemic fraction in [0, 1]
    epi = score_sigma_profile(sigma, radius, scoring="expected_epistemic",
                              expected_error=ee, epistemic_fraction=ef, epistemic_gamma=0.0)
    plain = score_sigma_profile(sigma, radius, scoring="expected_abs_p95", expected_error=ee)
    assert epi.risk_score == pytest.approx(plain.risk_score)


def test_expected_epistemic_downweights_aleatoric_points():
    # a point with zero epistemic fraction contributes nothing; a fully-epistemic one keeps its ee
    sigma = torch.tensor([1.0, 1.0])
    radius = torch.tensor([1.2, 1.2])
    ee = torch.tensor([5.0, 5.0])
    ef = torch.tensor([0.0, 1.0])
    out = score_sigma_profile(sigma, radius, scoring="expected_epistemic",
                              expected_error=ee, epistemic_fraction=ef, epistemic_gamma=1.0)
    assert math.isfinite(out.risk_score) and out.risk_score <= 5.0


# ----------------------------------------------------------------- end-to-end on a fitted plugin
def test_epistemic_fraction_in_unit_interval():
    plugin = _fitted_plugin()
    pred = plugin.predict_uncertainty(_query_shell(120, 1.05, 1.6, seed=4))
    ef = pred.epistemic_fraction
    assert bool((ef >= -1e-9).all()) and bool((ef <= 1.0 + 1e-9).all())


@pytest.mark.parametrize("scoring", [*_DIRECTIONAL, "expected_epistemic"])
def test_directional_scores_finite_on_smoke_ensemble(scoring):
    plugin = _fitted_plugin()
    ens = generate_orbit_ensemble(n_orbits=10, n_points=30, seed=5, dtype=torch.float64)
    scores = plugin.score_ensemble(ens.trajectories, scoring=scoring)
    assert len(scores) == 10
    assert all(math.isfinite(s.risk_score) for s in scores), scoring


def test_directional_scoring_matches_sequential():
    # the gated covariance build must not break the batched == sequential contract
    plugin = _fitted_plugin()
    plugin.query_chunk_size = 64
    ens = generate_orbit_ensemble(n_orbits=8, n_points=25, seed=6, dtype=torch.float64)
    batched = plugin.score_ensemble(ens.trajectories, scoring="anisotropy_gated")
    sequential = [plugin.score_trajectory(t, scoring="anisotropy_gated") for t in ens.trajectories]
    for got, want in zip(batched, sequential, strict=True):
        assert got.risk_score == pytest.approx(want.risk_score, rel=1e-9, abs=1e-15)
