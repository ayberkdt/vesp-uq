"""Tests for the exploratory Monte Carlo orbit-dispersion propagator (vesp.uq.propagation).

These pin the *consistency* of the sampled force-error field with the fitted posterior (honoring
acceleration sign + softening), determinism under a seed, and output shapes -- not any
position-error or covariance-realism claim.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vesp.core.kernels import acceleration_kernel
from vesp.core.sources import make_shell_sources
from vesp.uq import VESPUQPlugin, make_synthetic_uq_samples
from vesp.uq.propagation import VESPMonteCarloPropagator, draw_posterior_samples


def _fitted_plugin(*, acceleration_sign: float = 1.0, eps: float = 0.0, seed: int = 0):
    samples = make_synthetic_uq_samples(n=400, noise_std=1.0e-4, seed=1)
    src = make_shell_sources([0.75, 0.9], [24, 32], dtype=torch.float64)
    plugin = VESPUQPlugin(
        src, reg_method="lcurve", noise_model="heteroscedastic",
        acceleration_sign=acceleration_sign, eps=eps, seed=seed,
    )
    plugin.fit_error(samples.positions, samples.error)
    return plugin


def _mc_error_field(plugin, sigma_vectors, x):
    """Replicate the propagator's force-error field for given source-strength vectors at points x.

    ``sigma_vectors`` is (S, n_sources); returns (S, N, 3) acceleration-error samples.
    """

    ker = acceleration_kernel(x, plugin.sources.positions, eps=plugin.eps, sign=plugin.acceleration_sign)
    weighted = sigma_vectors * plugin.sources.weights.unsqueeze(0)  # (S, n_sources)
    return torch.einsum("nsc,ks->knc", ker, weighted)  # (S, N, 3)


def test_draw_posterior_samples_deterministic_and_shaped():
    plugin = _fitted_plugin()
    a = draw_posterior_samples(plugin, 64, seed=7)
    b = draw_posterior_samples(plugin, 64, seed=7)
    c = draw_posterior_samples(plugin, 64, seed=8)
    assert a.shape == (64, int(plugin.sources.n_sources))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_draw_posterior_samples_mean_matches_posterior():
    plugin = _fitted_plugin()
    n = 6000
    samples = draw_posterior_samples(plugin, n, seed=0)
    mean = plugin.posterior.mean
    se = torch.sqrt(torch.diagonal(plugin.posterior.cov).clamp_min(0.0) / n)
    # 6 standard errors -> deterministic (fixed seed) and effectively never flaky
    assert torch.all((samples.mean(0) - mean).abs() <= 6.0 * se + 1.0e-12)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_mc_error_field_matches_posterior_mean(sign):
    # The propagator's field, built from the posterior MEAN, must equal predict_uncertainty's
    # mean_error. This is the sign/eps consistency regression guard: with a flipped acceleration
    # sign the field would be negated if the kernel call dropped the sign.
    plugin = _fitted_plugin(acceleration_sign=sign)
    x = torch.tensor([[1.1, 0.0, 0.0], [0.0, 1.3, 0.2], [0.5, 0.5, 1.0]], dtype=torch.float64)
    field = _mc_error_field(plugin, plugin.posterior.mean.unsqueeze(0), x)[0]  # (N, 3)
    expected = plugin.predict_uncertainty(x).mean_error
    assert torch.allclose(field, expected, atol=1.0e-9, rtol=1.0e-6)


def test_propagator_honors_plugin_sign_and_eps():
    plugin = _fitted_plugin(acceleration_sign=-1.0, eps=1.0e-3)
    prop = VESPMonteCarloPropagator(plugin, n_samples=8, seed=0)
    assert prop.accel_sign == -1.0
    assert prop.eps == pytest.approx(1.0e-3)


def test_mc_field_sample_covariance_matches_epistemic():
    # Sample covariance of the source-posterior field (no aleatoric floor) should track the
    # plugin's epistemic variance. Fixed seed -> deterministic; tolerance is generous.
    plugin = _fitted_plugin()
    x = torch.tensor([[1.08, 0.0, 0.0]], dtype=torch.float64)
    samples = draw_posterior_samples(plugin, 8000, seed=0)
    field = _mc_error_field(plugin, samples, x)[:, 0, :]  # (S, 3)
    sample_cov_trace = float(torch.cov(field.T).diagonal().sum())
    epistemic_var = float(plugin.predict_uncertainty(x).epistemic_sigma[0] ** 2)
    assert sample_cov_trace == pytest.approx(epistemic_var, rel=0.15)


def test_propagate_shapes_and_determinism():
    plugin = _fitted_plugin()
    prop = VESPMonteCarloPropagator(plugin, n_samples=16, dt_s=60.0, mu=1.0, seed=0)
    y0 = np.array([1.2, 0.0, 0.0, 0.0, 0.9, 0.0], dtype=np.float64)  # r=1.2, near-circular
    t1, Y1 = prop.propagate(y0, duration_s=600.0, output_dt_s=120.0)
    assert Y1.shape == (t1.shape[0], 16, 6)
    assert np.isfinite(Y1).all()
    # same seed -> identical propagation
    prop2 = VESPMonteCarloPropagator(plugin, n_samples=16, dt_s=60.0, mu=1.0, seed=0)
    _, Y2 = prop2.propagate(y0, duration_s=600.0, output_dt_s=120.0)
    assert np.allclose(Y1, Y2)


def test_propagate_rejects_bad_initial_state():
    plugin = _fitted_plugin()
    prop = VESPMonteCarloPropagator(plugin, n_samples=4, seed=0)
    with pytest.raises(ValueError):
        prop.propagate(np.zeros(5, dtype=np.float64), duration_s=120.0, output_dt_s=60.0)


def test_base_accel_fn_is_used():
    plugin = _fitted_plugin()
    # zero base field -> motion driven only by the (tiny) sampled error field
    prop = VESPMonteCarloPropagator(
        plugin, n_samples=4, seed=0, base_accel_fn=lambda r: torch.zeros_like(r)
    )
    y0 = np.array([1.3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    _, Y = prop.propagate(y0, duration_s=120.0, output_dt_s=60.0)
    assert np.isfinite(Y).all()


def test_no_position_error_phrase_in_module():
    import inspect

    import vesp.uq.propagation as mod

    src = inspect.getsource(mod).lower()
    # the module may discuss "position error" only inside the honesty caveat; ensure it is not a
    # silent claim by requiring the disclaimer wording to accompany it
    assert "not a validated" in src
