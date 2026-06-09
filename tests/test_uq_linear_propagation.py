"""Tests for the exploratory linearized force-error covariance propagator.

The headline correctness check is that, in the linear regime, the deterministic STM covariance
matches the Monte Carlo sampler's empirical covariance (same static-force-error model).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vesp.core.sources import make_shell_sources
from vesp.uq import VESPUQPlugin, make_synthetic_uq_samples
from vesp.uq.linear_propagation import LinearForceErrorCovariancePropagator
from vesp.uq.propagation import VESPMonteCarloPropagator


def _fitted_plugin(seed: int = 0):
    samples = make_synthetic_uq_samples(n=400, noise_std=1.0e-4, seed=1)
    src = make_shell_sources([0.75, 0.9], [24, 32], dtype=torch.float64)
    plugin = VESPUQPlugin(src, reg_method="lcurve", noise_model="heteroscedastic", seed=seed)
    plugin.fit_error(samples.positions, samples.error)
    return plugin


def test_shapes_and_zero_initial_covariance():
    plugin = _fitted_plugin()
    prop = LinearForceErrorCovariancePropagator(plugin, dt_s=30.0, mu=1.0)
    y0 = np.array([1.2, 0.0, 0.0, 0.0, 0.9, 0.0])
    res = prop.propagate(y0, duration_s=600.0, output_dt_s=120.0)
    T = res.times.shape[0]
    assert res.states.shape == (T, 6)
    assert res.covariances.shape == (T, 6, 6)
    assert res.position_sigma.shape == (T,)
    # covariance starts at zero (J(0) = 0)
    assert np.allclose(res.covariances[0], 0.0)
    assert res.position_sigma[0] == 0.0


def test_covariance_symmetric_psd_and_grows():
    plugin = _fitted_plugin()
    prop = LinearForceErrorCovariancePropagator(plugin, dt_s=30.0, mu=1.0)
    res = prop.propagate(np.array([1.3, 0.0, 0.0, 0.0, 0.85, 0.0]), duration_s=600.0, output_dt_s=120.0)
    for P in res.covariances:
        assert np.allclose(P, P.T, atol=1.0e-12)
        w = np.linalg.eigvalsh(P)
        assert w.min() >= -1.0e-12  # PSD up to round-off
    # uncertainty accumulates from zero
    assert res.position_sigma[-1] > res.position_sigma[1] > 0.0


def test_determinism():
    plugin = _fitted_plugin()
    y0 = np.array([1.25, 0.0, 0.0, 0.0, 0.88, 0.0])
    r1 = LinearForceErrorCovariancePropagator(plugin, dt_s=30.0).propagate(y0, 480.0, 120.0)
    r2 = LinearForceErrorCovariancePropagator(plugin, dt_s=30.0).propagate(y0, 480.0, 120.0)
    assert np.allclose(r1.covariances, r2.covariances)
    assert np.allclose(r1.position_sigma, r2.position_sigma)


def test_linear_covariance_matches_monte_carlo_drift_regime():
    # mu=0, v0=0 -> pure drift under a (nearly) constant sampled force-error field, where the
    # dynamics are linear in the source strengths, so the STM covariance equals the MC sample
    # covariance up to sampling noise.
    plugin = _fitted_plugin()
    y0 = np.array([1.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    duration, out_dt, dt = 300.0, 300.0, 30.0

    lin = LinearForceErrorCovariancePropagator(plugin, dt_s=dt, mu=0.0).propagate(y0, duration, out_dt)

    mc = VESPMonteCarloPropagator(plugin, n_samples=8000, dt_s=dt, mu=0.0, seed=0)
    _, Y = mc.propagate(y0, duration_s=duration, output_dt_s=out_dt)
    final_pos = Y[-1, :, :3]  # (N, 3)
    mc_pos_sigma = float(np.sqrt(np.cov(final_pos.T).trace()))

    assert lin.position_sigma[-1] == pytest.approx(mc_pos_sigma, rel=0.1)


def test_custom_base_accel_fn_runs_with_fd_jacobian():
    plugin = _fitted_plugin()
    # constant base field -> zero gravity gradient via finite differences; should run and stay PSD
    prop = LinearForceErrorCovariancePropagator(
        plugin, dt_s=30.0, base_accel_fn=lambda r: torch.zeros_like(r)
    )
    res = prop.propagate(np.array([1.3, 0.0, 0.0, 0.0, 0.0, 0.0]), duration_s=240.0, output_dt_s=120.0)
    assert np.isfinite(res.covariances).all()
    assert res.position_sigma[-1] > 0.0


def test_rejects_bad_initial_state():
    plugin = _fitted_plugin()
    prop = LinearForceErrorCovariancePropagator(plugin, dt_s=30.0)
    with pytest.raises(ValueError):
        prop.propagate(np.zeros(5), duration_s=120.0, output_dt_s=60.0)


def test_requires_fitted_plugin():
    src = make_shell_sources([0.8], [16], dtype=torch.float64)
    plugin = VESPUQPlugin(src, seed=0)  # not fitted
    with pytest.raises(RuntimeError):
        LinearForceErrorCovariancePropagator(plugin)


def test_honest_caveat_in_module():
    import inspect

    import vesp.uq.linear_propagation as mod

    src = inspect.getsource(mod).lower()
    assert "not validated" in src
