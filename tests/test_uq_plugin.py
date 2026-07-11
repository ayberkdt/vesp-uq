"""Tests for the surrogate-agnostic VESP-UQ plugin (fit / predict / score)."""

from __future__ import annotations

import pytest
import torch

from vesp.core.operators import build_acceleration_operator
from vesp.core.sources import make_shell_sources
from vesp.uq import VESPUQPlugin


def _query_shell(n: int, r_lo: float, r_hi: float, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(n, 3, generator=g, dtype=torch.float64)
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)
    radii = (r_lo + (r_hi - r_lo) * torch.rand(n, generator=g, dtype=torch.float64)).unsqueeze(-1)
    return dirs * radii


def _plugin(**kw) -> VESPUQPlugin:
    sources = make_shell_sources([0.8], 64, dtype=torch.float64)
    defaults = dict(reg_method="fixed", lambda_l2=1.0e-8, noise_model="homoscedastic", val_fraction=0.25, seed=0)
    defaults.update(kw)
    return VESPUQPlugin(sources, **defaults)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        _plugin().predict_uncertainty(_query_shell(10, 1.1, 1.5))


def test_mean_error_reshape_matches_operator_times_posterior_mean():
    # internal consistency: predict_uncertainty's (N,3) mean_error must equal A @ mu reshaped
    positions = _query_shell(300, 1.05, 1.6)
    plugin = _plugin()
    sources = make_shell_sources([0.8], 64, dtype=torch.float64)
    sigma_true = 0.01 * torch.randn(sources.n_sources, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    A = build_acceleration_operator(positions, sources, eps=0.0, sign=1.0)
    error_flat = A @ sigma_true
    error = error_flat.reshape(3, positions.shape[0]).transpose(0, 1)  # (N,3)

    plugin.fit_error(positions, error)
    pred = plugin.predict_uncertainty(positions)
    mu = plugin.posterior.mean
    expected = (A @ mu).reshape(3, positions.shape[0]).transpose(0, 1)
    assert pred.mean_error.shape == (positions.shape[0], 3)
    assert torch.allclose(pred.mean_error, expected, atol=1.0e-10)


def test_recovers_in_span_error_field_with_small_lambda():
    # error generated exactly by the source model -> ridge mean should fit it well
    positions = _query_shell(600, 1.05, 1.6, seed=1)
    sources = make_shell_sources([0.8], 48, dtype=torch.float64)
    sigma_true = 0.02 * torch.randn(sources.n_sources, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    A = build_acceleration_operator(positions, sources, eps=0.0, sign=1.0)
    error = (A @ sigma_true).reshape(3, positions.shape[0]).transpose(0, 1)

    plugin = VESPUQPlugin(sources, reg_method="fixed", lambda_l2=1.0e-10, noise_model="homoscedastic", seed=2)
    plugin.fit_error(positions, error)
    pred = plugin.predict_uncertainty(positions)
    rel = torch.linalg.norm(pred.mean_error - error) / torch.linalg.norm(error)
    assert float(rel) < 0.05


def test_predict_shapes_and_nonnegative_uncertainty():
    positions = _query_shell(200, 1.05, 1.6, seed=4)
    error = 1.0e-3 * torch.randn(200, 3, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    plugin = _plugin().fit_error(positions, error)
    pred = plugin.predict_uncertainty(positions)
    assert pred.sigma.shape == (200,)
    assert pred.std_components.shape == (200, 3)
    assert torch.all(pred.sigma >= 0)
    assert torch.all(pred.sigma >= pred.epistemic_sigma - 1.0e-12)  # total >= epistemic
    assert torch.allclose(pred.risk_score, pred.sigma)
    # R2WP-5: canonical RMS accessor aliases expected_error, and the quantity really is the RMS
    # combination sqrt(||mu||^2 + tr(Sigma)) (upper bound on E||e||), not an expected norm.
    assert pred.rms_predictive_error is pred.expected_error
    recon = torch.sqrt(pred.mean_error_magnitude**2 + pred.sigma**2)
    assert torch.allclose(pred.expected_error, recon, rtol=1.0e-12, atol=0.0)


def test_epistemic_uncertainty_grows_at_low_altitude_ood():
    # train only on mid/high altitude; the posterior must be more (epistemic) uncertain where it
    # extrapolates downward -- the core value proposition of the layer.
    train = _query_shell(500, 1.25, 1.6, seed=10)
    sources = make_shell_sources([0.85], 80, dtype=torch.float64)
    sigma_true = 0.02 * torch.randn(sources.n_sources, generator=torch.Generator().manual_seed(11), dtype=torch.float64)
    A_tr = build_acceleration_operator(train, sources, eps=0.0, sign=1.0)
    err_tr = (A_tr @ sigma_true).reshape(3, train.shape[0]).transpose(0, 1)
    plugin = VESPUQPlugin(sources, reg_method="lcurve", noise_model="homoscedastic", seed=12)
    plugin.fit_error(train, err_tr)

    low = _query_shell(200, 1.03, 1.10, seed=13)
    high = _query_shell(200, 1.40, 1.55, seed=14)
    epi_low = float(plugin.predict_uncertainty(low).epistemic_sigma.mean())
    epi_high = float(plugin.predict_uncertainty(high).epistemic_sigma.mean())
    assert epi_low > epi_high


def test_score_trajectory_aggregates_consistently():
    positions = _query_shell(400, 1.04, 1.6, seed=20)
    error = 1.0e-3 * torch.randn(400, 3, generator=torch.Generator().manual_seed(21), dtype=torch.float64)
    plugin = _plugin(noise_model="heteroscedastic").fit_error(positions, error)
    traj = _query_shell(60, 1.05, 1.5, seed=22)
    score = plugin.score_trajectory(traj, scoring="max")
    assert score.n_points == 60
    assert score.max_sigma >= score.mean_sigma > 0.0
    assert score.risk_score == pytest.approx(score.max_sigma)
    assert score.min_radius <= score.mean_radius


def test_from_config_builds_multishell_sources():
    cfg = {
        "model": {"type": "multishell", "shell_alphas": [0.7, 0.9], "n_sources_per_shell": [32, 48]},
        "kernel": {"eps": 0.0, "acceleration_sign": 1.0, "source_chunk_size": 256},
        "uq": {"regularization": {"method": "lcurve"}, "noise_model": "heteroscedastic"},
    }
    plugin = VESPUQPlugin.from_config(cfg)
    assert plugin.sources.n_sources == 80
    assert plugin.reg_method == "lcurve"
    assert plugin.noise_model == "heteroscedastic"


def test_from_config_accepts_binned_noise_and_calibrated_supervisor_options():
    cfg = {
        "model": {"n_source": 16},
        "uq": {
            "regularization": {"method": "fixed", "lambda_l2": 1.0e-4},
            "noise_model": "altitude_binned",
            "noise": {"altitude_bins": 6},
            "risk": {
                "scoring": "calibrated_supervisor_p95",
                "calibrated_supervisor": {"enabled": True, "domain_weight": [0.0]},
            },
            "conformal": {"by_region": True, "min_region_n": 4},
        },
    }
    plugin = VESPUQPlugin.from_config(cfg)
    assert plugin.noise_model == "altitude_binned"
    assert plugin.altitude_noise_bins == 6
    assert plugin.calibrate_supervisor is True
    assert plugin.risk_scoring == "calibrated_supervisor_p95"
    assert plugin.conformal_by_region is True
    assert plugin.conformal_min_region_n == 4
