"""Tests for the N10 ``stm_dispersion`` exploratory diagnostic score.

Two contracts: (1) shape/finiteness/nonnegativity of the score vector; (2) the score uses the
FITTED posterior exactly -- ``P = J Sigma_sigma J^T`` is linear in ``Sigma_sigma``, so scaling
the posterior covariance by ``c`` must scale every dispersion score by exactly ``sqrt(c)``
(the nominal trajectory depends only on the posterior MEAN, which stays fixed).
"""

import numpy as np
import pytest
import torch

from vesp.core.operators import build_acceleration_operator
from vesp.core.sources import make_shell_sources
from vesp.extensions.probabilistic import LinearGaussianPosterior
from vesp.uq.linear_propagation import score_stm_dispersion
from vesp.uq.plugin import VESPUQPlugin
from vesp.uq.scoring import SCORING_FUNCTIONS


def _fitted_plugin() -> VESPUQPlugin:
    sources = make_shell_sources([0.8], 32, dtype=torch.float64)
    sigma_true = 0.02 * torch.randn(
        sources.n_sources, generator=torch.Generator().manual_seed(3), dtype=torch.float64
    )
    g = torch.Generator().manual_seed(1)
    dirs = torch.randn(200, 3, generator=g, dtype=torch.float64)
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)
    positions = dirs * (1.05 + 0.5 * torch.rand(200, 1, generator=g, dtype=torch.float64))
    A = build_acceleration_operator(positions, sources, eps=0.0, sign=1.0)
    error = (A @ sigma_true).reshape(3, positions.shape[0]).transpose(0, 1)
    plugin = VESPUQPlugin(sources, reg_method="fixed", lambda_l2=1.0e-8, noise_model="homoscedastic", seed=0)
    plugin.fit_error(positions, error)
    return plugin


INITIAL_STATES = np.array(
    [
        [1.2, 0.0, 0.0, 0.0, 0.9, 0.0],
        [1.3, 0.0, 0.0, 0.0, 0.85, 0.0],
        [0.0, 1.25, 0.0, -0.88, 0.0, 0.0],
    ]
)


def test_score_shape_finiteness_nonnegativity():
    plugin = _fitted_plugin()
    scores = score_stm_dispersion(plugin, INITIAL_STATES, duration_s=2.0, output_dt_s=0.25)
    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (3,)
    assert scores.dtype == torch.float64
    assert bool(torch.all(torch.isfinite(scores)))
    assert bool(torch.all(scores >= 0.0))
    assert bool(torch.all(scores > 0.0))  # nonzero posterior cov -> dispersion accumulates


def test_score_uses_fitted_posterior_exactly():
    # P = J Sigma J^T is linear in Sigma and J depends only on the (unchanged) posterior mean,
    # so cov * 4 must scale every dispersion score by exactly 2.
    plugin = _fitted_plugin()
    base = score_stm_dispersion(plugin, INITIAL_STATES, duration_s=2.0, output_dt_s=0.25)

    post = plugin.posterior
    plugin.posterior = LinearGaussianPosterior(
        mean=post.mean, cov=4.0 * post.cov, noise_var=post.noise_var, lambda_l2=post.lambda_l2
    )
    scaled = score_stm_dispersion(plugin, INITIAL_STATES, duration_s=2.0, output_dt_s=0.25)
    assert torch.allclose(scaled, 2.0 * base, rtol=1.0e-10, atol=0.0)


def test_score_honors_operator_sign_convention():
    # Flipping acceleration_sign flips K(r) and therefore J -> dispersion magnitude is identical
    # (P is quadratic in J), pinning that the score flows through the plugin's sign convention.
    plugin = _fitted_plugin()
    base = score_stm_dispersion(plugin, INITIAL_STATES[:1], duration_s=1.0, output_dt_s=0.25)
    plugin.acceleration_sign = -plugin.acceleration_sign
    plugin.posterior = LinearGaussianPosterior(
        mean=-plugin.posterior.mean,  # keep the nominal trajectory identical: K -> -K, mean -> -mean
        cov=plugin.posterior.cov,
        noise_var=plugin.posterior.noise_var,
        lambda_l2=plugin.posterior.lambda_l2,
    )
    flipped = score_stm_dispersion(plugin, INITIAL_STATES[:1], duration_s=1.0, output_dt_s=0.25)
    assert torch.allclose(flipped, base, rtol=1.0e-10, atol=0.0)


def test_rejects_bad_initial_state_shape():
    plugin = _fitted_plugin()
    with pytest.raises(ValueError, match=r"\(N, 6\)"):
        score_stm_dispersion(plugin, np.zeros((2, 5)), duration_s=1.0, output_dt_s=0.5)


def test_not_wired_into_default_scoring_modes():
    # N10 contract: the diagnostic stays a separate entry point unless it earns promotion.
    assert "stm_dispersion" not in SCORING_FUNCTIONS
