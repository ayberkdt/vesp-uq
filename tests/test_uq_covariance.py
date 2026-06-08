"""Tests for the 3x3 predictive covariance and covariance speed modes."""

from __future__ import annotations

import torch

from vesp.core.sources import make_shell_sources
from vesp.uq import VESPUQPlugin, make_synthetic_uq_samples


def _fitted(mode="exact", **kw):
    s = make_synthetic_uq_samples(n=400, seed=1)
    src = make_shell_sources([0.75, 0.9], [40, 60], dtype=torch.float64)
    plugin = VESPUQPlugin(src, reg_method="lcurve", noise_model="heteroscedastic", covariance_mode=mode, lowrank_rank=40, seed=0, **kw)
    plugin.fit_error(s.positions, s.error)
    return plugin, s


def test_covariance_shape_symmetry_and_psd():
    plugin, s = _fitted("exact")
    cov = plugin.predict_covariance_3x3(s.positions[:60])
    assert cov.covariance.shape == (60, 3, 3)
    sym = (cov.covariance - cov.covariance.transpose(-1, -2)).abs().max()
    assert float(sym) < 1.0e-10
    eigs = torch.linalg.eigvalsh(cov.covariance)
    assert float(eigs.min()) > -1.0e-9  # PSD up to tolerance


def test_covariance_diagonal_matches_std_components():
    plugin, s = _fitted("exact")
    cov = plugin.predict_covariance_3x3(s.positions[:60])
    diag = torch.diagonal(cov.covariance, dim1=-2, dim2=-1)
    assert torch.allclose(diag.sqrt(), cov.std_components, rtol=1.0e-5, atol=1.0e-12)
    # sigma is the sqrt of the trace
    assert torch.allclose(cov.sigma, diag.sum(dim=1).sqrt(), rtol=1.0e-5)


def test_diagonal_mode_returns_nonnegative_uncertainty():
    plugin, s = _fitted("diagonal")
    pred = plugin.predict_uncertainty(s.positions[:80])
    assert torch.all(pred.sigma >= 0)
    assert torch.all(pred.std_components >= 0)
    cov = plugin.predict_covariance_3x3(s.positions[:80])
    # diagonal mode -> off-diagonals are zero
    off = cov.covariance.clone()
    off[:, 0, 0] = off[:, 1, 1] = off[:, 2, 2] = 0.0
    assert float(off.abs().max()) == 0.0


def test_exact_and_diagonal_same_order_of_magnitude():
    plugin_e, s = _fitted("exact")
    plugin_d, _ = _fitted("diagonal")
    se = plugin_e.predict_uncertainty(s.positions[:100]).sigma.mean()
    sd = plugin_d.predict_uncertainty(s.positions[:100]).sigma.mean()
    ratio = float(se / sd)
    assert 0.1 < ratio < 10.0  # comparable order of magnitude


def test_lowrank_mode_runs_and_is_psd():
    plugin, s = _fitted("lowrank")
    cov = plugin.predict_covariance_3x3(s.positions[:50])
    eigs = torch.linalg.eigvalsh(cov.covariance)
    assert float(eigs.min()) > -1.0e-9
    pred = plugin.predict_uncertainty(s.positions[:50])
    assert torch.all(pred.sigma >= 0)
