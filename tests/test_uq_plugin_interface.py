"""Tests for the VESPUQPlugin fit/predict interface contract."""

from __future__ import annotations

import pytest
import torch

from vesp.core.sources import make_shell_sources
from vesp.uq import VESPUQPlugin, make_synthetic_uq_samples


def _plugin(**kw):
    src = make_shell_sources([0.8], 64, dtype=torch.float64)
    base = dict(reg_method="lcurve", noise_model="heteroscedastic", seed=0)
    base.update(kw)
    return VESPUQPlugin(src, **base)


def test_fit_accepts_surrogate_and_reference():
    s = make_synthetic_uq_samples(n=300, seed=1)
    surrogate = torch.zeros_like(s.error)
    reference = s.error  # error = reference - surrogate
    plugin = _plugin().fit(s.positions, surrogate, reference)
    assert plugin.posterior is not None
    assert plugin.fit_info["n_train"] + plugin.fit_info["n_val"] == s.n


def test_fit_error_accepts_direct_error():
    s = make_synthetic_uq_samples(n=300, seed=2)
    plugin = _plugin().fit_error(s.positions, s.error)
    pred = plugin.predict_uncertainty(s.positions[:20])
    assert pred.mean_error.shape == (20, 3)
    assert pred.sigma.shape == (20,)


def test_risk_score_equals_sigma_by_default():
    s = make_synthetic_uq_samples(n=200, seed=3)
    plugin = _plugin().fit_error(s.positions, s.error)
    pred = plugin.predict_uncertainty(s.positions[:50])
    assert torch.allclose(pred.risk_score, pred.sigma)


def test_explicit_validation_set_is_used():
    s = make_synthetic_uq_samples(n=400, seed=4)
    train = s.subset(torch.arange(0, 300))
    val = s.subset(torch.arange(300, 400))
    plugin = _plugin().fit(
        train.positions,
        train.surrogate,
        train.reference,
        val_positions=val.positions,
        val_surrogate_acceleration=val.surrogate,
        val_reference_acceleration=val.reference,
    )
    # the full train set is used for the mean (no internal hold-out), the explicit val for noise
    assert plugin.fit_info["n_train"] == 300
    assert plugin.fit_info["n_val"] == 100


def test_missing_val_reference_or_surrogate_raises():
    s = make_synthetic_uq_samples(n=200, seed=5)
    with pytest.raises(ValueError, match="val_reference_acceleration"):
        _plugin().fit(
            s.positions,
            s.surrogate,
            s.reference,
            val_positions=s.positions[:20],
            val_surrogate_acceleration=s.surrogate[:20],
            # val_reference_acceleration intentionally omitted
        )


def test_predict_before_fit_raises():
    s = make_synthetic_uq_samples(n=50, seed=6)
    with pytest.raises(RuntimeError):
        _plugin().predict_uncertainty(s.positions)
    with pytest.raises(RuntimeError):
        _plugin().predict_covariance_3x3(s.positions)
