"""Tests for the post-hoc conformal force-error calibration layer (vesp.uq.conformal)."""

from __future__ import annotations

import math

import pytest
import torch

from vesp.uq.conformal import (
    ConformalCalibrator,
    apply_conformal_scale,
    calibrate_trajectory_risk,
    coverage_before_after,
    fit_conformal_scale,
)


def test_scale_at_least_one_when_undercovering():
    # Predicted uncertainty 3x too small -> the predictive band under-covers -> scale must inflate.
    torch.manual_seed(0)
    n = 2000
    true = torch.rand(n).abs() + 0.1
    predicted = true / 3.0  # systematically too tight
    cal = fit_conformal_scale(predicted, true, alpha=0.10, mode="norm")
    assert isinstance(cal, ConformalCalibrator)
    assert cal.scale >= 1.0


def test_coverage_after_not_worse_on_controlled_example():
    torch.manual_seed(1)
    n = 3000
    true = torch.rand(n) + 0.05
    predicted = 0.5 * true  # under-covering by construction
    cal = fit_conformal_scale(predicted, true, alpha=0.10, mode="norm")
    calibrated = apply_conformal_scale(predicted, cal.scale)
    cov = coverage_before_after(predicted, true, calibrated, alpha=0.10, mode="norm")
    assert cov["coverage_after"] >= cov["coverage_before"]
    # under-covering input should reach (or essentially reach) the nominal target after scaling
    assert cov["coverage_after"] >= 0.90 - 1.0e-9


def test_calibrated_coverage_reaches_target_on_gaussian_norms():
    # 3D Gaussian error; predicted per-point sigma = sqrt(3) (E||e|| scale). Conformal should land
    # coverage near the 1-alpha target without assuming the (non-Gaussian) norm distribution.
    torch.manual_seed(2)
    n = 5000
    residual = torch.randn(n, 3, dtype=torch.float64)
    predicted = torch.full((n,), math.sqrt(3.0), dtype=torch.float64)
    cal = fit_conformal_scale(predicted, residual, alpha=0.10, mode="norm")
    calibrated = cal.apply(predicted)
    cov = coverage_before_after(predicted, residual, calibrated, alpha=0.10, mode="norm")
    assert abs(cov["coverage_after"] - 0.90) < 0.03


def test_component_max_mode_runs_and_covers():
    torch.manual_seed(3)
    n = 2000
    residual = torch.randn(n, 3, dtype=torch.float64)
    predicted = torch.ones(n, 3, dtype=torch.float64)  # per-component std too small for the max
    cal = fit_conformal_scale(predicted, residual, alpha=0.10, mode="component_max")
    calibrated = cal.apply(predicted)
    cov = coverage_before_after(predicted, residual, calibrated, alpha=0.10, mode="component_max")
    assert cov["coverage_after"] >= cov["coverage_before"]


def test_invalid_alpha_raises():
    true = torch.rand(50) + 0.1
    predicted = true.clone()
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            fit_conformal_scale(predicted, true, alpha=bad)
    with pytest.raises(ValueError):
        coverage_before_after(predicted, true, predicted, alpha=0.0)


def test_empty_and_nan_inputs_raise():
    with pytest.raises(ValueError):
        fit_conformal_scale(torch.empty(0), torch.empty(0))
    bad = torch.tensor([1.0, float("nan"), 2.0])
    ok = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        fit_conformal_scale(bad, ok)
    with pytest.raises(ValueError):
        fit_conformal_scale(ok, bad)


def test_incompatible_shapes_raise():
    predicted = torch.rand(10, 3) + 0.1
    true = torch.rand(8, 3) + 0.1
    with pytest.raises(ValueError):
        fit_conformal_scale(predicted, true, mode="norm")


def test_negative_scale_rejected():
    with pytest.raises(ValueError):
        apply_conformal_scale(torch.ones(5), -1.0)


def test_small_array_handled_gracefully():
    # n=1 -> finite-sample level clamps to 1.0 (use the max); should not raise.
    cal = fit_conformal_scale(torch.tensor([0.5]), torch.tensor([1.0]), alpha=0.10)
    assert cal.scale >= 1.0
    assert cal.quantile_level <= 1.0


def test_trajectory_risk_calibration_reports_coverage():
    torch.manual_seed(4)
    n = 200
    scores = torch.rand(n) + 0.1
    true_errors = 1.5 * scores  # risk under-bounds true error -> needs inflation
    out = calibrate_trajectory_risk(scores, true_errors, alpha=0.10, mode="p95")
    assert out["scale"] >= 1.0
    assert out["coverage_after"] >= out["coverage_before"]
    assert out["n_trajectories"] == n
    assert len(out["calibrated_scores"]) == n


def test_trajectory_risk_per_point_calibration():
    torch.manual_seed(5)
    scores = [torch.rand(10) + 0.1 for _ in range(6)]
    true_errors = [1.2 * s for s in scores]
    out = calibrate_trajectory_risk(scores, true_errors, alpha=0.10, mode="p95", per_point=True)
    assert out["per_point"] is True
    assert out["n_samples"] == 60
    assert out["coverage_after"] >= out["coverage_before"]


def test_no_position_error_references_in_module():
    import inspect

    import vesp.uq.conformal as mod

    src = inspect.getsource(mod).lower()
    assert "position error" not in src
    assert "position-error" not in src
