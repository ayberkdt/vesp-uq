"""Tests for vector (ellipsoid / Mahalanobis) calibration metrics."""

from __future__ import annotations

import torch

from vesp.uq.metrics import (
    chi2_3_cdf,
    chi2_3_ppf,
    diagonal_covariances,
    vector_calibration_metrics,
)


def test_chi2_3_cdf_ppf_roundtrip():
    for p in (0.5, 0.68, 0.9, 0.95):
        t = chi2_3_ppf(p)
        assert abs(float(chi2_3_cdf(t)) - p) < 1.0e-4
    # known value: chi2(3) 95% quantile ~ 7.815
    assert abs(chi2_3_ppf(0.95) - 7.8147) < 1.0e-2


def test_calibrated_gaussian_gives_nominal_ellipsoid_coverage():
    torch.manual_seed(0)
    n = 40000
    cov = torch.eye(3, dtype=torch.float64).expand(n, 3, 3).contiguous()
    residuals = torch.randn(n, 3, dtype=torch.float64)  # unit covariance -> calibrated
    m = vector_calibration_metrics(residuals, cov)
    assert abs(m["ellipsoid_picp_90"] - 0.90) < 0.02
    assert abs(m["ellipsoid_picp_95"] - 0.95) < 0.02
    assert abs(m["mean_mahalanobis_d2"] - 3.0) < 0.1  # E[chi2_3] = 3


def test_overconfident_covariance_undercovers_and_inflates_d2():
    torch.manual_seed(1)
    n = 20000
    residuals = torch.randn(n, 3, dtype=torch.float64)
    # predicted covariance 9x too small (std 3x too small) -> overconfident
    cov = (1.0 / 9.0) * torch.eye(3, dtype=torch.float64).expand(n, 3, 3).contiguous()
    m = vector_calibration_metrics(residuals, cov)
    assert m["ellipsoid_picp_90"] < 0.90
    assert m["mean_mahalanobis_d2"] > 3.0 * 3.0  # ~ 9x the calibrated mean


def test_diagonal_approximation_metrics_work():
    torch.manual_seed(2)
    n = 20000
    std = torch.full((n, 3), 1.0, dtype=torch.float64)
    residuals = torch.randn(n, 3, dtype=torch.float64)
    cov = diagonal_covariances(std)
    assert cov.shape == (n, 3, 3)
    m = vector_calibration_metrics(residuals, cov)
    assert abs(m["ellipsoid_picp_90"] - 0.90) < 0.02
