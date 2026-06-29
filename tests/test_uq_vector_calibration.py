"""Tests for vector (ellipsoid / Mahalanobis) calibration metrics."""

from __future__ import annotations

import pytest
import torch

from vesp.uq.metrics import (
    chi2_3_cdf,
    chi2_3_ppf,
    component_calibration_metrics,
    diagonal_covariances,
    local_radial_frame,
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


def test_local_radial_frame_is_orthonormal_and_radial():
    torch.manual_seed(3)
    pos = torch.randn(50, 3, dtype=torch.float64) * 1.3
    pos[0] = torch.tensor([0.0, 0.0, 1.4])  # near-pole case uses the e_x reference
    rot = local_radial_frame(pos)
    assert rot.shape == (50, 3, 3)
    # rows orthonormal: R R^T = I
    gram = torch.einsum("nij,nkj->nik", rot, rot)
    eye = torch.eye(3, dtype=torch.float64).expand(50, 3, 3)
    assert torch.allclose(gram, eye, atol=1e-10)
    # row 0 is the unit radial direction
    radial = pos / torch.linalg.norm(pos, dim=-1, keepdim=True)
    assert torch.allclose(rot[:, 0, :], radial, atol=1e-10)


def test_component_calibration_calibrated_gaussian_unit_zstd():
    torch.manual_seed(4)
    n = 40000
    pos = torch.randn(n, 3, dtype=torch.float64) * 1.2
    cov = torch.eye(3, dtype=torch.float64).expand(n, 3, 3).contiguous()
    residuals = torch.randn(n, 3, dtype=torch.float64)  # unit covariance -> calibrated
    m = component_calibration_metrics(residuals, cov, pos)
    assert abs(m["radial_z_std"] - 1.0) < 0.03
    assert abs(m["tangential_z_std"] - 1.0) < 0.03
    assert abs(m["radial_picp_90"] - 0.90) < 0.02
    assert abs(m["tangential_picp_90"] - 0.90) < 0.02
    assert m["calibration_error_90"] < 0.02


def test_component_calibration_detects_radial_only_miscalibration():
    torch.manual_seed(5)
    n = 40000
    # place all points on +x so the radial axis is exactly e_x and the rotation is identity-aligned
    pos = torch.zeros(n, 3, dtype=torch.float64)
    pos[:, 0] = 1.2
    residuals = torch.randn(n, 3, dtype=torch.float64)
    residuals[:, 0] *= 3.0  # inflate the radial (x) error only -> radial under-confident
    cov = torch.eye(3, dtype=torch.float64).expand(n, 3, 3).contiguous()
    m = component_calibration_metrics(residuals, cov, pos)
    assert m["radial_z_std"] > 2.0  # radial axis is badly under-covered
    assert abs(m["tangential_z_std"] - 1.0) < 0.05  # tangential stays calibrated


def test_component_calibration_winkler_minimized_at_true_scale():
    torch.manual_seed(6)
    n = 20000
    pos = torch.randn(n, 3, dtype=torch.float64) * 1.2
    residuals = torch.randn(n, 3, dtype=torch.float64)
    true_cov = torch.eye(3, dtype=torch.float64).expand(n, 3, 3).contiguous()
    too_small = 0.25 * true_cov
    too_big = 4.0 * true_cov
    w_true = component_calibration_metrics(residuals, true_cov, pos)["radial_winkler_90"]
    w_small = component_calibration_metrics(residuals, too_small, pos)["radial_winkler_90"]
    w_big = component_calibration_metrics(residuals, too_big, pos)["radial_winkler_90"]
    assert w_true < w_small and w_true < w_big  # proper score is minimized near the true scale


def test_component_calibration_shape_validation():
    pos = torch.randn(5, 3, dtype=torch.float64)
    with pytest.raises(ValueError):
        component_calibration_metrics(torch.randn(5, 3), torch.randn(4, 3, 3), pos)


def test_diagonal_approximation_metrics_work():
    torch.manual_seed(2)
    n = 20000
    std = torch.full((n, 3), 1.0, dtype=torch.float64)
    residuals = torch.randn(n, 3, dtype=torch.float64)
    cov = diagonal_covariances(std)
    assert cov.shape == (n, 3, 3)
    m = vector_calibration_metrics(residuals, cov)
    assert abs(m["ellipsoid_picp_90"] - 0.90) < 0.02
