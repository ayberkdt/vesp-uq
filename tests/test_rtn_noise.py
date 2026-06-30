from __future__ import annotations

import pytest
import torch

from vesp.uq.metrics import local_radial_frame
from vesp.uq.rtn_noise import (
    RTNVarianceScale,
    RTNVarianceScaleModel,
    apply_rtn_variance_scale,
    fit_rtn_variance_scale_model,
    local_frame_residual_and_variance,
    rtn_calibration_summary,
)


def _sphere_points(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n, 3, generator=g, dtype=torch.float64)
    return pos / torch.linalg.norm(pos, dim=-1, keepdim=True)


def test_fit_rtn_variance_scales_radial_and_tangential_differently():
    positions = _sphere_points(240, seed=3)
    frame = local_radial_frame(positions)
    t = torch.linspace(-1.0, 1.0, positions.shape[0], dtype=torch.float64)
    local_residual = torch.stack([2.0 * t, 0.5 * t, -0.5 * t], dim=-1)
    residual = torch.einsum("nji,nj->ni", frame, local_residual)
    cov = torch.eye(3, dtype=torch.float64).expand(positions.shape[0], 3, 3).clone()

    model = fit_rtn_variance_scale_model(positions, residual, cov)

    assert model.global_scale.radial_scale > 1.0
    assert model.global_scale.tangential_scale < 1.0
    adjusted = apply_rtn_variance_scale(positions, cov, model)
    before = rtn_calibration_summary(positions, residual, cov)
    after = rtn_calibration_summary(positions, residual, adjusted)
    assert abs(after["radial_z_std"] - 1.0) < abs(before["radial_z_std"] - 1.0)
    assert abs(after["tangential_z_std"] - 1.0) < abs(before["tangential_z_std"] - 1.0)


def test_apply_rtn_variance_scale_changes_local_diagonal_only():
    positions = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    cov = torch.eye(3, dtype=torch.float64).reshape(1, 3, 3)
    model = RTNVarianceScaleModel(
        global_scale=RTNVarianceScale(
            name="all",
            lo=1.0,
            hi=1.0,
            radial_scale=4.0,
            tangential_scale=0.25,
            n_points=1,
            radial_z_std_fit=2.0,
            tangential_z_std_fit=0.5,
        )
    )

    adjusted = apply_rtn_variance_scale(positions, cov, model)
    _res, local_var = local_frame_residual_and_variance(
        positions, torch.zeros_like(positions), adjusted
    )

    assert local_var[0, 0].item() == pytest.approx(4.0)
    assert local_var[0, 1].item() == pytest.approx(0.25)
    assert local_var[0, 2].item() == pytest.approx(0.25)
