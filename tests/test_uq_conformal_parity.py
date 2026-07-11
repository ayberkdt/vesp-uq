"""Serving-parity regression tests for the calibration report (R2WP-1).

The 2026-07-11 review found that ``evaluate_calibration`` re-applied the operational conformal
scale on top of ``_predict_covariance_block`` (which already applies it), so the report measured
``c^2 * sigma`` / ``c^4 * cov`` while serving predictions carried ``c * sigma`` / ``c^2 * cov``.
These tests pin the invariant: the arrays the calibration report is computed from must be exactly
the arrays ``predict_uncertainty`` / ``predict_covariance_3x3`` serve — with conformal on and off.
"""

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


def _fitted_plugin(*, conformal_apply: bool, **kw) -> VESPUQPlugin:
    """Plugin fitted on an in-span field plus heavy noise, so conformal must inflate (c > 1)."""

    sources = make_shell_sources([0.8], 48, dtype=torch.float64)
    positions = _query_shell(400, 1.05, 1.6, seed=1)
    sigma_true = 0.02 * torch.randn(
        sources.n_sources, generator=torch.Generator().manual_seed(3), dtype=torch.float64
    )
    A = build_acceleration_operator(positions, sources, eps=0.0, sign=1.0)
    error = (A @ sigma_true).reshape(3, positions.shape[0]).transpose(0, 1)
    error = error + 0.5 * error.abs().mean() * torch.randn(
        *error.shape, generator=torch.Generator().manual_seed(5), dtype=torch.float64
    )
    plugin = VESPUQPlugin(
        sources,
        reg_method="fixed",
        lambda_l2=1.0e-6,
        noise_model="homoscedastic",
        val_fraction=0.25,
        conformal_apply=conformal_apply,
        seed=0,
        **kw,
    )
    plugin.fit_error(positions, error)
    return plugin


@pytest.fixture(scope="module")
def plugins() -> dict[str, VESPUQPlugin]:
    return {
        "raw": _fitted_plugin(conformal_apply=False),
        "conformal": _fitted_plugin(conformal_apply=True),
    }


@pytest.mark.parametrize("which", ["raw", "conformal"])
def test_calibration_arrays_match_served_predictions(plugins, which):
    plugin = plugins[which]
    if which == "conformal":
        assert plugin.conformal_calibration is not None
    queries = _query_shell(120, 1.05, 1.6, seed=11)
    error = 1.0e-3 * torch.randn(120, 3, generator=torch.Generator().manual_seed(13), dtype=torch.float64)

    arrays = plugin._calibration_arrays(queries, error)
    pred = plugin.predict_uncertainty(queries)
    cov = plugin.predict_covariance_3x3(queries)

    n = queries.shape[0]
    report_std = arrays["std"].reshape(3, n).transpose(0, 1)  # rows are [x-all, y-all, z-all]
    assert torch.allclose(report_std, pred.std_components, rtol=1.0e-12, atol=0.0)
    assert torch.allclose(report_std, cov.std_components, rtol=1.0e-12, atol=0.0)
    assert torch.allclose(arrays["covariance"], cov.covariance, rtol=1.0e-12, atol=0.0)
    report_mean = arrays["mean"].reshape(3, n).transpose(0, 1)
    assert torch.allclose(report_mean, pred.mean_error, rtol=1.0e-12, atol=0.0)


def test_reported_sigma_scales_linearly_with_conformal_scale(plugins):
    # The directed version of the review bug: with global scale c, the reported mean predictive
    # sigma must be c * (raw sigma) — the double-application reported c^2 * (raw sigma).
    raw, conformal = plugins["raw"], plugins["conformal"]
    scale = float(conformal.conformal_calibration["global"]["scale"])
    assert scale > 1.05, "test setup must force a conformal scale meaningfully above 1"

    queries = _query_shell(150, 1.05, 1.6, seed=21)
    error = 1.0e-3 * torch.randn(150, 3, generator=torch.Generator().manual_seed(23), dtype=torch.float64)
    rep_raw = raw.evaluate_calibration(queries, error)
    rep_cal = conformal.evaluate_calibration(queries, error)
    assert rep_cal["all"]["mean_pred_sigma"] == pytest.approx(
        scale * rep_raw["all"]["mean_pred_sigma"], rel=1.0e-10
    )
