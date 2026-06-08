"""Tests for end-to-end trajectory risk screening with a fitted plugin."""

from __future__ import annotations

import torch

from vesp.core.sources import make_shell_sources
from vesp.uq import VESPUQPlugin, make_synthetic_uq_samples, run_risk_screening
from vesp.uq.ensemble import nearest_neighbor_error_magnitude


def _fitted_plugin():
    # interior-source error field naturally grows toward low altitude
    s = make_synthetic_uq_samples(n=600, noise_std=5.0e-5, seed=1)
    src = make_shell_sources([0.75, 0.9], [48, 64], dtype=torch.float64)
    plugin = VESPUQPlugin(src, reg_method="lcurve", noise_model="heteroscedastic", seed=0)
    plugin.fit_error(s.positions, s.error)
    return plugin, s


def _circular_orbit(radius: float, n: int = 40, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    theta = torch.linspace(0.0, 2.0 * torch.pi, n + 1, dtype=torch.float64)[:-1]
    plane = torch.stack([torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)], dim=-1)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    return radius * (plane @ q.T)


def test_low_altitude_trajectory_scores_higher_risk():
    plugin, _ = _fitted_plugin()
    low = _circular_orbit(1.05, seed=1)
    high = _circular_orbit(1.50, seed=1)
    for scoring in ("low_alt_integral", "combined", "max"):
        s_low = plugin.score_trajectory(low, scoring=scoring)
        s_high = plugin.score_trajectory(high, scoring=scoring)
        assert s_low.risk_score > s_high.risk_score, scoring


def test_screening_flags_requested_fraction_and_beats_random():
    plugin, samples = _fitted_plugin()
    radii = torch.linspace(1.04, 1.55, 60, dtype=torch.float64)
    trajectories = [_circular_orbit(float(r), n=36, seed=i) for i, r in enumerate(radii)]
    true_error = torch.tensor(
        [
            float(nearest_neighbor_error_magnitude(t, samples.positions, samples.error).max())
            for t in trajectories
        ],
        dtype=torch.float64,
    )

    result = run_risk_screening(plugin, trajectories, true_error=true_error, rerun_fraction=0.2, scoring="max")
    report = result["risk_screening_report"]
    assert len(result["trajectory_scores"]) == len(trajectories)
    assert 0.15 <= report.rerun_fraction <= 0.27  # ~requested 20%
    # the screen should beat the 0.2 random-capture baseline for the top decile
    assert report.capture_rate > 0.4
    assert report.mean_error_flagged > report.mean_error_accepted
    assert report.spearman_risk_vs_error > 0.3
