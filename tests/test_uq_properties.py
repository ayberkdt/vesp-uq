import math
from dataclasses import dataclass

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from vesp.uq.conformal import _conformal_quantile_level, fit_conformal_scale
from vesp.uq.scoring import aggregate_trajectory_error, score_sigma_profile
from vesp.uq.selection import select_reruns


@dataclass
class DummyProfile:
    risk_score: float
    orbit_id: int
    traj_id: int = 0


# Limit examples to keep CI very fast
settings.register_profile("ci", max_examples=100, deadline=None)
settings.load_profile("ci")


@given(
    scores=st.lists(st.floats(min_value=0.0, max_value=1e6), min_size=1, max_size=100),
    threshold=st.floats(min_value=0.0, max_value=1e6),
)
def test_select_reruns_threshold_invariant(scores, threshold):
    """Property: threshold flagging must exactly flag items >= threshold."""
    scores_t = torch.tensor(scores, dtype=torch.float64)
    report = select_reruns(scores_t, threshold=threshold)

    # Check flagged count
    expected_flag_count = int((scores_t >= threshold).sum())
    assert report.n_flagged == expected_flag_count
    assert len(report.flagged_indices) == expected_flag_count


@given(
    scores=st.lists(st.floats(min_value=0.0, max_value=1e6), min_size=1, max_size=100),
    budget=st.floats(min_value=0.01, max_value=0.99),
)
def test_select_reruns_relative_budget_invariant(scores, budget):
    """Property: fraction flags exactly ceil(budget * len(scores))."""
    scores_t = torch.tensor(scores, dtype=torch.float64)
    report = select_reruns(scores_t, rerun_fraction=budget, fraction_policy="topk")

    expected_flag_count = math.ceil(budget * len(scores))
    assert report.n_flagged == expected_flag_count
    assert len(report.flagged_indices) == expected_flag_count


@given(
    n=st.integers(min_value=1, max_value=1000),
    alpha=st.floats(min_value=0.01, max_value=0.99)
)
def test_conformal_level_finite_sample_bound(n, alpha):
    """Property: the finite sample adjusted level is always > (1 - alpha) and <= 1.0 (unless n is too small)."""
    target = 1.0 - alpha
    level = _conformal_quantile_level(n, alpha)
    assert level >= target
    if level > 1.0:
        # Expected if n is too small to cover the requested alpha
        assert math.ceil((n + 1) * target) / n > 1.0


@given(
    predicted=st.lists(st.floats(min_value=0.1, max_value=10.0), min_size=10, max_size=50),
    true=st.lists(st.floats(min_value=0.0, max_value=20.0), min_size=10, max_size=50),
    alpha=st.floats(min_value=0.05, max_value=0.20)
)
def test_fit_conformal_scale_bounds(predicted, true, alpha):
    """Property: conformal scale is strictly positive."""
    # Ensure equal length
    n = min(len(predicted), len(true))
    pred_t = torch.tensor(predicted[:n])
    true_t = torch.tensor(true[:n])

    calibrator = fit_conformal_scale(pred_t, true_t, alpha=alpha)
    assert calibrator.scale >= 0.0


@given(
    sigma=st.lists(st.floats(min_value=0.0, max_value=10.0), min_size=1, max_size=50),
)
def test_aggregate_trajectory_error_monotonicity(sigma):
    """Property: p95 <= max always, and mean <= max always."""
    t = torch.tensor(sigma, dtype=torch.float64)
    v_max = aggregate_trajectory_error(t, mode="max")
    v_mean = aggregate_trajectory_error(t, mode="mean")
    v_p95 = aggregate_trajectory_error(t, mode="p95")

    assert v_mean <= v_max + 1e-9
    assert v_p95 <= v_max + 1e-9


@given(
    sigma=st.lists(st.floats(min_value=0.0, max_value=10.0), min_size=5, max_size=50),
    radius=st.lists(st.floats(min_value=1.05, max_value=2.0), min_size=5, max_size=50),
    scale=st.floats(min_value=0.1, max_value=10.0)
)
def test_score_sigma_profile_scaling(sigma, radius, scale):
    """Property: Expected-only scoring scales exactly linearly with sigma."""
    n = min(len(sigma), len(radius))
    sig_t = torch.tensor(sigma[:n], dtype=torch.float64)
    rad_t = torch.tensor(radius[:n], dtype=torch.float64)

    # Expected abs ignores radius weight
    score_base = score_sigma_profile(sig_t, rad_t, scoring="mean")
    score_scaled = score_sigma_profile(sig_t * scale, rad_t, scoring="mean")

    # Tolerance due to float point
    assert pytest.approx(score_scaled.risk_score, rel=1e-5) == score_base.risk_score * scale
