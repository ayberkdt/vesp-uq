"""Tests for trajectory risk scoring and selective-rerun logic."""

from __future__ import annotations

import math

import pytest
import torch

from vesp.uq.trajectory import (
    RiskScreeningReport,
    aggregate_trajectory_error,
    score_sigma_profile,
    select_reruns,
)


def test_score_sigma_profile_basic_aggregations():
    sigma = torch.tensor([1.0, 3.0, 2.0])
    radius = torch.tensor([1.05, 1.50, 1.20])
    s = score_sigma_profile(sigma, radius, scoring="max", sigma_threshold=2.5, low_altitude_radius=1.15)
    assert s.max_sigma == pytest.approx(3.0)
    assert s.mean_sigma == pytest.approx(2.0)
    # only radius 1.05 is below 1.15 -> integral picks up sigma=1.0
    assert s.low_altitude_sigma_integral == pytest.approx(1.0)
    # one of three points exceeds threshold 2.5
    assert s.time_above_threshold == pytest.approx(1.0 / 3.0)
    assert s.risk_score == pytest.approx(3.0)
    assert s.min_radius == pytest.approx(1.05)


def test_combined_altitude_risk_rewards_low_uncertain_points():
    radius = torch.tensor([1.05, 1.50])
    # same sigma; the low-altitude point gets a far larger altitude weight
    low_heavy = score_sigma_profile(torch.tensor([2.0, 0.0]), radius, scoring="combined")
    high_heavy = score_sigma_profile(torch.tensor([0.0, 2.0]), radius, scoring="combined")
    assert low_heavy.combined_altitude_risk > high_heavy.combined_altitude_risk


def test_score_empty_or_invalid_raises():
    with pytest.raises(ValueError):
        score_sigma_profile(torch.tensor([]), torch.tensor([]))
    with pytest.raises(ValueError):
        score_sigma_profile(torch.tensor([1.0]), torch.tensor([1.0]), scoring="nope")


def test_select_reruns_by_fraction_flags_top_subset():
    risk = torch.arange(100, dtype=torch.float64)
    report = select_reruns(risk, rerun_fraction=0.2)
    assert isinstance(report, RiskScreeningReport)
    assert report.n_flagged == 20
    assert 0.18 <= report.rerun_fraction <= 0.22
    assert min(report.flagged_indices) >= 80


def test_select_reruns_threshold_path():
    risk = torch.tensor([0.1, 0.5, 0.9, 0.95])
    report = select_reruns(risk, threshold=0.6)
    assert report.flagged_indices == [2, 3]


def test_select_reruns_requires_exactly_one_budget():
    risk = torch.arange(10, dtype=torch.float64)
    with pytest.raises(ValueError):
        select_reruns(risk)
    with pytest.raises(ValueError):
        select_reruns(risk, rerun_fraction=0.2, threshold=0.5)


def test_capture_rate_and_spearman_when_risk_ranks_error_perfectly():
    n = 100
    err = torch.arange(n, dtype=torch.float64)
    report = select_reruns(err.clone(), rerun_fraction=0.2, true_error=err, true_error_quantile=0.9)
    # risk == error -> every top-decile high-error trajectory is flagged
    assert report.capture_rate == pytest.approx(1.0)
    assert report.spearman_risk_vs_error == pytest.approx(1.0)
    # 20 flagged, 10 of them truly high -> precision 0.5
    assert report.precision == pytest.approx(0.5)
    assert report.error_ratio_flagged_to_accepted > 1.0


def test_anticorrelated_risk_misses_high_error():
    n = 100
    err = torch.arange(n, dtype=torch.float64)
    risk = torch.flip(err, dims=[0])  # risk inversely ranks error
    report = select_reruns(risk, rerun_fraction=0.2, true_error=err)
    assert report.capture_rate == pytest.approx(0.0)
    assert report.spearman_risk_vs_error == pytest.approx(-1.0)


def test_true_error_length_mismatch_raises():
    risk = torch.arange(10, dtype=torch.float64)
    with pytest.raises(ValueError):
        select_reruns(risk, rerun_fraction=0.2, true_error=torch.arange(5, dtype=torch.float64))


def test_time_above_is_nan_without_threshold():
    s = score_sigma_profile(torch.tensor([1.0, 2.0]), torch.tensor([1.1, 1.2]), scoring="mean")
    assert math.isnan(s.time_above_threshold)


# ---------------------------------------------------------------- expected-error scoring (A)

def test_expected_scoring_uses_mean_error_not_just_sigma():
    # Two trajectories with identical (small) sigma but different posterior-mean bias. The
    # legacy sigma `max` mode cannot tell them apart; `expected` (sqrt(bias^2 + sigma^2)) must.
    radius = torch.full((4,), 1.30)
    sigma = torch.full((4,), 0.01)
    big_bias = torch.full((4,), 1.0)
    small_bias = torch.full((4,), 0.02)
    ee_high = torch.sqrt(big_bias**2 + sigma**2)
    ee_low = torch.sqrt(small_bias**2 + sigma**2)

    s_high = score_sigma_profile(sigma, radius, scoring="expected", expected_error=ee_high)
    s_low = score_sigma_profile(sigma, radius, scoring="expected", expected_error=ee_low)
    assert s_high.risk_score > s_low.risk_score
    # sigma-only modes are blind to the bias difference
    m_high = score_sigma_profile(sigma, radius, scoring="max")
    m_low = score_sigma_profile(sigma, radius, scoring="max")
    assert m_high.risk_score == pytest.approx(m_low.risk_score)


def test_expected_modes_require_expected_error_profile():
    for mode in ("expected", "expected_p95", "expected_low_alt", "supervisor", "supervisor_p95"):
        with pytest.raises(ValueError):
            score_sigma_profile(torch.ones(3), torch.full((3,), 1.2), scoring=mode)


def test_p95_expected_error_between_mean_and_max_on_a_spiky_profile():
    radius = torch.full((20,), 1.3)
    sigma = torch.full((20,), 0.01)
    ee = torch.cat([torch.full((18,), 0.1), torch.full((2,), 5.0)])
    s = score_sigma_profile(sigma, radius, scoring="expected_p95", expected_error=ee)
    assert s.mean_expected_error < s.p95_expected_error <= s.max_expected_error


# ---------------------------------------------------------------- supervisor point risk (D)

def test_supervisor_emphasizes_high_expected_error_at_low_altitude():
    # Same expected-error VALUES on both trajectories, same radius profile; only the altitude
    # at which the big error sits differs. Supervisor must rank the low-altitude placement higher.
    radius = torch.tensor([1.05, 1.55])
    sigma = torch.tensor([0.01, 0.01])
    ee_low_big = torch.tensor([1.0, 0.1])  # big error at the low point
    ee_high_big = torch.tensor([0.1, 1.0])  # big error at the high point
    s_low = score_sigma_profile(sigma, radius, scoring="supervisor", expected_error=ee_low_big)
    s_high = score_sigma_profile(sigma, radius, scoring="supervisor", expected_error=ee_high_big)
    assert s_low.risk_score > s_high.risk_score


def test_supervisor_increases_when_outside_support():
    radius = torch.full((4,), 1.30)
    sigma = torch.full((4,), 0.01)
    ee = torch.full((4,), 1.0)
    supported = score_sigma_profile(
        sigma, radius, scoring="supervisor", expected_error=ee, domain_risk=torch.zeros(4)
    )
    unsupported = score_sigma_profile(
        sigma, radius, scoring="supervisor", expected_error=ee,
        domain_risk=torch.full((4,), 2.0), domain_weight=1.0,
    )
    assert unsupported.risk_score > supported.risk_score
    assert unsupported.max_domain_risk == pytest.approx(2.0)
    assert unsupported.time_outside_support == pytest.approx(1.0)
    assert supported.max_domain_risk == pytest.approx(0.0)
    assert supported.time_outside_support == pytest.approx(0.0)
    # with no domain profile supplied at all, the domain metrics are nan (not 0)
    no_domain = score_sigma_profile(sigma, radius, scoring="supervisor", expected_error=ee)
    assert math.isnan(no_domain.max_domain_risk)


# ---------------------------------------------------------------- weighted scoring (item 7)

def test_weights_change_mean_but_none_preserves_legacy():
    sigma = torch.tensor([1.0, 3.0])
    radius = torch.tensor([1.05, 1.50])
    legacy = score_sigma_profile(sigma, radius, scoring="mean")
    assert legacy.mean_sigma == pytest.approx(2.0)
    # weight the first point 3x the second -> mean pulled toward 1.0
    weighted = score_sigma_profile(sigma, radius, scoring="mean", weights=torch.tensor([3.0, 1.0]))
    assert weighted.mean_sigma == pytest.approx((3 * 1.0 + 1 * 3.0) / 4.0)
    assert weighted.mean_sigma < legacy.mean_sigma


# ---------------------------------------------------------------- threshold + max fraction (B)

def test_threshold_plus_max_fraction_caps_when_too_many_above():
    risk = torch.ones(100, dtype=torch.float64)  # everything above an 0.5 threshold
    report = select_reruns(risk, threshold=0.5, max_rerun_fraction=0.1)
    assert report.selection_mode == "threshold+max_fraction"
    assert report.n_above_threshold == 100
    assert report.n_flagged == 10
    assert report.max_rerun_fraction == pytest.approx(0.1)


def test_threshold_plus_max_fraction_keeps_all_when_under_budget():
    risk = torch.cat([torch.zeros(95), torch.ones(5)]).to(torch.float64)
    report = select_reruns(risk, threshold=0.5, max_rerun_fraction=0.2)  # budget 20 > 5 above
    assert report.n_above_threshold == 5
    assert report.n_flagged == 5
    assert sorted(report.flagged_indices) == [95, 96, 97, 98, 99]


def test_threshold_only_can_flag_zero():
    risk = torch.tensor([0.1, 0.2, 0.3])
    report = select_reruns(risk, threshold=0.9)
    assert report.selection_mode == "threshold"
    assert report.n_flagged == 0
    assert report.flagged_indices == []
    assert report.n_above_threshold == 0


def test_max_rerun_fraction_requires_threshold():
    with pytest.raises(ValueError):
        select_reruns(torch.arange(10, dtype=torch.float64), max_rerun_fraction=0.1)


def test_fraction_only_behavior_is_unchanged():
    risk = torch.arange(100, dtype=torch.float64)
    report = select_reruns(risk, rerun_fraction=0.2)
    assert report.selection_mode == "fraction"
    assert report.n_flagged == 20
    assert report.max_rerun_fraction is None


# ---------------------------------------------------------------- safe-regime, no NaN crash (E)

def test_zero_flagged_with_true_error_does_not_crash():
    risk = torch.tensor([0.1, 0.2, 0.3, 0.4])
    err = torch.tensor([1.0, 2.0, 3.0, 4.0])
    report = select_reruns(risk, threshold=10.0, true_error=err)
    assert report.n_flagged == 0
    assert math.isnan(report.precision)
    assert math.isnan(report.mean_error_flagged)
    assert report.mean_error_accepted == pytest.approx(2.5)
    # the dict round-trips (report stays JSON-clean for the driver)
    d = report.to_dict()
    assert d["n_above_threshold"] == 0 and d["n_flagged"] == 0


# ---------------------------------------------------------------- p95 aggregator (F)

def test_p95_aggregator_robust_to_spike_but_tracks_sustained_pass():
    # a single nearest-neighbour spike: max is dominated by it, p95 ignores it
    spike = torch.cat([torch.ones(99), torch.tensor([100.0])])
    assert aggregate_trajectory_error(spike, "max") == pytest.approx(100.0)
    assert aggregate_trajectory_error(spike, "p95") < 5.0
    # a sustained ~10% high-error pass: p95 tracks the pass, mean dilutes it
    sustained = torch.cat([torch.ones(90), torch.full((10,), 10.0)])
    mean = aggregate_trajectory_error(sustained, "mean")
    p95 = aggregate_trajectory_error(sustained, "p95")
    mx = aggregate_trajectory_error(sustained, "max")
    assert mean < p95 <= mx
    assert p95 >= 9.0


def test_aggregate_trajectory_error_rejects_bad_mode():
    with pytest.raises(ValueError):
        aggregate_trajectory_error(torch.ones(4), "median")
