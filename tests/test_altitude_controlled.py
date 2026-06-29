"""Unit tests for the altitude-controlled incremental-value diagnostics (WP4)."""

from __future__ import annotations

import math

import torch

from vesp.uq.altitude_controlled import (
    matched_altitude_pairs,
    partial_correlations,
    partial_pearson_given_altitude,
    pearson,
    spearman,
    within_altitude_bin_ranking,
)


def test_pearson_spearman_basic():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert abs(pearson(x, 2.0 * x + 1.0) - 1.0) < 1e-12  # perfectly linear
    assert abs(spearman(x, torch.tensor([1.0, 4.0, 9.0, 16.0])) - 1.0) < 1e-12  # monotone
    # a constant input has no defined correlation
    assert math.isnan(pearson(x, torch.ones(4)))


def test_partial_correlation_removes_pure_altitude_signal():
    # true error and score are BOTH driven only by altitude -> partial correlation ~ 0.
    torch.manual_seed(0)
    min_radius = torch.linspace(1.0, 1.6, 200)
    true_error = -min_radius + 0.0 * torch.randn(200)
    score = -min_radius  # pure altitude proxy
    raw = pearson(score, true_error)
    partial = partial_pearson_given_altitude(score, true_error, min_radius)
    assert raw > 0.9  # strong raw correlation (both are altitude)
    assert abs(partial) < 0.2 or math.isnan(partial)  # vanishes after controlling altitude


def test_partial_correlation_keeps_altitude_independent_signal():
    torch.manual_seed(1)
    min_radius = torch.linspace(1.0, 1.6, 300)
    extra = torch.randn(300)  # altitude-independent component
    true_error = -min_radius + 0.5 * extra
    score = -min_radius + 0.5 * extra  # carries the same extra signal
    partial = partial_pearson_given_altitude(score, true_error, min_radius)
    assert partial > 0.5  # incremental signal survives altitude control


def test_within_altitude_bin_ranking_shape():
    torch.manual_seed(2)
    n = 120
    min_radius = torch.rand(n) + 1.0
    true_error = torch.rand(n)
    scores = {"a": torch.rand(n), "b": torch.rand(n)}
    out = within_altitude_bin_ranking(scores, true_error, min_radius, n_bins=5)
    assert out["n_bins"] == 5
    assert set(out["weighted_within_bin_spearman"]) == {"a", "b"}
    assert all("methods" in b and "true_error_var" in b for b in out["bins"])


def test_partial_correlations_keys():
    n = 50
    min_radius = torch.linspace(1.0, 1.5, n)
    te = torch.rand(n)
    out = partial_correlations({"s": torch.rand(n)}, te, min_radius)
    assert set(out["s"]) == {"pearson", "spearman", "partial_pearson_given_min_radius"}


def _matched_inputs(reverse: bool):
    # three altitude-matched pairs; within each pair idx-with-higher-score also has higher error
    min_radius = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    score = torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    base = torch.tensor([10.0, 20.0, 10.0, 20.0, 10.0, 20.0])
    true_error = (40.0 - base) if reverse else base
    return score, true_error, min_radius


def test_matched_pairs_perfect_concordance():
    score, true_error, min_radius = _matched_inputs(reverse=False)
    out = matched_altitude_pairs(score, true_error, min_radius, caliper=float("inf"))
    assert out["n_pairs"] == 3
    assert out["concordance_rate"] == 1.0
    assert out["mean_delta_true_error"] > 0
    assert out["sign_test_p_value_one_sided"] < 0.5


def test_matched_pairs_perfect_discordance():
    score, true_error, min_radius = _matched_inputs(reverse=True)
    out = matched_altitude_pairs(score, true_error, min_radius, caliper=float("inf"))
    assert out["concordance_rate"] == 0.0
    assert out["mean_delta_true_error"] < 0


def test_matched_pairs_caliper_drops_far_pairs():
    # a tiny caliper admits only the within-altitude ties, never cross-altitude matches
    min_radius = torch.tensor([1.0, 1.0, 5.0, 5.0])
    score = torch.tensor([1.0, 2.0, 1.0, 2.0])
    true_error = torch.tensor([10.0, 20.0, 10.0, 20.0])
    out = matched_altitude_pairs(score, true_error, min_radius, caliper=0.1)
    assert out["n_pairs"] == 2  # (0,1) and (2,3); both gaps are 0
    assert all(r["min_radius_gap"] <= 0.1 for r in out["rows"])
