"""Tests for G3 -- metric-range invariants (loud failure on out-of-domain metrics)."""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from vesp.uq.benchmarking import capture_auc, detection_metrics, oracle_regret
from vesp.uq.integrity.metric_invariants import (
    METRIC_DOMAINS,
    MetricRangeError,
    validate_metric,
    validate_row,
)


def test_in_domain_values_pass_through_unchanged():
    assert validate_metric("auroc", 0.73) == 0.73
    assert validate_metric("spearman", -1.0) == -1.0
    assert validate_metric("capture_auc_normalized", 1.0) == 1.0
    assert validate_metric("z_std", 12.5) == 12.5  # unbounded above is fine


def test_none_and_nan_pass_through():
    assert validate_metric("auroc", None) is None
    nan = float("nan")
    assert math.isnan(validate_metric("spearman", nan))


def test_unknown_metric_is_never_constrained():
    # z_mean / nll are legitimately unbounded and not in the domain table -> pass through.
    assert validate_metric("z_mean", -42.0) == -42.0
    assert validate_metric("nll", -1.0e6) == -1.0e6
    assert "z_mean" not in METRIC_DOMAINS


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("auroc", 1.5),
        ("auroc", -0.2),
        ("capture_rate", 1.2),
        ("spearman", 1.4),
        ("spearman", -1.4),
        ("oracle_regret", 1.5),
        ("z_std", -0.1),
        ("picp_90", 1.5),
    ],
)
def test_finite_out_of_domain_raises(name, value):
    with pytest.raises(MetricRangeError):
        validate_metric(name, value, where="unit")


def test_infinite_known_metric_raises():
    with pytest.raises(MetricRangeError):
        validate_metric("z_std", float("inf"), where="unit")


def test_floating_point_slack_is_tolerated():
    # round-off just past 1.0 / -1.0 is accepted; a real excursion is not.
    assert validate_metric("auroc", 1.0 + 1e-9) == pytest.approx(1.0 + 1e-9)
    assert validate_metric("spearman", -1.0 - 1e-9) == pytest.approx(-1.0 - 1e-9)


def test_validate_row_flags_the_offending_key():
    row = {"band": "L60", "selector": "x", "auroc": 0.8, "capture_rate": 2.0}
    with pytest.raises(MetricRangeError) as exc:
        validate_row(row, where="ranking")
    assert "capture_rate" in str(exc.value)


def test_validate_row_passes_a_clean_record():
    row = {"band": "L90", "seed": 0, "selector": "min_altitude", "auroc": 0.61,
           "auprc": 0.4, "capture_auc_normalized": 0.92, "oracle_regret": 0.3,
           "spearman": 0.55, "z_mean": -0.2}
    validate_row(row, where="ranking")  # no raise


# --- property: the real decision-quality metrics never emit an out-of-domain value -------------
@settings(max_examples=60, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    n=st.integers(min_value=5, max_value=60),
)
def test_decision_metrics_outputs_always_in_domain(seed, n):
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(n, generator=g, dtype=torch.float64)
    true_error = torch.rand(n, generator=g, dtype=torch.float64).abs()
    fractions = (0.05, 0.1, 0.2, 0.3, 0.4)
    det = detection_metrics(scores, true_error, high_quantile=0.90)
    cap = capture_auc(scores, true_error, fractions=fractions)
    reg = oracle_regret(scores, true_error, fraction=0.20)
    for name, value in {**det, **cap, **reg}.items():
        validate_metric(name, value, where=f"property[seed={seed} n={n}]")  # must not raise
