"""Tests for the WP-B decision-quality metrics (AUROC / AUPRC / capture-AUC / oracle-regret)."""

from __future__ import annotations

import math

import pytest
import torch

from vesp.uq.benchmarking import (
    DECISION_METRIC_KEYS,
    capture_auc,
    decision_quality_metrics,
    detection_metrics,
    oracle_regret,
)

# Distinct increasing true errors: top-decile class is well defined; oracle ranking is unambiguous.
TRUE_ERROR = torch.arange(1.0, 21.0, dtype=torch.float64)
FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.40)


def test_detection_metrics_perfect_and_inverse():
    perfect = detection_metrics(TRUE_ERROR, TRUE_ERROR, high_quantile=0.90)
    assert perfect["auroc"] == pytest.approx(1.0)
    assert perfect["auprc"] == pytest.approx(1.0)
    inverse = detection_metrics(-TRUE_ERROR, TRUE_ERROR, high_quantile=0.90)
    assert inverse["auroc"] == pytest.approx(0.0)


def test_detection_metrics_degenerate_and_nonfinite_safe():
    # all-equal true error -> high-error class is everything or nothing -> nan, never a crash
    flat = detection_metrics(TRUE_ERROR, torch.ones(20, dtype=torch.float64))
    assert math.isnan(flat["auroc"])
    bad = detection_metrics(torch.full((20,), float("nan"), dtype=torch.float64), TRUE_ERROR)
    assert math.isnan(bad["auroc"])


def test_detection_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        detection_metrics(torch.zeros(5), torch.zeros(6))


def test_capture_auc_oracle_normalized_is_one():
    out = capture_auc(TRUE_ERROR, TRUE_ERROR, fractions=FRACTIONS)
    assert out["capture_auc_normalized"] == pytest.approx(1.0)
    # an inverse score captures far less than the oracle
    worse = capture_auc(-TRUE_ERROR, TRUE_ERROR, fractions=FRACTIONS)
    assert worse["capture_auc_normalized"] < out["capture_auc_normalized"]


def test_oracle_regret_bounds():
    assert oracle_regret(TRUE_ERROR, TRUE_ERROR, fraction=0.20)["oracle_regret"] == pytest.approx(0.0)
    # worst-case ranking (smallest errors first) is clamped to the random/oracle gap upper bound
    assert oracle_regret(-TRUE_ERROR, TRUE_ERROR, fraction=0.20)["oracle_regret"] == pytest.approx(1.0)


def test_decision_quality_metrics_has_all_keys_and_snaps_reference():
    out = decision_quality_metrics(TRUE_ERROR, TRUE_ERROR, fractions=FRACTIONS,
                                   reference_fraction=0.18)
    for key in DECISION_METRIC_KEYS:
        assert key in out
    assert out["reference_fraction"] == pytest.approx(0.20)  # snapped to the nearest grid budget
    assert out["auroc"] == pytest.approx(1.0)
    assert out["oracle_regret"] == pytest.approx(0.0)
