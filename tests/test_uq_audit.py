"""Tests for the sentinel false-negative audit layer (vesp.uq.audit)."""

from __future__ import annotations

import pytest
import torch

from vesp.uq.audit import audit_summary_dict, evaluate_false_negatives, select_sentinel_audit


def test_sentinel_deterministic_with_seed():
    flagged = [0, 1, 2, 3, 4]
    a = select_sentinel_audit(flagged, n_total=100, audit_fraction=0.1, min_audit=5, seed=7)
    b = select_sentinel_audit(flagged, n_total=100, audit_fraction=0.1, min_audit=5, seed=7)
    c = select_sentinel_audit(flagged, n_total=100, audit_fraction=0.1, min_audit=5, seed=8)
    assert a == b
    assert a != c  # different seed -> (almost surely) different draw


def test_sentinel_does_not_overlap_flagged():
    flagged = list(range(0, 20))
    sentinel = select_sentinel_audit(flagged, n_total=80, audit_fraction=0.2, min_audit=5, seed=0)
    assert set(sentinel).isdisjoint(set(flagged))
    assert sentinel == sorted(sentinel)


def test_sentinel_size_respects_fraction_and_min():
    flagged = [0]
    # 99 accepted, fraction 0.1 -> ceil(9.9)=10, max with min_audit=5 -> 10
    sentinel = select_sentinel_audit(flagged, n_total=100, audit_fraction=0.1, min_audit=5, seed=0)
    assert len(sentinel) == 10
    # tiny fraction -> falls back to min_audit
    sentinel2 = select_sentinel_audit(flagged, n_total=100, audit_fraction=0.0, min_audit=5, seed=0)
    assert len(sentinel2) == 5


def test_too_few_accepted_selects_as_many_as_possible():
    flagged = list(range(0, 8))  # 8 flagged, 2 accepted
    sentinel = select_sentinel_audit(flagged, n_total=10, audit_fraction=0.5, min_audit=5, seed=0)
    assert len(sentinel) == 2
    assert set(sentinel).isdisjoint(set(flagged))


def test_no_accepted_returns_empty():
    sentinel = select_sentinel_audit(list(range(10)), n_total=10, seed=0)
    assert sentinel == []


def test_false_negatives_use_true_force_error_and_are_stable():
    # 10 trajectories; the two highest true force errors are at indices 8 and 9.
    true_error = torch.tensor([0.1, 0.2, 0.1, 0.3, 0.15, 0.25, 0.1, 0.2, 0.9, 1.0])
    flagged = [9]  # only the very top is rerun -> index 8 is a false negative
    sentinel = [0, 1, 8]
    out = evaluate_false_negatives(flagged, sentinel, true_error, high_error_quantile=0.90)
    assert out["error_basis"] == "true_force_model_error"
    assert out["n_high_error"] == 1  # 0.90 quantile keeps the single top trajectory
    # raise the bar lower to capture both tails
    out2 = evaluate_false_negatives(flagged, sentinel, true_error, high_error_quantile=0.80)
    assert out2["n_high_error"] == 2
    assert out2["n_high_error_flagged"] == 1
    assert out2["n_false_negatives"] == 1  # index 8 high-error but accepted
    assert out2["false_negative_rate"] == pytest.approx(0.5)
    # sentinel includes index 8 (a high-error accepted) -> one sentinel hit
    assert out2["n_sentinel_high_error"] == 1
    assert out2["sentinel_false_negative_rate"] == pytest.approx(1.0 / 3.0)
    # determinism: repeated evaluation is identical
    assert evaluate_false_negatives(flagged, sentinel, true_error, high_error_quantile=0.80) == out2


def test_false_negatives_reject_overlap_and_bad_inputs():
    true_error = torch.tensor([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError):
        evaluate_false_negatives([1], [1, 2], true_error)  # sentinel overlaps flagged
    with pytest.raises(ValueError):
        evaluate_false_negatives([0], [10], true_error)  # index out of range
    with pytest.raises(ValueError):
        evaluate_false_negatives([0], [1], torch.tensor([0.1, float("nan"), 0.3, 0.4]))
    with pytest.raises(ValueError):
        evaluate_false_negatives([], [], torch.empty(0))


def test_invalid_selection_params_raise():
    with pytest.raises(ValueError):
        select_sentinel_audit([], n_total=10, audit_fraction=1.5)
    with pytest.raises(ValueError):
        select_sentinel_audit([], n_total=-1)


def test_audit_summary_dict_is_serializable():
    import json

    true_error = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    flagged = [5]
    sentinel = select_sentinel_audit(flagged, n_total=6, audit_fraction=0.2, min_audit=2, seed=0)
    fn = evaluate_false_negatives(flagged, sentinel, true_error, high_error_quantile=0.80)
    summary = audit_summary_dict(
        6, flagged, sentinel, fn,
        audit_fraction=0.2, min_audit=2, high_error_quantile=0.80, seed=0,
    )
    assert summary["n_total"] == 6
    assert summary["n_flagged"] == 1
    assert summary["n_accepted"] == 5
    assert set(summary["sentinel_indices"]).isdisjoint(set(summary["flagged_indices"]))
    json.dumps(summary)  # must be JSON-serializable


def test_small_arrays_handled_gracefully():
    true_error = torch.tensor([0.5, 1.0])
    sentinel = select_sentinel_audit([0], n_total=2, audit_fraction=0.5, min_audit=5, seed=0)
    assert sentinel == [1]
    out = evaluate_false_negatives([0], sentinel, true_error, high_error_quantile=0.5)
    assert out["n_total"] == 2
    assert out["n_sentinel"] == 1


def test_no_position_error_references_in_module():
    import inspect

    import vesp.uq.audit as mod

    src = inspect.getsource(mod).lower()
    assert "position error" not in src
    assert "position-error" not in src
