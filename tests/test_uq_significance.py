"""Tests for the WP-A significance layer (paired bootstrap CI + seed Wilcoxon signed-rank)."""

from __future__ import annotations

import math

import pytest
import torch

from vesp.uq.significance import (
    paired_bootstrap_ci,
    resolve_metric,
    seed_paired_test,
)

TRUE_ERROR = torch.arange(1.0, 41.0, dtype=torch.float64)


def test_resolve_metric_names_and_callable():
    assert resolve_metric("spearman")(TRUE_ERROR, TRUE_ERROR) == pytest.approx(1.0)
    assert callable(resolve_metric("capture"))
    assert callable(resolve_metric("auroc"))
    with pytest.raises(ValueError):
        resolve_metric("nope")


def test_bootstrap_identical_scores_brackets_zero():
    a = TRUE_ERROR.clone()
    out = paired_bootstrap_ci(a, a.clone(), TRUE_ERROR, metric="spearman", n_boot=500, seed=0)
    assert out["delta"] == pytest.approx(0.0)
    assert out["ci_low"] <= 0.0 <= out["ci_high"]
    assert out["significant"] is False


def test_bootstrap_clear_separation_is_significant():
    # a ranks perfectly with truth, b ranks inversely -> delta strongly positive, CI excludes 0
    out = paired_bootstrap_ci(TRUE_ERROR, -TRUE_ERROR, TRUE_ERROR, metric="spearman",
                              n_boot=500, seed=1)
    assert out["delta"] > 0.0
    assert out["ci_low"] > 0.0
    assert out["significant"] is True
    assert out["p_value"] < 0.05


def test_bootstrap_is_deterministic_under_seed():
    args = (TRUE_ERROR, -TRUE_ERROR, TRUE_ERROR)
    a = paired_bootstrap_ci(*args, metric="auroc", n_boot=300, seed=7)
    b = paired_bootstrap_ci(*args, metric="auroc", n_boot=300, seed=7)
    assert a["ci_low"] == b["ci_low"] and a["ci_high"] == b["ci_high"]


def test_bootstrap_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_bootstrap_ci(torch.zeros(5), torch.zeros(5), torch.zeros(6))


def test_seed_paired_test_consistent_winner():
    a = [0.70, 0.72, 0.69, 0.75, 0.71]
    b = [0.60, 0.61, 0.59, 0.66, 0.62]
    out = seed_paired_test(a, b)
    assert out["n_seeds"] == 5
    assert out["mean_delta"] > 0.0
    assert out["a_wins"] is True
    assert 0.0 <= out["p_value"] <= 1.0


def test_seed_paired_test_no_difference():
    a = [0.5, 0.6, 0.7]
    out = seed_paired_test(a, list(a))
    assert out["n_seeds"] == 3  # finite pairs counted; zero diffs dropped inside Wilcoxon -> p nan
    assert out["mean_delta"] == pytest.approx(0.0)
    assert math.isnan(out["p_value"])  # no nonzero differences -> undefined signed-rank p
    assert out["a_wins"] is False


def test_seed_paired_test_drops_nonfinite_pairs():
    a = [0.7, float("nan"), 0.8]
    b = [0.6, 0.5, 0.7]
    out = seed_paired_test(a, b)
    assert out["n_seeds"] == 2


def test_seed_paired_test_length_mismatch_raises():
    with pytest.raises(ValueError):
        seed_paired_test([0.1, 0.2], [0.1])
