"""Tests for G4 -- built-in negative controls (placebos) at chance."""

from __future__ import annotations

import pytest
import torch

from vesp.uq.baselines import label_shuffled_scores
from vesp.uq.benchmarking import evaluate_score_against_true_error
from vesp.uq.suite import (
    PLACEBO_SELECTORS,
    PlaceboLeakageError,
    assert_placebos_at_chance,
    placebo_tolerance,
)


def _agg_entry(spearman, capture):
    """Minimal ranking_agg value carrying just the fields the placebo assertion inspects."""

    return {"spearman": {"mean": spearman}, "capture_rate": {"mean": capture}}


def test_label_shuffled_preserves_marginal_and_is_deterministic():
    te = torch.arange(1.0, 31.0, dtype=torch.float64)
    a = label_shuffled_scores(te, seed=3)
    b = label_shuffled_scores(te, seed=3)
    assert torch.equal(a, b)                                   # deterministic given the seed
    assert torch.equal(torch.sort(a).values, torch.sort(te).values)  # exact same marginal (permuted)
    assert not torch.equal(a, te)                              # but re-ordered


def test_label_shuffled_scores_at_chance():
    # a large independent permutation has ~0 rank correlation with the true error
    te = torch.rand(4000, dtype=torch.float64).abs()
    placebo = label_shuffled_scores(te, seed=0)
    m = evaluate_score_against_true_error(placebo, te, rerun_fraction=0.20)
    assert abs(float(m["spearman"])) < 0.1
    assert abs(float(m["capture_rate"]) - 0.20) < 0.1


def test_placebo_tolerance_scales_with_n():
    import math

    from vesp.uq.suite import PLACEBO_TOL_FLOOR

    assert placebo_tolerance(40) > placebo_tolerance(4000)         # tighter band at larger n
    # small runs keep the protective floor (statistical band would be noisy at that size) ...
    assert placebo_tolerance(500) == pytest.approx(PLACEBO_TOL_FLOOR)
    # ... but large journal runs drop the floor and use the tight statistical band, so a subtle
    # leak is not masked (the whole point of G4). The floor is small-n only.
    big = placebo_tolerance(280_000)
    assert big < PLACEBO_TOL_FLOOR
    assert big == pytest.approx(3.5 / math.sqrt(280_000))


def test_assert_placebos_catches_subtle_leak_at_journal_scale():
    # 0.05 aggregate Spearman is invisible under the old fixed 0.20 floor but is a clear leak at
    # journal-run scale (n_eff = 10000*28); the small-n-only floor now flags it.
    agg = {("L60", "random", 0.20): _agg_entry(0.05, 0.20)}
    with pytest.raises(PlaceboLeakageError):
        assert_placebos_at_chance(agg, primary_fraction=0.20, n_trajectories=10_000, n_seeds=28)


def test_assert_placebos_pass_on_real_run_aggregates():
    # the actual 28-seed placebo aggregates (|Spearman| <= 0.0025, capture within 0.005 of chance)
    # stay comfortably at chance under the tightened band -- no false positive on real runs.
    agg = {("L60", "label_shuffled", 0.20): _agg_entry(0.0014, 0.2020),
           ("L60", "random", 0.20): _agg_entry(0.0011, 0.1953)}
    checks = assert_placebos_at_chance(agg, primary_fraction=0.20, n_trajectories=10_000, n_seeds=28)
    assert all(c["ok"] for c in checks.values())


def test_assert_placebos_passes_at_chance():
    agg = {("L60", sel, 0.20): _agg_entry(0.01, 0.205) for sel in PLACEBO_SELECTORS}
    agg[("L60", "vespuq_supervisor", 0.20)] = _agg_entry(0.8, 0.7)  # a real method may beat chance
    checks = assert_placebos_at_chance(agg, primary_fraction=0.20, n_trajectories=500, n_seeds=3)
    assert all(c["ok"] for c in checks.values())
    assert set(checks) == {("L60", sel) for sel in PLACEBO_SELECTORS}


def test_assert_placebos_raises_on_leaked_spearman():
    # a placebo that ranks like the oracle (Spearman ~1) is the signature of a leak
    agg = {("L60", "label_shuffled", 0.20): _agg_entry(0.95, 0.9),
           ("L60", "random", 0.20): _agg_entry(0.0, 0.2)}
    with pytest.raises(PlaceboLeakageError):
        assert_placebos_at_chance(agg, primary_fraction=0.20, n_trajectories=500, n_seeds=3)


def test_assert_placebos_raises_on_capture_above_chance():
    agg = {("L60", "random", 0.20): _agg_entry(0.0, 0.95)}  # captures ~all error at a 20% budget
    with pytest.raises(PlaceboLeakageError):
        assert_placebos_at_chance(agg, primary_fraction=0.20, n_trajectories=500, n_seeds=3)


def test_assert_placebos_skips_missing_aggregates():
    agg = {("L60", "random", 0.20): _agg_entry(None, None),
           ("L60", "label_shuffled", 0.20): _agg_entry(float("nan"), float("nan"))}
    checks = assert_placebos_at_chance(agg, primary_fraction=0.20, n_trajectories=20, n_seeds=1)
    assert all(c["ok"] for c in checks.values())  # None / NaN are legitimately missing, not a leak


def test_other_budgets_are_ignored():
    # only the primary budget is asserted; an off-primary capture (== that fraction) is untouched
    agg = {("L60", "random", 0.40): _agg_entry(0.9, 0.9)}
    checks = assert_placebos_at_chance(agg, primary_fraction=0.20, n_trajectories=500, n_seeds=3)
    assert checks == {}
