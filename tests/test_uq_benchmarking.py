"""Tests for the score-vs-true-force-error comparison utilities and the comparison script."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

from vesp.uq.benchmarking import (
    METRIC_KEYS,
    compare_baselines,
    evaluate_score_against_true_error,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_evaluate_perfect_ranking_concentrates_error():
    n = 100
    err = torch.arange(n, dtype=torch.float64)
    m = evaluate_score_against_true_error(err.clone(), err, rerun_fraction=0.10)
    assert set(METRIC_KEYS).issubset(m.keys())
    assert m["spearman"] == pytest.approx(1.0)
    assert m["capture_rate"] == pytest.approx(1.0)  # flagged top-10% == truly-high top-10%
    assert m["lift_over_random"] > 1.0
    assert m["mean_true_error_flagged"] > m["mean_true_error_accepted"]


def test_evaluate_handles_ties_and_tiny_arrays_safely():
    # all-equal scores -> tie-broken (finite) Spearman, no crash; exactly one flagged at 10%
    m = evaluate_score_against_true_error(torch.ones(3), torch.tensor([1.0, 2.0, 3.0]), rerun_fraction=0.10)
    assert m["n_trajectories"] == 3
    assert m["n_flagged"] == 1
    assert isinstance(m["spearman"], float)  # finite tie-broken value, never a crash
    # n == 1 is safe too: Spearman is undefined (nan) for a single point, but does not crash
    m1 = evaluate_score_against_true_error(torch.tensor([5.0]), torch.tensor([2.0]))
    assert m1["n_trajectories"] == 1
    assert m1["spearman"] is None or math.isnan(float(m1["spearman"]))


def test_evaluate_length_mismatch_and_empty_raise():
    with pytest.raises(ValueError):
        evaluate_score_against_true_error(torch.ones(4), torch.ones(5))
    with pytest.raises(ValueError):
        evaluate_score_against_true_error(torch.tensor([]), torch.tensor([]))


def test_compare_baselines_returns_metrics_for_every_baseline():
    err = torch.arange(40, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    scores = {
        "random": torch.rand(40, generator=g, dtype=torch.float64),
        "perfect": err.clone(),
        "anti": torch.flip(err, dims=[0]),
    }
    results = compare_baselines(scores, err, rerun_fraction=0.1)
    assert set(results.keys()) == {"random", "perfect", "anti"}
    for m in results.values():
        assert set(METRIC_KEYS).issubset(m.keys())
    assert results["perfect"]["spearman"] == pytest.approx(1.0)
    assert results["anti"]["spearman"] == pytest.approx(-1.0)


def test_compare_baselines_empty_raises():
    with pytest.raises(ValueError):
        compare_baselines({}, torch.arange(5, dtype=torch.float64))


# ---- the comparison script runs on a tiny in-test config ----

def _tiny_config():
    return {
        "seed": 0,
        "device": "cpu",
        "dtype": "float64",
        "data": {"type": "synthetic", "n": 240, "noise_std": 1.0e-4, "train_fraction": 0.7},
        "model": {"type": "multishell", "shell_alphas": [0.75, 0.9], "n_sources_per_shell": [24, 32]},
        "kernel": {"eps": 0.0},
        "uq": {
            "risk": {"scoring": "supervisor_rel", "low_altitude_radius": 1.15, "domain_support": True},
            "screening": {"n_orbits": 30, "n_points": 24, "true_error_aggregator": "p95"},
        },
    }


def test_compare_script_runs_and_writes_outputs(tmp_path):
    import compare_risk_baselines as crb

    payload = crb.run_baseline_comparison(_tiny_config(), rerun_fraction=0.2)
    assert payload["n_trajectories"] == 30
    assert payload["true_force_error_source"] == "nn_oracle_heldout"
    # all baselines present (domain support enabled in the tiny config)
    expected = {
        "random",
        "min_altitude",
        "low_altitude_exposure",
        "knn_p95",
        "uncertainty_only",
        "altitude_residual_expected_ratio",
        "altitude_residual_expected_delta",
        "supervisor",
        "domain_support",
    }
    assert set(payload["baselines"].keys()) == expected
    assert payload["best_by_spearman"] in expected
    assert payload["best_by_lift"] in expected
    assert "altitude_incremental_value" in payload
    assert expected - {"random"} <= set(payload["altitude_incremental_value"]["summary"])

    crb.write_outputs(payload, tmp_path)
    for fname in (
        "baseline_comparison.json",
        "baseline_comparison.csv",
        "baseline_comparison_paper.csv",
        "altitude_incremental_value.csv",
        "altitude_incremental_sweep.csv",
        "baseline_comparison.md",
    ):
        assert (tmp_path / fname).exists()
    # markdown is force-error oriented and never claims position-error prediction
    md = (tmp_path / "baseline_comparison.md").read_text(encoding="utf-8")
    assert "force-model" in md
    assert "position error" in md.lower()  # only as the explicit "NOT position error" disclaimer
