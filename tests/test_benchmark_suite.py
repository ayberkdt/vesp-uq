"""Unit + smoke tests for the VESP-UQ benchmark suite (WP1-WP3)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from vesp.common.config import load_config
from vesp.uq.suite import (
    DEFAULT_SELECTORS,
    aggregate_ranking,
    band_label,
    compute_run,
    mean_std,
    run_suite,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "vespuq" / "vespuq_smoke.yaml"


def test_mean_std_ignores_none_and_nan():
    out = mean_std([1.0, 2.0, 3.0, None, float("nan")])
    assert out["n"] == 3
    assert abs(out["mean"] - 2.0) < 1e-12
    assert abs(out["std"] - math.sqrt(2.0 / 3.0)) < 1e-12
    assert mean_std([])["mean"] is None


def test_band_label():
    assert band_label({"data": {"path": "data/lunar_grail_gl0420a_L90_residual.csv"}}) == "L90"
    assert band_label({"data": {"path": "data/lunar_grail_gl0420a_L60_residual.csv"}}) == "L60"
    assert band_label({"output": {"run_name": "vespuq_real_lunar_L90"}}) == "L90"


def test_aggregate_ranking_groups_by_band_selector_fraction():
    rows = [
        {"band": "L60", "seed": 0, "selector": "x", "rerun_fraction": 0.1, "spearman": 0.4,
         "capture_rate": 0.5, "precision": 0.2, "lift_over_random": 2.0,
         "force_error_ratio_flagged_to_accepted": 1.1, "mean_true_error_flagged": 1.0,
         "mean_true_error_accepted": 0.9, "runtime_ms_per_traj": 1.0},
        {"band": "L60", "seed": 1, "selector": "x", "rerun_fraction": 0.1, "spearman": 0.6,
         "capture_rate": 0.5, "precision": 0.2, "lift_over_random": 2.0,
         "force_error_ratio_flagged_to_accepted": 1.3, "mean_true_error_flagged": 1.0,
         "mean_true_error_accepted": 0.9, "runtime_ms_per_traj": 1.0},
    ]
    agg = aggregate_ranking(rows)
    key = ("L60", "x", 0.1)
    assert key in agg
    assert agg[key]["n_seeds"] == 2
    assert abs(agg[key]["spearman"]["mean"] - 0.5) < 1e-12


@pytest.mark.skipif(not SMOKE_CONFIG.exists(), reason="smoke config missing")
def test_compute_run_smoke_structure_and_determinism():
    cfg = load_config(str(SMOKE_CONFIG))
    cfg["_config_path"] = str(SMOKE_CONFIG)
    fractions = (0.1, 0.2)
    a = compute_run(cfg, seed=0, rerun_fractions=fractions, selectors=DEFAULT_SELECTORS)
    assert a["ranking_rows"], "expected ranking rows"
    assert any(r["region"] == "all" for r in a["calibration_rows"])
    ac = a["altitude_controlled"]
    assert set(ac) == {"within_bin", "partial", "matched"}
    assert "vespuq_supervisor" in ac["partial"]

    # deterministic metrics for a fixed seed (timings excluded)
    b = compute_run(cfg, seed=0, rerun_fractions=fractions, selectors=DEFAULT_SELECTORS)

    def metric(run):
        return [(r["selector"], r["rerun_fraction"], r["spearman"], r["capture_rate"])
                for r in run["ranking_rows"]]

    assert metric(a) == metric(b)


@pytest.mark.skipif(not SMOKE_CONFIG.exists(), reason="smoke config missing")
def test_run_suite_writes_manifested_artifacts(tmp_path):
    cfg = load_config(str(SMOKE_CONFIG))
    cfg["_config_path"] = str(SMOKE_CONFIG)
    out = tmp_path / "suite"
    run_suite(
        [cfg],
        seeds=(0,),
        rerun_fractions=(0.1, 0.2),
        selectors=DEFAULT_SELECTORS,
        out_dir=out,
        make_plots=False,
    )
    for name in (
        "benchmark_runs.csv",
        "benchmark_summary.csv",
        "benchmark_summary.md",
        "decision_quality.csv",
        "decision_quality.md",
        "significance_summary.csv",
        "significance_summary.md",
        "rerun_budget_curves.csv",
        "selector_ablation.csv",
        "calibration_summary.csv",
        "partial_correlation_summary.csv",
        "matched_altitude_pairs.csv",
        "matched_altitude_summary.md",
        "manifest.json",
    ):
        assert (out / name).exists(), f"missing {name}"

    manifest = json.loads((out / "manifest.json").read_text())
    entry = manifest["artifacts"]["benchmark_runs.csv"]
    assert entry["sha256"] and entry["bytes"] > 0
