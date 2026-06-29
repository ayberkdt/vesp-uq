"""Unit + smoke tests for score variants (WP6) and expanded baselines (WP5)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vesp.common.config import load_config
from vesp.uq.ablation import expanded_baselines_run, score_variant_run
from vesp.uq.altitude_controlled import spearman
from vesp.uq.expanded_baselines import (
    altitude_bin_rmse_lookup,
    altitude_uncertainty_hybrid,
    apply_ridge_ranker,
    fit_ridge_ranker,
    zscore,
)
from vesp.uq.score_variants import SCORE_VARIANTS

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "vespuq" / "vespuq_smoke.yaml"


def test_zscore_standardizes():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    z = zscore(x)
    assert abs(float(z.mean())) < 1e-12
    assert abs(float(z.std()) - 1.0) < 1e-6


def test_altitude_uncertainty_hybrid_weight_zero():
    alt = torch.tensor([1.0, 2.0, 3.0, 4.0])
    unc = torch.tensor([4.0, 1.0, 2.0, 3.0])
    s0 = altitude_uncertainty_hybrid(alt, unc, 0.0)
    assert abs(spearman(s0, zscore(alt)) - 1.0) < 1e-9  # a=0 -> pure altitude order


def test_ridge_recovers_linear_ranking():
    torch.manual_seed(0)
    X = torch.randn(200, 3, dtype=torch.float64)
    w_true = torch.tensor([1.5, -2.0, 0.5], dtype=torch.float64)
    y = X @ w_true
    model = fit_ridge_ranker(X[:150], y[:150], lam=1e-3)
    pred = apply_ridge_ranker(model, X[150:])
    assert spearman(pred, y[150:]) > 0.95


def test_altitude_bin_rmse_lookup_tracks_altitude():
    # held-out error magnitude grows toward low radius
    n = 400
    r = torch.linspace(1.0, 1.6, n)
    pos = torch.zeros(n, 3, dtype=torch.float64)
    pos[:, 0] = r
    err = torch.zeros(n, 3, dtype=torch.float64)
    err[:, 0] = (1.6 - r)  # larger error at low radius
    low_traj = torch.tensor([[1.02, 0.0, 0.0], [1.05, 0.0, 0.0]], dtype=torch.float64)
    high_traj = torch.tensor([[1.5, 0.0, 0.0], [1.55, 0.0, 0.0]], dtype=torch.float64)
    scores = altitude_bin_rmse_lookup(pos, err, [low_traj, high_traj])
    assert scores.numel() == 2
    assert float(scores[0]) > float(scores[1])  # low-altitude trajectory flagged riskier


@pytest.mark.skipif(not SMOKE_CONFIG.exists(), reason="smoke config missing")
def test_score_variant_run_smoke():
    cfg = load_config(str(SMOKE_CONFIG))
    cfg["_config_path"] = str(SMOKE_CONFIG)
    run = score_variant_run(cfg, seed=0)
    variants = {r["variant"] for r in run["rows"]}
    assert variants == set(SCORE_VARIANTS)
    assert "mahalanobis_residual" in run["unavailable"]
    assert all(r["test_spearman"] is not None for r in run["rows"])


@pytest.mark.skipif(not SMOKE_CONFIG.exists(), reason="smoke config missing")
def test_expanded_baselines_run_smoke():
    cfg = load_config(str(SMOKE_CONFIG))
    cfg["_config_path"] = str(SMOKE_CONFIG)
    run = expanded_baselines_run(cfg, seed=0)
    names = {r["baseline"] for r in run["rows"]}
    assert "learned_ridge_supervisor" in names
    assert "altitude+uncertainty_hybrid" in names
    assert "altitude_bin_rmse_lookup" in names
    assert "learned_ridge_lambda" in run["selection"]
