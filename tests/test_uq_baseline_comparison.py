"""Tests for the WP-D GP UQ baseline + the VESP-UQ vs GP comparison runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vesp.common.config import load_config
from vesp.uq.baselines import GPResidualUQ

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "vespuq" / "vespuq_smoke.yaml"


def _calibrated_field(n=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = 1.1 + 0.4 * torch.rand(n, 3, generator=g, dtype=torch.float64)
    # smooth spatial mean error + homoscedastic noise -> a GP should recover calibrated std
    mean = torch.stack([torch.sin(pos[:, 0]), torch.cos(pos[:, 1]), 0.2 * pos[:, 2]], dim=-1)
    noise = 0.05 * torch.randn(n, 3, generator=g, dtype=torch.float64)
    return pos, mean + noise


def test_gp_predict_interface_shapes():
    pos, err = _calibrated_field()
    gp = GPResidualUQ(seed=0).fit(pos, err)
    pred = gp.predict(pos[:50])
    assert pred.mean_error.shape == (50, 3)
    assert pred.std_components.shape == (50, 3)
    assert pred.sigma.shape == (50,)
    assert pred.expected_error.shape == (50,)
    assert torch.all(pred.std_components > 0)


def test_gp_recovers_reasonable_calibration():
    pos, err = _calibrated_field(n=600, seed=1)
    split = 450
    gp = GPResidualUQ(seed=1).fit(pos[:split], err[:split])
    cal = gp.evaluate_calibration(pos[split:], err[split:],
                                  altitude_bands={"low": [1.0, 1.6]})
    z = cal["all"]["z_std"]
    assert 0.5 < z < 2.0  # a smooth field with homoscedastic noise -> roughly calibrated GP
    assert "radial_z_std" in cal["all"]  # component metrics flow through


def test_gp_score_trajectories_length_and_order():
    pos, err = _calibrated_field()
    gp = GPResidualUQ(seed=0).fit(pos, err)
    trajs = [1.1 + 0.3 * torch.rand(20, 3, dtype=torch.float64) for _ in range(5)]
    scores = gp.score_trajectories(trajs)
    assert scores.shape == (5,)
    assert torch.all(torch.isfinite(scores))


def test_gp_fit_validates_shape():
    with pytest.raises(ValueError):
        GPResidualUQ().fit(torch.randn(10, 3), torch.randn(10, 2))


def test_gp_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        GPResidualUQ().predict(torch.randn(3, 3, dtype=torch.float64))


@pytest.mark.skipif(not SMOKE_CONFIG.exists(), reason="smoke config missing")
def test_comparison_runner_writes_manifested_artifacts(tmp_path):
    from vesp.uq.uq_baseline_comparison import run_uq_baseline_comparison

    cfg = load_config(str(SMOKE_CONFIG))
    cfg["_config_path"] = str(SMOKE_CONFIG)
    out = tmp_path / "cmp"
    run_uq_baseline_comparison([cfg], seeds=(0,), out_dir=out)
    for name in ("uq_baseline_comparison.csv", "uq_baseline_decision.csv",
                 "uq_baseline_comparison.md", "manifest.json"):
        assert (out / name).exists(), f"missing {name}"
    manifest = json.loads((out / "manifest.json").read_text())
    entry = manifest["artifacts"]["uq_baseline_comparison.csv"]
    assert entry["sha256"] and entry["bytes"] > 0
