"""Unit + smoke tests for the learned supervisor (Design A)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vesp.common.config import load_config
from vesp.uq.ablation import learned_supervisor_run
from vesp.uq.altitude_controlled import spearman
from vesp.uq.learned_supervisor import (
    DEFAULT_BETAS,
    apply_learned_supervisor,
    fit_learned_supervisor,
    supervisor_components,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "vespuq" / "vespuq_smoke.yaml"


def _toy_components(n_traj=60, n_points=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    ee = torch.rand(n_traj, n_points, generator=g, dtype=torch.float64) + 0.1
    rel = torch.rand(n_traj, n_points, generator=g, dtype=torch.float64) + 0.1
    dr = torch.rand(n_traj, n_points, generator=g, dtype=torch.float64)
    return {"expected_error": ee, "rel_alt": rel, "domain_risk": dr, "n_points": n_points}


def test_default_betas_reproduce_handset_formula():
    c = _toy_components()
    score = apply_learned_supervisor(c, DEFAULT_BETAS)
    # explicit hand-set supervisor: p95(ee * rel_alt * (1 + 1*dr))
    explicit = torch.quantile(c["expected_error"] * c["rel_alt"] * (1.0 + c["domain_risk"]), 0.95, dim=1)
    assert torch.allclose(score, explicit, atol=1e-12)


def test_fit_recovers_a_better_exponent_set():
    # construct a target that depends mostly on expected_error^2 -> fit should raise b1, drop others
    c = _toy_components(n_traj=200, n_points=24, seed=1)
    target = torch.quantile(c["expected_error"] ** 2.0, 0.95, dim=1)
    fit = fit_learned_supervisor(c, target)
    assert fit["improves_on_fit_split"] is True
    assert fit["fit_spearman"] >= fit["baseline_spearman"]
    assert fit["betas"][0] >= 1.0  # expected-error exponent pushed up


def test_fit_includes_default_and_never_worse():
    c = _toy_components(seed=2)
    target = torch.rand(60, dtype=torch.float64)
    fit = fit_learned_supervisor(c, target)
    base = spearman(apply_learned_supervisor(c, DEFAULT_BETAS), target)
    assert fit["fit_spearman"] >= base - 1e-9  # grid contains (1,1,1)


@pytest.mark.skipif(not SMOKE_CONFIG.exists(), reason="smoke config missing")
def test_supervisor_components_match_plugin_supervisor():
    from vesp.uq.experiment import _build_trajectories
    from vesp.uq.risk_baselines import prepare

    cfg = load_config(str(SMOKE_CONFIG))
    cfg["_config_path"] = str(SMOKE_CONFIG)
    plugin, _s, _tr, _h, dtype, _ = prepare(cfg)
    traj = _build_trajectories(cfg["uq"]["screening"], seed=0, dtype=dtype, config=cfg)["trajectories"]
    comps = supervisor_components(plugin, traj)
    learned_default = apply_learned_supervisor(comps, DEFAULT_BETAS)
    # the plugin's own supervisor_rel_p95 should match beta=(1,1,1) closely (same formula)
    scored = plugin.score_ensemble(traj, scoring="supervisor_rel_p95")
    plugin_score = torch.tensor([s.risk_score for s in scored], dtype=torch.float64)
    assert spearman(learned_default, plugin_score) > 0.999


@pytest.mark.skipif(not SMOKE_CONFIG.exists(), reason="smoke config missing")
def test_learned_supervisor_run_smoke():
    cfg = load_config(str(SMOKE_CONFIG))
    cfg["_config_path"] = str(SMOKE_CONFIG)
    run = learned_supervisor_run(cfg, seed=0)
    methods = {r["method"] for r in run["rows"]}
    assert methods == {"supervisor_handtuned", "supervisor_learned"}
    assert len(run["betas"]) == 3
