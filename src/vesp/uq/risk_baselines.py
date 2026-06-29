"""Reusable core for trajectory force-risk baseline comparison.

These helpers were factored out of ``scripts/compare_risk_baselines.py`` so the same fit /
true-force-error / baseline-score-assembly logic can be reused by the benchmark suite
(:mod:`vesp.uq.suite`) without re-implementation or importing from a script module.

Everything targets trajectory-level true FORCE-MODEL error (``a_reference - a_surrogate``),
never position error.
"""

from __future__ import annotations

import torch

from vesp.common.config import get_dtype
from vesp.uq.baselines import (
    altitude_residual_expected_scores,
    domain_support_scores,
    fit_altitude_expected_curve,
    knn_p95_scores,
    low_altitude_exposure_scores,
    min_altitude_scores,
    vespuq_scores,
)
from vesp.uq.data import split_uq_samples
from vesp.uq.ensemble import nearest_neighbor_error_magnitude
from vesp.uq.experiment import _load_samples
from vesp.uq.integrity.split_guard import forbid_oracle
from vesp.uq.plugin import VESPUQPlugin
from vesp.uq.scoring import aggregate_trajectory_error

# Baseline scoring modes used for the two VESP-UQ entries (kept identical to the legacy script).
UNCERTAINTY_SCORING = "mean"  # mean predictive sigma -- uncertainty-only (no bias / altitude)
SUPERVISOR_SCORING = "supervisor_rel_p95"  # full supervisor (expected error * altitude * domain)
ALTITUDE_RESIDUAL_AGGREGATOR = "p95"

__all__ = [
    "UNCERTAINTY_SCORING",
    "SUPERVISOR_SCORING",
    "ALTITUDE_RESIDUAL_AGGREGATOR",
    "prepare",
    "true_force_error",
    "assemble_baseline_scores",
]


def prepare(config: dict):
    """Fit VESP-UQ on the train split; return (plugin, samples, train, held, dtype, seed).

    The split and trajectory generation are both keyed on ``config['seed']``, so cloning the
    config with a fresh ``seed`` yields an independent realization (used for multi-seed runs).
    """

    dtype = get_dtype(config)
    samples = _load_samples(config, dtype)
    seed = int(config.get("seed", 0))
    train, held = split_uq_samples(
        samples, train_fraction=float(config.get("data", {}).get("train_fraction", 0.7)), seed=seed
    )
    plugin = VESPUQPlugin.from_config(config)
    plugin.fit(train.positions, train.surrogate, train.reference)
    return plugin, samples, train, held, dtype, seed


def true_force_error(trajectories, *, residuals, held, aggregator, dtype, weights=None):
    """Trajectory-level true FORCE error: direct residual pairs if present, else held-out NN oracle.

    Returns ``(true_error, source)`` where ``source`` records which oracle produced the target.
    """

    if weights is not None:
        weights = list(weights)
        if len(weights) != len(trajectories):
            raise ValueError("weights must be None or one weight vector per trajectory")
    out = torch.empty(len(trajectories), dtype=torch.float64)
    if residuals is not None:
        for i, res in enumerate(residuals):
            mag = torch.linalg.norm(torch.as_tensor(res, dtype=torch.float64), dim=-1)
            out[i] = aggregate_trajectory_error(
                mag, aggregator, weights=None if weights is None else weights[i]
            )
        return out, "residual_csv"
    for i, traj in enumerate(trajectories):
        nn = nearest_neighbor_error_magnitude(traj.to(dtype), held.positions, held.error)
        out[i] = aggregate_trajectory_error(
            nn.to(torch.float64), aggregator, weights=None if weights is None else weights[i]
        )
    return out, "nn_oracle_heldout"


def assemble_baseline_scores(config: dict, plugin, trajectories, train_positions, *, weights=None):
    """Assemble the baseline -> per-trajectory-score mapping for a fitted plugin + trajectories.

    Mirrors the legacy ``baseline_scores_for`` exactly: the trivial altitude/kNN baselines, the
    uncertainty-only and full supervisor VESP-UQ scores, the altitude-residual P1 scores, and
    (when enabled on the plugin) the calibrated supervisor and domain-support scores.
    """

    # G2: assert the true-error oracle never reaches the score-assembly seam (untagged inputs pass).
    forbid_oracle(trajectories, train_positions, weights)
    low_alt = float(config.get("uq", {}).get("risk", {}).get("low_altitude_radius", 1.15))
    curve_positions = getattr(plugin, "val_positions", None)
    if curve_positions is None:
        curve_positions = train_positions
    curve = fit_altitude_expected_curve(plugin, curve_positions)
    scores = {
        "min_altitude": min_altitude_scores(trajectories),
        "low_altitude_exposure": low_altitude_exposure_scores(
            trajectories, low_altitude_radius=low_alt, weights=weights
        ),
        "knn_p95": knn_p95_scores(train_positions, trajectories),
        "uncertainty_only": vespuq_scores(plugin, trajectories, UNCERTAINTY_SCORING, weights=weights),
        "altitude_residual_expected_ratio": altitude_residual_expected_scores(
            plugin, trajectories, curve=curve, mode="ratio",
            aggregator=ALTITUDE_RESIDUAL_AGGREGATOR, weights=weights,
        ),
        "altitude_residual_expected_delta": altitude_residual_expected_scores(
            plugin, trajectories, curve=curve, mode="delta",
            aggregator=ALTITUDE_RESIDUAL_AGGREGATOR, weights=weights,
        ),
        "supervisor": vespuq_scores(plugin, trajectories, SUPERVISOR_SCORING, weights=weights),
    }
    if (getattr(plugin, "calibrated_supervisor", None) or {}).get("enabled", False):
        scores["calibrated_supervisor"] = vespuq_scores(
            plugin, trajectories, "calibrated_supervisor_p95", weights=weights
        )
    if getattr(plugin, "domain_support", False):
        scores["domain_support"] = domain_support_scores(plugin, trajectories, weights=weights)
    return scores
