"""Canonical baseline registry for VESP-UQ force-risk comparisons.

This package consolidates the formerly separate baseline modules:

- :mod:`vesp.uq.baselines.core` for cheap trajectory heuristics and VESP-UQ score wrappers;
- :mod:`vesp.uq.baselines.assembly` for fitted-plugin/run assembly helpers;
- :mod:`vesp.uq.baselines.expanded` for hybrid and learned comparison baselines;
- :mod:`vesp.uq.baselines.gp` for the exact GP residual-UQ baseline.

New baselines should be registered here, then consumed through this package instead of adding new
top-level ``vesp.uq.*_baselines`` modules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from vesp.uq.baselines.assembly import (
    ALTITUDE_RESIDUAL_AGGREGATOR,
    SUPERVISOR_SCORING,
    UNCERTAINTY_SCORING,
    assemble_baseline_scores,
    prepare,
    true_force_error,
)
from vesp.uq.baselines.core import (
    AltitudeExpectedCurve,
    altitude_residual_expected_scores,
    domain_support_scores,
    fit_altitude_expected_curve,
    independent_random_scores,
    knn_p95_scores,
    label_shuffled_scores,
    low_altitude_exposure_scores,
    min_altitude_scores,
    random_scores,
    vespuq_scores,
)
from vesp.uq.baselines.expanded import (
    HYBRID_WEIGHTS,
    altitude_bin_rmse_lookup,
    altitude_ood_hybrid,
    altitude_uncertainty_hybrid,
    apply_ridge_ranker,
    fit_ridge_ranker,
    zscore,
)
from vesp.uq.baselines.gp import GPPrediction, GPResidualUQ

BaselineKind = Literal["trajectory", "plugin", "oracle_control", "expanded", "probabilistic"]


@dataclass(frozen=True)
class BaselineEntry:
    """Registry entry documenting how a baseline is built and what inputs it consumes."""

    name: str
    kind: BaselineKind
    builder: Callable
    consumes_true_error: bool = False
    description: str = ""


BASELINE_REGISTRY: dict[str, BaselineEntry] = {
    "random": BaselineEntry(
        "random",
        "trajectory",
        independent_random_scores,
        description="chance-level random score decoupled from trajectory-generation RNG",
    ),
    "label_shuffled": BaselineEntry(
        "label_shuffled",
        "oracle_control",
        label_shuffled_scores,
        consumes_true_error=True,
        description="negative control: true force-error labels permuted by seed",
    ),
    "min_altitude": BaselineEntry(
        "min_altitude",
        "trajectory",
        min_altitude_scores,
        description="periapsis altitude heuristic",
    ),
    "low_altitude_exposure": BaselineEntry(
        "low_altitude_exposure",
        "trajectory",
        low_altitude_exposure_scores,
        description="fraction of trajectory below the low-altitude radius",
    ),
    "knn_p95": BaselineEntry(
        "knn_p95",
        "trajectory",
        knn_p95_scores,
        description="95th percentile kNN distance to calibration positions",
    ),
    "domain_support": BaselineEntry(
        "domain_support",
        "plugin",
        domain_support_scores,
        description="mean VESP-UQ domain-support/OOD score along trajectory",
    ),
    "vespuq_score": BaselineEntry(
        "vespuq_score",
        "plugin",
        vespuq_scores,
        description="VESP-UQ score_ensemble wrapper for a named scoring mode",
    ),
    "altitude_residual_expected": BaselineEntry(
        "altitude_residual_expected",
        "plugin",
        altitude_residual_expected_scores,
        description="expected force error after subtracting/dividing altitude-only trend",
    ),
    "altitude_uncertainty_hybrid": BaselineEntry(
        "altitude_uncertainty_hybrid",
        "expanded",
        altitude_uncertainty_hybrid,
        description="z(altitude) plus weighted z(uncertainty)",
    ),
    "altitude_ood_hybrid": BaselineEntry(
        "altitude_ood_hybrid",
        "expanded",
        altitude_ood_hybrid,
        description="z(altitude) plus weighted z(domain-support)",
    ),
    "ridge_ranker": BaselineEntry(
        "ridge_ranker",
        "expanded",
        fit_ridge_ranker,
        consumes_true_error=True,
        description="ridge ranker over validation features, applied to held-out splits",
    ),
    "altitude_bin_rmse": BaselineEntry(
        "altitude_bin_rmse",
        "expanded",
        altitude_bin_rmse_lookup,
        description="held-out altitude-bin residual-RMSE lookup",
    ),
    "gp_residual_uq": BaselineEntry(
        "gp_residual_uq",
        "probabilistic",
        GPResidualUQ,
        description="independent-output exact GP residual uncertainty baseline",
    ),
}

__all__ = [
    "ALTITUDE_RESIDUAL_AGGREGATOR",
    "BASELINE_REGISTRY",
    "BaselineEntry",
    "GPResidualUQ",
    "GPPrediction",
    "HYBRID_WEIGHTS",
    "SUPERVISOR_SCORING",
    "UNCERTAINTY_SCORING",
    "AltitudeExpectedCurve",
    "altitude_bin_rmse_lookup",
    "altitude_ood_hybrid",
    "altitude_residual_expected_scores",
    "altitude_uncertainty_hybrid",
    "apply_ridge_ranker",
    "assemble_baseline_scores",
    "domain_support_scores",
    "fit_altitude_expected_curve",
    "fit_ridge_ranker",
    "independent_random_scores",
    "knn_p95_scores",
    "label_shuffled_scores",
    "low_altitude_exposure_scores",
    "min_altitude_scores",
    "prepare",
    "random_scores",
    "true_force_error",
    "vespuq_scores",
    "zscore",
]
