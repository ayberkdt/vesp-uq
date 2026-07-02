"""VESP-UQ: equivalent-source uncertainty calibration layer for residual-gravity surrogates.

This package reframes the equivalent-source machinery as a surrogate-interface-agnostic
*uncertainty* layer (not a better point-estimate surrogate). The headline object is
:class:`VESPUQPlugin`, which fits the calibrated linear-Gaussian error posterior and scores
Monte Carlo trajectories for force-risk follow-up prioritization. See
``VESP_UQ_pipeline_and_usefulness_plan`` for the full positioning.
"""

from vesp.uq.data import (
    UQSamples,
    load_uq_samples_from_csv,
    make_synthetic_uq_samples,
    split_uq_samples,
    validate_uq_samples,
)
from vesp.uq.io import TrajectoryDataset, flatten_acceleration_pairs, load_trajectory_csv
from vesp.uq.metrics import (
    diagonal_covariances,
    mahalanobis_squared,
    vector_calibration_metrics,
)
from vesp.uq.plugin import CovariancePrediction, UncertaintyPrediction, VESPUQPlugin
from vesp.uq.scoring import (
    TrajectoryScore,
    aggregate_trajectory_error,
    calibrate_risk_threshold,
    canonical_scoring_name,
    is_absolute_scoring,
    is_expected_only_scoring,
    is_relative_scoring,
    score_sigma_profile,
)
from vesp.uq.selection import (
    RiskScreeningReport,
    run_risk_screening,
    select_reruns,
)

__all__ = [
    "VESPUQPlugin",
    "UncertaintyPrediction",
    "CovariancePrediction",
    "TrajectoryScore",
    "RiskScreeningReport",
    "score_sigma_profile",
    "select_reruns",
    "run_risk_screening",
    "aggregate_trajectory_error",
    "calibrate_risk_threshold",
    "canonical_scoring_name",
    "is_relative_scoring",
    "is_absolute_scoring",
    "is_expected_only_scoring",
    "UQSamples",
    "load_uq_samples_from_csv",
    "split_uq_samples",
    "validate_uq_samples",
    "make_synthetic_uq_samples",
    "vector_calibration_metrics",
    "mahalanobis_squared",
    "diagonal_covariances",
    "TrajectoryDataset",
    "load_trajectory_csv",
    "flatten_acceleration_pairs",
]
