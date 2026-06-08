"""VESP-UQ: equivalent-source uncertainty calibration layer for residual-gravity surrogates.

This package reframes the equivalent-source machinery as a surrogate-agnostic *uncertainty*
layer (not a better point-estimate surrogate). The headline object is :class:`VESPUQPlugin`,
which fits the calibrated linear-Gaussian error posterior and scores Monte Carlo trajectories
for selective high-fidelity rerun. See ``VESP_UQ_pipeline_and_usefulness_plan`` for the full
positioning.
"""

from vesp.uq.data import (
    UQSamples,
    load_uq_samples_from_csv,
    make_synthetic_uq_samples,
    split_uq_samples,
    validate_uq_samples,
)
from vesp.uq.metrics import (
    diagonal_covariances,
    mahalanobis_squared,
    vector_calibration_metrics,
)
from vesp.uq.plugin import CovariancePrediction, UncertaintyPrediction, VESPUQPlugin
from vesp.uq.trajectory import (
    RiskScreeningReport,
    TrajectoryScore,
    aggregate_trajectory_error,
    run_risk_screening,
    score_sigma_profile,
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
    "UQSamples",
    "load_uq_samples_from_csv",
    "split_uq_samples",
    "validate_uq_samples",
    "make_synthetic_uq_samples",
    "vector_calibration_metrics",
    "mahalanobis_squared",
    "diagonal_covariances",
]
