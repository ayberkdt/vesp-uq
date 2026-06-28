"""Backward-compatible facade for trajectory risk scoring + selective rerun.

The implementation was split into focused modules:

- :mod:`vesp.uq.scoring`   -- per-trajectory scoring (``score_sigma_profile``, ``TrajectoryScore``,
  scoring-mode classification, ``aggregate_trajectory_error``, ``calibrate_risk_threshold``);
- :mod:`vesp.uq.selection` -- rerun selection (``select_reruns``, ``run_risk_screening``,
  ``RiskScreeningReport``);
- :mod:`vesp.uq.domain_support` -- domain-support / OOD helpers and backends.

This module re-exports the historical public (and internal) names so existing imports such as
``from vesp.uq.trajectory import score_sigma_profile, select_reruns, run_risk_screening`` keep
working unchanged. Prefer importing from the focused modules in new code.
"""

from __future__ import annotations

from vesp.uq.scoring import (  # noqa: F401
    _CALIBRATED_SUPERVISOR_MODES,
    _CANONICAL_ALIASES,
    _EXPECTED_MODES,
    _EXPECTED_ONLY_MODES,
    _SIGMA_MODES,
    _SUPERVISOR_MODES,
    SCORING_FUNCTIONS,
    TRUE_ERROR_AGGREGATORS,
    TrajectoryScore,
    _absolute_altitude_weight,
    _altitude_weight,
    _as_1d,
    _normalize_weights,
    _relative_altitude_weight,
    _validate_scoring,
    _weighted_quantile,
    _wmean,
    aggregate_trajectory_error,
    calibrate_risk_threshold,
    canonical_scoring_name,
    is_absolute_scoring,
    is_expected_only_scoring,
    is_relative_scoring,
    score_sigma_profile,
)
from vesp.uq.selection import (  # noqa: F401
    FRACTION_POLICIES,
    RiskScreeningReport,
    _spearman,
    run_risk_screening,
    select_reruns,
)

__all__ = [
    "SCORING_FUNCTIONS",
    "TRUE_ERROR_AGGREGATORS",
    "FRACTION_POLICIES",
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
]
