"""G3 -- metric-range invariants: loud failure, never silent.

A fabricated or buggy metric almost always lands outside its mathematically valid range (an AUROC of
1.5, a capture rate of -0.2, a negative z-std). This module makes that abort at the moment the value
is recorded, with a precise ``where`` location, instead of letting it flow into a journal table.

Contract (binding on every metric-record point in the suite / baseline comparison):

* ``None`` and ``NaN`` pass through unchecked -- a band / pair can be *legitimately* absent (a
  degenerate split yields a ``nan`` Spearman by design), and that is not a fabrication.
* A *finite* value of a **known** metric outside its domain raises :class:`MetricRangeError`.
* A non-finite (``+/-inf``) value of a known metric also raises -- an infinite metric is never a
  legitimate "missing" marker, and the whole point of G3 is that it fails loudly.
* An **unknown** metric name passes through untouched (this module never invents a domain).

Domains carry a small absolute tolerance so a value that is in-range up to floating-point round-off
(e.g. an AUROC computed as ``1.0000000002``) is accepted; anything past the tolerance is a real
out-of-domain value and raises.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

# Absolute slack applied to both ends of every domain, to absorb floating-point round-off only.
_TOL = 1.0e-6

# Bounded probabilities / proportions in [0, 1].
_UNIT = (0.0, 1.0)
# Non-negative, unbounded above (magnitudes, dispersions, interval scores).
_NONNEG = (0.0, math.inf)

# Valid closed-interval domain per known metric name. Anything not listed is treated as
# free / pass-through (we never fabricate a domain for an unknown key). Keys mirror the metric
# columns emitted by ``benchmarking.py``, ``suite.py``, and ``uq_baseline_comparison.py``.
METRIC_DOMAINS: dict[str, tuple[float, float]] = {
    # --- ranking / detection (benchmarking.py) ---
    "spearman": (-1.0, 1.0),
    "capture_rate": _UNIT,
    "precision": _UNIT,
    "auroc": _UNIT,
    "auprc": _UNIT,
    "lift_over_random": _NONNEG,
    "force_error_ratio_flagged_to_accepted": _NONNEG,
    "mean_true_error_flagged": _NONNEG,
    "mean_true_error_accepted": _NONNEG,
    # --- budget-integrated decision quality (benchmarking.py) ---
    "capture_auc": _NONNEG,
    "capture_auc_normalized": _UNIT,
    "oracle_regret": _UNIT,
    # --- calibration (plugin.evaluate_calibration -> suite / baseline comparison) ---
    "z_std": _NONNEG,
    "radial_z_std": _NONNEG,
    "tangential_z_std": _NONNEG,
    "picp_50": _UNIT,
    "picp_68": _UNIT,
    "picp_90": _UNIT,
    "picp_95": _UNIT,
    "ellipsoid_picp_90": _UNIT,
    "radial_picp_90": _UNIT,
    "tangential_picp_90": _UNIT,
    "calibration_error_90": _NONNEG,
    "radial_winkler_90": _NONNEG,
    "tangential_winkler_90": _NONNEG,
    "rmse": _NONNEG,
    "mean_pred_std": _NONNEG,
    "mean_epistemic_std": _NONNEG,
    "mean_radius": _NONNEG,
    "crps": _NONNEG,
    "n": _NONNEG,
}


class MetricRangeError(ValueError):
    """Raised when a finite metric value lands outside its valid domain (a G3 violation)."""


def _coerce(value: Any) -> float | None:
    """Return ``float(value)`` or ``None`` if the value is not numeric (e.g. a label string)."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_metric(name: str, value: Any, *, where: str = "") -> Any:
    """Validate one metric value against its domain; return it unchanged when it passes.

    ``None`` and ``NaN`` pass through (legitimately missing). A finite value of a *known* metric
    outside ``METRIC_DOMAINS[name]`` (or a ``+/-inf`` value of a known metric) raises
    :class:`MetricRangeError` tagged with ``where``. Unknown metric names and non-numeric values
    pass through untouched.
    """

    if value is None:
        return value
    f = _coerce(value)
    if f is None:  # not a number (a label / string column): not our concern
        return value
    if math.isnan(f):  # legitimately missing
        return value

    domain = METRIC_DOMAINS.get(name)
    if domain is None:  # unknown metric: never fabricate a domain
        return value

    loc = f" at {where}" if where else ""
    if not math.isfinite(f):
        raise MetricRangeError(f"metric {name!r}={f} is not finite{loc}")
    lo, hi = domain
    if f < lo - _TOL or f > hi + _TOL:
        raise MetricRangeError(
            f"metric {name!r}={f!r} out of domain [{lo}, {hi}]{loc}"
        )
    return value


def validate_row(row: Mapping[str, Any], *, where: str = "") -> None:
    """Validate every *known* metric key present in ``row`` (a single result record).

    Unknown keys (``band``, ``seed``, ``selector``, label columns, ...) are ignored. Raises
    :class:`MetricRangeError` on the first out-of-domain finite value, with ``where`` extended by the
    offending key so the abort message points straight at it.
    """

    for key, value in row.items():
        if key in METRIC_DOMAINS:
            sub = f"{where}[{key}]" if where else key
            validate_metric(key, value, where=sub)
