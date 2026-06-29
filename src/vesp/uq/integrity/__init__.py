"""VESP-UQ integrity / anti-fabrication features.

This package collects the system-level guards from the *Method-Strengthening & Integrity Plan*
(Pillar II). They are designed so that a fabricated, leaked, or invalid metric becomes structurally
impossible or loudly detectable rather than silently written to a table:

* :mod:`vesp.uq.integrity.metric_invariants` (G3) -- metric-range invariants: any finite metric
  recorded outside its mathematically valid domain aborts with a precise location, instead of
  flowing into a CSV.
* :mod:`vesp.uq.integrity.split_guard` (G2) -- split-leakage & oracle-isolation guard: a thin,
  opt-in-at-the-seams way to tag the test split / true-error oracle and trip a loud error if either
  is read during a region that must not see them (e.g. score assembly / variant selection).

Negative-control placebos (G4) live with the cheap baselines in :mod:`vesp.uq.baselines`
(``label_shuffled_scores``) and are asserted at chance by the suite; see
:func:`vesp.uq.suite.assert_placebos_at_chance`.

All public functions take explicit arguments and carry no global state beyond the deliberately
thread-local leakage flag in :mod:`split_guard`.
"""

from __future__ import annotations

from vesp.uq.integrity.metric_invariants import (
    METRIC_DOMAINS,
    MetricRangeError,
    validate_metric,
    validate_row,
)
from vesp.uq.integrity.split_guard import (
    OracleLeakageError,
    Split,
    SplitLeakageError,
    Tagged,
    assert_no_test_access,
    forbid_oracle,
    reveal,
    tag,
)

__all__ = [
    # G3 -- metric-range invariants
    "METRIC_DOMAINS",
    "MetricRangeError",
    "validate_metric",
    "validate_row",
    # G2 -- split-leakage & oracle-isolation guard
    "Split",
    "Tagged",
    "SplitLeakageError",
    "OracleLeakageError",
    "tag",
    "reveal",
    "forbid_oracle",
    "assert_no_test_access",
]
