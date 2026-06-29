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
* :mod:`vesp.uq.integrity.number_audit` (G1) -- no-orphan-number auditor: every number in a report /
  LaTeX table must trace to a source CSV cell recorded in a checksummed manifest, or it is an orphan.
* :mod:`vesp.uq.integrity.reproducibility` (G5) -- determinism gate: rerun a (config, seed) and the
  data tables must be byte-identical after dropping timing columns.
* :mod:`vesp.uq.integrity.claim_lint` (G6) -- forbidden-claim linter: a report / manuscript may not
  carry phrasing the project has ruled out, unless the line is explicitly disclaimed.

The provenance-completeness checker (G7) lives with the manifest writer as
:func:`vesp.uq.io.run_artifacts.verify_manifest` and is re-exported here for convenience.

Negative-control placebos (G4) live with the cheap baselines in :mod:`vesp.uq.baselines`
(``label_shuffled_scores``) and are asserted at chance by the suite; see
:func:`vesp.uq.suite.assert_placebos_at_chance`.

All public functions take explicit arguments and carry no global state beyond the deliberately
thread-local leakage flag in :mod:`split_guard`.
"""

from __future__ import annotations

from vesp.uq.integrity.claim_lint import FORBIDDEN, lint_report, scan_text
from vesp.uq.integrity.metric_invariants import (
    METRIC_DOMAINS,
    MetricRangeError,
    validate_metric,
    validate_row,
)
from vesp.uq.integrity.number_audit import (
    audit_latex_tables,
    audit_report_numbers,
    collect_csv_values,
    verify_csv_manifests,
)
from vesp.uq.integrity.reproducibility import (
    REPRO_FILES,
    ReproducibilityError,
    assert_reproducible,
    compare_outputs,
    normalize_csv_text,
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
from vesp.uq.io.run_artifacts import verify_manifest

__all__ = [
    # G3 -- metric-range invariants
    "METRIC_DOMAINS",
    "MetricRangeError",
    "validate_metric",
    "validate_row",
    # G1 -- no-orphan-number auditor
    "audit_report_numbers",
    "audit_latex_tables",
    "collect_csv_values",
    "verify_csv_manifests",
    # G5 -- determinism / reproducibility gate
    "REPRO_FILES",
    "ReproducibilityError",
    "normalize_csv_text",
    "compare_outputs",
    "assert_reproducible",
    # G6 -- forbidden-claim linter
    "FORBIDDEN",
    "scan_text",
    "lint_report",
    # G7 -- provenance-completeness checker (lives in io.run_artifacts)
    "verify_manifest",
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
