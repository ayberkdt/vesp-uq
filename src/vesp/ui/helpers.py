"""Small display and file helpers shared by Mission Console pages."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# Canonical risk-scoring modes, mirroring ``vesp.uq.scoring.SCORING_FUNCTIONS`` with the
# backward-compatible aliases collapsed to their canonical names. Kept as plain strings so the UI
# shell stays torch-free at import time (see ``test_ui_imports_stay_light``); the Train and Screen
# pages prepend their own "use the config/model default" sentinel. ``test_vespuq_ui`` guards this
# list against backend drift so a newly added scoring mode cannot silently miss the UI.
SCORING_MODES: tuple[str, ...] = (
    # supervisor -- per-trajectory ranking (relative) and fixed-reference budget (absolute)
    "supervisor_rel", "supervisor_rel_p95", "supervisor_abs", "supervisor_abs_p95",
    "calibrated_supervisor", "calibrated_supervisor_p95",
    # expected force error on an absolute scale
    "expected_abs", "expected_abs_p95", "expected_low_alt",
    # directional / covariance-geometry (M1) -- within-band ranking beyond the scalar altitude cue
    "radial_expected", "radial_expected_diag", "anisotropy_gated", "largest_eigenvalue",
    # epistemic-targeted (M2) -- up-weight reducible uncertainty where a rerun actually helps
    "expected_epistemic",
    # legacy sigma-only
    "max", "mean", "low_alt_integral", "time_above", "combined",
)

# Short one-line tooltips, keyed by the base mode name (``_p95`` variants inherit their base hint).
SCORING_HINTS: dict[str, str] = {
    "supervisor_rel": "Per-trajectory altitude-normalized supervisor score (ranking; relative).",
    "supervisor_abs": "Supervisor score on a fixed altitude reference (absolute budget / zero-alarm).",
    "calibrated_supervisor": "Supervisor driven by the calibrated point risk.",
    "expected_abs": "Expected force error on an absolute scale (physical budgets).",
    "expected_low_alt": "Expected force error weighted toward low altitude.",
    "radial_expected": "M1: |bias . r_hat| + exact radial spread sqrt(r_hat' Sigma r_hat) from the 3x3 covariance.",
    "radial_expected_diag": "M1 ablation: radial_expected under the diagonal-covariance approximation.",
    "anisotropy_gated": "M1: expected error x anisotropy multiplier from the 3x3 covariance.",
    "largest_eigenvalue": "M1: worst-direction predictive std, sqrt(lambda_max).",
    "expected_epistemic": "M2: expected error weighted by the reducible (epistemic) fraction.",
    "max": "Legacy: peak per-point sigma.",
    "mean": "Legacy: mean per-point sigma.",
    "low_alt_integral": "Legacy: sigma integrated over the low-altitude arc.",
    "time_above": "Legacy: fraction of points whose sigma exceeds the policy threshold.",
    "combined": "Legacy: blended sigma summary.",
}


def scoring_hint(mode: str) -> str:
    """Tooltip for a scoring ``mode`` (``_p95`` variants fall back to their base hint)."""

    if mode in SCORING_HINTS:
        return SCORING_HINTS[mode]
    if mode.endswith("_p95"):
        base = SCORING_HINTS.get(mode[: -len("_p95")])
        if base:
            return f"{base} (p95-aggregated per trajectory)"
    return ""


def fmt(value: Any, digits: int = 4) -> str:
    """Format a numeric UI value, returning ``--`` for missing or invalid data."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    return f"{number:.{digits}g}" if math.isfinite(number) else "--"


def safe_read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read a JSON object without allowing filesystem or decode errors to escape."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, "expected a JSON object"
    return payload, None
