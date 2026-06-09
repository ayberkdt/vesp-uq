"""Absolute force-risk threshold resolution for VESP-UQ screening.

Resolves the screening threshold (and its provenance) from config, enforcing the scale rule that
a *pointwise* expected-force-error budget is only ever paired with an absolute-scale scoring mode
-- never a relative supervisor score (which is on a per-trajectory-normalized scale).

Sources (``uq.screening.threshold_source``):
- ``manual`` -- use ``uq.screening.threshold`` directly;
- ``pointwise_calibration_quantile`` -- quantile of held-out per-point ``expected_error``
  (absolute scoring only);
- ``trajectory_calibration_quantile`` -- quantile of the *same* trajectory score on a held-out
  calibration ensemble (safe for relative or absolute scoring);
- ``physical_budget`` -- a user-supplied physical acceleration-error budget (e.g. ``1e-8 m/s^2``)
  converted into model-score units (absolute scoring only; requires explicit unit metadata).
"""

from __future__ import annotations

from vesp.uq.ensemble import generate_orbit_ensemble
from vesp.uq.physical_units import acceleration_to_model_units, resolve_acceleration_scale
from vesp.uq.scoring import is_absolute_scoring, is_relative_scoring

THRESHOLD_SOURCES = (
    "manual",
    "pointwise_calibration_quantile",
    "trajectory_calibration_quantile",
    "physical_budget",
)

# Absolute-scale scoring modes a physical budget may be paired with (a physical budget is meaningless
# against a per-trajectory-normalized relative supervisor score).
PHYSICAL_BUDGET_SCORINGS = ("expected_abs", "expected_abs_p95", "supervisor_abs", "supervisor_abs_p95")


def resolve_physical_budget_threshold(physical_cfg, scale, scoring):
    """Convert a physical acceleration-error budget into a model-unit screening threshold.

    Returns ``(model_threshold, physical_value, physical_units)``. Enforces an absolute-scale
    scoring mode (relative supervisor scores are rejected) and requires an available physical
    acceleration scale -- otherwise a clear ``ValueError`` rather than a silent normalized fallback.
    """

    if is_relative_scoring(scoring) or not is_absolute_scoring(scoring):
        raise ValueError(
            f"physical_budget screening requires an absolute-scale scoring mode "
            f"{PHYSICAL_BUDGET_SCORINGS}; got scoring={scoring!r} (relative supervisor scores are "
            "per-trajectory-normalized and cannot be compared to a physical budget)"
        )
    value = physical_cfg.get("value")
    if value is None:
        raise ValueError(
            "physical_budget requires uq.physical_budget.value (the acceleration-error budget)"
        )
    units = str(physical_cfg.get("units", "m/s^2"))
    if not scale.physical:
        raise ValueError(
            "physical_budget requires an explicit acceleration scale; set body.acceleration_scale_m_s2 "
            "or a physical body.acceleration_units (no scale is ever inferred)"
        )
    model_threshold = float(acceleration_to_model_units(float(value), scale, source_units=units))
    return model_threshold, float(value), units


def resolve_threshold(screen_cfg, plugin, held, scoring, *, dtype, seed, config=None):
    """Resolve the absolute screening threshold and its provenance from config.

    Returns ``(threshold_or_None, meta)``. ``None`` means fall back to fraction mode. ``meta``
    records the threshold source / quantile / multiplier / calibration scoring + count, the physical
    budget (model + physical units) when applicable, and an optional backward-compatibility note.
    Enforces that a *pointwise* expected-error budget or a *physical* budget is only paired with
    absolute-scale scoring (never a relative supervisor score). ``config`` (the full config) is
    required for ``threshold_source=physical_budget``.
    """

    threshold = screen_cfg.get("threshold")
    threshold_quantile = screen_cfg.get("threshold_quantile")
    multiplier = float(screen_cfg.get("threshold_multiplier", 1.0))
    src = screen_cfg.get("threshold_source")

    meta = {
        "threshold_source": None,
        "threshold_quantile": None,
        "threshold_multiplier": multiplier,
        "threshold_calibration_scoring": None,
        "threshold_calibration_n": None,
        "threshold_compatibility_note": None,
        "threshold_model_units": None,
        "threshold_physical_value": None,
        "threshold_physical_units": None,
        "acceleration_scale_m_s2": None,
    }

    physical_cfg = ((config or {}).get("uq", {}) or {}).get("physical_budget", {}) or {}

    # Backward-compatible inference when threshold_source is omitted.
    if src is None:
        if bool(physical_cfg.get("enabled", False)):
            src = "physical_budget"
        elif threshold is not None:
            src = "manual"
        elif threshold_quantile is not None:
            src = "pointwise_calibration_quantile"
            meta["threshold_compatibility_note"] = (
                "legacy syntax: threshold_quantile without threshold_source -> inferred "
                "pointwise_calibration_quantile (requires absolute-like scoring)"
            )
        else:
            return None, meta  # no threshold configured -> fraction mode
    src = str(src).lower()
    if src not in THRESHOLD_SOURCES:
        raise ValueError(f"uq.screening.threshold_source must be one of {THRESHOLD_SOURCES}, got {src!r}")

    if src == "physical_budget":
        if config is None:
            raise ValueError("threshold_source=physical_budget requires the full config")
        scale = resolve_acceleration_scale(config)
        model_thr, phys_val, phys_units = resolve_physical_budget_threshold(physical_cfg, scale, scoring)
        meta.update(
            threshold_source=src,
            threshold_model_units=model_thr,
            threshold_physical_value=phys_val,
            threshold_physical_units=phys_units,
            acceleration_scale_m_s2=scale.scale_m_s2,
        )
        return model_thr, meta

    if src == "manual":
        if threshold is None:
            raise ValueError("threshold_source=manual requires uq.screening.threshold")
        meta["threshold_source"] = "manual"
        return float(threshold), meta

    if threshold_quantile is None:
        raise ValueError(f"threshold_source={src} requires uq.screening.threshold_quantile")

    if src == "pointwise_calibration_quantile":
        if not is_absolute_scoring(scoring):
            why = (
                " (a relative supervisor score is not on the pointwise expected-error scale; use "
                "trajectory_calibration_quantile instead)"
                if is_relative_scoring(scoring)
                else " (use an expected_abs*/supervisor_abs* score, or trajectory_calibration_quantile)"
            )
            raise ValueError(
                f"threshold_source=pointwise_calibration_quantile needs absolute-like scoring; "
                f"got scoring={scoring!r}{why}"
            )
        thr = plugin.calibrate_pointwise_expected_error_threshold(
            held.positions, quantile=float(threshold_quantile), multiplier=multiplier
        )
        meta.update(
            threshold_source=src,
            threshold_quantile=float(threshold_quantile),
            threshold_calibration_n=int(held.positions.shape[0]),
        )
        return thr, meta

    # trajectory_calibration_quantile: calibrate the SAME trajectory score -> safe for any scoring.
    default_n = min(int(screen_cfg.get("n_orbits", 200)), 200)
    cal = generate_orbit_ensemble(
        n_orbits=int(screen_cfg.get("calibration_n_orbits", default_n)),
        n_points=int(screen_cfg.get("calibration_n_points", int(screen_cfg.get("n_points", 48)))),
        r_peri_range=tuple(screen_cfg.get("calibration_r_peri_range", screen_cfg.get("r_peri_range", (1.02, 1.30)))),
        r_apo_range=tuple(screen_cfg.get("calibration_r_apo_range", screen_cfg.get("r_apo_range", (1.30, 1.60)))),
        seed=int(seed) + 1,  # a held-out calibration ensemble, distinct from the screening one
        dtype=dtype,
    )
    thr = plugin.calibrate_trajectory_risk_threshold(
        cal.trajectories, scoring=scoring, quantile=float(threshold_quantile), multiplier=multiplier
    )
    meta.update(
        threshold_source=src,
        threshold_quantile=float(threshold_quantile),
        threshold_calibration_scoring=scoring,
        threshold_calibration_n=len(cal.trajectories),
    )
    return thr, meta
