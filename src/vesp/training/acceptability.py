"""Fast triage classifier for deterministic Stage 1-2 VESP runs.

This turns a run's metrics + source diagnostics into a single coarse status so
sweeps and reports are scannable. It is intentionally a *screening* signal, NOT a
scientific verdict: a ``GOOD`` status only means no automatic red flag fired, and a
``REJECT_*`` status means one specific guardrail tripped. Always read the underlying
metrics before drawing physical conclusions.
"""
from __future__ import annotations

from typing import Any


STATUS_GOOD = "GOOD"
STATUS_CONDITIONAL = "CONDITIONAL"
STATUS_REJECT_REGULARIZATION = "REJECT_REGULARIZATION"
STATUS_REJECT_LOW_ALTITUDE = "REJECT_LOW_ALTITUDE"
STATUS_REJECT_SOURCE_COLLAPSE = "REJECT_SOURCE_COLLAPSE"
STATUS_REJECT_NUMERICAL = "REJECT_NUMERICAL"

ACCEPTANCE_DEFAULTS: dict[str, float] = {
    "max_relative_acceleration_rmse": 0.75,
    "max_low_altitude_rmse_factor": 5.0,
    "max_top5_source_contribution": 0.40,
    "max_dominant_shell_energy_fraction": 0.90,
    # sum(per-shell field RMS) / total field RMS; >> 1 means shells cancel (brittle).
    "max_shell_cancellation_ratio": 5.0,
    # ``sigma_l2`` is an absolute, coordinate-dependent magnitude (healthy ridge fits
    # routinely reach ~10). It is kept only as a coarse "gross blow-up" gate; the
    # scale-invariant screening signals below carry the real weight.
    "max_sigma_l2": 100.0,
    # Relative (dimensionless) moment leakage = absolute moment / total absolute source
    # mass (dipole additionally divided by mean source radius). These are invariant to
    # field magnitude and unit choice, unlike the deprecated absolute leakage gates.
    "max_relative_monopole_leakage": 0.05,
    "max_relative_dipole_leakage": 0.5,
}


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def classify_run_acceptability(metrics: dict, diagnostics: dict, config: dict | None = None) -> dict[str, Any]:
    """Return ``{"acceptability_status", "acceptability_reasons"}``.

    Reject reasons take priority over conditional ones; the most fundamental
    failure (numerical → regularization → source collapse → low altitude) wins the
    single returned status, while *all* triggered checks are listed in the reasons.
    """

    acc = dict(ACCEPTANCE_DEFAULTS)
    if config:
        acc.update({k: v for k, v in (config.get("acceptance", {}) or {}).items() if v is not None})

    diagnostics = diagnostics or {}
    reasons: list[str] = []

    rejects: list[str] = []

    monopole = _num(diagnostics.get("relative_monopole_leakage"))
    dipole = _num(diagnostics.get("relative_dipole_leakage"))
    if (monopole is not None and monopole > acc["max_relative_monopole_leakage"]) or (
        dipole is not None and dipole > acc["max_relative_dipole_leakage"]
    ):
        reasons.append(
            f"relative monopole/dipole leakage too high (monopole={monopole}, dipole={dipole}; "
            f"max={acc['max_relative_monopole_leakage']}/{acc['max_relative_dipole_leakage']})"
        )
        rejects.append(STATUS_REJECT_NUMERICAL)

    sigma_l2 = _num(diagnostics.get("sigma_l2"))
    if sigma_l2 is not None and sigma_l2 > acc["max_sigma_l2"]:
        reasons.append(f"sigma_l2={sigma_l2:.3e} exceeds max_sigma_l2={acc['max_sigma_l2']} (under-regularized)")
        rejects.append(STATUS_REJECT_REGULARIZATION)

    # Real multi-shell brittleness is near-redundant shells fitting the field with large
    # opposing source strengths that nearly cancel. The cancellation ratio captures this
    # directly and is the primary signal when available; the radius-biased shell-energy
    # fraction is only a fallback for runs that did not record the field-based metric.
    cancellation = _num(diagnostics.get("shell_cancellation_ratio"))
    dominant = _num(diagnostics.get("dominant_shell_energy_fraction"))
    collapse_flag = bool(diagnostics.get("shell_collapse_flag", False))
    if cancellation is not None:
        if cancellation > acc["max_shell_cancellation_ratio"]:
            reasons.append(
                f"shells cancel: sum(per-shell field RMS)/total field RMS={cancellation:.1f} "
                f"exceeds max={acc['max_shell_cancellation_ratio']} (brittle near-degenerate multi-shell fit)"
            )
            rejects.append(STATUS_REJECT_SOURCE_COLLAPSE)
    elif collapse_flag and dominant is not None and dominant > acc["max_dominant_shell_energy_fraction"]:
        reasons.append(
            f"dominant shell carries {dominant:.2%} of shell energy "
            f"(max={acc['max_dominant_shell_energy_fraction']:.0%}); shell energy collapsed"
        )
        rejects.append(STATUS_REJECT_SOURCE_COLLAPSE)

    ratio = _num(metrics.get("low_to_high_error_ratio"))
    if ratio is not None and ratio > acc["max_low_altitude_rmse_factor"]:
        reasons.append(
            f"low/high altitude error ratio={ratio:.2f} exceeds "
            f"max_low_altitude_rmse_factor={acc['max_low_altitude_rmse_factor']}"
        )
        rejects.append(STATUS_REJECT_LOW_ALTITUDE)

    # Conditional-level checks (do not, on their own, reject a run).
    conditional = False
    rel = _num(metrics.get("relative_acceleration_rmse"))
    if rel is not None and rel > acc["max_relative_acceleration_rmse"]:
        reasons.append(
            f"relative_acceleration_rmse={rel:.3f} exceeds max={acc['max_relative_acceleration_rmse']}"
        )
        conditional = True

    top5 = _num(diagnostics.get("top_5pct_source_contribution"))
    if top5 is not None and top5 > acc["max_top5_source_contribution"]:
        reasons.append(
            f"top 5% of sources carry {top5:.2%} of mass (max={acc['max_top5_source_contribution']:.0%}); "
            f"source concentration high"
        )
        conditional = True

    priority = [
        STATUS_REJECT_NUMERICAL,
        STATUS_REJECT_REGULARIZATION,
        STATUS_REJECT_SOURCE_COLLAPSE,
        STATUS_REJECT_LOW_ALTITUDE,
    ]
    status = STATUS_GOOD
    for candidate in priority:
        if candidate in rejects:
            status = candidate
            break
    else:
        if conditional:
            status = STATUS_CONDITIONAL

    if not reasons:
        reasons.append("no automatic red flags fired (screening only, not a scientific decision)")

    return {"acceptability_status": status, "acceptability_reasons": reasons}
