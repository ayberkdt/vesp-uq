"""Lunar constants and metadata-contract helpers.

This module is adapted from the LUNAR_SIMULATION/ST-LRPS architecture: keep
lunar physical constants in one place and reject ambiguous body metadata before
it can silently contaminate experiments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


MU_MOON_SI: float = 4_904_869_500_000.0
R_MOON_M: float = 1_738_000.0
R_MOON_MEAN_M: float = 1_737_400.0
OMEGA_MOON_RAD_S: float = 2.6616995e-06
R_MOON_KM: float = R_MOON_M / 1000.0
R_MOON_MEAN_KM: float = R_MOON_MEAN_M / 1000.0
MU_MOON_KM3_S2: float = MU_MOON_SI * 1.0e-9

LUNAR_ALIASES = {"moon", "lunar", "selene"}


@dataclass(frozen=True)
class LunarConstants:
    central_body: str = "moon"
    mu_si: float = MU_MOON_SI
    r_ref_m: float = R_MOON_M
    r_mean_m: float = R_MOON_MEAN_M
    omega_rad_s: float = OMEGA_MOON_RAD_S

    @property
    def mu_km3_s2(self) -> float:
        return self.mu_si * 1.0e-9

    @property
    def r_ref_km(self) -> float:
        return self.r_ref_m * 1.0e-3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_LUNAR_CONSTANTS = LunarConstants()


def canonical_scales(*, mu_si: float = MU_MOON_SI, du_m: float = R_MOON_M) -> tuple[float, float, float]:
    """Return canonical distance, time, and velocity scales.

    DU is the chosen distance unit in meters. TU = sqrt(DU^3 / mu), and
    VU = DU / TU.
    """

    mu = float(mu_si)
    du = float(du_m)
    if mu <= 0.0:
        raise ValueError(f"mu_si must be positive. Got {mu_si!r}.")
    if du <= 0.0:
        raise ValueError(f"du_m must be positive. Got {du_m!r}.")
    tu = (du**3 / mu) ** 0.5
    vu = du / tu
    return du, tu, vu


def normalize_body_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    name = str(value).strip().lower()
    return name or None


def _safe_float(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def is_lunar_body_signature(
    *,
    mu_si: float | None = None,
    mu_km3_s2: float | None = None,
    r_ref_m: float | None = None,
    r_ref_km: float | None = None,
    rel_tol: float = 0.20,
) -> bool:
    """Return True if supplied constants are compatible with the Moon.

    The tolerance is intentionally loose enough for rounded legacy metadata,
    but still rejects Earth-scale artifacts.
    """

    tol = max(float(rel_tol), 0.0)
    checks: list[bool] = []

    def _close(value: float, reference: float) -> bool:
        return abs(float(value) - reference) / max(abs(reference), 1.0) <= tol

    if mu_si is not None:
        checks.append(_close(float(mu_si), MU_MOON_SI))
    if mu_km3_s2 is not None:
        checks.append(_close(float(mu_km3_s2), MU_MOON_KM3_S2))
    if r_ref_m is not None:
        checks.append(_close(float(r_ref_m), R_MOON_M))
    if r_ref_km is not None:
        checks.append(_close(float(r_ref_km), R_MOON_KM))
    return bool(checks) and all(checks)


def metadata_lunar_signature(metadata: Mapping[str, Any]) -> bool:
    """Best-effort lunar numeric signature check for CSV/HDF5/run metadata."""

    mu_si = _safe_float(metadata, "mu_si", "GM_SI", "gm_si", "resolved_mu_si")
    mu_km3_s2 = _safe_float(metadata, "gm_km3_s2", "GM", "mu_km3_s2")
    r_ref_m = _safe_float(metadata, "r_ref_m", "resolved_r_ref_m")
    r_ref_km = _safe_float(metadata, "R_body", "reference_radius_km", "r_ref_km")
    units = str(metadata.get("R_body_units", "km")).strip().lower()
    if r_ref_km is not None and units in {"m", "meter", "metre", "meters", "metres"}:
        r_ref_m = r_ref_km
        r_ref_km = None
    return is_lunar_body_signature(mu_si=mu_si, mu_km3_s2=mu_km3_s2, r_ref_m=r_ref_m, r_ref_km=r_ref_km)


def looks_like_lunar_metadata(metadata: Mapping[str, Any]) -> bool:
    """Return True only when label and numeric evidence are not contradictory."""

    body = normalize_body_name(metadata.get("central_body") or metadata.get("target_name"))
    has_signature = metadata_lunar_signature(metadata)
    if body in LUNAR_ALIASES:
        return has_signature
    if body:
        return False
    return has_signature


def validate_lunar_metadata_contract(
    metadata: Mapping[str, Any],
    *,
    data_path: str | Path | None = None,
    require_lunar: bool = False,
) -> dict[str, Any]:
    """Validate lunar dataset/run metadata and return normalized fields.

    If ``require_lunar`` is False, metadata without body evidence is accepted.
    If a body label is present, non-lunar labels or contradictory constants are
    rejected. This mirrors the conservative contract checks in ST-LRPS.
    """

    body = normalize_body_name(metadata.get("central_body") or metadata.get("target_name"))
    has_signature = metadata_lunar_signature(metadata)
    label = Path(data_path).name if data_path is not None else "metadata"

    if body is not None and body not in LUNAR_ALIASES:
        raise ValueError(f"{label!r} declares central_body={body!r}, which is not lunar.")

    has_numeric_fields = any(
        key in metadata
        for key in (
            "mu_si",
            "GM_SI",
            "gm_si",
            "resolved_mu_si",
            "gm_km3_s2",
            "GM",
            "mu_km3_s2",
            "r_ref_m",
            "resolved_r_ref_m",
            "R_body",
            "reference_radius_km",
            "r_ref_km",
        )
    )

    if body in LUNAR_ALIASES and has_numeric_fields and not has_signature:
        raise ValueError(f"{label!r} is labeled lunar but its body constants do not look lunar.")

    if require_lunar and body is None and not has_signature:
        raise ValueError(f"{label!r} is missing reliable lunar metadata.")

    resolved_mu_si = _safe_float(metadata, "mu_si", "GM_SI", "gm_si", "resolved_mu_si")
    gm_km3_s2 = _safe_float(metadata, "gm_km3_s2", "GM", "mu_km3_s2")
    if resolved_mu_si is None and gm_km3_s2 is not None:
        resolved_mu_si = gm_km3_s2 * 1.0e9

    resolved_r_ref_m = _safe_float(metadata, "r_ref_m", "resolved_r_ref_m")
    r_body = _safe_float(metadata, "R_body", "reference_radius_km", "r_ref_km")
    units = str(metadata.get("R_body_units", "km")).strip().lower()
    if resolved_r_ref_m is None and r_body is not None:
        resolved_r_ref_m = r_body if units in {"m", "meter", "metre", "meters", "metres"} else r_body * 1000.0

    return {
        "central_body": "moon" if (body in LUNAR_ALIASES or has_signature) else body,
        "resolved_mu_si": float(resolved_mu_si) if resolved_mu_si is not None else None,
        "resolved_r_ref_m": float(resolved_r_ref_m) if resolved_r_ref_m is not None else None,
        "has_lunar_signature": bool(has_signature),
    }


__all__ = [
    "DEFAULT_LUNAR_CONSTANTS",
    "LUNAR_ALIASES",
    "LunarConstants",
    "MU_MOON_SI",
    "MU_MOON_KM3_S2",
    "OMEGA_MOON_RAD_S",
    "R_MOON_KM",
    "R_MOON_M",
    "R_MOON_MEAN_KM",
    "R_MOON_MEAN_M",
    "canonical_scales",
    "is_lunar_body_signature",
    "looks_like_lunar_metadata",
    "metadata_lunar_signature",
    "normalize_body_name",
    "validate_lunar_metadata_contract",
]
