"""Unit-aware physical acceleration-budget support for VESP-UQ.

VESP-UQ risk / expected-force-error scores are produced on the model's normalized-acceleration
scale. That scale is fine for internal ranking, but a *physical* screening budget (e.g. "rerun any
trajectory whose estimated force-model error exceeds 1e-8 m/s^2") must be expressed in physical
acceleration units. This module provides the small, explicit conversion utilities that bridge the
two -- and, crucially, it never invents a physical scale: conversion is only available when the
config supplies it explicitly.

Two ways the config can make conversion available (both are explicit author declarations, not
inferences from body radius or GM):

- ``body.acceleration_units`` is itself a supported physical unit -- the scores are declared to be
  in that unit, so the model<->physical scale is just that unit's factor; or
- ``body.acceleration_units: model_normalized_accel`` together with an explicit
  ``body.acceleration_scale_m_s2`` giving how many m/s^2 one model-score unit equals.

If neither is present, :func:`resolve_acceleration_scale` returns a non-physical scale and callers
must keep reporting in model-normalized units. Conversion requested without an available scale is a
clear error, never a silent fallback.

Everything here concerns force-model acceleration error; none of it is a position-accuracy quantity.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

# The model's internal (dimensionless) acceleration unit, plus its accepted aliases.
MODEL_UNITS = "model_normalized_accel"
_MODEL_ALIASES = {"model_normalized_accel", "model", "normalized", "model_accel"}

# Supported physical units and their conversion factor to m/s^2 (value_in_m_s2 = value * factor).
_UNIT_TO_M_S2 = {
    "m/s^2": 1.0,
    "km/s^2": 1.0e3,
    "mm/s^2": 1.0e-3,
    "um/s^2": 1.0e-6,
}
PHYSICAL_UNITS = tuple(_UNIT_TO_M_S2.keys())
SUPPORTED_UNITS = (MODEL_UNITS, *PHYSICAL_UNITS)


def _normalize_units(units: str) -> str:
    """Map a units string to its canonical supported name, raising on anything unsupported."""

    u = str(units).strip()
    if u.lower() in _MODEL_ALIASES:
        return MODEL_UNITS
    if u in _UNIT_TO_M_S2:
        return u
    raise ValueError(
        f"unsupported acceleration units {units!r}; supported units are {SUPPORTED_UNITS}"
    )


def _validate_scale_m_s2(value) -> float:
    s = float(value)
    if not math.isfinite(s):
        raise ValueError(f"acceleration_scale_m_s2 must be finite, got {value!r}")
    if s <= 0.0:
        raise ValueError(f"acceleration_scale_m_s2 must be positive, got {value!r}")
    return s


@dataclass(frozen=True)
class AccelerationScale:
    """Resolved model<->physical acceleration scale for VESP-UQ scores.

    ``scale_m_s2`` is how many m/s^2 one model-score unit equals; ``None`` (with ``physical=False``)
    means no explicit physical metadata was supplied and values must stay in model-normalized units.
    ``model_units`` is the declared unit of the scores themselves, and ``source`` records which
    config declaration enabled the conversion (or ``"none"``).
    """

    model_units: str
    scale_m_s2: float | None
    physical: bool
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_acceleration_scale(config: dict) -> AccelerationScale:
    """Resolve the model<->physical acceleration scale from a config's ``body`` block.

    Never infers a scale from body radius or GM. Returns a non-physical :class:`AccelerationScale`
    (``physical=False``, ``scale_m_s2=None``) when the required metadata is absent. Raises
    ``ValueError`` for unsupported units or an invalid explicit scale.
    """

    body = config.get("body", {}) or {}
    declared = _normalize_units(body.get("acceleration_units", MODEL_UNITS))
    scale_raw = body.get("acceleration_scale_m_s2", None)

    if declared in _UNIT_TO_M_S2:
        # Scores are explicitly declared to be in this physical unit -> the model<->physical scale
        # is just the unit's factor (an explicit author declaration, not an inference).
        return AccelerationScale(
            model_units=declared,
            scale_m_s2=_UNIT_TO_M_S2[declared],
            physical=True,
            source="declared_physical_units",
        )

    # model_normalized_accel: physical conversion only via an explicit scale.
    if scale_raw is not None:
        return AccelerationScale(
            model_units=MODEL_UNITS,
            scale_m_s2=_validate_scale_m_s2(scale_raw),
            physical=True,
            source="acceleration_scale_m_s2",
        )

    return AccelerationScale(model_units=MODEL_UNITS, scale_m_s2=None, physical=False, source="none")


def has_physical_acceleration_scale(config: dict) -> bool:
    """True iff the config supplies enough explicit metadata to convert scores to physical units."""

    return resolve_acceleration_scale(config).physical


def _require_physical(scale: AccelerationScale) -> float:
    if not scale.physical or scale.scale_m_s2 is None:
        raise ValueError(
            "physical acceleration conversion requested but no acceleration scale is available; "
            "set body.acceleration_scale_m_s2 or a physical body.acceleration_units"
        )
    return scale.scale_m_s2


def _as_finite_tensor(values, name: str):
    """Return (tensor_float64, was_scalar). Raises on NaN/Inf."""

    t = torch.as_tensor(values, dtype=torch.float64)
    if not bool(torch.isfinite(t).all()):
        raise ValueError(f"{name} contains NaN or infinite values")
    return t, (t.ndim == 0)


def acceleration_to_physical(values, scale: AccelerationScale, target_units: str = "m/s^2"):
    """Convert model-unit acceleration ``values`` into a physical unit using ``scale``.

    Requires an available physical scale (otherwise a clear ``ValueError``). ``target_units`` must be
    a supported *physical* unit. Scalars return a ``float``; arrays return a ``float64`` tensor.
    """

    scale_m_s2 = _require_physical(scale)
    target = _normalize_units(target_units)
    if target == MODEL_UNITS:
        raise ValueError("target_units must be a physical unit, not model_normalized_accel")
    t, scalar = _as_finite_tensor(values, "values")
    physical = (t * scale_m_s2) / _UNIT_TO_M_S2[target]
    return float(physical) if scalar else physical


def acceleration_to_model_units(values, scale: AccelerationScale, source_units: str = "m/s^2"):
    """Convert physical ``values`` (in ``source_units``) back to model-score units using ``scale``.

    Requires an available physical scale (otherwise a clear ``ValueError``). ``source_units`` must be
    a supported *physical* unit. Scalars return a ``float``; arrays return a ``float64`` tensor.
    """

    scale_m_s2 = _require_physical(scale)
    source = _normalize_units(source_units)
    if source == MODEL_UNITS:
        raise ValueError("source_units must be a physical unit, not model_normalized_accel")
    t, scalar = _as_finite_tensor(values, "values")
    model = (t * _UNIT_TO_M_S2[source]) / scale_m_s2
    return float(model) if scalar else model


def format_acceleration_value(value, units: str) -> str:
    """Format a scalar acceleration value with its units, e.g. ``"1.234e-08 m/s^2"``."""

    u = _normalize_units(units)
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"value must be finite, got {value!r}")
    return f"{v:.4e} {u}"
