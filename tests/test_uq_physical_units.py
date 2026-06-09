"""Tests for unit-aware physical acceleration conversion (vesp.uq.physical_units)."""

from __future__ import annotations

import pytest
import torch

from vesp.uq.physical_units import (
    AccelerationScale,
    acceleration_to_model_units,
    acceleration_to_physical,
    format_acceleration_value,
    has_physical_acceleration_scale,
    resolve_acceleration_scale,
)


def _cfg_with_scale(scale_m_s2):
    return {"body": {"acceleration_units": "model_normalized_accel", "acceleration_scale_m_s2": scale_m_s2}}


def test_model_to_m_s2_with_explicit_scale():
    scale = resolve_acceleration_scale(_cfg_with_scale(1.0e-6))
    assert scale.physical is True
    assert scale.scale_m_s2 == pytest.approx(1.0e-6)
    # 1 model unit -> 1e-6 m/s^2
    assert acceleration_to_physical(1.0, scale, "m/s^2") == pytest.approx(1.0e-6)
    # array input -> tensor, and km/s^2 target divides by 1e3
    out = acceleration_to_physical(torch.tensor([1.0, 2.0]), scale, "km/s^2")
    assert torch.allclose(out, torch.tensor([1.0e-9, 2.0e-9], dtype=torch.float64))


def test_m_s2_to_model_units():
    scale = resolve_acceleration_scale(_cfg_with_scale(1.0e-6))
    # 1e-8 m/s^2 budget -> 1e-8 / 1e-6 = 1e-2 model units
    assert acceleration_to_model_units(1.0e-8, scale, "m/s^2") == pytest.approx(1.0e-2)
    # round-trip model -> physical -> model
    back = acceleration_to_model_units(acceleration_to_physical(0.5, scale, "mm/s^2"), scale, "mm/s^2")
    assert back == pytest.approx(0.5)


def test_declared_physical_units_enable_conversion():
    scale = resolve_acceleration_scale({"body": {"acceleration_units": "km/s^2"}})
    assert scale.physical is True
    assert scale.model_units == "km/s^2"
    assert scale.scale_m_s2 == pytest.approx(1.0e3)
    assert scale.source == "declared_physical_units"


def test_missing_metadata_is_not_physical():
    scale = resolve_acceleration_scale({"body": {}})
    assert scale.physical is False
    assert scale.scale_m_s2 is None
    assert has_physical_acceleration_scale({"body": {}}) is False
    assert has_physical_acceleration_scale(_cfg_with_scale(1.0e-6)) is True


def test_unsupported_units_raise():
    with pytest.raises(ValueError):
        resolve_acceleration_scale({"body": {"acceleration_units": "furlongs/s^2"}})
    scale = resolve_acceleration_scale(_cfg_with_scale(1.0e-6))
    with pytest.raises(ValueError):
        acceleration_to_physical(1.0, scale, "furlongs/s^2")
    with pytest.raises(ValueError):
        format_acceleration_value(1.0, "lightyears/s^2")


def test_missing_scale_raises_when_conversion_requested():
    scale = resolve_acceleration_scale({"body": {}})  # not physical
    with pytest.raises(ValueError):
        acceleration_to_physical(1.0, scale, "m/s^2")
    with pytest.raises(ValueError):
        acceleration_to_model_units(1.0e-8, scale, "m/s^2")


def test_invalid_scale_raises():
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            resolve_acceleration_scale(_cfg_with_scale(bad))


def test_nan_inf_inputs_raise():
    scale = resolve_acceleration_scale(_cfg_with_scale(1.0e-6))
    with pytest.raises(ValueError):
        acceleration_to_physical(torch.tensor([1.0, float("nan")]), scale, "m/s^2")
    with pytest.raises(ValueError):
        acceleration_to_model_units(float("inf"), scale, "m/s^2")


def test_physical_target_cannot_be_model_units():
    scale = resolve_acceleration_scale(_cfg_with_scale(1.0e-6))
    with pytest.raises(ValueError):
        acceleration_to_physical(1.0, scale, "model_normalized_accel")
    with pytest.raises(ValueError):
        acceleration_to_model_units(1.0, scale, "model_normalized_accel")


def test_format_acceleration_value():
    assert format_acceleration_value(1.234e-8, "m/s^2") == "1.2340e-08 m/s^2"
    assert format_acceleration_value(2.0, "model") == "2.0000e+00 model_normalized_accel"


def test_acceleration_scale_dataclass_serializable():
    scale = AccelerationScale("model_normalized_accel", 1.0e-6, True, "acceleration_scale_m_s2")
    d = scale.to_dict()
    assert d["scale_m_s2"] == pytest.approx(1.0e-6)
    assert d["physical"] is True


def test_no_position_error_references_in_module():
    import inspect

    import vesp.uq.physical_units as mod

    src = inspect.getsource(mod).lower()
    assert "position error" not in src
    assert "position-error" not in src
