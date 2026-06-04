import torch

from experimental_vesp.units import (
    PositionScaler,
    UnitConfig,
    normalized_gradient_to_physical_acceleration,
    physical_acceleration_to_normalized_gradient,
)


def test_source_body_radius_normalized():
    units = UnitConfig(R_body=1737.4, normalize_positions=True)
    assert units.source_body_radius == 1.0


def test_source_body_radius_physical():
    units = UnitConfig(R_body=1737.4, normalize_positions=False)
    assert units.source_body_radius == 1737.4


def test_altitude_normalized():
    scaler = PositionScaler(UnitConfig(R_body=10.0, normalize_positions=True))
    x = torch.tensor([[1.2, 0.0, 0.0]])
    assert torch.allclose(scaler.altitude_from_model_positions(x), torch.tensor([0.2]))


def test_acceleration_conversion_roundtrip():
    grad = torch.tensor([[10.0, 0.0, -5.0]])
    acc = normalized_gradient_to_physical_acceleration(grad, 2.0)
    assert torch.allclose(acc, torch.tensor([[5.0, 0.0, -2.5]]))
    assert torch.allclose(physical_acceleration_to_normalized_gradient(acc, 2.0), grad)

