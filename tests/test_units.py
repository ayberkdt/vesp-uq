import json

import pytest
import torch

from vesp.data.dataset import ResidualGravityData, load_csv_dataset, prepare_data_for_model
from vesp.common.units import (
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


def test_prepare_physical_km_positions_for_normalized_model():
    data = ResidualGravityData(
        positions=torch.tensor([[1738.0, 0.0, 0.0]], dtype=torch.float64),
        potential=torch.zeros(1, 1, dtype=torch.float64),
        acceleration=torch.zeros(1, 3, dtype=torch.float64),
        metadata={"position_units": "km", "R_body": 1738.0, "R_body_units": "km"},
    )
    out = prepare_data_for_model(data, UnitConfig(R_body=1.0, normalize_positions=True, position_units="normalized"))
    assert torch.allclose(out.positions, torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64))
    assert out.metadata["original_position_units"] == "km"
    assert out.metadata["model_position_units"] == "normalized"
    assert out.metadata["R_body"] == 1738.0


def test_prepare_normalized_positions_for_physical_model():
    data = ResidualGravityData(
        positions=torch.tensor([[1.2, 0.0, 0.0]], dtype=torch.float64),
        potential=torch.zeros(1, 1, dtype=torch.float64),
        acceleration=torch.zeros(1, 3, dtype=torch.float64),
        metadata={"position_units": "normalized", "R_body": 10.0, "R_body_units": "km"},
    )
    out = prepare_data_for_model(data, UnitConfig(R_body=10.0, normalize_positions=False, position_units="km"))
    assert torch.allclose(out.positions, torch.tensor([[12.0, 0.0, 0.0]], dtype=torch.float64))
    assert out.metadata["model_position_units"] == "km"


def test_csv_loader_requires_metadata(tmp_path):
    path = tmp_path / "residual.csv"
    path.write_text("x,y,z,Delta U,Delta a_x,Delta a_y,Delta a_z\n1,0,0,0,0,0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="metadata sidecar is required"):
        load_csv_dataset(path)


def test_csv_loader_uses_sidecar_and_preserves_acceleration_units(tmp_path):
    path = tmp_path / "residual.csv"
    path.write_text("x,y,z,Delta U,Delta a_x,Delta a_y,Delta a_z\n1,0,0,2,3,4,5\n", encoding="utf-8")
    metadata = {
        "position_units": "normalized",
        "potential_units": "km^2/s^2",
        "acceleration_units": "km/s^2",
        "R_body": 1738.0,
        "R_body_units": "km",
    }
    path.with_suffix(path.suffix + ".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    data = load_csv_dataset(
        path,
        dtype=torch.float64,
        unit_config=UnitConfig(R_body=1.0, normalize_positions=True, position_units="normalized"),
    )
    assert data.metadata["acceleration_units"] == "km/s^2"
    assert data.metadata["original_position_units"] == "normalized"
    assert data.metadata["model_position_units"] == "normalized"


def _write_physical_residual_csv(tmp_path, *, acceleration_units, acceleration_output=None):
    path = tmp_path / "residual.csv"
    path.write_text(
        "x,y,z,Delta U,Delta a_x,Delta a_y,Delta a_z\n1.2,0,0,2,3,4,5\n",
        encoding="utf-8",
    )
    metadata = {
        "position_units": "normalized",
        "potential_units": "km^2/s^2",
        "acceleration_units": acceleration_units,
        "R_body": 1738.0,
        "R_body_units": "km",
    }
    if acceleration_output is not None:
        metadata["acceleration_output"] = acceleration_output
    path.with_suffix(path.suffix + ".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return path


def test_physical_acceleration_converted_to_normalized_gradient_for_normalized_model(tmp_path):
    """A physical km/s^2 target must be scaled by R_body for a normalized-coordinate model."""

    path = _write_physical_residual_csv(tmp_path, acceleration_units="km/s^2", acceleration_output="physical")
    data = load_csv_dataset(
        path,
        dtype=torch.float64,
        unit_config=UnitConfig(R_body=1.0, normalize_positions=True, position_units="normalized"),
    )
    expected = torch.tensor([[3.0, 4.0, 5.0]], dtype=torch.float64) * 1738.0
    assert torch.allclose(data.acceleration, expected)
    assert data.metadata["acceleration_conversion_factor"] == 1738.0
    assert data.metadata["csv_acceleration_kind"] == "physical"
    assert data.metadata["model_acceleration_kind"] == "normalized_gradient"


def test_normalized_gradient_acceleration_unchanged_for_normalized_model(tmp_path):
    path = _write_physical_residual_csv(
        tmp_path,
        acceleration_units="km^2/s^2 per normalized radius",
        acceleration_output="normalized_gradient",
    )
    data = load_csv_dataset(
        path,
        dtype=torch.float64,
        unit_config=UnitConfig(R_body=1.0, normalize_positions=True, position_units="normalized"),
    )
    assert torch.allclose(data.acceleration, torch.tensor([[3.0, 4.0, 5.0]], dtype=torch.float64))
    assert data.metadata["acceleration_conversion_factor"] == 1.0


def test_normalized_gradient_converted_to_physical_for_physical_model(tmp_path):
    path = _write_physical_residual_csv(
        tmp_path,
        acceleration_units="km^2/s^2 per normalized radius",
        acceleration_output="normalized_gradient",
    )
    data = load_csv_dataset(
        path,
        dtype=torch.float64,
        unit_config=UnitConfig(R_body=1738.0, normalize_positions=False, position_units="km"),
    )
    expected = torch.tensor([[3.0, 4.0, 5.0]], dtype=torch.float64) / 1738.0
    assert torch.allclose(data.acceleration, expected)
