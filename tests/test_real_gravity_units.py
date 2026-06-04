import json

import numpy as np
import torch

from experimental_vesp import real_gravity
from experimental_vesp.data import load_csv_dataset
from experimental_vesp.real_gravity import SphericalHarmonicGravityModel
from experimental_vesp.units import normalized_gradient_to_physical_acceleration


def test_normalized_gradient_to_physical_acceleration_factor():
    grad = torch.tensor([[1737.4, 0.0, 0.0]], dtype=torch.float64)
    acc = normalized_gradient_to_physical_acceleration(grad, 1737.4)
    assert torch.allclose(acc, torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64))


def _patch_tiny_gravity(monkeypatch):
    model = SphericalHarmonicGravityModel(
        name="tiny",
        reference_radius_km=1738.0,
        gm_km3_s2=4902.80012616,
        degree=2,
        order=2,
        c=np.zeros((3, 3)),
        s=np.zeros((3, 3)),
        normalization_state=1,
        source_path="tiny.tab",
        column_order="degree_order",
    )
    monkeypatch.setattr(real_gravity, "read_pds_sha", lambda *args, **kwargs: model)
    monkeypatch.setattr(real_gravity, "random_exterior_points", lambda *args, **kwargs: np.array([[1.1, 0.0, 0.0]]))
    monkeypatch.setattr(real_gravity, "residual_potential", lambda *args, **kwargs: np.array([2.0]))
    monkeypatch.setattr(
        real_gravity,
        "residual_acceleration_finite_difference",
        lambda *args, **kwargs: np.array([[1738.0, 0.0, -3476.0]]),
    )


def test_physical_acceleration_metadata_and_loader_contract(tmp_path, monkeypatch):
    _patch_tiny_gravity(monkeypatch)
    path = real_gravity.build_real_lunar_dataset(
        sha_path=tmp_path / "tiny.tab",
        output_path=tmp_path / "tiny.csv",
        n_query=1,
        acceleration_output="physical",
    )
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["acceleration_units"] == "km/s^2"
    assert metadata["acceleration_output"] == "physical"
    data = load_csv_dataset(path, dtype=torch.float64)
    assert data.metadata["acceleration_units"] == "km/s^2"
    assert torch.allclose(data.acceleration, torch.tensor([[1.0, 0.0, -2.0]], dtype=torch.float64))


def test_normalized_gradient_metadata_contract(tmp_path, monkeypatch):
    _patch_tiny_gravity(monkeypatch)
    path = real_gravity.build_real_lunar_dataset(
        sha_path=tmp_path / "tiny.tab",
        output_path=tmp_path / "tiny_gradient.csv",
        n_query=1,
        acceleration_output="normalized_gradient",
    )
    metadata = json.loads(path.with_suffix(path.suffix + ".metadata.json").read_text(encoding="utf-8"))
    assert metadata["acceleration_units"] == "km^2/s^2 per normalized radius"
    assert metadata["acceleration_output"] == "normalized_gradient"
