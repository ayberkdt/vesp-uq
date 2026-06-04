import json
import math

import torch

from experimental_vesp.data import ResidualGravityData
from experimental_vesp.target_scaling import (
    compute_target_scales,
    observation_row_weights,
    write_target_scales,
)


def _data(potential: torch.Tensor, acceleration: torch.Tensor) -> ResidualGravityData:
    n = potential.shape[0]
    return ResidualGravityData(
        positions=torch.zeros(n, 3, dtype=torch.float64),
        potential=potential.to(dtype=torch.float64).reshape(n, 1),
        acceleration=acceleration.to(dtype=torch.float64),
        metadata={"position_units": "normalized"},
    )


def test_normalize_targets_false_preserves_legacy_row_weights():
    scales = compute_target_scales(
        _data(torch.tensor([1.0]), torch.tensor([[1.0, 0.0, 0.0]])),
        {"loss": {"normalize_targets": False}},
    )
    weights = observation_row_weights(
        n_query=2,
        include_potential=True,
        include_acceleration=True,
        lambda_potential=0.25,
        lambda_acceleration=4.0,
        scales=scales,
        dtype=torch.float64,
        device="cpu",
    )
    expected = torch.tensor([0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=torch.float64)
    assert torch.allclose(weights, expected)


def test_auto_scales_are_computed_from_train_data():
    data = _data(
        torch.tensor([3.0, 4.0]),
        torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    scales = compute_target_scales(
        data,
        {
            "loss": {
                "normalize_targets": True,
                "potential_scale": "auto",
                "acceleration_scale": "auto",
            }
        },
    )
    expected = math.sqrt(12.5)
    assert math.isclose(scales.potential_scale, expected)
    assert math.isclose(scales.acceleration_scale, expected)


def test_zero_scales_are_clamped_to_eps():
    scales = compute_target_scales(
        _data(torch.zeros(3), torch.zeros(3, 3)),
        {"loss": {"normalize_targets": True, "target_scale_eps": 1.0e-6}},
    )
    assert scales.potential_scale == 1.0e-6
    assert scales.acceleration_scale == 1.0e-6


def test_target_scale_artifact_is_written(tmp_path):
    scales = compute_target_scales(
        _data(torch.tensor([2.0]), torch.tensor([[0.0, 3.0, 4.0]])),
        {"loss": {"normalize_targets": True}},
    )
    path = write_target_scales(tmp_path / "target_scales.json", scales)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "vesp_target_scales_v1"
    assert payload["potential_scale"] == 2.0
    assert payload["acceleration_scale"] == 5.0
