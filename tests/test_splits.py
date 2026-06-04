import pytest
import torch

from experimental_vesp.data import ResidualGravityData
from experimental_vesp.splits import make_splits


def _data():
    radii = torch.tensor([1.02, 1.08, 1.2, 1.4, 1.7], dtype=torch.float64)
    x = torch.stack([radii, torch.zeros_like(radii), torch.zeros_like(radii)], dim=-1)
    return ResidualGravityData(x, torch.zeros(5, 1, dtype=torch.float64), torch.zeros(5, 3, dtype=torch.float64))


def test_altitude_ood_ranges():
    splits = make_splits(
        _data(),
        {
            "split": {
                "type": "altitude_ood",
                "train_r_range": [1.05, 1.5],
                "test_high_r_range": [1.5, 1.8],
                "test_low_r_range": [1.0, 1.05],
                "train_fraction": 0.67,
            },
            "data": {"seed": 1},
        },
    )
    assert splits.test_high is not None
    assert splits.test_low is not None
    assert torch.all(splits.test_high.r >= 1.5)
    assert torch.all(splits.test_low.r <= 1.05)


def test_empty_band_raises():
    with pytest.raises(ValueError):
        make_splits(
            _data(),
            {
                "split": {
                    "type": "altitude_ood",
                    "train_r_range": [2.0, 3.0],
                    "test_high_r_range": [3.0, 4.0],
                }
            },
        )

