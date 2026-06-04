import torch

from experimental_vesp.units import normalized_gradient_to_physical_acceleration


def test_normalized_gradient_to_physical_acceleration_factor():
    grad = torch.tensor([[1737.4, 0.0, 0.0]], dtype=torch.float64)
    acc = normalized_gradient_to_physical_acceleration(grad, 1737.4)
    assert torch.allclose(acc, torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64))

