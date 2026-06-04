import torch

from experimental_vesp.losses import moment_losses, shell_energy


def test_monopole_and_dipole():
    s = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64)
    w = torch.ones(2, dtype=torch.float64)
    sigma = torch.tensor([1.0, -1.0], dtype=torch.float64)
    moments = moment_losses(s, w, sigma)
    assert torch.isclose(moments["monopole"], torch.tensor(0.0, dtype=torch.float64))
    assert torch.isclose(moments["dipole"], torch.tensor(4.0, dtype=torch.float64))


def test_shell_energy():
    sigma = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    weights = torch.ones(3, dtype=torch.float64)
    shell_ids = torch.tensor([0, 0, 1])
    energy = shell_energy(sigma, weights, shell_ids)
    assert torch.allclose(energy, torch.tensor([5.0, 9.0], dtype=torch.float64))

