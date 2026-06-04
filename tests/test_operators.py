import torch

from experimental_vesp.operators import build_acceleration_operator, build_joint_operator, build_potential_operator
from experimental_vesp.sources import single_shell_sources


def test_operator_shapes():
    x = torch.randn(6, 3, dtype=torch.float64)
    x[:, 2] += 1.5
    sources = single_shell_sources(0.8, 10, dtype=torch.float64)
    assert build_potential_operator(x, sources).shape == (6, 10)
    assert build_acceleration_operator(x, sources).shape == (18, 10)


def test_joint_operator_target_order():
    x = torch.randn(4, 3, dtype=torch.float64)
    x[:, 2] += 1.5
    sources = single_shell_sources(0.8, 5, dtype=torch.float64)
    potential = torch.arange(4, dtype=torch.float64).reshape(-1, 1)
    acceleration = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    bundle = build_joint_operator(x, sources, potential=potential, acceleration=acceleration)
    expected = torch.cat([potential.reshape(-1), acceleration[:, 0], acceleration[:, 1], acceleration[:, 2]])
    assert torch.allclose(bundle.target, expected)
    assert bundle.operator.shape == (16, 5)

