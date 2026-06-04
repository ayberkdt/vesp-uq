import torch

from experimental_vesp.sources import fibonacci_sphere, single_shell_sources, multi_shell_sources


def test_fibonacci_sphere_unit_norm():
    points = fibonacci_sphere(128)
    assert points.shape == (128, 3)
    assert torch.allclose(torch.linalg.norm(points, dim=-1), torch.ones(128), atol=1e-6)


def test_single_shell_radius_and_weights():
    sources = single_shell_sources(0.8, 64)
    assert sources.positions.shape == (64, 3)
    assert sources.weights.shape == (64,)
    assert torch.allclose(torch.linalg.norm(sources.positions, dim=-1), torch.full((64,), 0.8), atol=1e-6)


def test_multi_shell_ids():
    sources = multi_shell_sources([0.5, 0.8], [16, 32])
    assert sources.n_sources == 48
    assert int((sources.shell_ids == 0).sum()) == 16
    assert int((sources.shell_ids == 1).sum()) == 32

