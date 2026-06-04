import torch

from experimental_vesp.kernels import (
    acceleration_kernel,
    evaluate_potential_acceleration_chunked,
    potential_kernel,
)


def test_potential_kernel_shape():
    x = torch.randn(5, 3, dtype=torch.float64) + torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64)
    s = torch.randn(7, 3, dtype=torch.float64) * 0.2
    k = potential_kernel(x, s)
    assert k.shape == (5, 7)


def test_acceleration_matches_finite_difference_gradient():
    x = torch.tensor([[1.4, 0.2, -0.1]], dtype=torch.float64)
    s = torch.tensor([[0.1, -0.2, 0.3], [-0.3, 0.4, -0.1]], dtype=torch.float64)
    sigma = torch.tensor([0.7, -0.2], dtype=torch.float64)
    weights = torch.ones(2, dtype=torch.float64)
    _, accel = evaluate_potential_acceleration_chunked(x, s, sigma, weights)
    h = 1.0e-6
    grad = torch.zeros(3, dtype=torch.float64)
    for axis in range(3):
        xp = x.clone()
        xm = x.clone()
        xp[:, axis] += h
        xm[:, axis] -= h
        up, _ = evaluate_potential_acceleration_chunked(xp, s, sigma, weights)
        um, _ = evaluate_potential_acceleration_chunked(xm, s, sigma, weights)
        grad[axis] = ((up - um) / (2.0 * h)).reshape(())
    assert torch.allclose(accel.reshape(-1), grad, atol=1e-6)


def test_chunked_and_dense_match():
    x = torch.randn(8, 3, dtype=torch.float64) + torch.tensor([0.0, 0.0, 1.5], dtype=torch.float64)
    s = torch.randn(10, 3, dtype=torch.float64) * 0.3
    sigma = torch.randn(10, dtype=torch.float64)
    weights = torch.ones(10, dtype=torch.float64)
    u1, a1 = evaluate_potential_acceleration_chunked(x, s, sigma, weights)
    u2, a2 = evaluate_potential_acceleration_chunked(x, s, sigma, weights, source_chunk_size=3)
    assert torch.allclose(u1, u2)
    assert torch.allclose(a1, a2)


def test_acceleration_sign():
    x = torch.tensor([[1.2, 0.0, 0.0]], dtype=torch.float64)
    s = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    k_pos = acceleration_kernel(x, s, sign=1.0)
    k_neg = acceleration_kernel(x, s, sign=-1.0)
    assert torch.allclose(k_pos, -k_neg)

