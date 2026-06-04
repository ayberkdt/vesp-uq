import torch

from experimental_vesp.solvers import build_regularization_rows, solve_ridge


def test_column_scale_manual_recovery():
    K = torch.tensor([[1.0, 100.0], [2.0, -50.0], [0.5, 25.0]], dtype=torch.float64)
    sigma_true = torch.tensor([0.4, -0.02], dtype=torch.float64)
    y = K @ sigma_true
    scale = torch.linalg.norm(K, dim=0)
    sigma_scaled = solve_ridge(K / scale, y, method="augmented_lstsq")
    sigma = sigma_scaled / scale
    assert torch.allclose(sigma, sigma_true, atol=1e-10)


def test_moment_regularization_rows_reduce_monopole_capacity():
    positions = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64)
    weights = torch.ones(2, dtype=torch.float64)
    shell_ids = torch.zeros(2, dtype=torch.long)
    rows, targets = build_regularization_rows(
        source_positions=positions,
        source_weights=weights,
        shell_ids=shell_ids,
        lambda_moment=1.0,
    )
    assert rows
    assert targets[0].shape[0] == 4

