import torch

from experimental_vesp.solvers import (
    RidgeSolveConfig,
    build_regularization_rows,
    solve_discrete_ridge,
    solve_ridge,
)


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


def test_augmented_lstsq_same_family_recovery():
    K = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]], dtype=torch.float64)
    sigma_true = torch.tensor([0.3, -0.7], dtype=torch.float64)
    y = K @ sigma_true
    sigma = solve_discrete_ridge(
        K,
        y,
        torch.zeros(2, 3, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.long),
        RidgeSolveConfig(method="augmented_lstsq", column_normalize=False),
    )
    assert torch.allclose(sigma, sigma_true, atol=1.0e-12)


def test_normal_equation_same_family_recovery():
    K = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]], dtype=torch.float64)
    sigma_true = torch.tensor([0.3, -0.7], dtype=torch.float64)
    y = K @ sigma_true
    sigma = solve_discrete_ridge(
        K,
        y,
        torch.zeros(2, 3, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.long),
        RidgeSolveConfig(method="normal_equation", column_normalize=False),
    )
    assert torch.allclose(sigma, sigma_true, atol=1.0e-12)


def test_discrete_solver_column_normalize_unscales_solution():
    K = torch.tensor([[1.0, 100.0], [2.0, -50.0], [0.5, 25.0]], dtype=torch.float64)
    sigma_true = torch.tensor([0.4, -0.02], dtype=torch.float64)
    y = K @ sigma_true
    sigma = solve_discrete_ridge(
        K,
        y,
        torch.zeros(2, 3, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.long),
        RidgeSolveConfig(method="augmented_lstsq", column_normalize=True),
    )
    assert torch.allclose(sigma, sigma_true, atol=1.0e-10)


def test_moment_regularization_reduces_monopole_leakage():
    K = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    y = torch.tensor([1.0], dtype=torch.float64)
    source_positions = torch.zeros(2, 3, dtype=torch.float64)
    source_weights = torch.ones(2, dtype=torch.float64)
    shell_ids = torch.zeros(2, dtype=torch.long)
    no_moment = solve_discrete_ridge(
        K,
        y,
        source_positions,
        source_weights,
        shell_ids,
        RidgeSolveConfig(method="augmented_lstsq", column_normalize=False),
    )
    with_moment = solve_discrete_ridge(
        K,
        y,
        source_positions,
        source_weights,
        shell_ids,
        RidgeSolveConfig(method="augmented_lstsq", column_normalize=False, lambda_moment=1.0, lambda_dipole=0.0),
    )
    assert abs(torch.sum(source_weights * with_moment)) < abs(torch.sum(source_weights * no_moment))


def test_shell_energy_rows_shape():
    rows, targets = build_regularization_rows(
        source_positions=torch.zeros(3, 3, dtype=torch.float64),
        source_weights=torch.ones(3, dtype=torch.float64),
        shell_ids=torch.tensor([0, 0, 1]),
        shell_energy_weights=torch.tensor([0.1, 0.2], dtype=torch.float64),
    )
    assert rows[-1].shape == (3, 3)
    assert targets[-1].shape == (3,)
