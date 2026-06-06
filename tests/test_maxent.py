import torch

from vesp.core.diagnostics import source_diagnostics
from vesp.core.kernels import evaluate_kernel
from vesp.core.operators import build_joint_operator
from vesp.core.solvers import RidgeSolveConfig, solve_discrete_ridge
from vesp.extensions.entropy import shell_energy_balance_entropy
from vesp.training.maxent import MaxEntSolveConfig, solve_discrete_maxent


def _collapse_prone_system():
    """Two-shell setup whose true field is dominated by the inner shell."""

    from vesp.core.sources import make_shell_sources

    sources = make_shell_sources([0.5, 0.9], [32, 32], body_radius=1.0, dtype=torch.float64)
    gen = torch.Generator().manual_seed(3)
    directions = torch.randn((400, 3), generator=gen, dtype=torch.float64)
    directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
    radii = 1.1 + 0.4 * torch.rand((400, 1), generator=gen, dtype=torch.float64)
    x = directions * radii

    sigma_true = torch.randn(sources.n_sources, generator=gen, dtype=torch.float64)
    sigma_true = sigma_true - sigma_true.mean()
    # suppress the outer shell so the field strongly prefers the inner shell
    sigma_true[sources.shell_ids == 1] *= 0.02

    out = evaluate_kernel(x, sources.positions, sources.weights * sigma_true)
    bundle = build_joint_operator(
        x, sources, potential=out.potential, acceleration=out.acceleration,
        use_potential=True, use_acceleration=True, sign=1.0,
    )
    return sources, bundle.operator, bundle.target


def _ridge(sources, operator, target):
    return solve_discrete_ridge(
        operator, target, sources.positions, sources.weights, sources.shell_ids,
        RidgeSolveConfig(method="augmented_lstsq", lambda_l2=1e-10, column_normalize=True, lambda_moment=0.0),
    )


def test_maxent_zero_weight_matches_ridge_data_fit():
    sources, operator, target = _collapse_prone_system()
    ridge = _ridge(sources, operator, target)
    cfg = MaxEntSolveConfig(entropy_weight=0.0, lambda_l2=1e-10, optimizer="lbfgs", max_iter=300)
    sigma = solve_discrete_maxent(
        operator, target, sources.positions, sources.weights, sources.shell_ids, cfg, warm_start_sigma=ridge
    )
    ridge_residual = float(torch.mean((operator @ ridge - target) ** 2))
    maxent_residual = float(torch.mean((operator @ sigma - target) ** 2))
    assert maxent_residual <= ridge_residual * 1.05 + 1e-12


def test_shell_balance_entropy_reduces_collapse():
    sources, operator, target = _collapse_prone_system()
    ridge = _ridge(sources, operator, target)
    base_entropy = float(shell_energy_balance_entropy(ridge, sources.weights, sources.shell_ids))
    base_dom = source_diagnostics(
        source_positions=sources.positions, source_weights=sources.weights,
        shell_ids=sources.shell_ids, sigma=ridge,
    )["dominant_shell_energy_fraction"]

    cfg = MaxEntSolveConfig(entropy_weight=10.0, entropy_mode="shell_balance", optimizer="lbfgs", max_iter=300)
    sigma = solve_discrete_maxent(
        operator, target, sources.positions, sources.weights, sources.shell_ids, cfg, warm_start_sigma=ridge
    )
    new_entropy = float(shell_energy_balance_entropy(sigma, sources.weights, sources.shell_ids))
    new_dom = source_diagnostics(
        source_positions=sources.positions, source_weights=sources.weights,
        shell_ids=sources.shell_ids, sigma=sigma,
    )["dominant_shell_energy_fraction"]

    assert new_entropy > base_entropy + 1e-6
    assert new_dom < base_dom - 1e-6
    # the data fit must degrade (the Pareto trade-off), confirming a real tension
    base_residual = float(torch.mean((operator @ ridge - target) ** 2))
    new_residual = float(torch.mean((operator @ sigma - target) ** 2))
    assert new_residual >= base_residual


def test_maxent_config_from_dict():
    config = {
        "loss": {"entropy_weight": 0.25, "entropy_mode": "shell_balance", "lambda_l2": 1e-6, "lambda_moment": 1e-4},
        "maxent": {"optimizer": "lbfgs", "max_iter": 123, "warm_start": False},
    }
    cfg = MaxEntSolveConfig.from_config(config)
    assert cfg.entropy_weight == 0.25
    assert cfg.entropy_mode == "shell_balance"
    assert cfg.max_iter == 123
    assert cfg.warm_start is False
    assert cfg.optimizer == "lbfgs"
