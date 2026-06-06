"""Stage 3A: deterministic discrete MaxEnt regularization over source strengths.

This is the conservative first step of the Stage 3 roadmap. It keeps the Stage 1-2
ridge/Tikhonov solution as a warm start and baseline, then refines the source
strengths by adding a maximum-entropy term to the same target-normalized,
row-weighted data objective used by the ridge solve:

    minimize   mean( (A sigma - b)^2 )
             + lambda_l2 * mean(sigma^2)
             + lambda_moment * (monopole^2 + lambda_dipole * dipole^2)
             - entropy_weight * H(sigma)            # i.e. + entropy_weight * (-H)

``A`` and ``b`` are the row-weighted (potential/acceleration, target-normalized,
optionally altitude-weighted) operator and target. ``H`` is one of the deterministic
entropy functionals in ``vesp.extensions.entropy``. The data term is identical to the
ridge objective, so the entropy weight traces out a data-error vs entropy Pareto curve.

This is NOT the full MaxEnt posterior framework: it produces a single deterministic
entropy-regularized point estimate, not a calibrated distribution over sources.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from vesp.core.losses import moment_loss
from vesp.extensions.entropy import entropy_regularization_loss


@dataclass(frozen=True)
class MaxEntSolveConfig:
    entropy_weight: float = 0.0
    entropy_mode: str = "positive_negative"
    lambda_l2: float = 0.0
    lambda_moment: float = 0.0
    lambda_dipole: float = 1.0
    optimizer: str = "lbfgs"
    max_iter: int = 500
    lr: float = 1.0
    warm_start: bool = True
    tol: float = 1.0e-12
    verbose: bool = False

    @classmethod
    def from_config(cls, config: dict) -> "MaxEntSolveConfig":
        loss_cfg = config.get("loss", {})
        maxent_cfg = config.get("maxent", {})
        return cls(
            entropy_weight=float(loss_cfg.get("entropy_weight", maxent_cfg.get("entropy_weight", 0.0))),
            entropy_mode=str(loss_cfg.get("entropy_mode", maxent_cfg.get("entropy_mode", "positive_negative"))),
            lambda_l2=float(loss_cfg.get("lambda_l2", 0.0)),
            lambda_moment=float(loss_cfg.get("lambda_moment", 0.0)),
            lambda_dipole=float(loss_cfg.get("lambda_dipole", 1.0)),
            optimizer=str(maxent_cfg.get("optimizer", "lbfgs")).lower(),
            max_iter=int(maxent_cfg.get("max_iter", maxent_cfg.get("epochs", 500))),
            lr=float(maxent_cfg.get("lr", 1.0)),
            warm_start=bool(maxent_cfg.get("warm_start", True)),
            tol=float(maxent_cfg.get("tol", 1.0e-12)),
            verbose=bool(maxent_cfg.get("verbose", False)),
        )


def _objective(
    sigma: torch.Tensor,
    operator: torch.Tensor,
    target: torch.Tensor,
    positions: torch.Tensor,
    weights: torch.Tensor,
    shells: torch.Tensor,
    config: MaxEntSolveConfig,
) -> torch.Tensor:
    residual = operator @ sigma - target
    loss = torch.mean(residual * residual)
    if config.lambda_l2:
        loss = loss + config.lambda_l2 * torch.mean(sigma * sigma)
    if config.lambda_moment:
        loss = loss + config.lambda_moment * moment_loss(positions, weights, sigma, dipole_weight=config.lambda_dipole)
    if config.entropy_weight:
        loss = loss + config.entropy_weight * entropy_regularization_loss(
            sigma, weights, mode=config.entropy_mode, shell_ids=shells
        )
    return loss


def solve_discrete_maxent(
    operator: torch.Tensor,
    target: torch.Tensor,
    source_positions: torch.Tensor,
    source_weights: torch.Tensor,
    shell_ids: torch.Tensor,
    config: MaxEntSolveConfig,
    *,
    warm_start_sigma: torch.Tensor | None = None,
) -> torch.Tensor:
    """Solve the entropy-regularized source system.

    The objective is convex (quadratic data term + convex negative-entropy). The
    data term is ill-conditioned (equivalent sources), so the default optimizer is
    L-BFGS with a strong-Wolfe line search, which is robust to the conditioning and
    converges from the ridge warm start without learning-rate tuning. Adam is kept
    as a fallback for very large matrix-free problems.
    """

    device = operator.device
    dtype = operator.dtype
    n_sources = operator.shape[1]

    if warm_start_sigma is not None:
        sigma = warm_start_sigma.detach().clone().to(device=device, dtype=dtype)
    else:
        sigma = torch.zeros(n_sources, dtype=dtype, device=device)
    sigma.requires_grad_(True)

    positions = source_positions.to(device=device, dtype=dtype)
    weights = source_weights.to(device=device, dtype=dtype)
    shells = shell_ids.to(device)

    if config.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS(
            [sigma],
            lr=config.lr,
            max_iter=config.max_iter,
            history_size=50,
            line_search_fn="strong_wolfe",
            tolerance_grad=config.tol,
            tolerance_change=config.tol,
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = _objective(sigma, operator, target, positions, weights, shells, config)
            loss.backward()
            return loss

        loss_value = float(optimizer.step(closure).detach().cpu())
        if config.verbose:
            print(f"maxent lbfgs final loss={loss_value:.6e}")
        if not math.isfinite(loss_value):
            raise RuntimeError(f"MaxEnt L-BFGS solve diverged (loss={loss_value})")
        return sigma.detach()

    if config.optimizer == "adam":
        optimizer = torch.optim.Adam([sigma], lr=config.lr)
        previous = None
        for epoch in range(config.max_iter):
            optimizer.zero_grad(set_to_none=True)
            loss = _objective(sigma, operator, target, positions, weights, shells, config)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            if not math.isfinite(loss_value):
                raise RuntimeError(f"MaxEnt Adam solve diverged at epoch {epoch} (loss={loss_value})")
            if previous is not None and abs(previous - loss_value) <= config.tol * max(1.0, abs(previous)):
                break
            previous = loss_value
        return sigma.detach()

    raise ValueError(f"unknown maxent optimizer: {config.optimizer}")
