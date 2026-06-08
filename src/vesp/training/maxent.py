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
optionally altitude-weighted) operator and target — the SAME system the ridge solver
sees. ``H`` is one of the deterministic entropy functionals in
``vesp.extensions.entropy``.

The Pareto curve is anchored at the ridge baseline: ``entropy_weight = 0`` is
short-circuited to return the ridge warm-start solution exactly (the ridge solve uses a
column-normalized, sum-form Tikhonov system, whereas the entropy objective above uses a
mean-based L2/data normalization, so re-optimizing at zero entropy would needlessly
drift away from the true ridge minimum). For ``entropy_weight > 0`` the entropy weight
then traces out the data-error vs entropy trade-off from that ridge anchor.

This is NOT the full MaxEnt posterior framework: it produces a single deterministic
entropy-regularized point estimate, not a calibrated distribution over sources.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

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
    # Constrained ("equal-data-fit") MaxEnt — the principled Skilling-Bryan form.
    # mode="penalty"     -> minimize data + l2 + moment - entropy_weight*H  (fixed weight)
    # mode="constrained" -> MAXIMIZE entropy subject to data misfit <= misfit_factor * ridge_misfit,
    #                       auto-selecting the entropy weight by bisection. This isolates the
    #                       MaxEnt question: among solutions that fit the data equally well, is the
    #                       maximum-entropy one healthier / more generalizable than the ridge one?
    mode: str = "penalty"
    misfit_factor: float = 1.05
    search_iters: int = 16
    weight_bounds: tuple[float, float] = (1.0e-4, 1.0e3)

    @classmethod
    def from_config(cls, config: dict) -> "MaxEntSolveConfig":
        loss_cfg = config.get("loss", {})
        maxent_cfg = config.get("maxent", {})
        bounds = maxent_cfg.get("weight_bounds", [1.0e-4, 1.0e3])
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError("maxent.weight_bounds must be a [low, high] pair")
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
            mode=str(maxent_cfg.get("mode", "penalty")).lower(),
            misfit_factor=float(maxent_cfg.get("misfit_factor", 1.05)),
            search_iters=int(maxent_cfg.get("search_iters", 16)),
            weight_bounds=(float(bounds[0]), float(bounds[1])),
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

    # entropy_weight == 0 is, by definition, the ridge baseline anchor of the
    # data-error vs entropy Pareto curve. The ridge warm start already solves the
    # column-normalized Tikhonov system exactly; the MaxEnt objective here uses a
    # different (mean-based, un-column-normalized) L2/data normalization, so running
    # the optimizer from the warm start would drift AWAY from the true ridge solution.
    # Short-circuit so entropy_weight=0 reproduces the ridge fit exactly and the
    # Pareto baseline is genuinely comparable to the ridge baseline.
    if config.entropy_weight == 0.0 and warm_start_sigma is not None:
        return warm_start_sigma.detach().clone().to(device=device, dtype=dtype)

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


def data_misfit(sigma: torch.Tensor, operator: torch.Tensor, target: torch.Tensor) -> float:
    """Mean squared residual of the (row-weighted, target-normalized) data system."""

    residual = operator @ sigma - target
    return float(torch.mean(residual * residual).detach().cpu())


def solve_discrete_maxent_constrained(
    operator: torch.Tensor,
    target: torch.Tensor,
    source_positions: torch.Tensor,
    source_weights: torch.Tensor,
    shell_ids: torch.Tensor,
    config: MaxEntSolveConfig,
    *,
    warm_start_sigma: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Constrained MaxEnt: maximize entropy s.t. data misfit <= misfit_factor * ridge_misfit.

    This is the principled (Skilling-Bryan style) MaxEnt formulation. Instead of fixing an
    arbitrary entropy weight, it asks: *among all source distributions that fit the data
    about as well as the ridge solution (within ``misfit_factor`` of its mean-squared
    residual), what is the maximum-entropy one?*

    The entropy weight is selected by bisection in log-space: increasing the weight trades
    data fit for entropy, so we find the largest weight whose resulting misfit still
    satisfies the constraint. ``warm_start_sigma`` (the ridge solution) defines both the
    misfit target and the optimizer warm start.

    Returns ``(sigma, info)`` where ``info`` records the ridge misfit, the target misfit,
    the chosen entropy weight, and the achieved MaxEnt misfit.
    """

    if warm_start_sigma is None:
        raise ValueError("constrained MaxEnt requires a ridge warm start to define the misfit target")
    device = operator.device
    dtype = operator.dtype
    ridge_sigma = warm_start_sigma.detach().to(device=device, dtype=dtype)

    ridge_misfit = data_misfit(ridge_sigma, operator, target)
    target_misfit = max(float(config.misfit_factor), 1.0) * ridge_misfit

    low, high = config.weight_bounds
    if not (0.0 < low < high):
        raise ValueError("maxent.weight_bounds must satisfy 0 < low < high")
    log_low, log_high = math.log10(low), math.log10(high)

    penalty_cfg = replace(config, mode="penalty")
    best_sigma = ridge_sigma.clone()
    best_weight = 0.0
    for _ in range(max(1, int(config.search_iters))):
        mid = 0.5 * (log_low + log_high)
        weight = 10.0 ** mid
        sigma_w = solve_discrete_maxent(
            operator,
            target,
            source_positions,
            source_weights,
            shell_ids,
            replace(penalty_cfg, entropy_weight=float(weight)),
            warm_start_sigma=ridge_sigma,
        )
        misfit_w = data_misfit(sigma_w, operator, target)
        if misfit_w <= target_misfit:
            # constraint satisfied -> we can afford more entropy weight
            best_sigma = sigma_w.detach().clone()
            best_weight = float(weight)
            log_low = mid
        else:
            log_high = mid

    info = {
        "constrained": True,
        "ridge_misfit": ridge_misfit,
        "target_misfit": target_misfit,
        "misfit_factor": float(config.misfit_factor),
        "chosen_entropy_weight": best_weight,
        "maxent_misfit": data_misfit(best_sigma, operator, target),
    }
    if config.verbose:
        print(
            f"constrained maxent: ridge_misfit={ridge_misfit:.4e} target={target_misfit:.4e} "
            f"chosen_weight={best_weight:.4e} maxent_misfit={info['maxent_misfit']:.4e}"
        )
    return best_sigma, info
