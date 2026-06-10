"""Principled automatic selection of the Tikhonov weight ``lambda_l2``.

Replaces hand-picked regularization with the classical **L-curve corner**: sweep
``lambda_l2``, plot ``(log||A sigma - b||, log||sigma||)``, and pick the point of maximum
curvature — the corner where the solution stops fitting data (norm shrinks fast, residual
barely grows) and starts over-smoothing (residual grows fast). It needs only ridge solves
(no hat-matrix trace), so it is cheap and robust for the dense equivalent-source problem.
"""

from __future__ import annotations

import math
from dataclasses import replace

import torch

from vesp.core.solvers import RidgeSolveConfig, solve_discrete_ridge

DEFAULT_LAMBDA_GRID = [
    1.0e-10, 1.0e-8, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3,
    1.0e-2, 1.0e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0,
]


def _menger_curvature(p_prev, p_curr, p_next) -> float:
    """Menger curvature of three planar points (1/circumradius); larger = sharper corner."""

    (x1, y1), (x2, y2), (x3, y3) = p_prev, p_curr, p_next
    area2 = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))  # 2 * triangle area
    d12 = math.hypot(x2 - x1, y2 - y1)
    d23 = math.hypot(x3 - x2, y3 - y2)
    d13 = math.hypot(x3 - x1, y3 - y1)
    denom = d12 * d23 * d13
    if denom <= 0.0:
        return 0.0
    return 2.0 * area2 / denom


def select_lambda_l2(
    operator: torch.Tensor,
    target: torch.Tensor,
    *,
    source_positions: torch.Tensor,
    source_weights: torch.Tensor,
    shell_ids: torch.Tensor,
    base_config: RidgeSolveConfig,
    grid: list[float] | None = None,
) -> tuple[float, list[dict]]:
    """Pick ``lambda_l2`` at the L-curve corner. Returns ``(lambda_star, curve_points)``.

    ``base_config`` supplies the ridge method / column-normalize / moment settings; only its
    ``lambda_l2`` is overridden across the grid. ``curve_points`` records, per grid value, the
    residual norm, solution norm and Menger curvature (for plotting / auditing).
    """

    grid = sorted(float(g) for g in (grid or DEFAULT_LAMBDA_GRID))
    eps = torch.finfo(operator.dtype).tiny
    points: list[dict] = []
    for lam in grid:
        sigma = solve_discrete_ridge(
            operator=operator,
            target=target,
            source_positions=source_positions,
            source_weights=source_weights,
            shell_ids=shell_ids,
            config=replace(base_config, lambda_l2=lam),
        )
        residual_norm = float(torch.linalg.norm(operator @ sigma - target).detach().cpu())
        solution_norm = float(torch.linalg.norm(sigma).detach().cpu())
        points.append(
            {
                "lambda_l2": lam,
                "residual_norm": residual_norm,
                "solution_norm": solution_norm,
                "log_residual": math.log(max(residual_norm, eps)),
                "log_solution": math.log(max(solution_norm, eps)),
                "curvature": 0.0,
            }
        )

    # Maximum-curvature interior point of the (log residual, log solution) L-curve.
    best_idx = len(points) // 2
    best_curv = -1.0
    for i in range(1, len(points) - 1):
        curv = _menger_curvature(
            (points[i - 1]["log_residual"], points[i - 1]["log_solution"]),
            (points[i]["log_residual"], points[i]["log_solution"]),
            (points[i + 1]["log_residual"], points[i + 1]["log_solution"]),
        )
        points[i]["curvature"] = curv
        if curv > best_curv:
            best_curv = curv
            best_idx = i
    return points[best_idx]["lambda_l2"], points


def lcurve_lambda(
    operator: torch.Tensor,
    target: torch.Tensor,
    *,
    grid: list[float] | None = None,
) -> tuple[float, list[dict]]:
    """Operator-only L-curve corner selection of the Tikhonov weight ``lambda_l2``.

    Unlike :func:`select_lambda_l2`, this needs only ``(A, b)`` -- no source geometry or
    :class:`RidgeSolveConfig` -- so it is the natural entry point for the surrogate-agnostic
    VESP-UQ plugin. It uses one economy SVD of ``A`` and evaluates each grid ``lambda`` in
    closed form (no repeated solves, no ill-conditioning): for ``A = U S V^T`` and
    ``beta = U^T b`` the Tikhonov filter factors are ``f_i = s_i^2 / (s_i^2 + lambda)``, the
    solution norm is ``||sigma||^2 = sum (f_i beta_i / s_i)^2`` and the residual norm is
    ``||A sigma - b||^2 = sum (lambda/(s_i^2 + lambda))^2 beta_i^2 + (||b||^2 - ||beta||^2)``.
    Returns ``(lambda_star, curve_points)`` at maximum Menger curvature of the log-log curve.
    """

    grid = sorted(float(g) for g in (grid or DEFAULT_LAMBDA_GRID))
    eps = torch.finfo(operator.dtype).tiny
    # economy SVD; singular values descending. Drop numerically-zero modes.
    svd = torch.linalg.svd(operator, full_matrices=False)
    s = svd.S
    beta = svd.U.transpose(-1, -2) @ target  # (k,)
    s2 = s * s
    beta2 = beta * beta
    b_norm2 = float(torch.sum(target * target).detach().cpu())
    captured = float(torch.sum(beta2).detach().cpu())
    residual_floor = max(b_norm2 - captured, 0.0)  # energy of b outside the column space

    points: list[dict] = []
    for lam in grid:
        denom = s2 + lam
        f = s2 / denom
        sol_norm2 = float(torch.sum((f * beta / s.clamp_min(eps)) ** 2).detach().cpu())
        res_modes = float(torch.sum(((lam / denom) ** 2) * beta2).detach().cpu())
        residual_norm = math.sqrt(max(res_modes + residual_floor, 0.0))
        solution_norm = math.sqrt(max(sol_norm2, 0.0))
        points.append(
            {
                "lambda_l2": lam,
                "residual_norm": residual_norm,
                "solution_norm": solution_norm,
                "log_residual": math.log(max(residual_norm, eps)),
                "log_solution": math.log(max(solution_norm, eps)),
                "curvature": 0.0,
            }
        )

    best_idx = len(points) // 2
    best_curv = -1.0
    for i in range(1, len(points) - 1):
        curv = _menger_curvature(
            (points[i - 1]["log_residual"], points[i - 1]["log_solution"]),
            (points[i]["log_residual"], points[i]["log_solution"]),
            (points[i + 1]["log_residual"], points[i + 1]["log_solution"]),
        )
        points[i]["curvature"] = curv
        if curv > best_curv:
            best_curv = curv
            best_idx = i
    return points[best_idx]["lambda_l2"], points


def lambda_is_auto(config: dict) -> bool:
    """True if ``loss.lambda_l2`` (or ``solver.lambda_l2``) is the sentinel string ``auto``."""

    loss = config.get("loss", {})
    solver = config.get("solver", {})
    solver = solver if isinstance(solver, dict) else {}
    value = loss.get("lambda_l2", solver.get("lambda_l2", None))
    return isinstance(value, str) and value.strip().lower() == "auto"
