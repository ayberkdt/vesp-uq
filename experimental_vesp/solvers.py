"""Solvers for discrete VESP source strengths."""

from __future__ import annotations

from typing import Callable

import torch


def build_regularization_rows(
    *,
    source_positions: torch.Tensor,
    source_weights: torch.Tensor,
    shell_ids: torch.Tensor,
    lambda_l2: float = 0.0,
    lambda_moment: float = 0.0,
    lambda_dipole: float = 1.0,
    shell_energy_weights: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    dtype = source_positions.dtype
    device = source_positions.device
    n_sources = source_positions.shape[0]
    rows: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    if lambda_l2:
        rows.append(torch.sqrt(torch.tensor(float(lambda_l2), dtype=dtype, device=device)) * torch.eye(n_sources, dtype=dtype, device=device))
        targets.append(torch.zeros(n_sources, dtype=dtype, device=device))

    if lambda_moment:
        moment_scale = torch.sqrt(torch.tensor(float(lambda_moment), dtype=dtype, device=device))
        dipole_scale = torch.sqrt(torch.tensor(float(lambda_dipole), dtype=dtype, device=device))
        moment_rows = [source_weights]
        moment_rows.extend([dipole_scale * source_weights * source_positions[:, axis] for axis in range(3)])
        rows.append(moment_scale * torch.stack(moment_rows, dim=0))
        targets.append(torch.zeros(4, dtype=dtype, device=device))

    if shell_energy_weights is not None and shell_energy_weights.numel():
        diag = torch.zeros(n_sources, dtype=dtype, device=device)
        for shell_id, weight in enumerate(shell_energy_weights.to(device=device, dtype=dtype)):
            diag = diag + torch.where(shell_ids.to(device) == shell_id, weight * source_weights, torch.zeros_like(diag))
        active = diag > 0
        if torch.any(active):
            row = torch.zeros((int(active.sum().item()), n_sources), dtype=dtype, device=device)
            row[:, active] = torch.diag(torch.sqrt(diag[active]))
            rows.append(row)
            targets.append(torch.zeros(row.shape[0], dtype=dtype, device=device))
    return rows, targets


def solve_ridge_normal_equation(K: torch.Tensor, y: torch.Tensor, lambda_l2: float = 0.0) -> torch.Tensor:
    n = K.shape[1]
    lhs = K.T @ K
    rhs = K.T @ y
    if lambda_l2:
        lhs = lhs + float(lambda_l2) * torch.eye(n, dtype=K.dtype, device=K.device)
    return torch.linalg.solve(lhs, rhs)


def solve_ridge_augmented_lstsq(
    K: torch.Tensor,
    y: torch.Tensor,
    lambda_l2: float = 0.0,
    *,
    extra_rows: list[torch.Tensor] | None = None,
    extra_targets: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    rows = [K]
    targets = [y]
    if lambda_l2:
        n = K.shape[1]
        rows.append(torch.sqrt(torch.tensor(float(lambda_l2), dtype=K.dtype, device=K.device)) * torch.eye(n, dtype=K.dtype, device=K.device))
        targets.append(torch.zeros(n, dtype=K.dtype, device=K.device))
    if extra_rows:
        rows.extend(extra_rows)
        if extra_targets is None:
            targets.extend([torch.zeros(row.shape[0], dtype=K.dtype, device=K.device) for row in extra_rows])
        else:
            targets.extend(extra_targets)
    K_aug = torch.cat(rows, dim=0)
    y_aug = torch.cat(targets, dim=0)
    return torch.linalg.lstsq(K_aug, y_aug.unsqueeze(-1)).solution.squeeze(-1)


def solve_augmented_lstsq(K: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
    return solve_ridge_augmented_lstsq(K, y, **kwargs)


def solve_normal_equation(K: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
    return solve_ridge_normal_equation(K, y, **kwargs)


def solve_ridge(
    K: torch.Tensor,
    y: torch.Tensor,
    *,
    method: str = "augmented_lstsq",
    lambda_l2: float = 0.0,
    extra_rows: list[torch.Tensor] | None = None,
    extra_targets: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    if method == "normal_equation":
        if extra_rows:
            rows = [K, *extra_rows]
            targets = [y, *(extra_targets or [torch.zeros(r.shape[0], dtype=K.dtype, device=K.device) for r in extra_rows])]
            return solve_ridge_normal_equation(torch.cat(rows, dim=0), torch.cat(targets, dim=0), lambda_l2=0.0)
        return solve_ridge_normal_equation(K, y, lambda_l2=lambda_l2)
    if method == "augmented_lstsq":
        return solve_ridge_augmented_lstsq(K, y, lambda_l2=lambda_l2, extra_rows=extra_rows, extra_targets=extra_targets)
    raise ValueError(f"unknown ridge method: {method}")


def solve_torch_adam(
    parameters: list[torch.nn.Parameter],
    step_fn: Callable[[], torch.Tensor],
    *,
    epochs: int = 200,
    lr: float = 1.0e-2,
    grad_clip_norm: float = 0.0,
    log_every: int = 25,
) -> None:
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for epoch in range(epochs):
        loss = step_fn()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(parameters, grad_clip_norm)
        optimizer.step()
        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"epoch={epoch} loss={float(loss.detach().cpu()):.6e}")


solve_adam_matrix_free = solve_torch_adam
