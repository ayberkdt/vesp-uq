"""Solvers for discrete VESP source strengths."""

from __future__ import annotations

from typing import Callable

import torch


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

