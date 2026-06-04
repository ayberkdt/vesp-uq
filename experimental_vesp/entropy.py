"""Stage 3 entropy regularization building blocks.

This is not the full MaxEnt posterior framework. It provides deterministic
entropy diagnostics/regularizers over solved source strengths so Stage 3 can
start without discarding the Stage 1-2 ridge baseline.
"""

from __future__ import annotations

import torch


def normalized_abs_distribution(sigma: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    mass = torch.abs(sigma if weights is None else weights * sigma)
    total = torch.sum(mass)
    if total <= 0:
        return torch.full_like(mass, 1.0 / max(1, mass.numel()))
    return mass / total


def shannon_entropy(p: torch.Tensor, *, eps: float = 1.0e-30) -> torch.Tensor:
    p_safe = torch.clamp(p, min=eps)
    return -torch.sum(p_safe * torch.log(p_safe))


def effective_source_entropy(sigma: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    return shannon_entropy(normalized_abs_distribution(sigma, weights))


def positive_negative_entropy(
    sigma: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    eps: float = 1.0e-30,
) -> torch.Tensor:
    weighted = sigma if weights is None else weights * sigma
    pos = torch.clamp(weighted, min=0.0)
    neg = torch.clamp(-weighted, min=0.0)
    terms = []
    for mass in (pos, neg):
        total = torch.sum(mass)
        if total > eps:
            terms.append(shannon_entropy(mass / total, eps=eps))
    if not terms:
        return torch.zeros((), dtype=sigma.dtype, device=sigma.device)
    return torch.stack(terms).sum()


def relative_entropy_to_uniform(sigma: torch.Tensor, weights: torch.Tensor | None = None, *, eps: float = 1.0e-30) -> torch.Tensor:
    p = normalized_abs_distribution(sigma, weights)
    uniform = torch.full_like(p, 1.0 / p.numel())
    return torch.sum(torch.clamp(p, min=eps) * torch.log(torch.clamp(p / uniform, min=eps)))


def shellwise_entropy(
    sigma: torch.Tensor,
    shell_ids: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    values = []
    for shell_id in torch.unique(shell_ids):
        mask = shell_ids == shell_id
        values.append(effective_source_entropy(sigma[mask], None if weights is None else weights[mask]))
    if not values:
        return torch.zeros((), dtype=sigma.dtype, device=sigma.device)
    return torch.stack(values)


def entropy_regularization_loss(
    sigma: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    mode: str = "positive_negative",
) -> torch.Tensor:
    """Return a loss term to minimize.

    Entropy should be maximized, so this returns negative entropy for entropy
    modes and KL for the relative-entropy mode.
    """

    if mode == "abs":
        return -effective_source_entropy(sigma, weights)
    if mode == "positive_negative":
        return -positive_negative_entropy(sigma, weights)
    if mode == "relative_uniform":
        return relative_entropy_to_uniform(sigma, weights)
    raise ValueError(f"unknown entropy mode: {mode}")

