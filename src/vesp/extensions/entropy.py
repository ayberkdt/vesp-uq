"""Stage 3A entropy regularization building blocks.

This is NOT the full MaxEnt posterior framework. It provides deterministic entropy
diagnostics / regularizers over solved source strengths so Stage 3A can start without
discarding the Stage 1-2 ridge baseline.

Why not naive Shannon entropy on raw sigma?
-------------------------------------------
The source strengths ``sigma`` are *signed*. Shannon entropy ``-sum p log p`` is only
defined for a probability distribution (non-negative, sums to 1), so it cannot be
applied to signed values directly. Each mode below therefore maps the signed strengths
to a valid distribution first, and each choice of mapping has a different meaning and
limitation:

- ``abs``              — distribution over ``|w*sigma|``. Simple, but it discards sign
                         information, so it cannot tell a healthy spread-out solution
                         from a brittle +/- cancelling one.
- ``positive_negative``— split into positive and negative parts, normalize each side
                         separately, sum the two entropies. Sign-aware.
- ``relative_uniform`` — KL divergence of the ``|w*sigma|`` distribution to uniform;
                         "penalize unnecessary concentration".
- ``shell_balance``    — entropy of the per-shell *energy* fractions; a coarse,
                         shell-level quantity, NOT a pointwise source entropy.

Smoothness caveat: ``abs`` / ``clamp`` are not differentiable at 0. In practice the
optimizer (L-BFGS / Adam) only sees these kinks on a measure-zero set and converges
fine from the ridge warm start, but the objective is only piecewise-smooth. A smooth
surrogate (e.g. ``sqrt(sigma^2 + delta^2)``) could replace ``abs`` if exact-gradient
guarantees are ever needed; this is intentionally left as a future option rather than
the main refactor.
"""

from __future__ import annotations

import torch

from vesp.core.losses import shell_energy


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
    """``abs`` mode: Shannon entropy of the ``|w*sigma|`` distribution.

    A simple sign-blind diagnostic / regularizer. ``exp`` of this value is the
    effective source count. It is NOT a full signed MaxEnt formulation (see the module
    docstring): it cannot distinguish a healthy spread from a +/- cancelling solution.
    """

    return shannon_entropy(normalized_abs_distribution(sigma, weights))


def positive_negative_entropy(
    sigma: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    eps: float = 1.0e-30,
) -> torch.Tensor:
    """Sign-aware entropy: H(positive part) + H(negative part).

    The weighted strengths are split into their positive and negative parts; each part
    is normalized to a distribution and its Shannon entropy is summed. Maximizing this
    spreads source mass within each sign group, discouraging a few dominant sources.

    Edge cases are handled safely: an all-positive (or all-negative) solution simply
    contributes one entropy term, and an all-zero solution returns 0. Limitation: it
    treats the two sign groups independently, so it does not by itself penalize a large
    near-cancelling +/- pair if each side is internally well spread (use
    ``shell_cancellation_ratio`` / ``shell_balance`` for that failure mode).
    """

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
    """KL divergence of the absolute source-mass distribution to uniform.

    ``KL(p || uniform)`` is 0 for a perfectly uniform ``|w*sigma|`` and grows as mass
    concentrates, so it is used as a regularizer that "avoids unnecessary
    concentration". Like ``abs``, it ignores sign, so it is a concentration penalty
    rather than a full signed MaxEnt objective.
    """

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


def shell_energy_fractions(
    sigma: torch.Tensor,
    source_weights: torch.Tensor,
    shell_ids: torch.Tensor,
    *,
    eps: float = 1.0e-30,
) -> torch.Tensor:
    """Normalized per-shell energy distribution ``p_j = E_j / sum_j E_j``."""

    energies = shell_energy(sigma, source_weights, shell_ids)
    if energies.numel() == 0:
        return energies
    total = torch.sum(energies)
    if total <= eps:
        return torch.full_like(energies, 1.0 / energies.numel())
    return energies / total


def shell_energy_balance_entropy(
    sigma: torch.Tensor,
    source_weights: torch.Tensor,
    shell_ids: torch.Tensor,
    *,
    eps: float = 1.0e-30,
) -> torch.Tensor:
    """Entropy of the per-shell energy distribution.

    Maximizing this directly resists shell-energy collapse, the multi-shell
    failure mode where almost all energy concentrates on a single shell.
    """

    fractions = shell_energy_fractions(sigma, source_weights, shell_ids, eps=eps)
    if fractions.numel() <= 1:
        return torch.zeros((), dtype=sigma.dtype, device=sigma.device)
    return shannon_entropy(fractions, eps=eps)


def entropy_regularization_loss(
    sigma: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    mode: str = "positive_negative",
    shell_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a loss term to minimize.

    Entropy should be maximized, so this returns negative entropy for entropy
    modes and KL for the relative-entropy mode. ``shell_balance`` maximizes the
    entropy of the per-shell energy distribution and requires ``shell_ids``.
    """

    if mode == "abs":
        return -effective_source_entropy(sigma, weights)
    if mode == "positive_negative":
        return -positive_negative_entropy(sigma, weights)
    if mode == "relative_uniform":
        return relative_entropy_to_uniform(sigma, weights)
    if mode == "shell_balance":
        if shell_ids is None:
            raise ValueError("shell_balance entropy mode requires shell_ids")
        return -shell_energy_balance_entropy(sigma, weights if weights is not None else torch.ones_like(sigma), shell_ids)
    raise ValueError(f"unknown entropy mode: {mode}")

