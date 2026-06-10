"""Synthetic residual gravity scenarios."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from vesp.core.kernels import evaluate_kernel
from vesp.core.sources import make_shell_sources
from vesp.data.dataset import ResidualGravityData


def make_synthetic_dataset(
    *,
    n_query: int = 1024,
    n_truth_sources: int | Sequence[int] = 64,
    query_radius_min: float = 1.05,
    query_radius_max: float = 1.60,
    truth_shell_radius: float = 0.72,
    truth_shell_radii: Sequence[float] | None = None,
    noise_std: float = 0.0,
    seed: int = 7,
    dtype: torch.dtype = torch.float32,
) -> ResidualGravityData:
    generator = torch.Generator().manual_seed(seed)
    directions = torch.randn((n_query, 3), generator=generator, dtype=dtype)
    directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
    radii = query_radius_min + (query_radius_max - query_radius_min) * torch.rand((n_query, 1), generator=generator, dtype=dtype)
    positions = directions * radii

    shell_radii = list(truth_shell_radii) if truth_shell_radii is not None else [truth_shell_radius]
    truth_sources = make_shell_sources(shell_radii, n_truth_sources, dtype=dtype)
    sigma_truth = torch.randn(truth_sources.n_sources, generator=generator, dtype=dtype)
    sigma_truth = sigma_truth - sigma_truth.mean()
    out = evaluate_kernel(positions, truth_sources.positions, truth_sources.weights * sigma_truth)
    if out.potential is None or out.acceleration is None:
        raise RuntimeError("synthetic kernel evaluation failed")
    potential = out.potential
    acceleration = out.acceleration
    if noise_std:
        potential = potential + noise_std * torch.randn(potential.shape, generator=generator, dtype=dtype)
        acceleration = acceleration + noise_std * torch.randn(acceleration.shape, generator=generator, dtype=dtype)
    return ResidualGravityData(
        positions=positions,
        potential=potential,
        acceleration=acceleration,
        metadata={
            "type": "synthetic",
            "position_units": "normalized",
            "truth_shell_radii": shell_radii,
            "n_truth_sources": n_truth_sources,
            "noise_std": noise_std,
        },
    )


def make_same_family_recovery_case(**kwargs) -> ResidualGravityData:
    kwargs.setdefault("truth_shell_radius", 0.86)
    return make_synthetic_dataset(**kwargs)


def make_radius_mismatch_case(**kwargs) -> ResidualGravityData:
    kwargs.setdefault("truth_shell_radius", 0.72)
    return make_synthetic_dataset(**kwargs)


def make_multishell_truth_case(**kwargs) -> ResidualGravityData:
    kwargs.setdefault("truth_shell_radii", [0.50, 0.78, 0.86])
    kwargs.setdefault("n_truth_sources", [96, 128, 96])
    return make_synthetic_dataset(**kwargs)


def make_noisy_case(**kwargs) -> ResidualGravityData:
    kwargs.setdefault("noise_std", 1.0e-4)
    return make_radius_mismatch_case(**kwargs)


def make_altitude_ood_case(**kwargs) -> ResidualGravityData:
    kwargs.setdefault("query_radius_min", 1.01)
    kwargs.setdefault("query_radius_max", 2.00)
    return make_radius_mismatch_case(**kwargs)

