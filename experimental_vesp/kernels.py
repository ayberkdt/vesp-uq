"""Newtonian potential and analytic acceleration kernels."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KernelOutput:
    potential: torch.Tensor | None
    acceleration: torch.Tensor | None


def potential_kernel(
    x: torch.Tensor,
    s: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    eps: float = 0.0,
) -> torch.Tensor:
    """Dense potential kernel [N_query, N_source]."""

    diff = x.unsqueeze(1) - s.unsqueeze(0)
    r2 = torch.sum(diff * diff, dim=-1)
    if eps:
        r2 = r2 + float(eps) ** 2
    kernel = torch.rsqrt(torch.clamp(r2, min=torch.finfo(x.dtype).eps))
    if weights is not None:
        kernel = kernel * weights.unsqueeze(0)
    return kernel


def acceleration_kernel(
    x: torch.Tensor,
    s: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    eps: float = 0.0,
    sign: float = 1.0,
) -> torch.Tensor:
    """Dense analytic acceleration kernel [N_query, N_source, 3]."""

    diff = s.unsqueeze(0) - x.unsqueeze(1)
    r2 = torch.sum(diff * diff, dim=-1)
    if eps:
        r2 = r2 + float(eps) ** 2
    inv_r = torch.rsqrt(torch.clamp(r2, min=torch.finfo(x.dtype).eps))
    kernel = float(sign) * diff * (inv_r * inv_r * inv_r).unsqueeze(-1)
    if weights is not None:
        kernel = kernel * weights.unsqueeze(0).unsqueeze(-1)
    return kernel


def evaluate_kernel(
    query_points: torch.Tensor,
    source_positions: torch.Tensor,
    source_strength: torch.Tensor,
    *,
    source_chunk_size: int | None = None,
    softening: float = 0.0,
    compute_potential: bool = True,
    compute_acceleration: bool = True,
    acceleration_sign: float = 1.0,
) -> KernelOutput:
    """Evaluate potential and acceleration for a batch of query points.

    ``source_strength`` should already include quadrature weights, i.e.
    ``source_strength = weights * sigma``.
    """

    if query_points.ndim != 2 or query_points.shape[-1] != 3:
        raise ValueError("query_points must have shape [B, 3]")
    if source_positions.ndim != 2 or source_positions.shape[-1] != 3:
        raise ValueError("source_positions must have shape [N, 3]")
    if source_strength.ndim != 1:
        raise ValueError("source_strength must have shape [N]")
    if source_positions.shape[0] != source_strength.shape[0]:
        raise ValueError("source_positions and source_strength length mismatch")

    n_sources = source_positions.shape[0]
    chunk = source_chunk_size or n_sources
    if chunk <= 0:
        raise ValueError("source_chunk_size must be positive")

    potential = None
    acceleration = None
    if compute_potential:
        potential = torch.zeros((query_points.shape[0], 1), dtype=query_points.dtype, device=query_points.device)
    if compute_acceleration:
        acceleration = torch.zeros((query_points.shape[0], 3), dtype=query_points.dtype, device=query_points.device)

    eps2 = float(softening) ** 2
    for start in range(0, n_sources, chunk):
        end = min(start + chunk, n_sources)
        s = source_positions[start:end]
        strength = source_strength[start:end]

        diff = s.unsqueeze(0) - query_points.unsqueeze(1)
        r2 = torch.sum(diff * diff, dim=-1)
        if eps2:
            r2 = r2 + eps2
        inv_r = torch.rsqrt(torch.clamp(r2, min=torch.finfo(query_points.dtype).eps))

        weighted = inv_r * strength.unsqueeze(0)
        if potential is not None:
            potential = potential + weighted.sum(dim=1, keepdim=True)

        if acceleration is not None:
            inv_r3 = inv_r * inv_r * inv_r
            acceleration = acceleration + float(acceleration_sign) * (
                diff * (strength.unsqueeze(0) * inv_r3).unsqueeze(-1)
            ).sum(dim=1)

    return KernelOutput(potential=potential, acceleration=acceleration)


def build_dense_operator(
    query_points: torch.Tensor,
    source_positions: torch.Tensor,
    source_weights: torch.Tensor,
    *,
    source_chunk_size: int | None = None,
    softening: float = 0.0,
    include_potential: bool = True,
    include_acceleration: bool = True,
    acceleration_sign: float = 1.0,
) -> torch.Tensor:
    """Build a dense linear operator mapping sigma to observations.

    This is useful for small-to-medium ridge/Tikhonov prototypes. It should not
    be used for very large source/query counts.
    """

    if not include_potential and not include_acceleration:
        raise ValueError("at least one of potential or acceleration must be included")

    n_sources = source_positions.shape[0]
    chunk = source_chunk_size or n_sources
    blocks = []
    eps2 = float(softening) ** 2

    for start in range(0, n_sources, chunk):
        end = min(start + chunk, n_sources)
        s = source_positions[start:end]
        weights = source_weights[start:end]

        diff = s.unsqueeze(0) - query_points.unsqueeze(1)
        r2 = torch.sum(diff * diff, dim=-1)
        if eps2:
            r2 = r2 + eps2
        inv_r = torch.rsqrt(torch.clamp(r2, min=torch.finfo(query_points.dtype).eps))

        rows = []
        if include_potential:
            rows.append(inv_r * weights.unsqueeze(0))
        if include_acceleration:
            inv_r3 = inv_r * inv_r * inv_r
            accel_block = float(acceleration_sign) * diff * (weights.unsqueeze(0) * inv_r3).unsqueeze(-1)
            rows.extend([accel_block[:, :, axis] for axis in range(3)])

        block = torch.cat([row for row in rows], dim=0)
        blocks.append(block)

    return torch.cat(blocks, dim=1)


def stack_observations(
    potential: torch.Tensor | None,
    acceleration: torch.Tensor | None,
    *,
    include_potential: bool = True,
    include_acceleration: bool = True,
) -> torch.Tensor:
    """Stack observations in the same row order as ``build_dense_operator``."""

    rows = []
    if include_potential:
        if potential is None:
            raise ValueError("potential target is required")
        rows.append(potential.reshape(-1))
    if include_acceleration:
        if acceleration is None:
            raise ValueError("acceleration target is required")
        rows.extend([acceleration[:, axis].reshape(-1) for axis in range(3)])
    return torch.cat(rows, dim=0)


def evaluate_potential_chunked(
    x: torch.Tensor,
    s: torch.Tensor,
    sigma: torch.Tensor,
    weights: torch.Tensor,
    *,
    source_chunk_size: int | None = None,
    eps: float = 0.0,
) -> torch.Tensor:
    out = evaluate_kernel(
        x,
        s,
        weights * sigma,
        source_chunk_size=source_chunk_size,
        softening=eps,
        compute_potential=True,
        compute_acceleration=False,
    )
    if out.potential is None:
        raise RuntimeError("potential evaluation failed")
    return out.potential


def evaluate_acceleration_chunked(
    x: torch.Tensor,
    s: torch.Tensor,
    sigma: torch.Tensor,
    weights: torch.Tensor,
    *,
    source_chunk_size: int | None = None,
    eps: float = 0.0,
    sign: float = 1.0,
) -> torch.Tensor:
    out = evaluate_kernel(
        x,
        s,
        weights * sigma,
        source_chunk_size=source_chunk_size,
        softening=eps,
        compute_potential=False,
        compute_acceleration=True,
        acceleration_sign=sign,
    )
    if out.acceleration is None:
        raise RuntimeError("acceleration evaluation failed")
    return out.acceleration


def evaluate_potential_acceleration_chunked(
    x: torch.Tensor,
    s: torch.Tensor,
    sigma: torch.Tensor,
    weights: torch.Tensor,
    *,
    source_chunk_size: int | None = None,
    eps: float = 0.0,
    sign: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = evaluate_kernel(
        x,
        s,
        weights * sigma,
        source_chunk_size=source_chunk_size,
        softening=eps,
        compute_potential=True,
        compute_acceleration=True,
        acceleration_sign=sign,
    )
    if out.potential is None or out.acceleration is None:
        raise RuntimeError("joint kernel evaluation failed")
    return out.potential, out.acceleration
