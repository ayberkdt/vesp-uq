"""Source geometry utilities for discrete equivalent-source models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class SourceSet:
    """Fixed source geometry and quadrature-like weights."""

    positions: torch.Tensor
    weights: torch.Tensor
    shell_ids: torch.Tensor
    shell_radii: tuple[float, ...]

    def to(self, device: torch.device | str) -> "SourceSet":
        return SourceSet(
            positions=self.positions.to(device),
            weights=self.weights.to(device),
            shell_ids=self.shell_ids.to(device),
            shell_radii=self.shell_radii,
        )

    @property
    def n_sources(self) -> int:
        return int(self.positions.shape[0])


SourceGeometry = SourceSet


def fibonacci_sphere(
    n_points: int,
    radius: float = 1.0,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Generate approximately uniform points on a sphere."""

    if n_points <= 0:
        raise ValueError("n_points must be positive")

    i = torch.arange(n_points, dtype=dtype, device=device)
    golden_angle = torch.pi * (3.0 - torch.sqrt(torch.tensor(5.0, dtype=dtype, device=device)))

    z = 1.0 - 2.0 * (i + 0.5) / n_points
    r_xy = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    theta = golden_angle * i

    x = torch.cos(theta) * r_xy
    y = torch.sin(theta) * r_xy
    points = torch.stack((x, y, z), dim=-1)
    return radius * points


def _as_points_per_shell(
    points_per_shell: int | Sequence[int],
    n_shells: int,
) -> list[int]:
    if isinstance(points_per_shell, int):
        return [points_per_shell] * n_shells

    counts = [int(v) for v in points_per_shell]
    if len(counts) != n_shells:
        raise ValueError("points_per_shell must be an int or match len(shell_radii)")
    if any(v <= 0 for v in counts):
        raise ValueError("all source counts must be positive")
    return counts


def make_shell_sources(
    shell_radii: Sequence[float],
    points_per_shell: int | Sequence[int],
    *,
    body_radius: float = 1.0,
    weight_mode: str = "surface_area",
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> SourceSet:
    """Create one or more spherical source shells.

    The initial framework uses surface-like shell weights. For normalized units,
    use ``body_radius=1`` and shell radii such as ``0.80`` or ``0.95``.
    """

    radii = tuple(float(r) for r in shell_radii)
    if not radii:
        raise ValueError("at least one shell radius is required")
    if any(r <= 0.0 or r >= 1.0 for r in radii):
        raise ValueError("shell radii are expected in the interior: 0 < alpha < 1")

    counts = _as_points_per_shell(points_per_shell, len(radii))
    positions = []
    weights = []
    shell_ids = []

    for shell_id, (alpha, count) in enumerate(zip(radii, counts)):
        radius = alpha * body_radius
        points = fibonacci_sphere(count, radius=radius, dtype=dtype, device=device)

        if weight_mode == "surface_area":
            weight_value = 4.0 * torch.pi * radius * radius / count
        elif weight_mode == "uniform":
            weight_value = 1.0 / count
        elif weight_mode == "none":
            weight_value = 1.0
        else:
            raise ValueError(f"unknown weight_mode: {weight_mode}")

        positions.append(points)
        weights.append(torch.full((count,), float(weight_value), dtype=dtype, device=device))
        shell_ids.append(torch.full((count,), shell_id, dtype=torch.long, device=device))

    return SourceSet(
        positions=torch.cat(positions, dim=0),
        weights=torch.cat(weights, dim=0),
        shell_ids=torch.cat(shell_ids, dim=0),
        shell_radii=radii,
    )


def single_shell_sources(
    alpha: float,
    n_source: int,
    *,
    R_body: float = 1.0,
    weight_mode: str = "surface_area",
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> SourceGeometry:
    """Compatibility wrapper for one spherical source shell."""

    return make_shell_sources(
        [alpha],
        n_source,
        body_radius=R_body,
        weight_mode=weight_mode,
        dtype=dtype,
        device=device,
    )


def multi_shell_sources(
    alphas: Sequence[float],
    n_sources_per_shell: int | Sequence[int],
    *,
    R_body: float = 1.0,
    weight_mode: str = "surface_area",
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> SourceGeometry:
    """Compatibility wrapper for multiple spherical source shells."""

    return make_shell_sources(
        alphas,
        n_sources_per_shell,
        body_radius=R_body,
        weight_mode=weight_mode,
        dtype=dtype,
        device=device,
    )
