"""Discrete VESP model definitions."""

from __future__ import annotations

import torch
from torch import nn

from .kernels import evaluate_kernel
from .diagnostics import source_diagnostics
from .sources import SourceSet


class DiscreteVESP(nn.Module):
    """Fixed-source equivalent-source model with learnable source strengths."""

    def __init__(
        self,
        source_set: SourceSet,
        *,
        init_scale: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.register_buffer("source_positions", source_set.positions.to(dtype=dtype))
        self.register_buffer("source_weights", source_set.weights.to(dtype=dtype))
        self.register_buffer("shell_ids", source_set.shell_ids)
        self.shell_radii = source_set.shell_radii

        sigma = torch.empty(source_set.n_sources, dtype=dtype)
        if init_scale == 0.0:
            sigma.zero_()
        else:
            sigma.normal_(mean=0.0, std=init_scale)
        self.sigma = nn.Parameter(sigma)

    @property
    def n_sources(self) -> int:
        return int(self.sigma.shape[0])

    def forward(
        self,
        query_points: torch.Tensor,
        *,
        source_chunk_size: int | None = None,
        softening: float = 0.0,
        compute_potential: bool = True,
        compute_acceleration: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        query_points = query_points.to(dtype=self.sigma.dtype, device=self.sigma.device)
        strength = self.source_weights * self.sigma
        out = evaluate_kernel(
            query_points,
            self.source_positions,
            strength,
            source_chunk_size=source_chunk_size,
            softening=softening,
            compute_potential=compute_potential,
            compute_acceleration=compute_acceleration,
        )
        return out.potential, out.acceleration

    def predict_potential(self, query_points: torch.Tensor, **kwargs) -> torch.Tensor:
        potential, _ = self.forward(query_points, compute_potential=True, compute_acceleration=False, **kwargs)
        if potential is None:
            raise RuntimeError("potential prediction failed")
        return potential

    def predict_acceleration(self, query_points: torch.Tensor, **kwargs) -> torch.Tensor:
        _, acceleration = self.forward(query_points, compute_potential=False, compute_acceleration=True, **kwargs)
        if acceleration is None:
            raise RuntimeError("acceleration prediction failed")
        return acceleration

    def get_sigma(self) -> torch.Tensor:
        return self.sigma.detach().clone()

    def set_sigma(self, sigma: torch.Tensor) -> None:
        if sigma.shape != self.sigma.shape:
            raise ValueError(f"sigma shape mismatch: expected {tuple(self.sigma.shape)}, got {tuple(sigma.shape)}")
        with torch.no_grad():
            self.sigma.copy_(sigma.to(dtype=self.sigma.dtype, device=self.sigma.device))

    def source_diagnostics(self) -> dict:
        return source_diagnostics(
            source_positions=self.source_positions,
            source_weights=self.source_weights,
            shell_ids=self.shell_ids,
            sigma=self.sigma,
        )


class MultiShellDiscreteVESP(DiscreteVESP):
    """Semantic alias for a discrete model with multiple source shells."""

    pass


def save_checkpoint(
    path: str,
    model: DiscreteVESP,
    config: dict,
    metrics: dict | None = None,
) -> None:
    torch.save(
        {
            "source_positions": model.source_positions.detach().cpu(),
            "source_weights": model.source_weights.detach().cpu(),
            "shell_ids": model.shell_ids.detach().cpu(),
            "shell_radii": model.shell_radii,
            "sigma": model.sigma.detach().cpu(),
            "config": config,
            "metrics": metrics or {},
        },
        path,
    )


def load_checkpoint(path: str, *, map_location: str | torch.device = "cpu") -> DiscreteVESP:
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    source_set = SourceSet(
        positions=ckpt["source_positions"],
        weights=ckpt["source_weights"],
        shell_ids=ckpt["shell_ids"],
        shell_radii=tuple(float(v) for v in ckpt["shell_radii"]),
    )
    cls = MultiShellDiscreteVESP if len(source_set.shell_radii) > 1 else DiscreteVESP
    model = cls(source_set, dtype=ckpt["sigma"].dtype)
    model.set_sigma(ckpt["sigma"])
    return model
