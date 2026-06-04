"""Propagator-facing residual force model interface."""

from __future__ import annotations

import torch

from .models import DiscreteVESP


class VESPForceModel:
    def __init__(self, model: DiscreteVESP, *, source_chunk_size: int | None = None) -> None:
        self.model = model
        self.source_chunk_size = source_chunk_size

    @torch.no_grad()
    def predict_residual_accel(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.predict_acceleration(x, source_chunk_size=self.source_chunk_size)

    @torch.no_grad()
    def predict_residual_accel_with_uncertainty(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        accel = self.predict_residual_accel(x)
        return accel, None

