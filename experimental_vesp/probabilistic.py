"""Probabilistic VESP scaffolding for future posterior experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GaussianSourcePosterior:
    """Diagonal Gaussian posterior over source strengths.

    This is a scaffold, not calibrated uncertainty inference.
    """

    mean: torch.Tensor
    log_std: torch.Tensor

    def sample(self, n_samples: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
        eps = torch.randn((n_samples, *self.mean.shape), dtype=self.mean.dtype, device=self.mean.device, generator=generator)
        return self.mean.unsqueeze(0) + eps * torch.exp(self.log_std).unsqueeze(0)

    def covariance_diag(self) -> torch.Tensor:
        return torch.exp(2.0 * self.log_std)


def empirical_acceleration_covariance(samples: torch.Tensor) -> torch.Tensor:
    """Compute covariance from acceleration samples [S, B, 3]."""

    centered = samples - samples.mean(dim=0, keepdim=True)
    return torch.einsum("sbi,sbj->bij", centered, centered) / max(1, samples.shape[0] - 1)

