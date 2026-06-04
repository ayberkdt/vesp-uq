"""Neural source-density scaffolds for future Stage 3 experiments."""

from __future__ import annotations

import math

import torch
from torch import nn


class AngularMLP(nn.Module):
    """Simple angular source-density network f(u)->sigma."""

    def __init__(self, hidden_dim: int = 64, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 3
        for _ in range(depth):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, unit_vectors: torch.Tensor) -> torch.Tensor:
        return self.net(unit_vectors).squeeze(-1)


class SineLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, omega: float = 30.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.omega = omega
        bound = math.sqrt(6.0 / in_dim) / omega
        nn.init.uniform_(self.linear.weight, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.linear(x))


class AngularSIREN(nn.Module):
    """SIREN-style angular source-density scaffold."""

    def __init__(self, hidden_dim: int = 64, depth: int = 3, omega: float = 30.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [SineLayer(3, hidden_dim, omega=omega)]
        for _ in range(max(0, depth - 1)):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega=omega))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, unit_vectors: torch.Tensor) -> torch.Tensor:
        return self.net(unit_vectors).squeeze(-1)


class NeuralDensityShell(nn.Module):
    """Maps fixed shell source directions to source strengths."""

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, source_positions: torch.Tensor) -> torch.Tensor:
        directions = source_positions / torch.clamp(torch.linalg.norm(source_positions, dim=-1, keepdim=True), min=1.0e-12)
        return self.network(directions)

