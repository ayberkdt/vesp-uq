"""Dataset and synthetic-data helpers for VESP experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from .kernels import evaluate_kernel
from .sources import make_shell_sources


COLUMN_ALIASES = {
    "x": ("x", "X"),
    "y": ("y", "Y"),
    "z": ("z", "Z"),
    "potential": ("Delta U", "DeltaU", "Delta_U", "delta_u", "dU", "du", "potential"),
    "acceleration_x": (
        "Delta a_x",
        "Delta_a_x",
        "Delta ax",
        "Deltaa_x",
        "Deltaax",
        "delta_a_x",
        "dax",
        "acceleration_x",
    ),
    "acceleration_y": (
        "Delta a_y",
        "Delta_a_y",
        "Delta ay",
        "Deltaa_y",
        "Deltayay",
        "Deltaay",
        "delta_a_y",
        "day",
        "acceleration_y",
    ),
    "acceleration_z": (
        "Delta a_z",
        "Delta_a_z",
        "Delta az",
        "Deltaa_z",
        "Deltaz",
        "Deltaaz",
        "delta_a_z",
        "daz",
        "acceleration_z",
    ),
}


@dataclass
class ResidualGravityData:
    positions: torch.Tensor
    potential: torch.Tensor
    acceleration: torch.Tensor
    metadata: dict | None = None

    def to(self, device: torch.device | str) -> "ResidualGravityData":
        return ResidualGravityData(
            positions=self.positions.to(device),
            potential=self.potential.to(device),
            acceleration=self.acceleration.to(device),
            metadata=self.metadata,
        )

    def subset(self, indices: torch.Tensor) -> "ResidualGravityData":
        return ResidualGravityData(
            positions=self.positions[indices],
            potential=self.potential[indices],
            acceleration=self.acceleration[indices],
            metadata=self.metadata,
        )

    @property
    def r(self) -> torch.Tensor:
        return torch.linalg.norm(self.positions, dim=-1)

    @property
    def altitude(self) -> torch.Tensor:
        return self.r - 1.0


class ResidualGravityDataset(Dataset):
    def __init__(self, data: ResidualGravityData) -> None:
        self.data = data

    def __len__(self) -> int:
        return int(self.data.positions.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.data.positions[idx],
            "potential": self.data.potential[idx],
            "acceleration": self.data.acceleration[idx],
        }


def load_csv_dataset(path: str | Path, *, dtype: torch.dtype = torch.float32) -> ResidualGravityData:
    path = Path(path)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        fieldnames = set(reader.fieldnames)

        selected = {}
        missing = []
        for logical_name, aliases in COLUMN_ALIASES.items():
            match = next((alias for alias in aliases if alias in fieldnames), None)
            if match is None:
                missing.append(f"{logical_name} aliases={aliases}")
            else:
                selected[logical_name] = match

        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        x_rows = []
        u_rows = []
        a_rows = []
        for row in reader:
            x_rows.append([float(row[selected["x"]]), float(row[selected["y"]]), float(row[selected["z"]])])
            u_rows.append([float(row[selected["potential"]])])
            a_rows.append(
                [
                    float(row[selected["acceleration_x"]]),
                    float(row[selected["acceleration_y"]]),
                    float(row[selected["acceleration_z"]]),
                ]
            )

    if not x_rows:
        raise ValueError(f"CSV file has no data rows: {path}")

    return ResidualGravityData(
        positions=torch.tensor(x_rows, dtype=dtype),
        potential=torch.tensor(u_rows, dtype=dtype),
        acceleration=torch.tensor(a_rows, dtype=dtype),
        metadata={"path": str(path), "position_units": "normalized"},
    )


def split_data(
    data: ResidualGravityData,
    *,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> tuple[ResidualGravityData, ResidualGravityData]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    generator = torch.Generator().manual_seed(seed)
    n = data.positions.shape[0]
    perm = torch.randperm(n, generator=generator)
    n_train = int(round(train_fraction * n))
    return data.subset(perm[:n_train]), data.subset(perm[n_train:])


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
    """Generate a deterministic synthetic residual field from hidden sources."""

    generator = torch.Generator().manual_seed(seed)
    directions = torch.randn((n_query, 3), generator=generator, dtype=dtype)
    directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
    radii = query_radius_min + (query_radius_max - query_radius_min) * torch.rand((n_query, 1), generator=generator, dtype=dtype)
    positions = directions * radii

    shell_radii = list(truth_shell_radii) if truth_shell_radii is not None else [truth_shell_radius]
    truth_sources = make_shell_sources(shell_radii, n_truth_sources, dtype=dtype)
    sigma_truth = torch.randn(truth_sources.n_sources, generator=generator, dtype=dtype)
    sigma_truth = sigma_truth - sigma_truth.mean()
    strength = truth_sources.weights * sigma_truth

    out = evaluate_kernel(positions, truth_sources.positions, strength)
    potential = out.potential
    acceleration = out.acceleration
    if potential is None or acceleration is None:
        raise RuntimeError("synthetic kernel evaluation failed")

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
