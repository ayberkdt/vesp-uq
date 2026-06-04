"""Dataset and synthetic-data helpers for VESP experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from .lunar import validate_lunar_metadata_contract
from .units import PositionScaler, UnitConfig


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
        metadata = self.metadata or {}
        units = UnitConfig(
            R_body=float(metadata.get("R_body", 1.0)),
            normalize_positions=metadata.get("position_units", "normalized") == "normalized",
            position_units=str(metadata.get("position_units", "normalized")),
            potential_units=str(metadata.get("potential_units", "model")),
            acceleration_units=str(metadata.get("acceleration_units", "model")),
        )
        return PositionScaler(units).altitude_from_model_positions(self.positions)


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
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    if not metadata_path.exists():
        metadata_path = path.with_suffix(".metadata.json")
    metadata = {"path": str(path), "position_units": "normalized"}
    if metadata_path.exists():
        metadata.update(json.loads(metadata_path.read_text(encoding="utf-8")))
    contract = validate_lunar_metadata_contract(metadata, data_path=path, require_lunar=False)
    if contract.get("has_lunar_signature") or contract.get("central_body") == "moon":
        metadata.setdefault("central_body", "moon")
        if contract.get("resolved_mu_si") is not None:
            metadata.setdefault("resolved_mu_si", contract["resolved_mu_si"])
        if contract.get("resolved_r_ref_m") is not None:
            metadata.setdefault("resolved_r_ref_m", contract["resolved_r_ref_m"])
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
        metadata=metadata,
    )


def write_dataset_metadata(path: str | Path, metadata: dict) -> Path:
    output = Path(path)
    if output.suffix.lower() == ".csv":
        output = output.with_suffix(output.suffix + ".metadata.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


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
    """Legacy wrapper; synthetic generation lives in experimental_vesp.synthetic."""

    from .synthetic import make_synthetic_dataset as _make

    return _make(
        n_query=n_query,
        n_truth_sources=n_truth_sources,
        query_radius_min=query_radius_min,
        query_radius_max=query_radius_max,
        truth_shell_radius=truth_shell_radius,
        truth_shell_radii=truth_shell_radii,
        noise_std=noise_std,
        seed=seed,
        dtype=dtype,
    )
