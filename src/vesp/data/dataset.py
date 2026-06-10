"""Dataset and synthetic-data helpers for VESP experiments."""

from __future__ import annotations

import csv
import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from vesp.common.lunar import validate_lunar_metadata_contract
from vesp.common.units import PositionScaler, UnitConfig

PHYSICAL_POSITION_UNITS = {"km", "m"}
POSITION_UNIT_ALIASES = {
    "normalized": "normalized",
    "normalised": "normalized",
    "dimensionless": "normalized",
    "model": "normalized",
    "km": "km",
    "kilometer": "km",
    "kilometers": "km",
    "kilometre": "km",
    "kilometres": "km",
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
}


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


def canonical_position_units(units: str | None) -> str:
    if units is None:
        raise ValueError("CSV metadata must include position_units")
    key = str(units).strip().lower()
    if key not in POSITION_UNIT_ALIASES:
        raise ValueError("position_units must be one of: normalized, km, m")
    return POSITION_UNIT_ALIASES[key]


def _distance_factor(from_units: str, to_units: str) -> float:
    from_units = canonical_position_units(from_units)
    to_units = canonical_position_units(to_units)
    if from_units == to_units:
        return 1.0
    if from_units == "m" and to_units == "km":
        return 1.0e-3
    if from_units == "km" and to_units == "m":
        return 1.0e3
    raise ValueError(f"cannot convert distance from {from_units!r} to {to_units!r}")


def _metadata_sidecar_candidates(path: Path) -> list[Path]:
    return [path.with_suffix(path.suffix + ".metadata.json"), path.with_suffix(".metadata.json")]


def _read_csv_metadata(path: Path, *, require_metadata: bool) -> dict:
    metadata_path = next((candidate for candidate in _metadata_sidecar_candidates(path) if candidate.exists()), None)
    if metadata_path is None:
        expected = " or ".join(str(candidate) for candidate in _metadata_sidecar_candidates(path))
        if require_metadata:
            raise ValueError(
                f"CSV metadata sidecar is required for unit-safe loading: expected {expected}"
            )
        warnings.warn(
            "CSV metadata sidecar is missing; assuming normalized positions only because "
            "require_metadata=False was explicitly requested.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {"path": str(path), "position_units": "normalized", "metadata_missing": True}
    metadata = {"path": str(path), "metadata_path": str(metadata_path)}
    metadata.update(json.loads(metadata_path.read_text(encoding="utf-8")))
    return metadata


def _metadata_body_radius(metadata: dict) -> tuple[float, str] | None:
    if "R_body" in metadata:
        return float(metadata["R_body"]), canonical_position_units(str(metadata.get("R_body_units", "km")))
    if "reference_radius_km" in metadata:
        return float(metadata["reference_radius_km"]), "km"
    if "r_ref_km" in metadata:
        return float(metadata["r_ref_km"]), "km"
    if "r_ref_m" in metadata:
        return float(metadata["r_ref_m"]), "m"
    if "resolved_r_ref_m" in metadata:
        return float(metadata["resolved_r_ref_m"]), "m"
    return None


def _config_body_radius(units: UnitConfig) -> tuple[float, str] | None:
    if units.physical_R_body is not None:
        return float(units.physical_R_body), canonical_position_units(units.physical_R_body_units)
    if units.R_body > 0.0 and (not units.normalize_positions or units.R_body != 1.0):
        unit_name = canonical_position_units(units.position_units)
        if unit_name in PHYSICAL_POSITION_UNITS:
            return float(units.R_body), unit_name
    return None


def _resolve_body_radius(metadata: dict, units: UnitConfig, target_units: str) -> float:
    target_units = canonical_position_units(target_units)
    if target_units not in PHYSICAL_POSITION_UNITS:
        raise ValueError("body radius conversion target must be physical units")
    radius = _metadata_body_radius(metadata) or _config_body_radius(units)
    if radius is None:
        raise ValueError(
            "metadata must include R_body/R_body_units (or r_ref_m/reference_radius_km) "
            "to convert physical and normalized positions"
        )
    value, radius_units = radius
    return value * _distance_factor(radius_units, target_units)


@dataclass
class ResidualGravityData:
    positions: torch.Tensor
    potential: torch.Tensor
    acceleration: torch.Tensor
    metadata: dict | None = None

    def to(self, device: torch.device | str) -> ResidualGravityData:
        return ResidualGravityData(
            positions=self.positions.to(device),
            potential=self.potential.to(device),
            acceleration=self.acceleration.to(device),
            metadata=self.metadata,
        )

    def subset(self, indices: torch.Tensor) -> ResidualGravityData:
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
        position_units = str(metadata.get("position_units", metadata.get("model_position_units", "normalized")))
        units = UnitConfig(
            R_body=float(metadata.get("R_body", 1.0)),
            normalize_positions=canonical_position_units(position_units) == "normalized",
            position_units=position_units,
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


def _canonical_acceleration_kind(metadata: dict) -> str:
    """Classify the CSV acceleration semantics.

    Returns ``"physical"`` (an acceleration ``dU/d(distance)``), ``"normalized_gradient"``
    (``dU/d(x / R_body)``), or ``"model"`` when the dataset does not declare an
    acceleration convention (no reconciliation is then attempted). The model itself
    always predicts ``dU/d(model coordinate)``.
    """

    output = str(metadata.get("acceleration_output", "")).strip().lower()
    if output in {"physical", "normalized_gradient"}:
        return output
    units_str = str(metadata.get("acceleration_units", "model")).strip().lower()
    if "normalized" in units_str:  # e.g. "km^2/s^2 per normalized radius"
        return "normalized_gradient"
    if "/s^2" in units_str or "/s2" in units_str or "s^-2" in units_str:
        return "physical"
    return "model"


def _physical_acceleration_unit_km(metadata: dict) -> float:
    """Length (in km) of the distance unit a physical acceleration differentiates by."""

    units_str = str(metadata.get("acceleration_units", "")).strip().lower().replace(" ", "")
    if units_str in {"m/s^2", "m/s2", "ms^-2"}:
        return 1.0e-3
    return 1.0  # km/s^2 is the default physical acceleration unit


def prepare_data_for_model(data: ResidualGravityData, units: UnitConfig) -> ResidualGravityData:
    """Convert CSV positions and acceleration into the model coordinate system.

    The input dataset must carry explicit ``position_units`` metadata. Physical
    position data additionally needs a reference body radius whenever model
    positions are normalized.

    The model always predicts ``dU/d(model coordinate)``. When the CSV stored a
    physical acceleration (``dU/d(distance)``) while the model works in normalized
    coordinates -- or vice versa -- the acceleration target is rescaled by the body
    radius. Without this the joint potential+acceleration solve is internally
    inconsistent by a factor of ``R_body`` and silently abandons the potential fit.
    """

    metadata = dict(data.metadata or {})
    if "position_units" not in metadata:
        raise ValueError("CSV metadata must include position_units")

    original_units = canonical_position_units(str(metadata["position_units"]))
    config_position_units = canonical_position_units(units.position_units)
    if units.normalize_positions:
        model_units = "normalized"
    else:
        if config_position_units == "normalized":
            raise ValueError("body.normalize_positions=false requires body.position_units to be 'km' or 'm'")
        model_units = config_position_units

    positions = data.positions
    model_r_body = 1.0
    physical_r_body: float | None = None

    if original_units == "normalized":
        if units.normalize_positions:
            model_positions = positions.clone()
        else:
            physical_r_body = _resolve_body_radius(metadata, units, model_units)
            model_r_body = physical_r_body
            model_positions = positions * physical_r_body
    else:
        if units.normalize_positions:
            physical_r_body = _resolve_body_radius(metadata, units, original_units)
            model_positions = positions / physical_r_body
        else:
            factor = _distance_factor(original_units, model_units)
            model_positions = positions * factor
            physical_r_body = _resolve_body_radius(metadata, units, model_units)
            model_r_body = physical_r_body

    # Reconcile acceleration units with the model coordinate system.
    acceleration = data.acceleration
    csv_acceleration_kind = _canonical_acceleration_kind(metadata)
    model_acceleration_kind = "normalized_gradient" if units.normalize_positions else "physical"
    acceleration_conversion_factor = 1.0
    if csv_acceleration_kind in ("physical", "normalized_gradient") and csv_acceleration_kind != model_acceleration_kind:
        r_body_km = _resolve_body_radius(metadata, units, "km")
        if units.normalize_positions:
            model_unit_km = r_body_km
        else:
            model_unit_km = 1.0 if config_position_units == "km" else 1.0e-3
        csv_unit_km = r_body_km if csv_acceleration_kind == "normalized_gradient" else _physical_acceleration_unit_km(metadata)
        acceleration_conversion_factor = model_unit_km / csv_unit_km
        acceleration = acceleration * acceleration_conversion_factor

    prepared_metadata = dict(metadata)
    prepared_metadata["original_position_units"] = original_units
    prepared_metadata["model_position_units"] = model_units
    prepared_metadata["position_units"] = model_units
    prepared_metadata["position_prepared_for_model"] = True
    prepared_metadata["model_R_body"] = model_r_body
    prepared_metadata["config_R_body"] = float(units.R_body)
    if physical_r_body is not None:
        prepared_metadata["physical_R_body_in_model_conversion_units"] = physical_r_body
    prepared_metadata["original_acceleration_units"] = metadata.get("acceleration_units", "model")
    prepared_metadata["csv_acceleration_kind"] = csv_acceleration_kind
    prepared_metadata["model_acceleration_kind"] = model_acceleration_kind
    prepared_metadata["acceleration_conversion_factor"] = acceleration_conversion_factor
    prepared_metadata["acceleration_prepared_for_model"] = True

    return ResidualGravityData(
        positions=model_positions,
        potential=data.potential,
        acceleration=acceleration,
        metadata=prepared_metadata,
    )


def load_csv_dataset(
    path: str | Path,
    *,
    dtype: torch.dtype = torch.float32,
    unit_config: UnitConfig | None = None,
    require_metadata: bool = True,
) -> ResidualGravityData:
    path = Path(path)
    metadata = _read_csv_metadata(path, require_metadata=require_metadata)
    canonical_position_units(metadata.get("position_units"))
    contract = validate_lunar_metadata_contract(metadata, data_path=path, require_lunar=False)
    if contract.get("has_lunar_signature") or contract.get("central_body") == "moon":
        metadata.setdefault("central_body", "moon")
        if contract.get("resolved_mu_si") is not None:
            metadata.setdefault("resolved_mu_si", contract["resolved_mu_si"])
        if contract.get("resolved_r_ref_m") is not None:
            metadata.setdefault("resolved_r_ref_m", contract["resolved_r_ref_m"])
    with open(path, encoding="utf-8-sig", newline="") as f:
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

    data = ResidualGravityData(
        positions=torch.tensor(x_rows, dtype=dtype),
        potential=torch.tensor(u_rows, dtype=dtype),
        acceleration=torch.tensor(a_rows, dtype=dtype),
        metadata=metadata,
    )
    if unit_config is not None:
        return prepare_data_for_model(data, unit_config)
    return data


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
    """Legacy wrapper; synthetic generation lives in vesp.data.synthetic."""

    from vesp.data.synthetic import make_synthetic_dataset as _make

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
