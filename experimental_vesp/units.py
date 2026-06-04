"""Unit and normalization handling for VESP experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class UnitConfig:
    R_body: float
    normalize_positions: bool = True
    position_units: str = "normalized"
    potential_units: str = "model"
    acceleration_units: str = "model"
    physical_R_body: float | None = None
    physical_R_body_units: str = "km"

    @classmethod
    def from_config(cls, config: dict) -> "UnitConfig":
        body = config.get("body", {})
        physical_r_body = body.get("physical_R_body", body.get("R_body_physical"))
        return cls(
            R_body=float(body.get("R_body", 1.0)),
            normalize_positions=bool(body.get("normalize_positions", True)),
            position_units=str(body.get("position_units", "normalized")),
            potential_units=str(body.get("potential_units", "model")),
            acceleration_units=str(body.get("acceleration_units", "model")),
            physical_R_body=float(physical_r_body) if physical_r_body is not None else None,
            physical_R_body_units=str(body.get("physical_R_body_units", body.get("R_body_units", "km"))),
        )

    @property
    def source_body_radius(self) -> float:
        return 1.0 if self.normalize_positions else self.R_body


class PositionScaler:
    def __init__(self, units: UnitConfig) -> None:
        self.units = units

    def to_model_positions(self, x: torch.Tensor) -> torch.Tensor:
        if self.units.normalize_positions:
            return x / float(self.units.R_body) if self.units.position_units != "normalized" else x
        return x

    def from_model_positions(self, x_model: torch.Tensor) -> torch.Tensor:
        if self.units.normalize_positions:
            return x_model * float(self.units.R_body) if self.units.position_units != "normalized" else x_model
        return x_model

    def radius_to_model_units(self, r: float | torch.Tensor) -> float | torch.Tensor:
        if self.units.normalize_positions and self.units.position_units != "normalized":
            return r / float(self.units.R_body)
        return r

    def altitude_from_model_positions(self, x_model: torch.Tensor) -> torch.Tensor:
        r = torch.linalg.norm(x_model, dim=-1)
        body_radius = 1.0 if self.units.normalize_positions else self.units.R_body
        return r - body_radius


def normalized_gradient_to_physical_acceleration(grad: torch.Tensor, R_ref: float) -> torch.Tensor:
    return grad / float(R_ref)


def physical_acceleration_to_normalized_gradient(acc: torch.Tensor, R_ref: float) -> torch.Tensor:
    return acc * float(R_ref)
