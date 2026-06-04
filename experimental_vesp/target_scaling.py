"""Target scaling utilities for mixed potential/acceleration fitting."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .artifacts import atomic_write_json
from .data import ResidualGravityData


TARGET_SCALES_SCHEMA_VERSION = "vesp_target_scales_v1"


@dataclass(frozen=True)
class TargetScales:
    normalize_targets: bool
    potential_scale: float = 1.0
    acceleration_scale: float = 1.0
    potential_source: str = "disabled"
    acceleration_source: str = "disabled"
    eps: float = 1.0e-12

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = TARGET_SCALES_SCHEMA_VERSION
        return payload


def potential_rms(potential: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(potential.reshape(-1) ** 2)).detach().cpu())


def acceleration_vector_rms(acceleration: torch.Tensor) -> float:
    squared_norm = torch.sum(acceleration * acceleration, dim=-1)
    return float(torch.sqrt(torch.mean(squared_norm)).detach().cpu())


def _clamp_scale(value: float, eps: float) -> float:
    if not math.isfinite(value):
        return eps
    return max(abs(float(value)), eps)


def _resolve_scale(value: Any, auto_value: float, *, eps: float) -> tuple[float, str]:
    if value is None:
        value = "auto"
    if isinstance(value, str) and value.strip().lower() == "auto":
        return _clamp_scale(auto_value, eps), "auto_rms"
    return _clamp_scale(float(value), eps), "configured"


def compute_target_scales(data: ResidualGravityData, config: dict) -> TargetScales:
    loss_cfg = config.get("loss", config)
    normalize_targets = bool(loss_cfg.get("normalize_targets", False))
    eps = float(loss_cfg.get("target_scale_eps", 1.0e-12))
    if not normalize_targets:
        return TargetScales(normalize_targets=False, eps=eps)

    use_potential = bool(loss_cfg.get("use_potential", True))
    use_acceleration = bool(loss_cfg.get("use_acceleration", True))
    potential_scale = 1.0
    acceleration_scale = 1.0
    potential_source = "unused"
    acceleration_source = "unused"

    if use_potential:
        potential_scale, potential_source = _resolve_scale(
            loss_cfg.get("potential_scale", "auto"),
            potential_rms(data.potential),
            eps=eps,
        )
    if use_acceleration:
        acceleration_scale, acceleration_source = _resolve_scale(
            loss_cfg.get("acceleration_scale", "auto"),
            acceleration_vector_rms(data.acceleration),
            eps=eps,
        )
    return TargetScales(
        normalize_targets=True,
        potential_scale=potential_scale,
        acceleration_scale=acceleration_scale,
        potential_source=potential_source,
        acceleration_source=acceleration_source,
        eps=eps,
    )


def apply_target_scales_to_config(config: dict, scales: TargetScales) -> None:
    loss_cfg = config.setdefault("loss", {})
    loss_cfg["resolved_potential_scale"] = scales.potential_scale
    loss_cfg["resolved_acceleration_scale"] = scales.acceleration_scale
    loss_cfg["resolved_normalize_targets"] = scales.normalize_targets


def observation_row_weights(
    *,
    n_query: int,
    include_potential: bool,
    include_acceleration: bool,
    lambda_potential: float,
    lambda_acceleration: float,
    scales: TargetScales,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    blocks: list[torch.Tensor] = []
    if include_potential:
        factor = math.sqrt(float(lambda_potential))
        if scales.normalize_targets:
            factor /= scales.potential_scale
        blocks.append(torch.full((n_query,), factor, dtype=dtype, device=device))
    if include_acceleration:
        factor = math.sqrt(float(lambda_acceleration))
        if scales.normalize_targets:
            factor /= scales.acceleration_scale
        blocks.extend([torch.full((n_query,), factor, dtype=dtype, device=device) for _ in range(3)])
    if not blocks:
        raise ValueError("at least one target block is required for row weighting")
    return torch.cat(blocks, dim=0)


def write_target_scales(path: str | Path, scales: TargetScales) -> Path:
    output = Path(path)
    atomic_write_json(output, scales.as_dict())
    return output

