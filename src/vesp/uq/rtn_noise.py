"""Prototype diagonal RTN covariance calibration for VESP-UQ.

The production VESP-UQ covariance is physics-driven by the equivalent-source posterior plus a
scalar altitude-aware aleatoric floor. This module implements the deliberately smaller next step
suggested by the gate diagnostics: fit a held-out, diagonal local-frame variance scaling and report
whether it improves radial/tangential calibration without creating over-confidence.

The pointwise calibration data used here contains positions but no velocity, so the local frame is
radial plus two pooled tangential directions rather than a true trajectory RTN frame. That makes the
prototype appropriate for a before/after calibration gate, not for final trajectory-frame claims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from vesp.uq.metrics import component_calibration_metrics, local_radial_frame, vector_calibration_metrics

_NORMAL_HALF_WIDTH_90 = 1.6448536269514722


@dataclass(frozen=True)
class RTNVarianceScale:
    """One radial + pooled-tangential variance scale for a radius interval."""

    name: str
    lo: float
    hi: float
    radial_scale: float
    tangential_scale: float
    n_points: int
    radial_z_std_fit: float
    tangential_z_std_fit: float


@dataclass(frozen=True)
class RTNVarianceScaleModel:
    """Piecewise radius-binned diagonal local-frame variance scaling."""

    global_scale: RTNVarianceScale
    bands: tuple[RTNVarianceScale, ...] = ()
    allow_shrink: bool = True
    min_scale: float = 0.25
    max_scale: float = 4.0
    target_z_std: float = 1.0

    def scale_for_radius(self, radius: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(radial_scale, tangential_scale)`` per query radius."""

        r = torch.as_tensor(radius, dtype=torch.float64)
        radial = torch.full_like(r, float(self.global_scale.radial_scale))
        tangential = torch.full_like(r, float(self.global_scale.tangential_scale))
        for band in self.bands:
            mask = (r >= float(band.lo)) & (r <= float(band.hi))
            radial = torch.where(mask, torch.as_tensor(band.radial_scale, dtype=r.dtype, device=r.device), radial)
            tangential = torch.where(
                mask, torch.as_tensor(band.tangential_scale, dtype=r.dtype, device=r.device), tangential
            )
        return radial, tangential

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": "position-only radial + pooled tangential",
            "allow_shrink": self.allow_shrink,
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
            "target_z_std": self.target_z_std,
            "global": self.global_scale.__dict__,
            "bands": [band.__dict__ for band in self.bands],
        }


def local_frame_residual_and_variance(
    positions: torch.Tensor,
    residuals: torch.Tensor,
    covariances: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate residual vectors and covariance diagonals into the local radial/tangential frame."""

    if positions.shape != residuals.shape or positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError("positions and residuals must both have shape (N, 3)")
    if covariances.shape != (positions.shape[0], 3, 3):
        raise ValueError("covariances must have shape (N, 3, 3)")
    pos = positions.to(torch.float64)
    res = residuals.to(torch.float64)
    cov = covariances.to(torch.float64)
    frame = local_radial_frame(pos)
    local_res = torch.einsum("nij,nj->ni", frame, res)
    local_cov = torch.einsum("nij,njk,nlk->nil", frame, cov, frame)
    local_var = torch.diagonal(local_cov, dim1=-2, dim2=-1).clamp_min(0.0)
    return local_res, local_var


def _z_std(errors: torch.Tensor, variances: torch.Tensor) -> float:
    std = torch.sqrt(variances.to(torch.float64).clamp_min(torch.finfo(torch.float64).tiny))
    z = errors.to(torch.float64).reshape(-1) / std.reshape(-1)
    z = z[torch.isfinite(z)]
    if z.numel() < 2:
        return float("nan")
    return float(torch.std(z, unbiased=False).detach().cpu())


def _scale_from_z(
    z_std: float,
    *,
    target_z_std: float,
    allow_shrink: bool,
    min_scale: float,
    max_scale: float,
) -> float:
    if not math.isfinite(z_std) or z_std <= 0.0:
        return 1.0
    scale = (z_std / max(float(target_z_std), 1.0e-12)) ** 2
    if not allow_shrink:
        scale = max(1.0, scale)
    return min(max(float(scale), float(min_scale)), float(max_scale))


def _fit_one_scale(
    name: str,
    lo: float,
    hi: float,
    positions: torch.Tensor,
    residuals: torch.Tensor,
    covariances: torch.Tensor,
    *,
    target_z_std: float,
    allow_shrink: bool,
    min_scale: float,
    max_scale: float,
) -> RTNVarianceScale:
    local_res, local_var = local_frame_residual_and_variance(positions, residuals, covariances)
    radial_z = _z_std(local_res[:, 0], local_var[:, 0])
    tangential_z = _z_std(local_res[:, 1:], local_var[:, 1:])
    return RTNVarianceScale(
        name=str(name),
        lo=float(lo),
        hi=float(hi),
        radial_scale=_scale_from_z(
            radial_z,
            target_z_std=target_z_std,
            allow_shrink=allow_shrink,
            min_scale=min_scale,
            max_scale=max_scale,
        ),
        tangential_scale=_scale_from_z(
            tangential_z,
            target_z_std=target_z_std,
            allow_shrink=allow_shrink,
            min_scale=min_scale,
            max_scale=max_scale,
        ),
        n_points=int(positions.shape[0]),
        radial_z_std_fit=radial_z,
        tangential_z_std_fit=tangential_z,
    )


def fit_rtn_variance_scale_model(
    positions: torch.Tensor,
    residuals: torch.Tensor,
    covariances: torch.Tensor,
    *,
    altitude_bands: dict[str, list[float]] | None = None,
    min_band_points: int = 30,
    target_z_std: float = 1.0,
    allow_shrink: bool = True,
    min_scale: float = 0.25,
    max_scale: float = 4.0,
) -> RTNVarianceScaleModel:
    """Fit global and optional radius-binned radial/tangential variance scales."""

    if positions.shape[0] < 2:
        raise ValueError("at least two calibration points are required")
    radius = torch.linalg.norm(positions.to(torch.float64), dim=-1)
    global_scale = _fit_one_scale(
        "all",
        float(radius.min().detach().cpu()),
        float(radius.max().detach().cpu()),
        positions,
        residuals,
        covariances,
        target_z_std=target_z_std,
        allow_shrink=allow_shrink,
        min_scale=min_scale,
        max_scale=max_scale,
    )

    bands: list[RTNVarianceScale] = []
    for name, rng in (altitude_bands or {}).items():
        if rng is None or len(rng) != 2:
            continue
        lo, hi = float(rng[0]), float(rng[1])
        mask = (radius >= lo) & (radius <= hi)
        if int(mask.sum()) < int(min_band_points):
            continue
        bands.append(
            _fit_one_scale(
                name,
                lo,
                hi,
                positions[mask],
                residuals[mask],
                covariances[mask],
                target_z_std=target_z_std,
                allow_shrink=allow_shrink,
                min_scale=min_scale,
                max_scale=max_scale,
            )
        )

    return RTNVarianceScaleModel(
        global_scale=global_scale,
        bands=tuple(bands),
        allow_shrink=bool(allow_shrink),
        min_scale=float(min_scale),
        max_scale=float(max_scale),
        target_z_std=float(target_z_std),
    )


def apply_rtn_variance_scale(
    positions: torch.Tensor,
    covariances: torch.Tensor,
    model: RTNVarianceScaleModel,
) -> torch.Tensor:
    """Apply the fitted local-frame variance scaling and rotate back to world coordinates."""

    if covariances.shape != (positions.shape[0], 3, 3):
        raise ValueError("covariances must have shape (N, 3, 3)")
    pos = positions.to(torch.float64)
    cov = covariances.to(torch.float64)
    frame = local_radial_frame(pos)
    local_cov = torch.einsum("nij,njk,nlk->nil", frame, cov, frame)
    radial_scale, tangential_scale = model.scale_for_radius(torch.linalg.norm(pos, dim=-1))
    scales = torch.stack([radial_scale, tangential_scale, tangential_scale], dim=-1)
    scaled_local = local_cov * torch.sqrt(scales).unsqueeze(-1) * torch.sqrt(scales).unsqueeze(-2)
    world = torch.einsum("nai,nab,nbj->nij", frame, scaled_local, frame)
    return 0.5 * (world + world.transpose(-1, -2))


def rtn_calibration_summary(
    positions: torch.Tensor,
    residuals: torch.Tensor,
    covariances: torch.Tensor,
) -> dict[str, float]:
    """Compact radial/tangential plus vector calibration summary."""

    comp = component_calibration_metrics(residuals, covariances, positions)
    vec = vector_calibration_metrics(residuals, covariances)
    local_res, local_var = local_frame_residual_and_variance(positions, residuals, covariances)
    radial_picp = float(
        torch.mean(
            (local_res[:, 0].abs() <= _NORMAL_HALF_WIDTH_90 * torch.sqrt(local_var[:, 0].clamp_min(0.0))).to(
                torch.float64
            )
        ).detach().cpu()
    )
    tangential_picp = float(
        torch.mean(
            (
                local_res[:, 1:].abs()
                <= _NORMAL_HALF_WIDTH_90 * torch.sqrt(local_var[:, 1:].clamp_min(0.0))
            ).to(torch.float64)
        ).detach().cpu()
    )
    return {
        "n": int(positions.shape[0]),
        "radial_z_std": float(comp.get("radial_z_std", float("nan"))),
        "tangential_z_std": float(comp.get("tangential_z_std", float("nan"))),
        "radial_picp_90": radial_picp,
        "tangential_picp_90": tangential_picp,
        "ellipsoid_picp_90": float(vec.get("ellipsoid_picp_90", float("nan"))),
        "mean_mahalanobis_d2": float(vec.get("mean_mahalanobis_d2", float("nan"))),
    }


def calibration_error_score(summary: dict[str, float]) -> float:
    """Single scalar used only to compare before/after prototype calibration."""

    terms = [
        abs(float(summary["radial_z_std"]) - 1.0),
        abs(float(summary["tangential_z_std"]) - 1.0),
        abs(float(summary["ellipsoid_picp_90"]) - 0.90),
    ]
    return float(sum(terms) / len(terms))


def creates_overconfidence(
    summary: dict[str, float],
    *,
    max_z_std: float = 1.10,
    min_picp_90: float = 0.88,
) -> bool:
    """Conservative guardrail for deciding whether the prototype is admissible."""

    return (
        float(summary["radial_z_std"]) > float(max_z_std)
        or float(summary["tangential_z_std"]) > float(max_z_std)
        or float(summary["ellipsoid_picp_90"]) < float(min_picp_90)
        or float(summary["radial_picp_90"]) < float(min_picp_90)
        or float(summary["tangential_picp_90"]) < float(min_picp_90)
    )
