"""Research metrics for VESP feasibility decisions."""

from __future__ import annotations

import warnings

import torch


def rmse_potential(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred.reshape_as(target) - target) ** 2)).detach().cpu())


def rmse_acceleration(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2)).detach().cpu())


def rmse_acceleration_components(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = pred - target
    return {
        "acc_x_rmse": float(torch.sqrt(torch.mean(error[:, 0] ** 2)).detach().cpu()),
        "acc_y_rmse": float(torch.sqrt(torch.mean(error[:, 1] ** 2)).detach().cpu()),
        "acc_z_rmse": float(torch.sqrt(torch.mean(error[:, 2] ** 2)).detach().cpu()),
    }


def rmse_acceleration_norm(pred: torch.Tensor, target: torch.Tensor) -> float:
    norm_error = torch.linalg.norm(pred - target, dim=-1)
    return float(torch.sqrt(torch.mean(norm_error * norm_error)).detach().cpu())


def relative_rmse_acceleration(pred: torch.Tensor, target: torch.Tensor) -> float:
    num = torch.sqrt(torch.mean((pred - target) ** 2))
    den = torch.sqrt(torch.mean(target ** 2))
    return float((num / torch.clamp(den, min=torch.finfo(target.dtype).eps)).detach().cpu())


def radial_cross_radial_error(
    positions: torch.Tensor,
    pred_acceleration: torch.Tensor,
    target_acceleration: torch.Tensor,
) -> dict[str, float]:
    radial = positions / torch.clamp(torch.linalg.norm(positions, dim=-1, keepdim=True), min=torch.finfo(positions.dtype).eps)
    error = pred_acceleration - target_acceleration
    radial_scalar = torch.sum(error * radial, dim=-1)
    radial_error = radial_scalar.unsqueeze(-1) * radial
    cross_error = error - radial_error
    return {
        "radial_scalar_rmse": float(torch.sqrt(torch.mean(radial_scalar * radial_scalar)).detach().cpu()),
        "cross_norm_rmse": float(torch.sqrt(torch.mean(torch.linalg.norm(cross_error, dim=-1) ** 2)).detach().cpu()),
        "total_acceleration_norm_rmse": rmse_acceleration_norm(pred_acceleration, target_acceleration),
    }


def vector_angle_error(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_norm = torch.linalg.norm(pred, dim=-1)
    target_norm = torch.linalg.norm(target, dim=-1)
    denom = torch.clamp(pred_norm * target_norm, min=torch.finfo(pred.dtype).eps)
    cosang = torch.clamp(torch.sum(pred * target, dim=-1) / denom, min=-1.0, max=1.0)
    deg = torch.rad2deg(torch.arccos(cosang))
    return {
        "angle_deg_mean": float(torch.mean(deg).detach().cpu()),
        "angle_deg_p95": float(torch.quantile(deg, 0.95).detach().cpu()),
    }


DEFAULT_ALTITUDE_BANDS: dict[str, list[float]] = {
    "low": [1.03, 1.15],
    "mid": [1.15, 1.35],
    "high": [1.35, 1.60],
}


def altitude_band_errors(
    positions: torch.Tensor,
    pred_acceleration: torch.Tensor,
    target_acceleration: torch.Tensor,
    *,
    bands: dict[str, list[float]] | None = None,
    warn_empty: bool = True,
) -> dict[str, float | int | None]:
    """Acceleration RMSE inside named radial bands (low/mid/high) + ratio.

    Real lunar residual fits are dominated by low-altitude error, so we report
    each band separately instead of only a single global RMSE. A band with no
    samples yields ``None`` (and a warning) rather than a misleading number.
    """

    # ``None`` means "use the default bands"; an explicit empty dict means "no bands"
    # (used for single-band OOD subsets to avoid spurious empty-band warnings).
    bands = DEFAULT_ALTITUDE_BANDS if bands is None else bands
    radii = torch.linalg.norm(positions, dim=-1)
    result: dict[str, float | int | None] = {}
    band_rmse: dict[str, float | None] = {}
    for name, band in bands.items():
        if band is None:
            band_rmse[name] = None
            result[f"{name}_altitude_acceleration_rmse"] = None
            result[f"{name}_altitude_count"] = 0
            continue
        lo, hi = float(band[0]), float(band[1])
        mask = (radii >= lo) & (radii <= hi)
        count = int(mask.sum().detach().cpu())
        result[f"{name}_altitude_count"] = count
        if count == 0:
            band_rmse[name] = None
            result[f"{name}_altitude_acceleration_rmse"] = None
            if warn_empty:
                warnings.warn(
                    f"altitude band '{name}'={band} contains no evaluation samples; reporting null",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue
        rmse = rmse_acceleration(pred_acceleration[mask], target_acceleration[mask])
        band_rmse[name] = rmse
        result[f"{name}_altitude_acceleration_rmse"] = rmse

    low = band_rmse.get("low")
    high = band_rmse.get("high")
    if low is not None and high is not None and high > 0.0:
        result["low_to_high_error_ratio"] = float(low / high)
    else:
        result["low_to_high_error_ratio"] = None
    return result


def altitude_binned_error(
    positions: torch.Tensor,
    pred_acceleration: torch.Tensor,
    target_acceleration: torch.Tensor,
    *,
    n_bins: int = 6,
) -> list[dict[str, float]]:
    radii = torch.linalg.norm(positions, dim=-1)
    bins = torch.linspace(float(radii.min()), float(radii.max()), n_bins + 1, device=positions.device)
    rows = []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (radii >= bins[i]) & (radii <= bins[i + 1])
        else:
            mask = (radii >= bins[i]) & (radii < bins[i + 1])
        if torch.any(mask):
            rows.append(
                {
                    "r_min": float(bins[i].detach().cpu()),
                    "r_max": float(bins[i + 1].detach().cpu()),
                    "count": int(mask.sum().detach().cpu()),
                    "acceleration_rmse": rmse_acceleration(pred_acceleration[mask], target_acceleration[mask]),
                }
            )
    return rows
