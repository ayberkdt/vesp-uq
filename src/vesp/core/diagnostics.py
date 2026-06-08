"""Diagnostics for source distributions and fitted fields."""

from __future__ import annotations

import time

import torch

from vesp.core.losses import moment_losses, shell_energy


def effective_source_count(weighted_sigma: torch.Tensor) -> float:
    mass = torch.abs(weighted_sigma)
    total = torch.sum(mass)
    if float(total.detach().cpu()) <= 0.0:
        return 0.0
    p = mass / total
    entropy = -torch.sum(p * torch.log(torch.clamp(p, min=torch.finfo(p.dtype).tiny)))
    return float(torch.exp(entropy).detach().cpu())


def topk_source_contribution(weighted_sigma: torch.Tensor, fraction: float = 0.05) -> float:
    mass = torch.abs(weighted_sigma)
    total = torch.sum(mass)
    if float(total.detach().cpu()) <= 0.0:
        return 0.0
    k = max(1, int(round(float(fraction) * mass.numel())))
    return float(torch.sum(torch.sort(mass, descending=True).values[:k] / total).detach().cpu())


def condition_number_estimate(operator: torch.Tensor | None = None) -> float | None:
    if operator is None:
        return None
    try:
        return float(torch.linalg.cond(operator).detach().cpu())
    except RuntimeError:
        return None


def shell_energy_collapse(
    energies: torch.Tensor,
    shell_rows: list[dict],
    *,
    threshold: float = 0.90,
) -> dict[str, float | int | bool | None]:
    """Summarize how concentrated shell energy is across shells.

    Used to catch the multi-shell failure mode where almost all energy collapses
    onto a single (usually innermost) shell.
    """

    if energies.numel() == 0:
        return {
            "dominant_shell_id": None,
            "dominant_shell_alpha": None,
            "dominant_shell_energy_fraction": None,
            "shell_energy_entropy": None,
            "shell_energy_effective_count": None,
            "shell_collapse_flag": False,
        }
    total = torch.clamp(torch.sum(energies), min=torch.finfo(energies.dtype).eps)
    p = energies / total
    eps = torch.finfo(energies.dtype).tiny
    entropy = float(-torch.sum(p * torch.log(p + eps)).detach().cpu())
    dominant_id = int(torch.argmax(p).detach().cpu())
    dominant_fraction = float(p[dominant_id].detach().cpu())
    n_shells = int(energies.numel())
    return {
        "dominant_shell_id": dominant_id,
        "dominant_shell_alpha": shell_rows[dominant_id]["shell_alpha"] if dominant_id < len(shell_rows) else None,
        "dominant_shell_energy_fraction": dominant_fraction,
        "shell_energy_entropy": entropy,
        "shell_energy_effective_count": float(torch.exp(torch.tensor(entropy)).item()),
        # A single-shell model trivially has fraction 1.0; collapse only applies to
        # genuine multi-shell models.
        "shell_collapse_flag": bool(n_shells > 1 and dominant_fraction > float(threshold)),
    }


def source_diagnostics(
    *,
    source_positions: torch.Tensor,
    source_weights: torch.Tensor,
    shell_ids: torch.Tensor,
    sigma: torch.Tensor,
    shell_collapse_threshold: float = 0.90,
    sigma_l2_warning_threshold: float = 1.0,
) -> dict[str, float | list[float]]:
    moments = moment_losses(source_positions, source_weights, sigma)
    energies = shell_energy(sigma, source_weights, shell_ids)
    weighted_sigma = source_weights * sigma
    total_energy = torch.sum(energies)
    shell_rows = []
    for shell_id, energy in enumerate(energies):
        mask = shell_ids == shell_id
        shell_rows.append(
            {
                "shell_id": shell_id,
                "shell_alpha": float(torch.mean(torch.linalg.norm(source_positions[mask], dim=-1)).detach().cpu()) if torch.any(mask) else None,
                "n_source": int(mask.sum().detach().cpu()),
                "energy": float(energy.detach().cpu()),
                "energy_fraction": float((energy / torch.clamp(total_energy, min=torch.finfo(sigma.dtype).eps)).detach().cpu()),
                "sigma_norm": float(torch.linalg.norm(sigma[mask]).detach().cpu()),
            }
        )
    sigma_l2 = float(torch.linalg.norm(sigma).detach().cpu())
    collapse = shell_energy_collapse(energies, shell_rows, threshold=shell_collapse_threshold)

    # Absolute moment leakage scales with both the source magnitude and the
    # coordinate convention, so it is reported for continuity but is unsuitable as a
    # screening threshold. The relative versions normalize by the total absolute
    # source mass (and, for the dipole, the mean source radius), giving dimensionless
    # quantities that are invariant to field magnitude and unit choice.
    monopole_leakage = float(torch.sqrt(moments["monopole"]).detach().cpu())
    dipole_leakage = float(torch.sqrt(moments["dipole"]).detach().cpu())
    total_abs_source_mass = float(torch.sum(torch.abs(weighted_sigma)).detach().cpu())
    mean_source_radius = (
        float(torch.mean(torch.linalg.norm(source_positions, dim=-1)).detach().cpu())
        if source_positions.numel()
        else 0.0
    )
    relative_monopole_leakage = monopole_leakage / total_abs_source_mass if total_abs_source_mass > 0.0 else 0.0
    relative_dipole_leakage = (
        dipole_leakage / (total_abs_source_mass * mean_source_radius)
        if total_abs_source_mass > 0.0 and mean_source_radius > 0.0
        else 0.0
    )
    return {
        "sigma_l2": sigma_l2,
        "sigma_linf": float(torch.max(torch.abs(sigma)).detach().cpu()),
        "sigma_rms": float(torch.sqrt(torch.mean(sigma * sigma)).detach().cpu()),
        "sigma_abs_sum": float(torch.sum(torch.abs(sigma)).detach().cpu()),
        "sigma_norm_warning": bool(sigma_l2 > float(sigma_l2_warning_threshold)),
        "weighted_sigma_l2": float(torch.linalg.norm(weighted_sigma).detach().cpu()),
        "sigma_abs_max": float(torch.max(torch.abs(sigma)).detach().cpu()),
        "effective_source_count": effective_source_count(weighted_sigma),
        "top_1pct_source_contribution": topk_source_contribution(weighted_sigma, 0.01),
        "top_5pct_source_contribution": topk_source_contribution(weighted_sigma, 0.05),
        "monopole_leakage": monopole_leakage,
        "dipole_leakage": dipole_leakage,
        "total_abs_source_mass": total_abs_source_mass,
        "mean_source_radius": mean_source_radius,
        "relative_monopole_leakage": relative_monopole_leakage,
        "relative_dipole_leakage": relative_dipole_leakage,
        "shell_energy": [float(v.detach().cpu()) for v in energies],
        "shell_energy_distribution": shell_rows,
        **collapse,
    }


def time_inference(
    model,
    x: torch.Tensor,
    *,
    source_chunk_size: int | None,
    softening: float = 0.0,
    acceleration_sign: float = 1.0,
    repeats: int = 5,
) -> float:
    if x.shape[0] == 0:
        return 0.0

    with torch.no_grad():
        model(x, source_chunk_size=source_chunk_size, softening=softening, acceleration_sign=acceleration_sign)
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(x, source_chunk_size=source_chunk_size, softening=softening, acceleration_sign=acceleration_sign)
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
    return (end - start) / repeats
