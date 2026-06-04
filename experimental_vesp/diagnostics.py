"""Diagnostics for source distributions and fitted fields."""

from __future__ import annotations

import time

import torch

from .losses import moment_losses, shell_energy


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


def source_diagnostics(
    *,
    source_positions: torch.Tensor,
    source_weights: torch.Tensor,
    shell_ids: torch.Tensor,
    sigma: torch.Tensor,
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
    return {
        "sigma_l2": float(torch.linalg.norm(sigma).detach().cpu()),
        "weighted_sigma_l2": float(torch.linalg.norm(weighted_sigma).detach().cpu()),
        "sigma_abs_max": float(torch.max(torch.abs(sigma)).detach().cpu()),
        "effective_source_count": effective_source_count(weighted_sigma),
        "top_1pct_source_contribution": topk_source_contribution(weighted_sigma, 0.01),
        "top_5pct_source_contribution": topk_source_contribution(weighted_sigma, 0.05),
        "monopole_leakage": float(torch.sqrt(moments["monopole"]).detach().cpu()),
        "dipole_leakage": float(torch.sqrt(moments["dipole"]).detach().cpu()),
        "shell_energy": [float(v.detach().cpu()) for v in energies],
        "shell_energy_distribution": shell_rows,
    }


def time_inference(model, x: torch.Tensor, *, source_chunk_size: int | None, repeats: int = 5) -> float:
    if x.shape[0] == 0:
        return 0.0

    with torch.no_grad():
        model(x, source_chunk_size=source_chunk_size)
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(x, source_chunk_size=source_chunk_size)
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
    return (end - start) / repeats
