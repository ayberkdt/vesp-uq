"""Loss and regularization utilities for discrete VESP models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def potential_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred.reshape_as(target), target)


def acceleration_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def joint_data_loss(
    *,
    pred_potential: torch.Tensor | None,
    pred_acceleration: torch.Tensor | None,
    target_potential: torch.Tensor | None,
    target_acceleration: torch.Tensor | None,
    potential_weight: float = 1.0,
    acceleration_weight: float = 1.0,
) -> torch.Tensor:
    total = torch.zeros((), dtype=(pred_acceleration if pred_acceleration is not None else pred_potential).dtype)
    if pred_potential is not None and target_potential is not None and potential_weight:
        total = total.to(pred_potential.device) + float(potential_weight) * potential_mse(pred_potential, target_potential)
    if pred_acceleration is not None and target_acceleration is not None and acceleration_weight:
        total = total.to(pred_acceleration.device) + float(acceleration_weight) * acceleration_mse(pred_acceleration, target_acceleration)
    return total


def source_l2(sigma: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return torch.mean(sigma * sigma)
    return torch.sum(weights * sigma * sigma) / torch.clamp(torch.sum(weights), min=torch.finfo(sigma.dtype).eps)


source_l2_loss = source_l2


def moment_losses(
    source_positions: torch.Tensor,
    source_weights: torch.Tensor,
    sigma: torch.Tensor,
) -> dict[str, torch.Tensor]:
    weighted_sigma = source_weights * sigma
    monopole = torch.sum(weighted_sigma)
    dipole = torch.sum(weighted_sigma.unsqueeze(-1) * source_positions, dim=0)
    return {
        "monopole": monopole * monopole,
        "dipole": torch.sum(dipole * dipole),
    }


def moment_loss(
    source_positions: torch.Tensor,
    source_weights: torch.Tensor,
    sigma: torch.Tensor,
    *,
    dipole_weight: float = 1.0,
) -> torch.Tensor:
    losses = moment_losses(source_positions, source_weights, sigma)
    return losses["monopole"] + float(dipole_weight) * losses["dipole"]


def shell_energy(
    sigma: torch.Tensor,
    source_weights: torch.Tensor,
    shell_ids: torch.Tensor,
) -> torch.Tensor:
    n_shells = int(torch.max(shell_ids).item()) + 1 if shell_ids.numel() else 0
    energies = []
    for shell_id in range(n_shells):
        mask = shell_ids == shell_id
        energies.append(torch.sum(source_weights[mask] * sigma[mask] * sigma[mask]))
    return torch.stack(energies) if energies else torch.empty(0, dtype=sigma.dtype, device=sigma.device)


shell_energy_loss = shell_energy

# TODO Stage 3:
#     entropy_regularization
#     signed_entropy
#     relative_entropy
#     shellwise_entropy


def composite_loss(
    *,
    pred_potential: torch.Tensor | None,
    pred_acceleration: torch.Tensor | None,
    target_potential: torch.Tensor | None,
    target_acceleration: torch.Tensor | None,
    sigma: torch.Tensor,
    source_positions: torch.Tensor,
    source_weights: torch.Tensor,
    shell_ids: torch.Tensor,
    lambda_potential: float = 1.0,
    lambda_acceleration: float = 1.0,
    lambda_l2: float = 0.0,
    lambda_moment: float = 0.0,
    lambda_dipole: float = 1.0,
    shell_energy_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    total = torch.zeros((), dtype=sigma.dtype, device=sigma.device)
    values: dict[str, float] = {}

    if lambda_potential and pred_potential is not None and target_potential is not None:
        l_pot = potential_mse(pred_potential, target_potential)
        total = total + float(lambda_potential) * l_pot
        values["potential_mse"] = float(l_pot.detach().cpu())

    if lambda_acceleration and pred_acceleration is not None and target_acceleration is not None:
        l_acc = acceleration_mse(pred_acceleration, target_acceleration)
        total = total + float(lambda_acceleration) * l_acc
        values["acceleration_mse"] = float(l_acc.detach().cpu())

    if lambda_l2:
        l_l2 = source_l2(sigma, source_weights)
        total = total + float(lambda_l2) * l_l2
        values["source_l2"] = float(l_l2.detach().cpu())

    if lambda_moment:
        moments = moment_losses(source_positions, source_weights, sigma)
        l_moment = moments["monopole"] + float(lambda_dipole) * moments["dipole"]
        total = total + float(lambda_moment) * l_moment
        values["moment"] = float(l_moment.detach().cpu())

    if shell_energy_weights is not None and shell_energy_weights.numel() > 0:
        energies = shell_energy(sigma, source_weights, shell_ids)
        if energies.shape[0] != shell_energy_weights.shape[0]:
            raise ValueError("shell_energy_weights must match number of shells")
        l_shell = torch.sum(shell_energy_weights.to(device=sigma.device, dtype=sigma.dtype) * energies)
        total = total + l_shell
        values["shell_energy"] = float(l_shell.detach().cpu())

    values["total"] = float(total.detach().cpu())
    return total, values
