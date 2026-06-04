"""Evaluation metrics for VESP field predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from .data import ResidualGravityData, ResidualGravityDataset, load_csv_dataset
from .diagnostics import source_diagnostics, time_inference
from .metrics import (
    altitude_binned_error,
    radial_cross_radial_error,
    relative_rmse_acceleration,
    rmse_acceleration,
    rmse_potential,
    vector_angle_error,
)
from .models import DiscreteVESP, load_checkpoint


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2)).detach().cpu())


def radial_cross_errors(
    positions: torch.Tensor,
    pred_acceleration: torch.Tensor,
    target_acceleration: torch.Tensor,
) -> dict[str, float]:
    radial = positions / torch.clamp(torch.linalg.norm(positions, dim=-1, keepdim=True), min=torch.finfo(positions.dtype).eps)
    error = pred_acceleration - target_acceleration
    radial_error = torch.sum(error * radial, dim=-1, keepdim=True) * radial
    cross_error = error - radial_error
    return {
        "radial_acceleration_rmse": float(torch.sqrt(torch.mean(radial_error * radial_error)).detach().cpu()),
        "cross_radial_acceleration_rmse": float(torch.sqrt(torch.mean(cross_error * cross_error)).detach().cpu()),
    }


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
        if not torch.any(mask):
            continue
        rows.append(
            {
                "r_min": float(bins[i].detach().cpu()),
                "r_max": float(bins[i + 1].detach().cpu()),
                "count": int(mask.sum().detach().cpu()),
                "acceleration_rmse": rmse(pred_acceleration[mask], target_acceleration[mask]),
            }
        )
    return rows


def evaluate_model(
    model: DiscreteVESP,
    data: ResidualGravityData,
    *,
    batch_size: int = 4096,
    source_chunk_size: int | None = None,
    device: str | torch.device = "cpu",
    n_altitude_bins: int = 6,
) -> dict:
    model = model.to(device)
    data = data.to(device)
    loader = DataLoader(ResidualGravityDataset(data), batch_size=batch_size, shuffle=False)

    pred_u = []
    pred_a = []
    true_u = []
    true_a = []
    xs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            u, a = model(x, source_chunk_size=source_chunk_size)
            if u is None or a is None:
                raise RuntimeError("model did not return both potential and acceleration")
            pred_u.append(u.detach())
            pred_a.append(a.detach())
            true_u.append(batch["potential"].to(device))
            true_a.append(batch["acceleration"].to(device))
            xs.append(x)

    pred_u_t = torch.cat(pred_u, dim=0)
    pred_a_t = torch.cat(pred_a, dim=0)
    true_u_t = torch.cat(true_u, dim=0)
    true_a_t = torch.cat(true_a, dim=0)
    xs_t = torch.cat(xs, dim=0)

    rc = radial_cross_radial_error(xs_t, pred_a_t, true_a_t)
    metrics = {
        "potential_rmse": rmse_potential(pred_u_t, true_u_t),
        "acceleration_rmse": rmse_acceleration(pred_a_t, true_a_t),
        "relative_acceleration_rmse": relative_rmse_acceleration(pred_a_t, true_a_t),
        **rc,
        "radial_acceleration_rmse": rc["radial_rmse"],
        "cross_radial_acceleration_rmse": rc["cross_radial_rmse"],
        **vector_angle_error(pred_a_t, true_a_t),
        "altitude_binned_error": altitude_binned_error(xs_t, pred_a_t, true_a_t, n_bins=n_altitude_bins),
        "diagnostics": source_diagnostics(
            source_positions=model.source_positions,
            source_weights=model.source_weights,
            shell_ids=model.shell_ids,
            sigma=model.sigma,
        ),
        "inference_seconds_per_batch": time_inference(
            model,
            xs_t[: min(batch_size, xs_t.shape[0])],
            source_chunk_size=source_chunk_size,
        ),
    }
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        metrics["cuda_max_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return metrics


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    return value


def write_evaluation_artifacts(output_dir: str | Path, metrics: dict, config: dict | None = None) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if config is not None:
        import yaml

        (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    diagnostics = metrics.get("diagnostics", {})
    metrics_without_nested = {k: v for k, v in metrics.items() if k not in {"diagnostics", "altitude_binned_error"}}

    (output / "metrics.json").write_text(json.dumps(metrics_without_nested, indent=2, default=_json_default), encoding="utf-8")
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, default=_json_default), encoding="utf-8")

    with (output / "altitude_binned_error.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["r_min", "r_max", "count", "acceleration_rmse"])
        writer.writeheader()
        writer.writerows(metrics.get("altitude_binned_error", []))

    shell_rows = diagnostics.get("shell_energy_distribution", [])
    with (output / "shell_energy.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["shell_id", "shell_alpha", "n_source", "energy", "energy_fraction", "sigma_norm"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(shell_rows)

    summary_lines = [
        "VESP Run Summary",
        "",
        f"potential_rmse: {metrics.get('potential_rmse')}",
        f"acceleration_rmse: {metrics.get('acceleration_rmse')}",
        f"relative_acceleration_rmse: {metrics.get('relative_acceleration_rmse')}",
        f"radial_rmse: {metrics.get('radial_rmse', metrics.get('radial_acceleration_rmse'))}",
        f"cross_radial_rmse: {metrics.get('cross_radial_rmse', metrics.get('cross_radial_acceleration_rmse'))}",
        f"angle_deg_p95: {metrics.get('angle_deg_p95')}",
        f"sigma_l2: {diagnostics.get('sigma_l2')}",
        f"effective_source_count: {diagnostics.get('effective_source_count')}",
        f"monopole_leakage: {diagnostics.get('monopole_leakage')}",
        f"dipole_leakage: {diagnostics.get('dipole_leakage')}",
    ]
    (output / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def print_metrics(metrics: dict) -> None:
    for key, value in metrics.items():
        if key == "altitude_binned_error":
            print("altitude_binned_error:")
            for row in value:
                print(f"  r=[{row['r_min']:.4f}, {row['r_max']:.4f}] n={row['count']} acc_rmse={row['acceleration_rmse']:.6e}")
        elif isinstance(value, dict):
            print(f"{key}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"{key}: {value}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--source-chunk-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    model = load_checkpoint(args.checkpoint, map_location=args.device)
    data = load_csv_dataset(Path(args.data))
    metrics = evaluate_model(
        model,
        data,
        batch_size=args.batch_size,
        source_chunk_size=args.source_chunk_size,
        device=args.device,
    )
    print_metrics(metrics)


if __name__ == "__main__":
    main()
