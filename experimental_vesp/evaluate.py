"""Evaluation metrics for VESP field predictions."""

from __future__ import annotations

import argparse
import csv
import json
import io
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from .artifacts import atomic_write_json, atomic_write_text, ensure_run_layout, write_run_manifest
from .data import ResidualGravityData, ResidualGravityDataset, load_csv_dataset
from .diagnostics import source_diagnostics, time_inference
from .metrics import (
    altitude_binned_error,
    radial_cross_radial_error,
    relative_rmse_acceleration,
    rmse_acceleration_components,
    rmse_acceleration_norm,
    rmse_acceleration,
    rmse_potential,
    vector_angle_error,
)
from .models import DiscreteVESP, load_checkpoint


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
        "acceleration_norm_rmse": rmse_acceleration_norm(pred_a_t, true_a_t),
        **rmse_acceleration_components(pred_a_t, true_a_t),
        "relative_acceleration_rmse": relative_rmse_acceleration(pred_a_t, true_a_t),
        **rc,
        "radial_rmse": rc["radial_scalar_rmse"],
        "cross_radial_rmse": rc["cross_norm_rmse"],
        "radial_acceleration_rmse": rc["radial_scalar_rmse"],
        "cross_radial_acceleration_rmse": rc["cross_norm_rmse"],
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


def _csv_text(fieldnames: list[str], rows: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_evaluation_artifacts(
    output_dir: str | Path,
    metrics: dict,
    config: dict | None = None,
    *,
    extra_artifacts: dict[str, str | Path] | None = None,
) -> None:
    layout = ensure_run_layout(output_dir)
    if config is not None:
        import yaml

        atomic_write_text(layout.config_yaml, yaml.safe_dump(config, sort_keys=False))

    diagnostics = metrics.get("diagnostics", {})
    metrics_without_nested = {k: v for k, v in metrics.items() if k not in {"diagnostics", "altitude_binned_error"}}

    atomic_write_json(layout.metrics_json, metrics_without_nested)
    atomic_write_json(layout.diagnostics_json, diagnostics)

    atomic_write_text(
        layout.altitude_binned_error_csv,
        _csv_text(["r_min", "r_max", "count", "acceleration_rmse"], metrics.get("altitude_binned_error", [])),
    )

    shell_rows = diagnostics.get("shell_energy_distribution", [])
    shell_fieldnames = ["shell_id", "shell_alpha", "n_source", "energy", "energy_fraction", "sigma_norm"]
    atomic_write_text(layout.shell_energy_csv, _csv_text(shell_fieldnames, shell_rows))

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
    atomic_write_text(layout.summary_txt, "\n".join(summary_lines) + "\n")

    artifacts = {
        "config": layout.config_yaml,
        "metrics": layout.metrics_json,
        "diagnostics": layout.diagnostics_json,
        "altitude_binned_error": layout.altitude_binned_error_csv,
        "shell_energy": layout.shell_energy_csv,
        "summary": layout.summary_txt,
    }
    artifacts.update(extra_artifacts or {})
    write_run_manifest(layout.run_dir, config=config, metrics=metrics_without_nested, artifacts=artifacts)


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
