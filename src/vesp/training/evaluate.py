"""Evaluation metrics for VESP field predictions."""

from __future__ import annotations

import argparse
import csv
import json
import io
import warnings
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from vesp.common.artifacts import atomic_write_json, atomic_write_text, ensure_run_layout, write_run_manifest
from vesp.common.config import load_config
from vesp.data.dataset import ResidualGravityData, ResidualGravityDataset, load_csv_dataset
from vesp.core.diagnostics import source_diagnostics, time_inference
from vesp.core.kernels import evaluate_kernel
from vesp.core.metrics import (
    altitude_band_errors,
    altitude_binned_error,
    radial_cross_radial_error,
    relative_rmse_acceleration,
    rmse_acceleration_components,
    rmse_acceleration_norm,
    rmse_acceleration,
    rmse_potential,
    vector_angle_error,
)
from vesp.core.models import DiscreteVESP, load_checkpoint
from vesp.data.target_scaling import TargetScales
from vesp.common.units import UnitConfig


def evaluate_model(
    model: DiscreteVESP,
    data: ResidualGravityData,
    *,
    batch_size: int = 4096,
    source_chunk_size: int | None = None,
    softening: float = 0.0,
    acceleration_sign: float = 1.0,
    device: str | torch.device = "cpu",
    n_altitude_bins: int = 6,
    altitude_bands: dict | None = None,
    shell_collapse_threshold: float = 0.90,
    sigma_l2_warning_threshold: float = 1.0,
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
            u, a = model(
                x,
                source_chunk_size=source_chunk_size,
                softening=softening,
                acceleration_sign=acceleration_sign,
            )
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
        **altitude_band_errors(xs_t, pred_a_t, true_a_t, bands=altitude_bands),
        "diagnostics": source_diagnostics(
            source_positions=model.source_positions,
            source_weights=model.source_weights,
            shell_ids=model.shell_ids,
            sigma=model.sigma,
            shell_collapse_threshold=shell_collapse_threshold,
            sigma_l2_warning_threshold=sigma_l2_warning_threshold,
        ),
        "inference_seconds_per_batch": time_inference(
            model,
            xs_t[: min(batch_size, xs_t.shape[0])],
            source_chunk_size=source_chunk_size,
            softening=softening,
            acceleration_sign=acceleration_sign,
        ),
    }
    # Shell cancellation diagnostic (multi-shell only). The energy-fraction collapse
    # metric is radius-biased and blind to the real multi-shell pathology: adjacent
    # near-redundant shells fit the field with large opposing source strengths that
    # nearly cancel. cancellation_ratio = sum_j RMS(field_j) / RMS(field_total); it is
    # ~1-n_shells for a healthy fit and >> 1 when shells cancel (a brittle, ill-
    # conditioned solution), independent of which shell holds the most sigma^2 energy.
    shell_unique = torch.unique(model.shell_ids)
    if shell_unique.numel() > 1:
        eps = torch.finfo(pred_a_t.dtype).eps
        strength = (model.source_weights * model.sigma).detach()
        total_rms = float(torch.sqrt(torch.mean(torch.sum(pred_a_t * pred_a_t, dim=-1))))
        per_shell_rms: list[float] = []
        with torch.no_grad():
            for shell_value in shell_unique:
                mask = model.shell_ids == shell_value
                shell_out = evaluate_kernel(
                    xs_t,
                    model.source_positions[mask],
                    strength[mask],
                    source_chunk_size=source_chunk_size,
                    softening=softening,
                    acceleration_sign=acceleration_sign,
                    compute_potential=False,
                    compute_acceleration=True,
                )
                per_shell_rms.append(float(torch.sqrt(torch.mean(torch.sum(shell_out.acceleration ** 2, dim=-1)))))
        metrics["diagnostics"]["per_shell_field_rms"] = per_shell_rms
        metrics["diagnostics"]["shell_cancellation_ratio"] = float(sum(per_shell_rms) / max(total_rms, eps))

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
    target_scales: TargetScales | dict | None = None,
    extra_artifacts: dict[str, str | Path] | None = None,
) -> None:
    layout = ensure_run_layout(output_dir)
    if config is not None:
        import yaml

        atomic_write_text(layout.config_yaml, yaml.safe_dump(config, sort_keys=False))

    diagnostics = metrics.get("diagnostics", {})
    metrics_without_nested = {k: v for k, v in metrics.items() if k not in {"diagnostics", "altitude_binned_error"}}
    scale_payload: dict = {}
    if target_scales is not None:
        scale_payload = target_scales.as_dict() if hasattr(target_scales, "as_dict") else dict(target_scales)
    elif config is not None:
        loss_cfg = config.get("loss", {})
        scale_payload = {
            "normalize_targets": bool(loss_cfg.get("resolved_normalize_targets", loss_cfg.get("normalize_targets", False))),
            "potential_scale": float(loss_cfg.get("resolved_potential_scale", 1.0)),
            "acceleration_scale": float(loss_cfg.get("resolved_acceleration_scale", 1.0)),
            "potential_source": str(loss_cfg.get("resolved_potential_source", "unknown")),
            "acceleration_source": str(loss_cfg.get("resolved_acceleration_source", "unknown")),
        }

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
        f"acceptability_status: {metrics.get('acceptability_status')}",
        f"potential_rmse: {metrics.get('potential_rmse')}",
        f"acceleration_rmse: {metrics.get('acceleration_rmse')}",
        f"relative_acceleration_rmse: {metrics.get('relative_acceleration_rmse')}",
        f"radial_rmse: {metrics.get('radial_rmse', metrics.get('radial_acceleration_rmse'))}",
        f"cross_radial_rmse: {metrics.get('cross_radial_rmse', metrics.get('cross_radial_acceleration_rmse'))}",
        f"angle_deg_p95: {metrics.get('angle_deg_p95')}",
        "",
        f"low_altitude_acceleration_rmse: {metrics.get('low_altitude_acceleration_rmse')}",
        f"mid_altitude_acceleration_rmse: {metrics.get('mid_altitude_acceleration_rmse')}",
        f"high_altitude_acceleration_rmse: {metrics.get('high_altitude_acceleration_rmse')}",
        f"low_to_high_error_ratio: {metrics.get('low_to_high_error_ratio')}",
        "",
        f"normalize_targets: {scale_payload.get('normalize_targets')}",
        f"potential_scale: {scale_payload.get('potential_scale')}",
        f"acceleration_scale: {scale_payload.get('acceleration_scale')}",
        f"potential_scale_source: {scale_payload.get('potential_source')}",
        f"acceleration_scale_source: {scale_payload.get('acceleration_source')}",
        f"metrics_units: {metrics.get('metrics_units', 'raw target units')}",
        f"acceleration_metric_units: {metrics.get('acceleration_metric_units')}",
        f"training_loss_units: {metrics.get('training_loss_units')}",
        "",
        f"sigma_l2: {diagnostics.get('sigma_l2')}",
        f"sigma_norm_warning: {diagnostics.get('sigma_norm_warning')}",
        f"effective_source_count: {diagnostics.get('effective_source_count')}",
        f"top_5pct_source_contribution: {diagnostics.get('top_5pct_source_contribution')}",
        f"dominant_shell_alpha: {diagnostics.get('dominant_shell_alpha')}",
        f"dominant_shell_energy_fraction: {diagnostics.get('dominant_shell_energy_fraction')}",
        f"shell_energy_entropy: {diagnostics.get('shell_energy_entropy')}",
        f"shell_collapse_flag: {diagnostics.get('shell_collapse_flag')}",
        f"shell_cancellation_ratio: {diagnostics.get('shell_cancellation_ratio')}",
        f"per_shell_field_rms: {diagnostics.get('per_shell_field_rms')}",
        f"relative_monopole_leakage: {diagnostics.get('relative_monopole_leakage')}",
        f"relative_dipole_leakage: {diagnostics.get('relative_dipole_leakage')}",
        f"monopole_leakage_abs: {diagnostics.get('monopole_leakage')}",
        f"dipole_leakage_abs: {diagnostics.get('dipole_leakage')}",
    ]
    reasons = metrics.get("acceptability_reasons") or []
    if reasons:
        summary_lines.append("")
        summary_lines.append("acceptability_reasons:")
        summary_lines.extend(f"  - {reason}" for reason in reasons)
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


def _load_checkpoint_payload(path: str | Path, *, map_location: str | torch.device) -> dict:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    return payload


def _resolve_eval_config(checkpoint_payload: dict, explicit_config_path: str | None) -> dict:
    checkpoint_config = checkpoint_payload.get("config")
    explicit_config = load_config(explicit_config_path) if explicit_config_path else None
    if explicit_config is not None:
        if checkpoint_config is not None and checkpoint_config != explicit_config:
            warnings.warn(
                "explicit --config differs from checkpoint config; using explicit config for unit-safe evaluation",
                RuntimeWarning,
                stacklevel=2,
            )
        return explicit_config
    if checkpoint_config is None:
        raise ValueError("checkpoint does not contain config; unit-safe evaluation requires config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("checkpoint config must be a mapping")
    return checkpoint_config


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--source-chunk-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    payload = _load_checkpoint_payload(args.checkpoint, map_location=args.device)
    config = _resolve_eval_config(payload, args.config)
    model = load_checkpoint(args.checkpoint, map_location=args.device)
    data = load_csv_dataset(Path(args.data), dtype=model.sigma.dtype, unit_config=UnitConfig.from_config(config))
    kernel_cfg = config.get("kernel", {})
    metrics = evaluate_model(
        model,
        data,
        batch_size=args.batch_size,
        source_chunk_size=args.source_chunk_size,
        softening=float(kernel_cfg.get("softening", kernel_cfg.get("eps", 0.0))),
        acceleration_sign=float(kernel_cfg.get("acceleration_sign", 1.0)),
        device=args.device,
    )
    print_metrics(metrics)


if __name__ == "__main__":
    main()
