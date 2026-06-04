"""Train or solve a single-shell discrete equivalent-source baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
import yaml
from torch.utils.data import DataLoader

from .config import get_device, get_dtype, load_config as load_standard_config, merge_defaults, validate_config
from .data import ResidualGravityDataset, load_csv_dataset
from .evaluate import evaluate_model, print_metrics, write_evaluation_artifacts
from .kernels import build_dense_operator, stack_observations
from .losses import composite_loss
from .models import DiscreteVESP, save_checkpoint
from .solvers import build_regularization_rows, solve_ridge as solve_ridge_system
from .splits import DataSplits, make_splits
from .synthetic import make_synthetic_dataset
from .units import UnitConfig
from .sources import make_shell_sources


def load_config(path: str | Path) -> dict:
    return load_standard_config(path)


def make_data_splits(config: dict, *, dtype: torch.dtype) -> DataSplits:
    data_cfg = config.get("data", {})
    if data_cfg.get("path"):
        data = load_csv_dataset(data_cfg["path"], dtype=dtype)
    else:
        truth_shell_radii = data_cfg.get("synthetic_truth_shell_radii")
        data = make_synthetic_dataset(
            n_query=int(data_cfg.get("synthetic_n_query", 1024)),
            n_truth_sources=data_cfg.get("synthetic_n_truth_sources", 64),
            query_radius_min=float(data_cfg.get("synthetic_query_radius_min", 1.05)),
            query_radius_max=float(data_cfg.get("synthetic_query_radius_max", 1.60)),
            truth_shell_radius=float(data_cfg.get("synthetic_truth_shell_radius", 0.72)),
            truth_shell_radii=truth_shell_radii,
            noise_std=float(data_cfg.get("synthetic_noise_std", 0.0)),
            seed=int(data_cfg.get("seed", 7)),
            dtype=dtype,
        )
    return make_splits(data, config)


def make_data(config: dict, *, dtype: torch.dtype) -> tuple:
    splits = make_data_splits(config, dtype=dtype)
    return splits.train, splits.val


def _source_config(config: dict) -> dict:
    config = merge_defaults(config)
    if "model" in config:
        model_cfg = config.get("model", {})
        if model_cfg.get("type") == "multishell":
            return {
                "shell_radii": model_cfg.get("shell_alphas", model_cfg.get("shell_radii", [0.5, 0.8, 0.95])),
                "points_per_shell": model_cfg.get("n_sources_per_shell", model_cfg.get("points_per_shell", 512)),
                "body_radius": config.get("body", {}).get("R_body", 1.0 if config.get("body", {}).get("normalize_positions", True) else 1.0),
                "weight_mode": model_cfg.get("weight_mode", "surface_area"),
                "init_scale": model_cfg.get("init_scale", 0.0),
            }
        return {
            "shell_radii": [model_cfg.get("shell_alpha", 0.86)],
            "points_per_shell": model_cfg.get("n_source", 1024),
            "body_radius": config.get("body", {}).get("R_body", 1.0 if config.get("body", {}).get("normalize_positions", True) else 1.0),
            "weight_mode": model_cfg.get("weight_mode", "surface_area"),
            "init_scale": model_cfg.get("init_scale", 0.0),
        }
    return config.get("sources", {})


def make_model(config: dict, *, dtype: torch.dtype, model_cls=DiscreteVESP) -> DiscreteVESP:
    config = merge_defaults(config)
    src_cfg = _source_config(config)
    units = UnitConfig.from_config(config)
    shells = src_cfg.get("shell_radii", [0.8])
    points = src_cfg.get("points_per_shell", 512)
    source_set = make_shell_sources(
        shells,
        points,
        body_radius=units.source_body_radius,
        weight_mode=str(src_cfg.get("weight_mode", "surface_area")),
        dtype=dtype,
    )
    return model_cls(source_set, init_scale=float(src_cfg.get("init_scale", 0.0)), dtype=dtype)


def _regularization_rows(
    *,
    model: DiscreteVESP,
    config: dict,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    loss_cfg = config.get("loss", {})
    shell_weights = torch.as_tensor(loss_cfg.get("shell_energy_weights", []), dtype=dtype, device=device)
    return build_regularization_rows(
        source_positions=model.source_positions.to(device=device, dtype=dtype),
        source_weights=model.source_weights.to(device=device, dtype=dtype),
        shell_ids=model.shell_ids.to(device),
        lambda_l2=float(loss_cfg.get("lambda_l2", 0.0)),
        lambda_moment=float(loss_cfg.get("lambda_moment", 0.0)),
        lambda_dipole=float(loss_cfg.get("lambda_dipole", 1.0)),
        shell_energy_weights=shell_weights,
    )


def _normal_equation_solve(
    *,
    operator: torch.Tensor,
    target: torch.Tensor,
    model: DiscreteVESP,
    config: dict,
) -> torch.Tensor:
    loss_cfg = config.get("loss", {})
    dtype = operator.dtype
    device = operator.device
    n_sources = operator.shape[1]

    lhs = operator.T @ operator
    rhs = operator.T @ target

    lambda_l2 = float(loss_cfg.get("lambda_l2", 0.0))
    if lambda_l2:
        lhs = lhs + lambda_l2 * torch.eye(n_sources, dtype=dtype, device=device)

    lambda_moment = float(loss_cfg.get("lambda_moment", 0.0))
    if lambda_moment:
        weights = model.source_weights.to(device=device, dtype=dtype)
        positions = model.source_positions.to(device=device, dtype=dtype)
        dipole_weight = float(loss_cfg.get("lambda_dipole", 1.0))
        vectors = [weights]
        vectors.extend([torch.sqrt(torch.tensor(dipole_weight, dtype=dtype, device=device)) * weights * positions[:, axis] for axis in range(3)])
        for vec in vectors:
            lhs = lhs + lambda_moment * torch.outer(vec, vec)

    shell_weights = torch.as_tensor(loss_cfg.get("shell_energy_weights", []), dtype=dtype, device=device)
    if shell_weights.numel():
        shell_diag = torch.zeros(n_sources, dtype=dtype, device=device)
        for shell_id, weight in enumerate(shell_weights):
            shell_diag = shell_diag + torch.where(model.shell_ids.to(device) == shell_id, weight, torch.zeros_like(shell_diag))
        lhs = lhs + torch.diag(shell_diag * model.source_weights.to(device=device, dtype=dtype))

    return torch.linalg.solve(lhs, rhs)


def _augmented_lstsq_solve(
    *,
    operator: torch.Tensor,
    target: torch.Tensor,
    model: DiscreteVESP,
    config: dict,
) -> torch.Tensor:
    """Solve the regularized problem without squaring the condition number."""

    train_cfg = config.get("training", {})
    dtype = operator.dtype
    device = operator.device
    rows, targets = _regularization_rows(model=model, config=config, dtype=dtype, device=device)
    if bool(train_cfg.get("column_normalize", True)):
        op_for_scale = torch.cat([operator, *rows], dim=0) if rows else operator
        col_norm = torch.linalg.norm(op_for_scale, dim=0)
        col_scale = torch.clamp(col_norm, min=torch.finfo(dtype).eps)
        scaled_operator = operator / col_scale.unsqueeze(0)
        scaled_rows = [row / col_scale.unsqueeze(0) for row in rows]
        solution_scaled = solve_ridge_system(
            scaled_operator,
            target,
            method="augmented_lstsq",
            lambda_l2=0.0,
            extra_rows=scaled_rows,
            extra_targets=targets,
        )
        return solution_scaled / col_scale

    return solve_ridge_system(operator, target, method="augmented_lstsq", lambda_l2=0.0, extra_rows=rows, extra_targets=targets)


def solve_ridge(model: DiscreteVESP, train_data, config: dict, *, device: torch.device) -> None:
    kernel_cfg = config.get("kernel", {})
    loss_cfg = config.get("loss", {})
    solver_cfg = config.get("solver", {})
    if not isinstance(solver_cfg, dict):
        solver_cfg = {"type": solver_cfg}
    include_potential = bool(loss_cfg.get("use_potential", True)) and float(loss_cfg.get("lambda_potential", loss_cfg.get("potential_weight", 1.0))) > 0.0
    include_acceleration = bool(loss_cfg.get("use_acceleration", True)) and float(loss_cfg.get("lambda_acceleration", loss_cfg.get("acceleration_weight", 1.0))) > 0.0

    model = model.to(device)
    train_data = train_data.to(device)
    operator = build_dense_operator(
        train_data.positions,
        model.source_positions,
        model.source_weights,
        source_chunk_size=kernel_cfg.get("source_chunk_size"),
        softening=float(kernel_cfg.get("softening", kernel_cfg.get("eps", 0.0))),
        include_potential=include_potential,
        include_acceleration=include_acceleration,
        acceleration_sign=float(kernel_cfg.get("acceleration_sign", 1.0)),
    )
    target = stack_observations(
        train_data.potential,
        train_data.acceleration,
        include_potential=include_potential,
        include_acceleration=include_acceleration,
    )

    row_weights = []
    if include_potential:
        row_weights.append(torch.full((train_data.positions.shape[0],), float(loss_cfg.get("lambda_potential", loss_cfg.get("potential_weight", 1.0))), dtype=operator.dtype, device=device))
    if include_acceleration:
        row_weights.extend(
            [
                torch.full((train_data.positions.shape[0],), float(loss_cfg.get("lambda_acceleration", loss_cfg.get("acceleration_weight", 1.0))), dtype=operator.dtype, device=device)
                for _ in range(3)
            ]
        )
    if row_weights:
        weights = torch.sqrt(torch.cat(row_weights, dim=0))
        operator = operator * weights.unsqueeze(-1)
        target = target * weights

    ridge_method = str(config.get("training", {}).get("ridge_method", solver_cfg.get("ridge_method", "augmented_lstsq"))).lower()
    if ridge_method == "normal_equation":
        sigma = _normal_equation_solve(operator=operator, target=target, model=model, config=config)
    elif ridge_method == "augmented_lstsq":
        sigma = _augmented_lstsq_solve(operator=operator, target=target, model=model, config=config)
    else:
        raise ValueError("training.ridge_method must be 'augmented_lstsq' or 'normal_equation'")
    model.set_sigma(sigma)


def train_adam(model: DiscreteVESP, train_data, config: dict, *, device: torch.device) -> None:
    train_cfg = config.get("training", {})
    kernel_cfg = config.get("kernel", {})
    loss_cfg = config.get("loss", {})

    model = model.to(device)
    train_data = train_data.to(device)
    loader = DataLoader(
        ResidualGravityDataset(train_data),
        batch_size=int(train_cfg.get("batch_size", 2048)),
        shuffle=True,
    )
    optimizer = torch.optim.Adam([model.sigma], lr=float(train_cfg.get("lr", 1e-2)))
    shell_weights = torch.as_tensor(loss_cfg.get("shell_energy_weights", []), dtype=model.sigma.dtype, device=device)

    for epoch in range(int(train_cfg.get("epochs", 200))):
        last = None
        for batch in loader:
            x = batch["x"].to(device)
            potential = batch["potential"].to(device)
            acceleration = batch["acceleration"].to(device)
            pred_u, pred_a = model(
                x,
                source_chunk_size=kernel_cfg.get("source_chunk_size"),
                softening=float(kernel_cfg.get("softening", 0.0)),
            )
            loss, values = composite_loss(
                pred_potential=pred_u,
                pred_acceleration=pred_a,
                target_potential=potential,
                target_acceleration=acceleration,
                sigma=model.sigma,
                source_positions=model.source_positions,
                source_weights=model.source_weights,
                shell_ids=model.shell_ids,
                lambda_potential=float(loss_cfg.get("lambda_potential", 1.0)),
                lambda_acceleration=float(loss_cfg.get("lambda_acceleration", 1.0)),
                lambda_l2=float(loss_cfg.get("lambda_l2", 0.0)),
                lambda_moment=float(loss_cfg.get("lambda_moment", 0.0)),
                lambda_dipole=float(loss_cfg.get("lambda_dipole", 1.0)),
                shell_energy_weights=shell_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0))
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_([model.sigma], grad_clip_norm)
            optimizer.step()
            last = values
        if epoch % int(train_cfg.get("log_every", 25)) == 0 or epoch == int(train_cfg.get("epochs", 200)) - 1:
            print(f"epoch={epoch} {last}")


def run(config: dict, *, model_cls=DiscreteVESP) -> dict:
    config = merge_defaults(config)
    validate_config(config)
    dtype = get_dtype(config)
    device = get_device(config)
    splits = make_data_splits(config, dtype=dtype)
    train_data, val_data = splits.train, splits.val
    model = make_model(config, dtype=dtype, model_cls=model_cls)

    solver_cfg = config.get("solver", {})
    if not isinstance(solver_cfg, dict):
        solver_cfg = {"type": solver_cfg}
    solver = str(solver_cfg.get("type", config.get("solver", "ridge"))).lower() if isinstance(solver_cfg, dict) else str(solver_cfg).lower()
    if solver == "ridge":
        solve_ridge(model, train_data, config, device=device)
    elif solver == "adam":
        train_adam(model, train_data, config, device=device)
    else:
        raise ValueError("solver must be 'ridge' or 'adam'")

    eval_cfg = config.get("evaluation", {})
    metrics = evaluate_model(
        model,
        val_data,
        batch_size=int(eval_cfg.get("batch_size", 4096)),
        source_chunk_size=config.get("kernel", {}).get("source_chunk_size"),
        device=device,
        n_altitude_bins=int(eval_cfg.get("n_altitude_bins", 6)),
    )
    if splits.test_high is not None and splits.test_high.positions.shape[0] > 0:
        high_metrics = evaluate_model(
            model,
            splits.test_high,
            batch_size=int(eval_cfg.get("batch_size", 4096)),
            source_chunk_size=config.get("kernel", {}).get("source_chunk_size"),
            device=device,
            n_altitude_bins=int(eval_cfg.get("n_altitude_bins", 6)),
        )
        metrics["test_high_acceleration_rmse"] = high_metrics["acceleration_rmse"]
        metrics["test_high_potential_rmse"] = high_metrics["potential_rmse"]
    if splits.test_low is not None and splits.test_low.positions.shape[0] > 0:
        low_metrics = evaluate_model(
            model,
            splits.test_low,
            batch_size=int(eval_cfg.get("batch_size", 4096)),
            source_chunk_size=config.get("kernel", {}).get("source_chunk_size"),
            device=device,
            n_altitude_bins=int(eval_cfg.get("n_altitude_bins", 6)),
        )
        metrics["test_low_acceleration_rmse"] = low_metrics["acceleration_rmse"]
        metrics["test_low_potential_rmse"] = low_metrics["potential_rmse"]

    output_cfg = config.get("output", {})
    output_dir = Path(output_cfg.get("output_dir", config.get("output_dir", "outputs")))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = str(output_cfg.get("run_name", Path(str(config.get("checkpoint_name", "vesp_discrete.pt"))).stem))
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{run_name}.pt"
    save_checkpoint(str(checkpoint_path), model, config, metrics)
    save_checkpoint(str(run_dir / "sigma.pt"), model, config, metrics)
    write_evaluation_artifacts(
        run_dir,
        metrics,
        config,
        extra_artifacts={"checkpoint": checkpoint_path, "sigma_checkpoint": run_dir / "sigma.pt"},
    )
    print(f"saved_checkpoint: {checkpoint_path}")
    print(f"saved_run_dir: {run_dir}")
    print_metrics(metrics)
    return metrics


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experimental_vesp/configs/discrete_single_shell.yaml")
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
