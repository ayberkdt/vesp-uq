"""Train or solve a single-shell discrete equivalent-source baseline."""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from vesp.common.config import get_device, get_dtype, load_config as load_standard_config, merge_defaults, validate_config
from vesp.data.dataset import ResidualGravityDataset, load_csv_dataset
from vesp.training.acceptability import classify_run_acceptability
from vesp.training.evaluate import evaluate_model, print_metrics, write_evaluation_artifacts
from vesp.core.losses import composite_loss
from vesp.core.models import DiscreteVESP, save_checkpoint
from vesp.core.operators import build_joint_operator
from vesp.core.regularization import lambda_is_auto, select_lambda_l2
from vesp.core.solvers import RidgeSolveConfig, solve_discrete_ridge
from vesp.training.maxent import MaxEntSolveConfig, solve_discrete_maxent, solve_discrete_maxent_constrained
from vesp.extensions.entropy import (
    effective_source_entropy,
    positive_negative_entropy,
    relative_entropy_to_uniform,
    shell_energy_balance_entropy,
)
from vesp.data.splits import DataSplits, make_splits
from vesp.data.synthetic import make_synthetic_dataset
from vesp.data.target_scaling import (
    TargetScales,
    altitude_row_weights,
    apply_target_scales_to_config,
    compute_target_scales,
    observation_row_weights,
    write_target_scales,
)
from vesp.common.units import UnitConfig
from vesp.core.sources import SourceSet, make_shell_sources


def load_config(path: str | Path) -> dict:
    return load_standard_config(path)


def make_data_splits(config: dict, *, dtype: torch.dtype) -> DataSplits:
    data_cfg = config.get("data", {})
    if data_cfg.get("path"):
        data = load_csv_dataset(data_cfg["path"], dtype=dtype, unit_config=UnitConfig.from_config(config))
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


def _build_weighted_system(
    model: DiscreteVESP,
    train_data,
    config: dict,
    *,
    device: torch.device,
    target_scales: TargetScales,
) -> tuple[torch.Tensor, torch.Tensor, SourceSet]:
    """Build the row-weighted, target-normalized (A, b) linear system + sources.

    Shared by the ridge and MaxEnt solvers so the data term is identical and the
    entropy weight traces out a comparable data-error vs entropy Pareto curve.
    """

    kernel_cfg = config.get("kernel", {})
    loss_cfg = config.get("loss", {})
    include_potential = bool(loss_cfg.get("use_potential", True)) and float(loss_cfg.get("lambda_potential", loss_cfg.get("potential_weight", 1.0))) > 0.0
    include_acceleration = bool(loss_cfg.get("use_acceleration", True)) and float(loss_cfg.get("lambda_acceleration", loss_cfg.get("acceleration_weight", 1.0))) > 0.0

    sources = SourceSet(
        positions=model.source_positions,
        weights=model.source_weights,
        shell_ids=model.shell_ids,
        shell_radii=tuple(float(v) for v in model.shell_radii),
    )
    bundle = build_joint_operator(
        train_data.positions,
        sources,
        potential=train_data.potential,
        acceleration=train_data.acceleration,
        source_chunk_size=kernel_cfg.get("source_chunk_size"),
        eps=float(kernel_cfg.get("softening", kernel_cfg.get("eps", 0.0))),
        use_potential=include_potential,
        use_acceleration=include_acceleration,
        potential_weight=1.0,
        acceleration_weight=1.0,
        sign=float(kernel_cfg.get("acceleration_sign", 1.0)),
        column_normalize=False,
    )
    operator = bundle.operator
    target = bundle.target

    # Altitude weighting (if enabled) is a solve-time row reweighting only; it never
    # touches evaluation/metrics. It applies to both the ridge and MaxEnt solvers.
    altitude_weights = altitude_row_weights(train_data.positions, config, dtype=operator.dtype)
    weights = observation_row_weights(
        n_query=train_data.positions.shape[0],
        include_potential=include_potential,
        include_acceleration=include_acceleration,
        lambda_potential=float(loss_cfg.get("lambda_potential", loss_cfg.get("potential_weight", 1.0))),
        lambda_acceleration=float(loss_cfg.get("lambda_acceleration", loss_cfg.get("acceleration_weight", 1.0))),
        scales=target_scales,
        dtype=operator.dtype,
        device=device,
        altitude_weights=altitude_weights,
    )
    operator = operator * weights.unsqueeze(-1)
    target = target * weights
    return operator, target, sources


def solve_ridge(
    model: DiscreteVESP,
    train_data,
    config: dict,
    *,
    device: torch.device,
    target_scales: TargetScales | None = None,
) -> dict | None:
    """Solve the ridge system. If ``lambda_l2: auto``, pick it at the L-curve corner.

    Returns an info dict (``selected_lambda_l2`` + the L-curve) when auto-selection ran,
    else ``None``.
    """

    model = model.to(device)
    train_data = train_data.to(device)
    target_scales = target_scales or compute_target_scales(train_data, config)
    operator, target, _ = _build_weighted_system(
        model, train_data, config, device=device, target_scales=target_scales
    )

    info: dict | None = None
    if lambda_is_auto(config):
        loss_cfg = config.setdefault("loss", {})
        solver_cfg = config.get("solver", {})
        # placeholder so RidgeSolveConfig.from_config does not choke on the "auto" sentinel
        loss_cfg["lambda_l2"] = 1.0
        if isinstance(solver_cfg, dict):
            solver_cfg["lambda_l2"] = 1.0
        base_ridge = RidgeSolveConfig.from_config(config)
        lambda_star, curve = select_lambda_l2(
            operator,
            target,
            source_positions=model.source_positions,
            source_weights=model.source_weights,
            shell_ids=model.shell_ids,
            base_config=base_ridge,
        )
        ridge_cfg = replace(base_ridge, lambda_l2=lambda_star)
        # write the resolved value back so artifacts / metrics / summary report a number
        loss_cfg["lambda_l2"] = lambda_star
        if isinstance(solver_cfg, dict):
            solver_cfg["lambda_l2"] = lambda_star
        info = {"selected_lambda_l2": lambda_star, "selection": "L-curve", "curve": curve}
        print(f"auto lambda_l2 (L-curve corner) = {lambda_star:g}")
    else:
        ridge_cfg = RidgeSolveConfig.from_config(config)

    sigma = solve_discrete_ridge(
        operator=operator,
        target=target,
        source_positions=model.source_positions,
        source_weights=model.source_weights,
        shell_ids=model.shell_ids,
        config=ridge_cfg,
    )
    model.set_sigma(sigma)
    return info


def solve_maxent(
    model: DiscreteVESP,
    train_data,
    config: dict,
    *,
    device: torch.device,
    target_scales: TargetScales | None = None,
) -> dict | None:
    """Stage 3A: refine the source strengths with deterministic entropy regularization.

    The ridge solution is used as a warm start (and remains the entropy_weight=0
    baseline), then the source strengths are optimized against the same data term
    plus a maximum-entropy regularizer.

    Two modes (``maxent.mode``):

    - ``penalty`` (default): fixed entropy weight, minimize data + l2 + moment - weight*H.
    - ``constrained``: the principled MaxEnt — maximize entropy subject to keeping the data
      misfit within ``maxent.misfit_factor`` of the ridge misfit, auto-selecting the weight.

    Returns an optional info dict (constrained mode reports the misfit target and the
    auto-selected entropy weight); ``None`` for the plain penalty mode.
    """

    model = model.to(device)
    train_data = train_data.to(device)
    target_scales = target_scales or compute_target_scales(train_data, config)
    operator, target, _ = _build_weighted_system(
        model, train_data, config, device=device, target_scales=target_scales
    )
    maxent_config = MaxEntSolveConfig.from_config(config)
    constrained = maxent_config.mode == "constrained"

    warm_start_sigma = None
    if maxent_config.warm_start or constrained:
        warm_start_sigma = solve_discrete_ridge(
            operator=operator,
            target=target,
            source_positions=model.source_positions,
            source_weights=model.source_weights,
            shell_ids=model.shell_ids,
            config=RidgeSolveConfig.from_config(config),
        )

    if constrained:
        sigma, info = solve_discrete_maxent_constrained(
            operator,
            target,
            model.source_positions,
            model.source_weights,
            model.shell_ids,
            maxent_config,
            warm_start_sigma=warm_start_sigma,
        )
        model.set_sigma(sigma)
        return info

    sigma = solve_discrete_maxent(
        operator,
        target,
        model.source_positions,
        model.source_weights,
        model.shell_ids,
        maxent_config,
        warm_start_sigma=warm_start_sigma,
    )
    model.set_sigma(sigma)
    return None


def train_adam(
    model: DiscreteVESP,
    train_data,
    config: dict,
    *,
    device: torch.device,
    target_scales: TargetScales | None = None,
) -> None:
    train_cfg = config.get("training", {})
    kernel_cfg = config.get("kernel", {})
    loss_cfg = config.get("loss", {})

    model = model.to(device)
    train_data = train_data.to(device)
    target_scales = target_scales or compute_target_scales(train_data, config)
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
                softening=float(kernel_cfg.get("softening", kernel_cfg.get("eps", 0.0))),
                acceleration_sign=float(kernel_cfg.get("acceleration_sign", 1.0)),
            )
            potential_scale = target_scales.potential_scale if target_scales.normalize_targets else 1.0
            acceleration_scale = target_scales.acceleration_scale if target_scales.normalize_targets else 1.0
            loss, values = composite_loss(
                pred_potential=pred_u / potential_scale if pred_u is not None else None,
                pred_acceleration=pred_a / acceleration_scale if pred_a is not None else None,
                target_potential=potential / potential_scale,
                target_acceleration=acceleration / acceleration_scale,
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
    target_scales = compute_target_scales(train_data, config)
    apply_target_scales_to_config(config, target_scales)
    model = make_model(config, dtype=dtype, model_cls=model_cls)

    solver_cfg = config.get("solver", {})
    if not isinstance(solver_cfg, dict):
        solver_cfg = {"type": solver_cfg}
    solver = str(solver_cfg.get("type", config.get("solver", "ridge"))).lower() if isinstance(solver_cfg, dict) else str(solver_cfg).lower()
    maxent_info: dict | None = None
    ridge_info: dict | None = None
    if solver == "ridge":
        ridge_info = solve_ridge(model, train_data, config, device=device, target_scales=target_scales)
    elif solver == "maxent":
        maxent_info = solve_maxent(model, train_data, config, device=device, target_scales=target_scales)
    elif solver == "adam":
        train_adam(model, train_data, config, device=device, target_scales=target_scales)
    else:
        raise ValueError("solver must be 'ridge', 'maxent', or 'adam'")

    eval_cfg = config.get("evaluation", {})
    kernel_cfg = config.get("kernel", {})
    diag_cfg = config.get("diagnostics", {})
    shell_collapse_threshold = float(diag_cfg.get("shell_collapse_threshold", 0.90))
    sigma_l2_warning_threshold = float(diag_cfg.get("sigma_l2_warning_threshold", 1.0))
    altitude_bands = eval_cfg.get("altitude_bands")
    eval_softening = float(kernel_cfg.get("softening", kernel_cfg.get("eps", 0.0)))
    eval_acceleration_sign = float(kernel_cfg.get("acceleration_sign", 1.0))

    def _evaluate(data, *, bands):
        return evaluate_model(
            model,
            data,
            batch_size=int(eval_cfg.get("batch_size", 4096)),
            source_chunk_size=kernel_cfg.get("source_chunk_size"),
            softening=eval_softening,
            acceleration_sign=eval_acceleration_sign,
            device=device,
            n_altitude_bins=int(eval_cfg.get("n_altitude_bins", 6)),
            altitude_bands=bands,
            shell_collapse_threshold=shell_collapse_threshold,
            sigma_l2_warning_threshold=sigma_l2_warning_threshold,
        )

    metrics = _evaluate(val_data, bands=altitude_bands)
    if splits.test_high is not None and splits.test_high.positions.shape[0] > 0:
        # OOD subsets only span one band; skip band computation to avoid empty-band noise.
        high_metrics = _evaluate(splits.test_high, bands={})
        metrics["test_high_acceleration_rmse"] = high_metrics["acceleration_rmse"]
        metrics["test_high_potential_rmse"] = high_metrics["potential_rmse"]
    if splits.test_low is not None and splits.test_low.positions.shape[0] > 0:
        low_metrics = _evaluate(splits.test_low, bands={})
        metrics["test_low_acceleration_rmse"] = low_metrics["acceleration_rmse"]
        metrics["test_low_potential_rmse"] = low_metrics["potential_rmse"]
    metrics["metrics_units"] = "raw target units (model coordinate system)"
    units = UnitConfig.from_config(config)
    if units.normalize_positions:
        metrics["acceleration_metric_units"] = "model normalized-gradient: dU/d(x / R_body)"
    else:
        metrics["acceleration_metric_units"] = f"physical: {config.get('body', {}).get('acceleration_units', 'model')}"
    metrics["training_loss_units"] = "target-normalized" if target_scales.normalize_targets else "raw target units"
    metrics["acceleration_sign"] = eval_acceleration_sign

    # Stage 3A entropy diagnostics: reported for every run so the ridge baseline
    # (entropy_weight=0) and MaxEnt runs are directly comparable on the
    # data-error vs entropy Pareto curve.
    with torch.no_grad():
        sigma_cpu = model.sigma.detach()
        weights_cpu = model.source_weights.detach()
        metrics["source_entropy_nats"] = float(effective_source_entropy(sigma_cpu, weights_cpu))
        metrics["positive_negative_entropy_nats"] = float(positive_negative_entropy(sigma_cpu, weights_cpu))
        metrics["relative_entropy_to_uniform"] = float(relative_entropy_to_uniform(sigma_cpu, weights_cpu))
        metrics["shell_energy_balance_entropy_nats"] = float(
            shell_energy_balance_entropy(sigma_cpu, weights_cpu, model.shell_ids)
        )
        metrics["max_possible_source_entropy_nats"] = float(math.log(model.n_sources)) if model.n_sources > 0 else 0.0
    loss_cfg = config.get("loss", {})
    maxent_cfg = config.get("maxent", {})
    metrics["solver"] = solver
    metrics["entropy_weight"] = float(loss_cfg.get("entropy_weight", maxent_cfg.get("entropy_weight", 0.0)))
    metrics["entropy_mode"] = str(loss_cfg.get("entropy_mode", maxent_cfg.get("entropy_mode", "positive_negative")))
    if maxent_info:
        # Constrained MaxEnt auto-selects the entropy weight; report the chosen value and
        # the misfit target so the equal-data-fit comparison against ridge is auditable.
        metrics["maxent_mode"] = "constrained"
        metrics["maxent_ridge_misfit"] = maxent_info.get("ridge_misfit")
        metrics["maxent_target_misfit"] = maxent_info.get("target_misfit")
        metrics["maxent_misfit"] = maxent_info.get("maxent_misfit")
        metrics["entropy_weight"] = float(maxent_info.get("chosen_entropy_weight", metrics["entropy_weight"]))
    if ridge_info:
        # lambda_l2 was auto-selected at the L-curve corner; report the chosen value.
        metrics["selected_lambda_l2"] = ridge_info.get("selected_lambda_l2")
        metrics["lambda_l2_selection"] = ridge_info.get("selection")

    acceptability = classify_run_acceptability(metrics, metrics.get("diagnostics", {}), config)
    metrics.update(acceptability)

    output_cfg = config.get("output", {})
    output_dir = Path(output_cfg.get("output_dir", config.get("output_dir", "outputs")))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = str(output_cfg.get("run_name", Path(str(config.get("checkpoint_name", "vesp_discrete.pt"))).stem))
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{run_name}.pt"
    save_checkpoint(str(checkpoint_path), model, config, metrics)
    save_checkpoint(str(run_dir / "sigma.pt"), model, config, metrics)
    extra_artifacts = {"checkpoint": checkpoint_path, "sigma_checkpoint": run_dir / "sigma.pt"}
    target_scales_path = write_target_scales(run_dir / "target_scales.json", target_scales)
    extra_artifacts["target_scales"] = target_scales_path
    write_evaluation_artifacts(
        run_dir,
        metrics,
        config,
        target_scales=target_scales,
        extra_artifacts=extra_artifacts,
    )
    print(f"saved_checkpoint: {checkpoint_path}")
    print(f"saved_run_dir: {run_dir}")
    print_metrics(metrics)
    return metrics


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/discrete_single_shell.yaml")
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
