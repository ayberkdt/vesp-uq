"""Central configuration loading, defaults, and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import warnings

import torch
import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "device": "cpu",
    "dtype": "float64",
    "body": {
        "R_body": 1.0,
        "normalize_positions": True,
        "position_units": "normalized",
        "potential_units": "model",
        "acceleration_units": "model",
        "physical_R_body": None,
        "physical_R_body_units": "km",
    },
    "data": {
        "type": "synthetic",
        "path": None,
        "train_fraction": 0.8,
        "seed": 42,
    },
    "model": {
        "type": "discrete",
        "shell_alpha": 0.86,
        "n_source": 512,
        "weight_mode": "surface_area",
    },
    "kernel": {
        "eps": 0.0,
        "acceleration_sign": 1.0,
        "source_chunk_size": 512,
    },
    "solver": {
        "type": "ridge",
        "ridge_method": "augmented_lstsq",
        "lambda_l2": 1.0e-8,
        "column_normalize": True,
    },
    "loss": {
        "use_potential": True,
        "use_acceleration": True,
        "normalize_targets": False,
        "potential_scale": "auto",
        "acceleration_scale": "auto",
        "target_scale_eps": 1.0e-12,
        "lambda_potential": 0.2,
        "lambda_acceleration": 1.0,
        "lambda_l2": 1.0e-8,
        "lambda_moment": 1.0e-6,
        "lambda_dipole": 1.0,
        "shell_energy_weights": [],
        "altitude_weighting": {
            "enabled": False,
            "mode": "low_altitude_boost",
            "r_threshold": 1.15,
            "boost": 3.0,
        },
    },
    "split": {
        "type": "random",
        "train_fraction": 0.8,
        "radius_units": "normalized",
    },
    "evaluation": {
        "batch_size": 4096,
        "n_altitude_bins": 6,
        "altitude_bands": {
            "low": [1.03, 1.15],
            "mid": [1.15, 1.35],
            "high": [1.35, 1.60],
        },
    },
    "diagnostics": {
        "shell_collapse_threshold": 0.90,
        # sigma_l2 is a coordinate-dependent magnitude; healthy ridge fits reach ~10.
        "sigma_l2_warning_threshold": 100.0,
    },
    "acceptance": {
        "max_relative_acceleration_rmse": 0.75,
        "max_low_altitude_rmse_factor": 5.0,
        "max_top5_source_contribution": 0.40,
        "max_dominant_shell_energy_fraction": 0.90,
        "max_shell_cancellation_ratio": 5.0,
        "max_sigma_l2": 100.0,
        "max_relative_monopole_leakage": 0.05,
        "max_relative_dipole_leakage": 0.5,
    },
    "output": {
        "output_dir": "outputs",
        "run_name": "vesp_run",
        "save_plots": False,
    },
}


def _deep_merge(base: dict, update: dict) -> dict:
    out = deepcopy(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _normalize_legacy(config: dict) -> dict:
    cfg = deepcopy(config)
    if "output_dir" in cfg:
        cfg.setdefault("output", {})["output_dir"] = cfg.pop("output_dir")
    if "checkpoint_name" in cfg:
        checkpoint_name = str(cfg["checkpoint_name"])
        cfg.setdefault("output", {}).setdefault("run_name", Path(checkpoint_name).stem)
    if "sources" in cfg and "model" not in cfg:
        sources = cfg.pop("sources")
        shell_radii = sources.get("shell_radii", [0.86])
        points = sources.get("points_per_shell", 512)
        if len(shell_radii) > 1:
            cfg["model"] = {
                "type": "multishell",
                "shell_alphas": shell_radii,
                "n_sources_per_shell": points,
                "weight_mode": sources.get("weight_mode", "surface_area"),
            }
        else:
            cfg["model"] = {
                "type": "discrete",
                "shell_alpha": shell_radii[0],
                "n_source": points if isinstance(points, int) else points[0],
                "weight_mode": sources.get("weight_mode", "surface_area"),
            }
    if isinstance(cfg.get("solver"), str):
        cfg["solver"] = {"type": cfg["solver"]}
    if "softening" in cfg.get("kernel", {}):
        cfg["kernel"]["eps"] = cfg["kernel"].pop("softening")
    return cfg


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if isinstance(loaded, str):
        ref = (path.parent / loaded).resolve()
        with ref.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
    cfg = merge_defaults(_normalize_legacy(loaded or {}))
    validate_config(cfg)
    return cfg


def merge_defaults(config: dict) -> dict:
    return _deep_merge(DEFAULT_CONFIG, _normalize_legacy(config))


def validate_config(config: dict) -> None:
    body = config.get("body", {})
    if "R_body" not in body or float(body["R_body"]) <= 0.0:
        raise ValueError("body.R_body must be present and positive")
    if bool(body.get("normalize_positions", True)) and abs(float(body.get("R_body", 1.0)) - 1.0) > 1.0e-12:
        warnings.warn(
            "body.normalize_positions=true usually expects body.R_body=1.0 in model coordinates",
            RuntimeWarning,
            stacklevel=2,
        )
    if bool(body.get("normalize_positions", True)) and body.get("physical_R_body") is None and config.get("data", {}).get("path"):
        warnings.warn(
            "body.normalize_positions=true with CSV data needs physical_R_body in config or dataset metadata for physical position conversion",
            RuntimeWarning,
            stacklevel=2,
        )

    model = config.get("model", {})
    model_type = model.get("type")
    if model_type not in {"discrete", "multishell"}:
        raise ValueError("model.type must be 'discrete' or 'multishell'")
    if model_type == "discrete":
        if "shell_alpha" not in model or "n_source" not in model:
            raise ValueError("discrete model requires model.shell_alpha and model.n_source")
        if float(model["shell_alpha"]) <= 0.0 or int(model["n_source"]) <= 0:
            raise ValueError("discrete model requires positive model.shell_alpha and model.n_source")
    if model_type == "multishell":
        shells = model.get("shell_alphas")
        counts = model.get("n_sources_per_shell")
        if not shells or counts is None:
            raise ValueError("multishell model requires model.shell_alphas and model.n_sources_per_shell")
        if isinstance(counts, list) and len(counts) != len(shells):
            raise ValueError("model.n_sources_per_shell must match model.shell_alphas length")
        if any(float(shell) <= 0.0 for shell in shells):
            raise ValueError("model.shell_alphas must be positive")
        if isinstance(counts, list):
            if any(int(count) <= 0 for count in counts):
                raise ValueError("model.n_sources_per_shell values must be positive")
        elif int(counts) <= 0:
            raise ValueError("model.n_sources_per_shell must be positive")

    loss = config.get("loss", {})
    if not bool(loss.get("use_potential", True)) and not bool(loss.get("use_acceleration", True)):
        raise ValueError("at least one of loss.use_potential or loss.use_acceleration must be true")
    if bool(loss.get("normalize_targets", False)):
        for key in ("potential_scale", "acceleration_scale"):
            value = loss.get(key, "auto")
            if isinstance(value, str):
                if value.strip().lower() != "auto":
                    raise ValueError(f"loss.{key} must be 'auto' or a positive number")
            elif float(value) <= 0.0:
                raise ValueError(f"loss.{key} must be 'auto' or a positive number")

    altitude_weighting = loss.get("altitude_weighting", {})
    if isinstance(altitude_weighting, dict) and bool(altitude_weighting.get("enabled", False)):
        if float(altitude_weighting.get("boost", 1.0)) <= 0.0:
            raise ValueError("loss.altitude_weighting.boost must be positive")
        if float(altitude_weighting.get("r_threshold", 0.0)) <= 0.0:
            raise ValueError("loss.altitude_weighting.r_threshold must be positive")

    bands = config.get("evaluation", {}).get("altitude_bands", {})
    if isinstance(bands, dict):
        for name, band in bands.items():
            if band is None:
                continue
            lo, hi = band
            if float(lo) >= float(hi):
                raise ValueError(f"evaluation.altitude_bands.{name} must be [min, max] with min < max")

    diagnostics_cfg = config.get("diagnostics", {})
    if isinstance(diagnostics_cfg, dict):
        if not 0.0 < float(diagnostics_cfg.get("shell_collapse_threshold", 0.9)) <= 1.0:
            raise ValueError("diagnostics.shell_collapse_threshold must be in (0, 1]")
        if float(diagnostics_cfg.get("sigma_l2_warning_threshold", 1.0)) <= 0.0:
            raise ValueError("diagnostics.sigma_l2_warning_threshold must be positive")

    kernel = config.get("kernel", {})
    if float(kernel.get("eps", kernel.get("softening", 0.0))) < 0.0:
        raise ValueError("kernel.eps must be non-negative")

    solver = config.get("solver", {})
    if not isinstance(solver, dict):
        solver = {"type": solver}
    solver_type = str(solver.get("type", "ridge")).lower()
    if solver_type not in {"ridge", "maxent", "adam"}:
        raise ValueError("solver.type must be 'ridge', 'maxent', or 'adam'")
    if solver_type == "ridge":
        method = str(solver.get("ridge_method", config.get("training", {}).get("ridge_method", "augmented_lstsq"))).lower()
        if method not in {"augmented_lstsq", "normal_equation"}:
            raise ValueError("solver.ridge_method must be 'augmented_lstsq' or 'normal_equation'")
    if solver_type == "maxent":
        maxent_cfg = config.get("maxent", {})
        mode = str(loss.get("entropy_mode", maxent_cfg.get("entropy_mode", "positive_negative"))).lower()
        if mode not in {"abs", "positive_negative", "relative_uniform", "shell_balance"}:
            raise ValueError(
                "loss.entropy_mode must be one of 'abs', 'positive_negative', 'relative_uniform', 'shell_balance'"
            )
        if float(loss.get("entropy_weight", maxent_cfg.get("entropy_weight", 0.0))) < 0.0:
            raise ValueError("loss.entropy_weight must be non-negative")

    split = config.get("split", {})
    for key in ("train_r_range", "val_r_range", "test_high_r_range", "test_low_r_range"):
        if key in split and split[key] is not None:
            lo, hi = split[key]
            if float(lo) >= float(hi):
                raise ValueError(f"split.{key} must be [min, max] with min < max")
    ranges = [
        (key, split.get(key))
        for key in ("train_r_range", "val_r_range", "test_high_r_range", "test_low_r_range")
        if split.get(key) is not None
    ]
    for idx, (left_name, left_range) in enumerate(ranges):
        left_lo, left_hi = map(float, left_range)
        for right_name, right_range in ranges[idx + 1 :]:
            right_lo, right_hi = map(float, right_range)
            if max(left_lo, right_lo) < min(left_hi, right_hi):
                warnings.warn(
                    f"split.{left_name} overlaps split.{right_name}; verify this is intentional",
                    RuntimeWarning,
                    stacklevel=2,
                )


def get_dtype(config: dict) -> torch.dtype:
    name = str(config.get("dtype", "float32")).lower()
    if name in {"float64", "double"}:
        return torch.float64
    if name in {"float32", "single"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def get_device(config: dict) -> torch.device:
    return torch.device(config.get("device", "cpu"))
