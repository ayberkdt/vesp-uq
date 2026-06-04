"""Central configuration loading, defaults, and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

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
    },
    "split": {
        "type": "random",
        "train_fraction": 0.8,
        "radius_units": "normalized",
    },
    "evaluation": {
        "batch_size": 4096,
        "n_altitude_bins": 6,
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

    model = config.get("model", {})
    model_type = model.get("type")
    if model_type not in {"discrete", "multishell"}:
        raise ValueError("model.type must be 'discrete' or 'multishell'")
    if model_type == "discrete":
        if "shell_alpha" not in model or "n_source" not in model:
            raise ValueError("discrete model requires model.shell_alpha and model.n_source")
    if model_type == "multishell":
        shells = model.get("shell_alphas")
        counts = model.get("n_sources_per_shell")
        if not shells or counts is None:
            raise ValueError("multishell model requires model.shell_alphas and model.n_sources_per_shell")
        if isinstance(counts, list) and len(counts) != len(shells):
            raise ValueError("model.n_sources_per_shell must match model.shell_alphas length")

    loss = config.get("loss", {})
    if not bool(loss.get("use_potential", True)) and not bool(loss.get("use_acceleration", True)):
        raise ValueError("at least one of loss.use_potential or loss.use_acceleration must be true")

    split = config.get("split", {})
    for key in ("train_r_range", "val_r_range", "test_high_r_range", "test_low_r_range"):
        if key in split and split[key] is not None:
            lo, hi = split[key]
            if float(lo) >= float(hi):
                raise ValueError(f"split.{key} must be [min, max] with min < max")


def get_dtype(config: dict) -> torch.dtype:
    name = str(config.get("dtype", "float32")).lower()
    if name in {"float64", "double"}:
        return torch.float64
    if name in {"float32", "single"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def get_device(config: dict) -> torch.device:
    return torch.device(config.get("device", "cpu"))
