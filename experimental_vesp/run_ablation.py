"""Run Stage 1-2 ablation sweeps and collect comparison tables."""

from __future__ import annotations

import argparse
import csv
import time
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import yaml

from .models import MultiShellDiscreteVESP
from .train_discrete import load_config, run


def _set_nested(config: dict, path: str, value) -> None:
    node = config
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _trial_configs(config: dict) -> list[dict]:
    if "trials" in config:
        trials = []
        base = deepcopy(config.get("base_config", {}))
        for trial in config["trials"]:
            cfg = deepcopy(base)
            for key, value in trial.items():
                if key == "name":
                    continue
                _set_nested(cfg, key, value)
            cfg.setdefault("output", {})["run_name"] = trial.get("name", "ablation_trial")
            cfg["checkpoint_name"] = f"{cfg['output']['run_name']}.pt"
            trials.append(cfg)
        return trials

    base = deepcopy(config)
    base.pop("ablation", None)
    trials = []
    ablation = config.get("ablation", {})
    for alpha in ablation.get("single_shell_alphas", [0.70, 0.80, 0.90, 0.95]):
        for n_source in ablation.get("n_sources", [256]):
            cfg = deepcopy(base)
            cfg["model"] = {"type": "discrete", "shell_alpha": alpha, "n_source": n_source}
            cfg.setdefault("output", {})["run_name"] = f"single_a{alpha:.2f}_n{n_source}".replace(".", "")
            cfg["checkpoint_name"] = f"{cfg['output']['run_name']}.pt"
            trials.append(cfg)
    for shells in ablation.get("multishell_sets", [[0.50, 0.80, 0.95]]):
        cfg = deepcopy(base)
        cfg["model"] = {
            "type": "multishell",
            "shell_alphas": shells,
            "n_sources_per_shell": [ablation.get("multi_shell_n_source", 256)] * len(shells),
        }
        cfg.setdefault("output", {})["run_name"] = "multi_" + "_".join(str(v).replace(".", "") for v in shells)
        cfg["checkpoint_name"] = f"{cfg['output']['run_name']}.pt"
        trials.append(cfg)
    return trials


def _row_from_metrics(config: dict, metrics: dict, runtime_sec: float) -> dict:
    model_cfg = config.get("model", {})
    diagnostics = metrics.get("diagnostics", {})
    shell_radii = model_cfg.get("shell_alphas", [model_cfg.get("shell_alpha", None)])
    n_source = model_cfg.get("n_sources_per_shell", model_cfg.get("n_source", None))
    return {
        "run_name": config.get("output", {}).get("run_name", config.get("checkpoint_name", "run")),
        "model_type": model_cfg.get("type", "discrete"),
        "shells": shell_radii,
        "n_source": n_source,
        "lambda_l2": config.get("solver", {}).get("lambda_l2", config.get("loss", {}).get("lambda_l2", "")),
        "train_acc_rmse": "",
        "val_acc_rmse": metrics.get("acceleration_rmse", ""),
        "test_high_acc_rmse": metrics.get("test_high_acceleration_rmse", ""),
        "test_low_acc_rmse": metrics.get("test_low_acceleration_rmse", ""),
        "potential_rmse": metrics.get("potential_rmse", ""),
        "radial_rmse": metrics.get("radial_rmse", metrics.get("radial_acceleration_rmse", "")),
        "cross_radial_rmse": metrics.get("cross_radial_rmse", metrics.get("cross_radial_acceleration_rmse", "")),
        "source_norm": diagnostics.get("sigma_l2", ""),
        "effective_source_count": diagnostics.get("effective_source_count", ""),
        "monopole_leakage": diagnostics.get("monopole_leakage", ""),
        "dipole_leakage": diagnostics.get("dipole_leakage", ""),
        "runtime_sec": runtime_sec,
        "gpu_memory_mb": metrics.get("cuda_max_memory_mb", ""),
    }


def run_ablation(config: dict) -> Path:
    output_dir = Path(config.get("ablation_output_dir", config.get("output_dir", "outputs/ablation")))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for trial in _trial_configs(config):
        trial.setdefault("output_dir", str(output_dir))
        model_type = trial.get("model", {}).get("type", "discrete")
        print(f"===== {trial.get('output', {}).get('run_name', trial.get('checkpoint_name'))} =====")
        start = time.perf_counter()
        if model_type == "multishell":
            metrics = run(trial, model_cls=MultiShellDiscreteVESP)
        else:
            metrics = run(trial)
        runtime = time.perf_counter() - start
        rows.append(_row_from_metrics(trial, metrics, runtime))

    csv_path = output_dir / "ablation_results.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best = min(rows, key=lambda row: float(row["val_acc_rmse"])) if rows else None
    summary = ["# Ablation Summary", ""]
    if best:
        summary.append(f"Best validation acceleration RMSE: `{best['run_name']}` = `{best['val_acc_rmse']}`")
    summary.extend(
        [
            "",
            "## Continue / Redesign Criteria",
            "",
            "- Continue if synthetic recovery succeeds, OOD errors stay controlled, and source/moment diagnostics stay bounded.",
            "- Redesign if same-family recovery fails, multi-shell never helps, or low-altitude instability dominates.",
        ]
    )
    (output_dir / "ablation_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return csv_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic_stress_multishell.yaml")
    args = parser.parse_args(argv)
    path = run_ablation(load_config(args.config))
    print(f"ablation_results: {path}")


if __name__ == "__main__":
    main()
