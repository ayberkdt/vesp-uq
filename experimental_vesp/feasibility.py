"""End-of-day feasibility suite for deciding whether Stage 3 is justified."""

from __future__ import annotations

import argparse
import csv
import time
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import yaml

from .models import MultiShellDiscreteVESP
from .train_discrete import run


def _base_config() -> dict:
    return {
        "seed": 42,
        "device": "cpu",
        "dtype": "float64",
        "output": {"output_dir": "outputs/feasibility"},
        "kernel": {"eps": 0.0, "acceleration_sign": 1.0, "source_chunk_size": 512},
        "solver": {"type": "ridge", "ridge_method": "augmented_lstsq", "lambda_l2": 1.0e-8, "column_normalize": True},
        "loss": {
            "use_potential": True,
            "use_acceleration": True,
            "normalize_targets": False,
            "potential_scale": "auto",
            "acceleration_scale": "auto",
            "lambda_potential": 0.2,
            "lambda_acceleration": 1.0,
            "lambda_l2": 1.0e-8,
            "lambda_moment": 1.0e-6,
            "lambda_dipole": 1.0,
            "shell_energy_weights": [],
        },
        "split": {"type": "random", "train_fraction": 0.8},
        "evaluation": {"batch_size": 2048, "n_altitude_bins": 6},
    }


def _deep_update(base: dict, update: dict) -> dict:
    out = deepcopy(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_suite_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError("feasibility config must be a YAML mapping")
    return cfg


def _scenario_configs(base: dict) -> list[dict]:
    scenarios: list[dict] = []

    def add(name: str, data: dict, model: dict, *, split: dict | None = None, loss_updates: dict | None = None) -> None:
        cfg = deepcopy(base)
        cfg["data"] = data
        cfg["model"] = model
        cfg["checkpoint_name"] = f"{name}.pt"
        cfg["output"] = {"run_name": name, "save_plots": False}
        if split is not None:
            cfg["split"] = split
        if loss_updates:
            cfg["loss"].update(loss_updates)
        scenarios.append(cfg)

    common_data = {
        "type": "synthetic",
        "path": None,
        "train_fraction": 0.8,
        "seed": 41,
        "synthetic_n_query": 384,
        "synthetic_query_radius_min": 1.05,
        "synthetic_query_radius_max": 1.60,
        "synthetic_noise_std": 0.0,
    }

    add(
        "same_family_single",
        {**common_data, "synthetic_truth_shell_radius": 0.86, "synthetic_n_truth_sources": 256},
        {"type": "discrete", "shell_alpha": 0.86, "n_source": 256},
    )
    add(
        "radius_mismatch_single",
        {**common_data, "synthetic_truth_shell_radius": 0.72, "synthetic_n_truth_sources": 128},
        {"type": "discrete", "shell_alpha": 0.86, "n_source": 256},
    )
    multi_truth_data = {
        **common_data,
        "synthetic_truth_shell_radii": [0.50, 0.78, 0.86],
        "synthetic_n_truth_sources": [96, 128, 96],
    }
    add(
        "multishell_truth_single",
        multi_truth_data,
        {"type": "discrete", "shell_alpha": 0.86, "n_source": 256},
    )
    add(
        "multishell_truth_multi",
        multi_truth_data,
        {"type": "multishell", "shell_alphas": [0.50, 0.78, 0.86], "n_sources_per_shell": [96, 128, 96]},
        loss_updates={"shell_energy_weights": [1.0e-9, 1.0e-9, 5.0e-9]},
    )
    add(
        "noisy_radius_mismatch_single",
        {**common_data, "synthetic_truth_shell_radius": 0.72, "synthetic_n_truth_sources": 128, "synthetic_noise_std": 1.0e-4},
        {"type": "discrete", "shell_alpha": 0.86, "n_source": 256},
        loss_updates={"lambda_l2": 1.0e-6, "lambda_moment": 1.0e-4},
    )
    add(
        "altitude_ood_single",
        {
            **common_data,
            "synthetic_n_query": 768,
            "synthetic_query_radius_min": 1.01,
            "synthetic_query_radius_max": 2.00,
            "synthetic_truth_shell_radius": 0.72,
            "synthetic_n_truth_sources": 128,
        },
        {"type": "discrete", "shell_alpha": 0.86, "n_source": 256},
        split={
            "type": "altitude_ood",
            "train_fraction": 0.8,
            "train_r_range": [1.05, 1.50],
            "val_r_range": [1.05, 1.50],
            "test_high_r_range": [1.50, 2.00],
            "test_low_r_range": [1.01, 1.05],
        },
    )
    return scenarios


def _metrics_row(config: dict, metrics: dict, runtime_sec: float) -> dict:
    diagnostics = metrics.get("diagnostics", {})
    model_cfg = config["model"]
    return {
        "scenario": config["output"]["run_name"],
        "model_type": model_cfg["type"],
        "shells": model_cfg.get("shell_alphas", [model_cfg.get("shell_alpha")]),
        "n_source": model_cfg.get("n_sources_per_shell", model_cfg.get("n_source")),
        "val_acc_rmse": metrics.get("acceleration_rmse"),
        "val_potential_rmse": metrics.get("potential_rmse"),
        "relative_acc_rmse": metrics.get("relative_acceleration_rmse"),
        "test_high_acc_rmse": metrics.get("test_high_acceleration_rmse", ""),
        "test_low_acc_rmse": metrics.get("test_low_acceleration_rmse", ""),
        "angle_deg_p95": metrics.get("angle_deg_p95"),
        "source_norm": diagnostics.get("sigma_l2"),
        "effective_source_count": diagnostics.get("effective_source_count"),
        "top_5pct_source_contribution": diagnostics.get("top_5pct_source_contribution"),
        "monopole_leakage": diagnostics.get("monopole_leakage"),
        "dipole_leakage": diagnostics.get("dipole_leakage"),
        "runtime_sec": runtime_sec,
    }


def _float(row: dict, key: str, default: float = float("nan")) -> float:
    value = row.get(key, default)
    if value == "":
        return default
    return float(value)


def _decision(rows: list[dict], thresholds: dict | None = None) -> tuple[str, list[str], list[str]]:
    thresholds = thresholds or {}
    by_name = {row["scenario"]: row for row in rows}
    positives: list[str] = []
    risks: list[str] = []

    same = _float(by_name.get("same_family_single", {}), "val_acc_rmse")
    same_family_tol = float(thresholds.get("same_family_tol", 2.0e-6))
    if same < same_family_tol:
        positives.append("Same-family recovery is excellent; kernel sign and ridge solver are trustworthy.")
    else:
        risks.append(f"Same-family recovery is not tight enough (`{same:.3e}`).")

    mismatch = _float(by_name.get("radius_mismatch_single", {}), "val_acc_rmse")
    if mismatch < float(thresholds.get("radius_mismatch_tol", 5.0e-3)):
        positives.append("Single-shell radius mismatch remains representable at useful error.")
    else:
        risks.append(f"Radius mismatch error is high (`{mismatch:.3e}`).")

    single_multi_truth = _float(by_name.get("multishell_truth_single", {}), "val_acc_rmse")
    multi_multi_truth = _float(by_name.get("multishell_truth_multi", {}), "val_acc_rmse")
    if multi_multi_truth < 0.9 * single_multi_truth:
        positives.append("Multi-shell improves on multi-shell truth; Stage 2 has a real use case.")
    else:
        risks.append("Multi-shell did not clearly beat single-shell on multi-shell truth.")

    noisy = _float(by_name.get("noisy_radius_mismatch_single", {}), "val_acc_rmse")
    if noisy < float(thresholds.get("noisy_tol", 2.0e-2)):
        positives.append("Noisy observation test remains stable under ridge/moment regularization.")
    else:
        risks.append(f"Noisy test error is high (`{noisy:.3e}`).")

    ood = by_name.get("altitude_ood_single", {})
    high = _float(ood, "test_high_acc_rmse")
    low = _float(ood, "test_low_acc_rmse")
    val = _float(ood, "val_acc_rmse")
    if high < float(thresholds.get("high_ood_factor", 5.0)) * val:
        positives.append("High-altitude OOD error is controlled relative to validation.")
    else:
        risks.append("High-altitude OOD error grows too much relative to validation.")
    if low < float(thresholds.get("low_ood_factor", 25.0)) * val:
        positives.append("Low-altitude OOD error is within the current watch band.")
    else:
        risks.append("Low-altitude OOD error is the dominant instability risk.")

    max_top5 = max(_float(row, "top_5pct_source_contribution", 0.0) for row in rows)
    if max_top5 < float(thresholds.get("max_top5_source_contribution", 0.45)):
        positives.append("Source concentration is not extreme in the feasibility suite.")
    else:
        risks.append("Some runs localize too much source mass in the top 5%; MaxEnt regularization may be useful.")

    if len(risks) <= 2 and same < same_family_tol:
        decision = "GO_STAGE3_PREP"
    elif same < same_family_tol:
        decision = "CONDITIONAL"
    else:
        decision = "REDESIGN"
    return decision, positives, risks


def run_feasibility_suite(config: dict | None = None) -> Path:
    base = _base_config()
    if config:
        base = _deep_update(base, config.get("base", {}))
    output_dir = Path(base.get("output", {}).get("output_dir", "outputs/feasibility"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for trial in _scenario_configs(base):
        model_type = trial["model"]["type"]
        trial.setdefault("output", {})["output_dir"] = str(output_dir)
        print(f"===== feasibility: {trial['output']['run_name']} =====")
        start = time.perf_counter()
        if model_type == "multishell":
            metrics = run(trial, model_cls=MultiShellDiscreteVESP)
        else:
            metrics = run(trial)
        rows.append(_metrics_row(trial, metrics, time.perf_counter() - start))

    results_path = output_dir / "feasibility_results.csv"
    with results_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    decision, positives, risks = _decision(rows, (config or {}).get("feasibility", {}))
    report_lines = [
        "# MaxEnt-VESP Feasibility Report",
        "",
        f"Decision: **{decision}**",
        "",
        "## Positive Signals",
        "",
        *[f"- {item}" for item in positives],
        "",
        "## Risks",
        "",
        *[f"- {item}" for item in risks],
        "",
        "## Scenario Results",
        "",
        "| Scenario | Model | Val Acc RMSE | High OOD | Low OOD | Eff. Sources | Top 5% |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['scenario']} | {row['model_type']} | {row['val_acc_rmse']:.3e} | "
            f"{row['test_high_acc_rmse'] or '-'} | {row['test_low_acc_rmse'] or '-'} | "
            f"{row['effective_source_count']:.1f} | {row['top_5pct_source_contribution']:.2%} |"
        )
    report_lines.extend(
        [
            "",
            "## MaxEnt Readiness",
            "",
            "- Start with deterministic entropy regularization over solved `sigma`, not a full posterior.",
            "- Prioritize signed positive/negative entropy and shell-wise entropy diagnostics.",
            "- Keep ridge/Tikhonov as the baseline comparator for every MaxEnt experiment.",
            "- Do not claim uncertainty-aware inference until probabilistic posterior calibration is tested.",
            "",
        ]
    )
    (output_dir / "maxent_readiness_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return results_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_suite_config(args.config) if args.config else None
    path = run_feasibility_suite(cfg)
    print(f"feasibility_results: {path}")


if __name__ == "__main__":
    main()
