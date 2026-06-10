"""End-of-day feasibility suite for deciding whether Stage 3 is justified."""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path

import yaml

from vesp.core.models import MultiShellDiscreteVESP
from vesp.feasibility.training.train_discrete import run


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
    loss_cfg = config.get("loss", {})
    kernel_cfg = config.get("kernel", {})
    return {
        "scenario": config["output"]["run_name"],
        "model_type": model_cfg["type"],
        "shells": model_cfg.get("shell_alphas", [model_cfg.get("shell_alpha")]),
        "n_source": model_cfg.get("n_sources_per_shell", model_cfg.get("n_source")),
        "target_normalization": loss_cfg.get("resolved_normalize_targets", loss_cfg.get("normalize_targets", False)),
        "potential_scale": loss_cfg.get("resolved_potential_scale", 1.0),
        "acceleration_scale": loss_cfg.get("resolved_acceleration_scale", 1.0),
        "acceleration_sign": kernel_cfg.get("acceleration_sign", 1.0),
        "source_weight_mode": model_cfg.get("weight_mode", "surface_area"),
        "kernel_softening": kernel_cfg.get("softening", kernel_cfg.get("eps", 0.0)),
        "val_acc_rmse": metrics.get("acceleration_rmse"),
        "val_potential_rmse": metrics.get("potential_rmse"),
        "relative_acc_rmse": metrics.get("relative_acceleration_rmse"),
        "test_high_acc_rmse": metrics.get("test_high_acceleration_rmse", ""),
        "test_low_acc_rmse": metrics.get("test_low_acceleration_rmse", ""),
        "angle_deg_p95": metrics.get("angle_deg_p95"),
        "source_norm": diagnostics.get("sigma_l2"),
        "effective_source_count": diagnostics.get("effective_source_count"),
        "top_5pct_source_contribution": diagnostics.get("top_5pct_source_contribution"),
        "relative_monopole_leakage": diagnostics.get("relative_monopole_leakage"),
        "relative_dipole_leakage": diagnostics.get("relative_dipole_leakage"),
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


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_float(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def _best(rows: list[dict], key: str, *, predicate=None) -> dict | None:
    candidates = []
    for row in rows:
        if predicate is not None and not predicate(row):
            continue
        value = _row_float(row, key)
        if value is not None:
            candidates.append((value, row))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def real_lunar_diagnostics_section(search_root: str | Path = "outputs") -> list[str]:
    """Build the Real Lunar Stage 1-2 section from any ablation CSVs that exist.

    Returns markdown lines. If no ablation results are present yet, a short notice is
    returned instead so the readiness report is still self-explanatory.
    """

    root = Path(search_root)
    csv_paths = sorted(root.glob("ablation_real_lunar_*/ablation_results.csv"))
    lines = ["", "## Real Lunar Stage 1-2 Diagnostics", ""]
    if not csv_paths:
        lines += [
            "- No real lunar ablation results found yet. Run the regularization, shell-set, and",
            "  low-altitude weighting ablations to populate this section.",
        ]
        return lines

    rows: list[dict] = []
    for path in csv_paths:
        try:
            rows.extend(_read_csv_rows(path))
        except OSError:
            continue
    succeeded = [r for r in rows if r.get("acceptability_status") not in ("", "FAILED")]

    is_multishell = lambda r: str(r.get("model_type")) == "multishell"
    not_collapsed = lambda r: str(r.get("shell_collapse_flag")) not in ("True", "true")

    best_rel = _best(succeeded, "relative_acceleration_rmse")
    best_low = _best(succeeded, "low_altitude_acceleration_rmse")
    best_noncollapse = _best(succeeded, "relative_acceleration_rmse", predicate=lambda r: is_multishell(r) and not_collapsed(r))
    best_single = _best(succeeded, "relative_acceleration_rmse", predicate=lambda r: str(r.get("model_type")) == "discrete")
    collapsed = [r for r in succeeded if str(r.get("shell_collapse_flag")) in ("True", "true")]

    def _fmt(row: dict | None, *cols: str) -> str:
        if row is None:
            return "none"
        return f"`{row.get('run_name')}` (" + ", ".join(f"{c}={row.get(c)}" for c in cols) + ")"

    lines += [
        f"- Ablation tables scanned: {len(csv_paths)} ({sum(len(_read_csv_rows(p)) for p in csv_paths)} runs).",
        f"- Best single-shell run: {_fmt(best_single, 'relative_acceleration_rmse', 'low_to_high_error_ratio')}",
        f"- Best multi-shell run: {_fmt(best_rel if best_rel and is_multishell(best_rel) else best_noncollapse, 'relative_acceleration_rmse', 'dominant_shell_energy_fraction')}",
        f"- Low-altitude bottleneck (best low-altitude RMSE): {_fmt(best_low, 'low_altitude_acceleration_rmse', 'low_to_high_error_ratio')}",
        f"- Shell collapse status: {len(collapsed)} of {len(succeeded)} runs flagged as collapsed.",
        f"- Best non-collapsed multi-shell run: {_fmt(best_noncollapse, 'relative_acceleration_rmse', 'sigma_l2', 'acceptability_status')}",
        "",
        "Recommendation:",
    ]
    if best_noncollapse is not None and best_noncollapse.get("acceptability_status") == "GOOD":
        lines.append("- A non-collapsed multi-shell candidate passes screening: continue deterministic ablation.")
    elif best_noncollapse is not None:
        lines.append(
            "- The best non-collapsed deterministic run still fails low-altitude / concentration screening: "
            "proceed to Stage 3A Discrete MaxEnt regularization. Do not proceed to NN yet."
        )
    else:
        lines.append(
            "- No non-collapsed multi-shell run found: tighten regularization first; if collapse persists, "
            "Stage 3A Discrete MaxEnt is justified. Do not proceed to NN yet."
        )
    return lines


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
        "| Scenario | Model | Norm | Pot Scale | Acc Scale | Sign | Softening | Val Acc RMSE | High OOD | Low OOD | Eff. Sources | Top 5% |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['scenario']} | {row['model_type']} | {row['target_normalization']} | "
            f"{float(row['potential_scale']):.3e} | {float(row['acceleration_scale']):.3e} | "
            f"{float(row['acceleration_sign']):.1f} | {float(row['kernel_softening']):.1e} | "
            f"{row['val_acc_rmse']:.3e} | "
            f"{row['test_high_acc_rmse'] or '-'} | {row['test_low_acc_rmse'] or '-'} | "
            f"{row['effective_source_count']:.1f} | {row['top_5pct_source_contribution']:.2%} |"
        )
    report_lines.extend(real_lunar_diagnostics_section(output_dir.parent if output_dir.name == "feasibility" else "outputs"))
    report_lines.extend(
        [
            "",
            "## MaxEnt Readiness",
            "",
            "- Do not start Stage 3 MaxEnt until deterministic Stage 1-2 checks pass on hard synthetic and small real residual datasets.",
            "- Keep ridge/Tikhonov as the baseline comparator for every future MaxEnt experiment.",
            "- Do not claim uncertainty-aware inference until probabilistic posterior calibration exists.",
            "",
        ]
    )
    (output_dir / "maxent_readiness_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return results_path


def main(argv: Iterable[str] | None = None) -> None:
    from vesp.common.deprecation import warn_superseded

    warn_superseded(
        "vesp.feasibility.training.feasibility",
        "python scripts/run_experiment_suite.py --suite synthetic  (E0-E4) "
        "or --experiment E0",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_suite_config(args.config) if args.config else None
    path = run_feasibility_suite(cfg)
    print(f"feasibility_results: {path}")


if __name__ == "__main__":
    main()
