"""Stage 3A MaxEnt Pareto sweep: data-error vs entropy over the entropy weight.

Runs the discrete MaxEnt solver for a list of entropy weights (``0.0`` is the pure
ridge baseline) and records, for each, the data-error metrics and the entropy /
source-concentration / shell-collapse diagnostics. This is the README step
"Compare data-error vs entropy Pareto curves" and is the deterministic gate before
any neural-density or probabilistic posterior experiments.

The report highlights the knee: the smallest entropy weight that clears the
deterministic failure modes (shell collapse and source concentration) without
degrading the relative acceleration RMSE beyond a tolerance factor of the baseline.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import yaml

from vesp.common.config import merge_defaults
from vesp.core.models import MultiShellDiscreteVESP
from vesp.training.train_discrete import run


FIELDNAMES = [
    "entropy_weight",
    "entropy_mode",
    "acceptability_status",
    "relative_acceleration_rmse",
    "potential_rmse",
    "low_altitude_acceleration_rmse",
    "low_to_high_error_ratio",
    "top_5pct_source_contribution",
    "dominant_shell_energy_fraction",
    "shell_collapse_flag",
    "source_entropy_nats",
    "max_possible_source_entropy_nats",
    "shell_energy_balance_entropy_nats",
    "relative_entropy_to_uniform",
    "sigma_l2",
]


def load_pareto_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict) or "base_config" not in cfg:
        raise ValueError("MaxEnt Pareto config must be a mapping with a 'base_config' key")
    return cfg


def _trial_configs(config: dict) -> list[tuple[float, str, dict]]:
    base = merge_defaults(deepcopy(config["base_config"]))
    base.setdefault("solver", {})
    base["solver"]["type"] = "maxent"
    sweep = config.get("maxent_pareto", {})
    weights = sweep.get("entropy_weights", [0.0, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2])
    mode = str(sweep.get("entropy_mode", base.get("loss", {}).get("entropy_mode", "positive_negative")))
    output_dir = sweep.get("output_dir") or base.get("output", {}).get("output_dir", "outputs/maxent_pareto")
    trials: list[tuple[float, str, dict]] = []
    for weight in weights:
        cfg = deepcopy(base)
        cfg.setdefault("loss", {})["entropy_weight"] = float(weight)
        cfg["loss"]["entropy_mode"] = mode
        name = f"ent_{float(weight):g}".replace(".", "p").replace("-", "m")
        cfg.setdefault("output", {})["run_name"] = name
        cfg["output"]["output_dir"] = str(output_dir)
        cfg["checkpoint_name"] = f"{name}.pt"
        trials.append((float(weight), mode, cfg))
    return trials


def _row(weight: float, mode: str, metrics: dict) -> dict:
    diagnostics = metrics.get("diagnostics", {})
    row = {name: "" for name in FIELDNAMES}
    row.update(
        {
            "entropy_weight": weight,
            "entropy_mode": mode,
            "acceptability_status": metrics.get("acceptability_status", ""),
            "relative_acceleration_rmse": metrics.get("relative_acceleration_rmse", ""),
            "potential_rmse": metrics.get("potential_rmse", ""),
            "low_altitude_acceleration_rmse": metrics.get("low_altitude_acceleration_rmse", ""),
            "low_to_high_error_ratio": metrics.get("low_to_high_error_ratio", ""),
            "top_5pct_source_contribution": diagnostics.get("top_5pct_source_contribution", ""),
            "dominant_shell_energy_fraction": diagnostics.get("dominant_shell_energy_fraction", ""),
            "shell_collapse_flag": diagnostics.get("shell_collapse_flag", ""),
            "source_entropy_nats": metrics.get("source_entropy_nats", ""),
            "max_possible_source_entropy_nats": metrics.get("max_possible_source_entropy_nats", ""),
            "shell_energy_balance_entropy_nats": metrics.get("shell_energy_balance_entropy_nats", ""),
            "relative_entropy_to_uniform": metrics.get("relative_entropy_to_uniform", ""),
            "sigma_l2": diagnostics.get("sigma_l2", ""),
        }
    )
    return row


def _to_float(value):
    if value in ("", None):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def _select_knee(rows: list[dict], config: dict) -> dict | None:
    """Smallest entropy weight that clears collapse/concentration within a data tolerance."""

    sweep = config.get("maxent_pareto", {})
    max_dominant = float(sweep.get("max_dominant_shell_energy_fraction", 0.90))
    max_top5 = float(sweep.get("max_top5_source_contribution", 0.40))
    tol_factor = float(sweep.get("data_tolerance_factor", 1.25))

    baseline = next((r for r in rows if _to_float(r["entropy_weight"]) == 0.0), None)
    baseline_rel = _to_float(baseline["relative_acceleration_rmse"]) if baseline else None

    candidates = []
    for row in rows:
        weight = _to_float(row["entropy_weight"])
        if weight is None or weight <= 0.0:
            continue
        dominant = _to_float(row["dominant_shell_energy_fraction"])
        top5 = _to_float(row["top_5pct_source_contribution"])
        rel = _to_float(row["relative_acceleration_rmse"])
        collapse = str(row["shell_collapse_flag"]) in ("True", "true")
        clears_collapse = (dominant is None or dominant <= max_dominant) and not collapse
        clears_conc = top5 is None or top5 <= max_top5
        within_data = (
            baseline_rel is None or rel is None or rel <= tol_factor * baseline_rel
        )
        if clears_collapse and clears_conc and within_data:
            candidates.append((weight, row))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _build_report(rows: list[dict], config: dict) -> str:
    lines = ["# MaxEnt Stage 3A Pareto Report", "", "Data-error vs entropy over the entropy weight.", ""]
    lines += [
        "| entropy_weight | status | rel_acc_rmse | low/high | top5 | dominant_shell | source_entropy | sigma_l2 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['entropy_weight']} | {row['acceptability_status']} | "
            f"{row['relative_acceleration_rmse']} | {row['low_to_high_error_ratio']} | "
            f"{row['top_5pct_source_contribution']} | {row['dominant_shell_energy_fraction']} | "
            f"{row['source_entropy_nats']} | {row['sigma_l2']} |"
        )
    knee = _select_knee(rows, config)
    lines += ["", "## Recommended Operating Point", ""]
    if knee is not None:
        lines.append(
            f"- entropy_weight=`{knee['entropy_weight']}` (mode=`{knee['entropy_mode']}`) clears shell collapse "
            f"and source concentration while keeping relative acceleration RMSE within tolerance "
            f"(rel_acc_rmse={knee['relative_acceleration_rmse']}, dominant_shell="
            f"{knee['dominant_shell_energy_fraction']}, top5={knee['top_5pct_source_contribution']}, "
            f"status={knee['acceptability_status']})."
        )
        lines.append(
            "- If this point also clears low-altitude error, deterministic Stage 3A is sufficient; otherwise it is "
            "the conservative baseline before Stage 3B neural density / Stage 3C probabilistic posterior experiments."
        )
    else:
        lines.append(
            "- No entropy weight cleared collapse/concentration within the data-error tolerance. Widen the entropy "
            "weight grid or revisit shell geometry/regularization before escalating to neural density."
        )
    return "\n".join(lines) + "\n"


def run_maxent_pareto(config: dict) -> Path:
    trials = _trial_configs(config)
    output_dir = Path(
        config.get("maxent_pareto", {}).get("output_dir")
        or config["base_config"].get("output", {}).get("output_dir", "outputs/maxent_pareto")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for weight, mode, trial in trials:
        print(f"===== maxent entropy_weight={weight} (mode={mode}) =====")
        model_type = trial.get("model", {}).get("type", "discrete")
        if model_type == "multishell":
            metrics = run(trial, model_cls=MultiShellDiscreteVESP)
        else:
            metrics = run(trial)
        rows.append(_row(weight, mode, metrics))

    csv_path = output_dir / "pareto_curve.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "maxent_pareto_report.md").write_text(_build_report(rows, config), encoding="utf-8")
    return csv_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a Stage 3A MaxEnt data-error vs entropy Pareto sweep.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    path = run_maxent_pareto(load_pareto_config(args.config))
    print(f"maxent_pareto_results: {path}")


if __name__ == "__main__":
    main()
