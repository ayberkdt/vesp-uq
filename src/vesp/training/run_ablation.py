"""Run Stage 1-2 ablation sweeps and collect comparison tables.

Two config schemas are supported:

* **New schema** (recommended) — a ``base_config`` plus ``quick`` / ``full`` trial
  specs selected by ``ablation.mode`` (overridable with ``--mode``). A trial spec is
  either an explicit list of ``{name, "<dotted.path>": value, ...}`` overrides or a
  ``{grid: {"<dotted.path>": [values...]}}`` cartesian product.
* **Legacy schema** — top-level ``ablation`` with ``single_shell_alphas`` /
  ``multishell_sets`` (kept for the synthetic stress configs).

Each run is screened with ``classify_run_acceptability`` and the summary highlights the
best non-collapsed candidates and the runs rejected by collapse / sigma blow-up.
"""

from __future__ import annotations

import argparse
import csv
import time
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Iterable

from vesp.core.models import MultiShellDiscreteVESP
from vesp.training.train_discrete import load_config, run


def _set_nested(config: dict, path: str, value) -> None:
    node = config
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _fmt_value(value) -> str:
    if isinstance(value, (list, tuple)):
        return "-".join(_fmt_value(v) for v in value)
    if isinstance(value, float):
        return f"{value:g}".replace(".", "p").replace("-", "m")
    return str(value).replace(".", "p")


def _apply_overrides(base: dict, overrides: dict, *, default_name: str) -> dict:
    cfg = deepcopy(base)
    name = overrides.get("name", default_name)
    for key, value in overrides.items():
        if key == "name":
            continue
        _set_nested(cfg, key, value)
    cfg.setdefault("output", {})["run_name"] = name
    cfg["checkpoint_name"] = f"{name}.pt"
    return cfg


def _expand_spec(spec, base: dict) -> list[dict]:
    """Turn a quick/full spec (explicit list or grid) into concrete trial configs."""

    trials: list[dict] = []
    if isinstance(spec, dict) and "grid" in spec:
        grid = spec["grid"]
        keys = list(grid.keys())
        value_lists = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
        for combo in product(*value_lists):
            overrides = {k: v for k, v in zip(keys, combo)}
            leaf_name = "__".join(f"{k.split('.')[-1]}_{_fmt_value(v)}" for k, v in zip(keys, combo))
            trials.append(_apply_overrides(base, overrides, default_name=leaf_name))
        return trials
    if isinstance(spec, list):
        for idx, overrides in enumerate(spec):
            trials.append(_apply_overrides(base, overrides, default_name=f"trial_{idx}"))
        return trials
    raise ValueError("ablation spec must be a list of trials or a {grid: {...}} mapping")


def _legacy_trial_configs(config: dict) -> list[dict]:
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


def _is_new_schema(config: dict) -> bool:
    return "base_config" in config and ("quick" in config or "full" in config)


def _trial_configs(config: dict, *, mode: str | None = None) -> list[dict]:
    if not _is_new_schema(config):
        return _legacy_trial_configs(config)
    ablation_meta = config.get("ablation", {}) if isinstance(config.get("ablation"), dict) else {}
    selected = (mode or ablation_meta.get("mode", "quick")).lower()
    if selected not in {"quick", "full"}:
        raise ValueError("ablation mode must be 'quick' or 'full'")
    spec = config.get(selected)
    if spec is None:
        # fall back to the other mode if only one is provided
        other = "full" if selected == "quick" else "quick"
        spec = config.get(other)
        if spec is None:
            raise ValueError("ablation config must define a 'quick' or 'full' trial spec")
    from vesp.common.config import merge_defaults

    base = merge_defaults(deepcopy(config["base_config"]))
    return _expand_spec(spec, base)


FIELDNAMES = [
    "run_name",
    "model_type",
    "shell_alphas",
    "n_source_total",
    "lambda_l2",
    "lambda_moment",
    "shell_energy_weights",
    "acceleration_rmse",
    "relative_acceleration_rmse",
    "potential_rmse",
    "low_altitude_acceleration_rmse",
    "high_altitude_acceleration_rmse",
    "low_to_high_error_ratio",
    "test_high_acceleration_rmse",
    "test_low_acceleration_rmse",
    "sigma_l2",
    "sigma_norm_warning",
    "effective_source_count",
    "top_1pct_source_contribution",
    "top_5pct_source_contribution",
    "relative_monopole_leakage",
    "relative_dipole_leakage",
    "monopole_leakage",
    "dipole_leakage",
    "dominant_shell_alpha",
    "dominant_shell_energy_fraction",
    "shell_energy_entropy",
    "shell_collapse_flag",
    "shell_cancellation_ratio",
    "acceptability_status",
    "runtime_sec",
    "error",
]


def _n_source_total(model_cfg: dict) -> int | str:
    counts = model_cfg.get("n_sources_per_shell")
    if isinstance(counts, list):
        return int(sum(counts))
    if counts is not None:
        return int(counts) * len(model_cfg.get("shell_alphas", [1]))
    return model_cfg.get("n_source", "")


def _row_from_metrics(config: dict, metrics: dict, runtime_sec: float) -> dict:
    model_cfg = config.get("model", {})
    loss_cfg = config.get("loss", {})
    diagnostics = metrics.get("diagnostics", {})
    shell_alphas = model_cfg.get("shell_alphas", [model_cfg.get("shell_alpha")])
    row = {name: "" for name in FIELDNAMES}
    row.update(
        {
            "run_name": config.get("output", {}).get("run_name", config.get("checkpoint_name", "run")),
            "model_type": model_cfg.get("type", "discrete"),
            "shell_alphas": shell_alphas,
            "n_source_total": _n_source_total(model_cfg),
            "lambda_l2": loss_cfg.get("lambda_l2", config.get("solver", {}).get("lambda_l2", "")),
            "lambda_moment": loss_cfg.get("lambda_moment", ""),
            "shell_energy_weights": loss_cfg.get("shell_energy_weights", ""),
            "acceleration_rmse": metrics.get("acceleration_rmse", ""),
            "relative_acceleration_rmse": metrics.get("relative_acceleration_rmse", ""),
            "potential_rmse": metrics.get("potential_rmse", ""),
            "low_altitude_acceleration_rmse": metrics.get("low_altitude_acceleration_rmse", ""),
            "high_altitude_acceleration_rmse": metrics.get("high_altitude_acceleration_rmse", ""),
            "low_to_high_error_ratio": metrics.get("low_to_high_error_ratio", ""),
            "test_high_acceleration_rmse": metrics.get("test_high_acceleration_rmse", ""),
            "test_low_acceleration_rmse": metrics.get("test_low_acceleration_rmse", ""),
            "sigma_l2": diagnostics.get("sigma_l2", ""),
            "sigma_norm_warning": diagnostics.get("sigma_norm_warning", ""),
            "effective_source_count": diagnostics.get("effective_source_count", ""),
            "top_1pct_source_contribution": diagnostics.get("top_1pct_source_contribution", ""),
            "top_5pct_source_contribution": diagnostics.get("top_5pct_source_contribution", ""),
            "relative_monopole_leakage": diagnostics.get("relative_monopole_leakage", ""),
            "relative_dipole_leakage": diagnostics.get("relative_dipole_leakage", ""),
            "monopole_leakage": diagnostics.get("monopole_leakage", ""),
            "dipole_leakage": diagnostics.get("dipole_leakage", ""),
            "dominant_shell_alpha": diagnostics.get("dominant_shell_alpha", ""),
            "dominant_shell_energy_fraction": diagnostics.get("dominant_shell_energy_fraction", ""),
            "shell_energy_entropy": diagnostics.get("shell_energy_entropy", ""),
            "shell_collapse_flag": diagnostics.get("shell_collapse_flag", ""),
            "shell_cancellation_ratio": diagnostics.get("shell_cancellation_ratio", ""),
            "acceptability_status": metrics.get("acceptability_status", ""),
            "runtime_sec": runtime_sec,
            "error": "",
        }
    )
    return row


def _failed_row(config: dict, error: Exception, runtime_sec: float) -> dict:
    model_cfg = config.get("model", {})
    row = {name: "" for name in FIELDNAMES}
    row.update(
        {
            "run_name": config.get("output", {}).get("run_name", config.get("checkpoint_name", "run")),
            "model_type": model_cfg.get("type", "discrete"),
            "shell_alphas": model_cfg.get("shell_alphas", [model_cfg.get("shell_alpha")]),
            "n_source_total": _n_source_total(model_cfg),
            "acceptability_status": "FAILED",
            "runtime_sec": runtime_sec,
            "error": str(error)[:300],
        }
    )
    return row


def _to_float(value) -> float | None:
    if value in ("", None):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def _best_row(rows: list[dict], column: str, *, predicate=None) -> dict | None:
    candidates = []
    for row in rows:
        if predicate is not None and not predicate(row):
            continue
        value = _to_float(row.get(column))
        if value is not None:
            candidates.append((value, row))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _build_summary(rows: list[dict]) -> str:
    succeeded = [r for r in rows if r.get("acceptability_status") not in ("", "FAILED")]
    failed = [r for r in rows if r.get("acceptability_status") == "FAILED"]

    def _line(row: dict | None, *cols: str) -> str:
        if row is None:
            return "- none"
        bits = ", ".join(f"{c}={row.get(c)}" for c in cols)
        return f"- `{row['run_name']}` ({bits})"

    is_multishell = lambda r: str(r.get("model_type")) == "multishell"
    not_collapsed = lambda r: str(r.get("shell_collapse_flag")) not in ("True", "true")

    best_rel = _best_row(succeeded, "relative_acceleration_rmse")
    best_low = _best_row(
        succeeded,
        "low_altitude_acceleration_rmse",
    ) or _best_row(succeeded, "test_low_acceleration_rmse")
    best_noncollapse = _best_row(
        succeeded, "relative_acceleration_rmse", predicate=lambda r: is_multishell(r) and not_collapsed(r)
    )
    collapsed = [r for r in succeeded if str(r.get("shell_collapse_flag")) in ("True", "true")]
    sigma_flagged = [r for r in succeeded if str(r.get("sigma_norm_warning")) in ("True", "true")]

    lines = ["# Ablation Summary", "", f"Total runs: {len(rows)} (succeeded {len(succeeded)}, failed {len(failed)})", ""]
    lines += ["## Best by Relative Acceleration RMSE", "", _line(best_rel, "relative_acceleration_rmse", "acceptability_status"), ""]
    lines += ["## Best by Low-Altitude RMSE", "", _line(best_low, "low_altitude_acceleration_rmse", "low_to_high_error_ratio", "acceptability_status"), ""]
    lines += ["## Best Non-Collapsed Multi-Shell Run", "", _line(best_noncollapse, "relative_acceleration_rmse", "dominant_shell_energy_fraction", "sigma_l2"), ""]
    lines += ["## Runs Rejected by Shell Collapse", ""]
    lines += [f"- `{r['run_name']}` (dominant_fraction={r.get('dominant_shell_energy_fraction')})" for r in collapsed] or ["- none"]
    lines += ["", "## Runs Rejected by Sigma Norm", ""]
    lines += [f"- `{r['run_name']}` (sigma_l2={r.get('sigma_l2')})" for r in sigma_flagged] or ["- none"]
    if failed:
        lines += ["", "## Failed Runs", ""]
        lines += [f"- `{r['run_name']}`: {r.get('error')}" for r in failed]

    lines += ["", "## Recommendation", ""]
    if best_noncollapse is not None:
        ratio = _to_float(best_noncollapse.get("low_to_high_error_ratio"))
        rel = _to_float(best_noncollapse.get("relative_acceleration_rmse"))
        if (ratio is None or ratio <= 5.0) and (rel is None or rel <= 0.75) and best_noncollapse.get("acceptability_status") == "GOOD":
            lines.append(
                "- A non-collapsed multi-shell candidate passes screening. Continue deterministic "
                "ablation to confirm stability before considering Stage 3."
            )
        else:
            lines.append(
                "- The best non-collapsed deterministic run still fails low-altitude error and/or source "
                "concentration screening. If further regularization / shell-set / low-altitude ablation does "
                "not fix this, proceed to Stage 3A: Discrete MaxEnt regularization. Do not jump directly to "
                "neural density."
            )
    else:
        lines.append(
            "- No non-collapsed multi-shell run was found. Tighten shell-energy / l2 regularization first; "
            "if collapse persists, Stage 3A Discrete MaxEnt is justified. Do not jump directly to neural density."
        )
    return "\n".join(lines) + "\n"


def run_ablation(config: dict, *, mode: str | None = None) -> Path:
    ablation_meta = config.get("ablation", {}) if isinstance(config.get("ablation"), dict) else {}
    output_dir = Path(
        ablation_meta.get("output_dir")
        or config.get("ablation_output_dir")
        or config.get("output", {}).get("output_dir", "outputs/ablation")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for trial in _trial_configs(config, mode=mode):
        trial.setdefault("output_dir", str(output_dir))
        trial.setdefault("output", {})["output_dir"] = str(output_dir)
        model_type = trial.get("model", {}).get("type", "discrete")
        run_name = trial.get("output", {}).get("run_name", trial.get("checkpoint_name"))
        print(f"===== {run_name} =====")
        start = time.perf_counter()
        try:
            if model_type == "multishell":
                metrics = run(trial, model_cls=MultiShellDiscreteVESP)
            else:
                metrics = run(trial)
            rows.append(_row_from_metrics(trial, metrics, time.perf_counter() - start))
        except Exception as exc:  # noqa: BLE001 - record and continue the sweep
            print(f"FAILED {run_name}: {exc}")
            rows.append(_failed_row(trial, exc, time.perf_counter() - start))

    csv_path = output_dir / "ablation_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "ablation_summary.md").write_text(_build_summary(rows), encoding="utf-8")
    return csv_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic_stress_multishell.yaml")
    parser.add_argument("--mode", choices=["quick", "full"], default=None, help="override ablation.mode")
    args = parser.parse_args(argv)
    path = run_ablation(load_config(args.config), mode=args.mode)
    print(f"ablation_results: {path}")


if __name__ == "__main__":
    main()
