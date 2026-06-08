"""Flatten run metrics into the standard summary row and write suite artifacts.

The summary row is the single source of truth for cross-experiment comparison. It
pulls scientific signals from both the top-level metrics and the nested
``diagnostics`` so that ridge baselines and MaxEnt runs are directly comparable along
the L2 / entropy sweeps. ``acceptability_status`` is included only as a screening
flag — never read it as a scientific verdict (see ``docs/SCIENTIFIC_CLAIMS.md``).
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

# Columns in report order. Provenance columns are appended after the scientific ones.
SCIENTIFIC_COLUMNS = [
    "run_name",
    "solver",
    "model_type",
    "shell_alphas",
    "n_sources_total",
    "lambda_l2",
    "lambda_moment",
    "entropy_weight",
    "entropy_mode",
    "potential_rmse",
    "acceleration_rmse",
    "relative_acceleration_rmse",
    "radial_rmse",
    "cross_radial_rmse",
    "angle_deg_p95",
    "low_altitude_acceleration_rmse",
    "mid_altitude_acceleration_rmse",
    "high_altitude_acceleration_rmse",
    "low_to_high_error_ratio",
    "test_low_acceleration_rmse",
    "test_high_acceleration_rmse",
    "source_entropy_nats",
    "positive_negative_entropy_nats",
    "shell_energy_balance_entropy_nats",
    "dominant_shell_energy_fraction",
    "shell_cancellation_ratio",
    "top_5pct_source_contribution",
    "effective_source_count",
    "sigma_l2",
    "relative_monopole_leakage",
    "relative_dipole_leakage",
    "acceptability_status",
    "acceptability_reasons",
]

PROVENANCE_COLUMNS = [
    "selected_lambda_l2",
    "experiment",
    "config_path",
    "git_commit",
    "timestamp",
    "runtime_sec",
    "inference_seconds_per_batch",
    "error",
]

SUMMARY_COLUMNS = SCIENTIFIC_COLUMNS + PROVENANCE_COLUMNS

# Columns kept for the Pareto / trade-off artifact and plots.
PARETO_COLUMNS = [
    "run_name",
    "experiment",
    "solver",
    "lambda_l2",
    "entropy_weight",
    "entropy_mode",
    "relative_acceleration_rmse",
    "acceleration_rmse",
    "potential_rmse",
    "source_entropy_nats",
    "shell_energy_balance_entropy_nats",
    "shell_cancellation_ratio",
    "dominant_shell_energy_fraction",
    "top_5pct_source_contribution",
    "effective_source_count",
    "sigma_l2",
    "acceptability_status",
]


def _shell_alphas(model_cfg: dict) -> list:
    if model_cfg.get("type") == "multishell":
        return list(model_cfg.get("shell_alphas", model_cfg.get("shell_radii", [])))
    alpha = model_cfg.get("shell_alpha")
    return [alpha] if alpha is not None else []


def _n_sources_total(model_cfg: dict) -> int | str:
    counts = model_cfg.get("n_sources_per_shell")
    if isinstance(counts, list):
        return int(sum(counts))
    if counts is not None:
        return int(counts) * len(model_cfg.get("shell_alphas", [1]))
    n_source = model_cfg.get("n_source")
    return int(n_source) if n_source is not None else ""


def summary_row(
    run_name: str,
    config: dict,
    metrics: dict,
    *,
    experiment: str | None = None,
    config_path: str | None = None,
    git_commit: str | None = None,
    timestamp: str | None = None,
    runtime_sec: float | None = None,
    error: str = "",
) -> dict:
    """Build one standardized summary row from a run's config + metrics.

    ``metrics`` is the dict returned by ``vesp.training.train_discrete.run`` (which
    still carries the nested ``diagnostics``). An empty ``metrics`` (a failed run)
    yields a row with blanks plus the ``error`` text.
    """

    diagnostics = metrics.get("diagnostics", {}) if metrics else {}
    model_cfg = config.get("model", {})
    loss_cfg = config.get("loss", {})
    solver_cfg = config.get("solver", {}) if isinstance(config.get("solver"), dict) else {}

    reasons = metrics.get("acceptability_reasons") if metrics else None
    reasons_text = " | ".join(reasons) if isinstance(reasons, list) else (reasons or "")

    row = {col: "" for col in SUMMARY_COLUMNS}
    row.update(
        {
            "run_name": run_name,
            "solver": metrics.get("solver", solver_cfg.get("type", "")) if metrics else solver_cfg.get("type", ""),
            "model_type": model_cfg.get("type", "discrete"),
            "shell_alphas": _shell_alphas(model_cfg),
            "n_sources_total": _n_sources_total(model_cfg),
            "lambda_l2": loss_cfg.get("lambda_l2", solver_cfg.get("lambda_l2", "")),
            "lambda_moment": loss_cfg.get("lambda_moment", ""),
            "entropy_weight": metrics.get("entropy_weight", loss_cfg.get("entropy_weight", "")) if metrics else loss_cfg.get("entropy_weight", ""),
            "entropy_mode": metrics.get("entropy_mode", loss_cfg.get("entropy_mode", "")) if metrics else loss_cfg.get("entropy_mode", ""),
            "experiment": experiment or "",
            "config_path": config_path or "",
            "git_commit": git_commit or "",
            "timestamp": timestamp or "",
            "runtime_sec": runtime_sec if runtime_sec is not None else "",
            "error": error,
        }
    )

    if metrics:
        for key in (
            "potential_rmse",
            "acceleration_rmse",
            "relative_acceleration_rmse",
            "radial_rmse",
            "cross_radial_rmse",
            "angle_deg_p95",
            "low_altitude_acceleration_rmse",
            "mid_altitude_acceleration_rmse",
            "high_altitude_acceleration_rmse",
            "low_to_high_error_ratio",
            "test_low_acceleration_rmse",
            "test_high_acceleration_rmse",
            "source_entropy_nats",
            "positive_negative_entropy_nats",
            "shell_energy_balance_entropy_nats",
            "inference_seconds_per_batch",
            "selected_lambda_l2",
        ):
            row[key] = metrics.get(key, "")
        for key in (
            "dominant_shell_energy_fraction",
            "shell_cancellation_ratio",
            "top_5pct_source_contribution",
            "effective_source_count",
            "sigma_l2",
            "relative_monopole_leakage",
            "relative_dipole_leakage",
        ):
            row[key] = diagnostics.get(key, "")
        row["acceptability_status"] = metrics.get("acceptability_status", "")
        row["acceptability_reasons"] = reasons_text
    else:
        row["acceptability_status"] = "FAILED"
    return row


def _csv_text(fieldnames: list[str], rows: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _csv_cell(row.get(k, "")) for k in fieldnames})
    return buffer.getvalue()


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    return value


def _to_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def _md_table(rows: list[dict], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                cells.append(f"{value:.4g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


_MD_COLUMNS = [
    "run_name",
    "solver",
    "lambda_l2",
    "entropy_weight",
    "entropy_mode",
    "relative_acceleration_rmse",
    "low_to_high_error_ratio",
    "shell_cancellation_ratio",
    "top_5pct_source_contribution",
    "source_entropy_nats",
    "sigma_l2",
    "acceptability_status",
]


def write_suite_artifacts(
    suite_dir: str | Path,
    rows: list[dict],
    *,
    suite_name: str,
    experiments: list[str] | None = None,
    git_commit: str | None = None,
    make_plots: bool = True,
    readme_extra: str | None = None,
) -> dict[str, Path]:
    """Write ``suite_summary.csv``, ``suite_summary.md``, ``pareto_data.csv`` and ``README.md``.

    Returns a mapping of artifact name -> path. Plot generation is optional and never
    raises into the caller (headless / missing-matplotlib safe).
    """

    suite_dir = Path(suite_dir)
    suite_dir.mkdir(parents=True, exist_ok=True)

    csv_path = suite_dir / "suite_summary.csv"
    csv_path.write_text(_csv_text(SUMMARY_COLUMNS, rows), encoding="utf-8")

    pareto_path = suite_dir / "pareto_data.csv"
    pareto_path.write_text(_csv_text(PARETO_COLUMNS, rows), encoding="utf-8")

    md_lines = [
        f"# Experiment Suite: {suite_name}",
        "",
        f"Total runs: {len(rows)}",
    ]
    if git_commit:
        md_lines.append(f"Git commit: `{git_commit}`")
    if experiments:
        md_lines += ["", "Experiments: " + ", ".join(f"`{e}`" for e in experiments)]
    md_lines += ["", "## Runs", ""]
    md_lines += _md_table(rows, _MD_COLUMNS)
    md_lines += [
        "",
        "_`acceptability_status` is a screening flag only, not a scientific verdict._",
        "_Headline comparison metric: `relative_acceleration_rmse` (unit-invariant)._",
        "",
    ]
    md_path = suite_dir / "suite_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    readme_path = suite_dir / "README.md"
    readme_path.write_text(
        _suite_readme(suite_name, experiments or [], rows, readme_extra), encoding="utf-8"
    )

    artifacts = {
        "suite_summary_csv": csv_path,
        "suite_summary_md": md_path,
        "pareto_data_csv": pareto_path,
        "readme": readme_path,
    }
    if make_plots:
        artifacts.update(_maybe_plot(suite_dir, rows))
    return artifacts


def _suite_readme(suite_name: str, experiments: list[str], rows: list[dict], extra: str | None) -> str:
    lines = [
        f"# {suite_name}",
        "",
        "Generated by `vesp.experiments`. This directory aggregates one or more",
        "experiment configs into a single comparison.",
        "",
        "## Files",
        "",
        "- `suite_summary.csv` — full standardized metrics row per run.",
        "- `suite_summary.md` — human-readable subset of the same table.",
        "- `pareto_data.csv` — columns needed for L2 / entropy trade-off plots.",
        "- `runs/<run_name>/` — per-run artifacts (`metrics.json`, `diagnostics.json`,",
        "  `summary.txt`, `shell_energy.csv`, `altitude_binned_error.csv`,",
        "  `target_scales.json`, `run_manifest.json`, `config.yaml`).",
        "- `*.png` — optional plots (only if matplotlib is available).",
        "",
    ]
    if experiments:
        lines += ["## Experiments", "", *[f"- `{e}`" for e in experiments], ""]
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[str(row.get("acceptability_status", ""))] = statuses.get(str(row.get("acceptability_status", "")), 0) + 1
    lines += ["## Acceptability screening tally", ""]
    lines += [f"- {status or '(blank)'}: {count}" for status, count in sorted(statuses.items())]
    lines += [
        "",
        "Acceptability is a *screening* signal only. Read the underlying metrics before",
        "drawing physical conclusions. See `docs/SCIENTIFIC_CLAIMS.md`.",
        "",
    ]
    if extra:
        lines += [extra, ""]
    return "\n".join(lines) + "\n"


def _maybe_plot(suite_dir: Path, rows: list[dict]) -> dict[str, Path]:
    """Best-effort plots; silently skipped on any failure (headless safe)."""

    out: dict[str, Path] = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - plotting is strictly optional
        return out

    def _xy(x_key: str, y_key: str, *, predicate=None) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            if predicate is not None and not predicate(row):
                continue
            x = _to_float(row.get(x_key))
            y = _to_float(row.get(y_key))
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        return [xs[i] for i in order], [ys[i] for i in order]

    has_l2_sweep = len({_to_float(r.get("lambda_l2")) for r in rows if _to_float(r.get("lambda_l2")) is not None}) > 1
    has_entropy_sweep = len({_to_float(r.get("entropy_weight")) for r in rows if _to_float(r.get("entropy_weight")) is not None}) > 1

    plot_specs = []
    if has_l2_sweep:
        plot_specs += [
            ("lambda_l2", "relative_acceleration_rmse", "acc_rmse_vs_lambda_l2.png", True),
            ("lambda_l2", "shell_cancellation_ratio", "cancellation_vs_lambda_l2.png", True),
        ]
    if has_entropy_sweep:
        plot_specs += [
            ("entropy_weight", "relative_acceleration_rmse", "acc_rmse_vs_entropy_weight.png", False),
            ("entropy_weight", "source_entropy_nats", "source_entropy_vs_entropy_weight.png", False),
            # data error vs entropy Pareto
            ("source_entropy_nats", "relative_acceleration_rmse", "data_error_vs_entropy_pareto.png", False),
        ]

    for x_key, y_key, fname, logx in plot_specs:
        xs, ys = _xy(x_key, y_key)
        if len(xs) < 2:
            continue
        try:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(xs, ys, marker="o")
            if logx and all(x > 0 for x in xs):
                ax.set_xscale("log")
            ax.set_xlabel(x_key)
            ax.set_ylabel(y_key)
            ax.set_title(f"{y_key} vs {x_key}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = suite_dir / fname
            fig.savefig(path, dpi=120)
            plt.close(fig)
            out[fname] = path
        except Exception:  # noqa: BLE001
            try:
                plt.close("all")
            except Exception:
                pass
            continue
    return out
