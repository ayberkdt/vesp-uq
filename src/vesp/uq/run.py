"""CLI driver for VESP-UQ: calibration + trajectory force-risk screening.

    python -m vesp.uq.run --config configs/vespuq/vespuq_real_lunar.yaml
    python -m vesp.uq.run --config configs/vespuq/vespuq_smoke.yaml

The pipeline lives in :mod:`vesp.uq.experiment` (fit / calibrate / score / screen), threshold
resolution in :mod:`vesp.uq.thresholds`, and report/CSV construction in :mod:`vesp.uq.reporting`.
This module is the thin CLI + artifact writer. The historical symbols ``run_vespuq``,
``build_report_md``, ``_resolve_threshold`` and ``_resolve_time_weighting`` are re-exported here
for backward compatibility.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from vesp.common.artifacts import (
    atomic_write_json,
    atomic_write_text,
    compute_file_sha256,
    ensure_run_layout,
    utc_now_iso,
    write_run_manifest,
)
from vesp.common.config import load_config
from vesp.uq.experiment import _resolve_time_weighting, _time_weights, run_vespuq
from vesp.uq.plugin import PLUGIN_STATE_VERSION
from vesp.uq.reporting import build_model_card, build_report_md, calibration_table, csv_text
from vesp.uq.thresholds import resolve_threshold as _resolve_threshold

__all__ = [
    "run",
    "run_vespuq",
    "build_report_md",
    "main",
    "_resolve_threshold",
    "_resolve_time_weighting",
    "_time_weights",
]


def run(config: dict) -> dict:
    report, plugin = run_vespuq(config, return_plugin=True)
    tables = report.pop("_tables")
    output_cfg = config.get("output", {})
    output_dir = Path(output_cfg.get("output_dir", "outputs"))
    run_name = str(output_cfg.get("run_name", "vespuq"))
    layout = ensure_run_layout(output_dir / run_name)
    run_dir = layout.run_dir

    atomic_write_json(run_dir / "vespuq_report.json", report)
    atomic_write_json(run_dir / "fit_summary.json", report["fit"])
    markdown = build_report_md(report)
    atomic_write_text(run_dir / "vespuq_report.md", markdown)

    cal_header, cal_rows = calibration_table(report["experiment_1_calibration"])
    atomic_write_text(run_dir / "calibration_by_band.csv", csv_text(cal_header, cal_rows))
    atomic_write_text(
        run_dir / "trajectory_scores.csv", csv_text(tables["trajectory_header"], tables["trajectory_rows"])
    )
    atomic_write_text(
        run_dir / "flagged_trajectories.csv", csv_text(tables["trajectory_header"], tables["flagged_rows"])
    )

    artifacts = {
        "vespuq_report_json": run_dir / "vespuq_report.json",
        "vespuq_report_md": run_dir / "vespuq_report.md",
        "fit_summary_json": run_dir / "fit_summary.json",
        "calibration_by_band_csv": run_dir / "calibration_by_band.csv",
        "trajectory_scores_csv": run_dir / "trajectory_scores.csv",
        "flagged_trajectories_csv": run_dir / "flagged_trajectories.csv",
    }

    # Inputs the run consumed (datasets, external trajectory CSVs) -- checksummed into the
    # manifest so a result is traceable to the exact input bytes, not just a path.
    inputs: dict[str, Path] = {}
    data_path = config.get("data", {}).get("path")
    if data_path:
        inputs["dataset_csv"] = Path(data_path)
    trajectory_path = report.get("experiment_3_screening", {}).get("trajectory_path")
    if trajectory_path:
        inputs["trajectory_csv"] = Path(trajectory_path)

    # output.save_model: true persists the fitted plugin so downstream consumers (the
    # `python -m vesp.uq.screen` serve CLI, CorrectedForceField, the MC/STM propagators) can
    # VESPUQPlugin.load(...) without refitting. The training run's decision policy and data
    # provenance are embedded in the artifact, and a model card is written next to it.
    if bool(output_cfg.get("save_model", False)):
        model_path = run_dir / "vespuq_plugin.pt"
        screen = report["experiment_3_screening"]
        metadata = {
            "kind": "vespuq_training_run",
            "state_version": PLUGIN_STATE_VERSION,
            "decision_policy": {
                "scoring": screen.get("scoring"),
                "scoring_canonical": screen.get("scoring_canonical"),
                "scoring_scale": screen.get("scoring_scale"),
                "threshold": (
                    screen["screen"].get("threshold") if screen.get("threshold_source") else None
                ),
                "threshold_source": screen.get("threshold_source"),
                "threshold_quantile": screen.get("threshold_quantile"),
                "threshold_multiplier": screen.get("threshold_multiplier"),
                "threshold_model_units": screen.get("threshold_model_units"),
                "threshold_physical_value": screen.get("threshold_physical_value"),
                "threshold_physical_units": screen.get("threshold_physical_units"),
                "rerun_fraction": screen["screen"].get("requested_rerun_fraction")
                or screen["screen"].get("rerun_fraction"),
                "fraction_policy": screen.get("fraction_policy"),
                "max_rerun_fraction": screen["screen"].get("max_rerun_fraction"),
                "time_weighting": screen.get("time_weighting"),
            },
            "units": report.get("units", {}),
            "provenance": {
                "created_at_utc": utc_now_iso(),
                "dataset": report.get("dataset"),
                "dataset_sha256": (
                    compute_file_sha256(data_path) if data_path and Path(data_path).is_file() else None
                ),
                "run_name": run_name,
            },
        }
        plugin.save(model_path, extra_metadata=metadata)
        card = build_model_card(report, model_filename=model_path.name, metadata=metadata)
        atomic_write_text(run_dir / "vespuq_plugin_card.md", card)
        artifacts["vespuq_plugin_pt"] = model_path
        artifacts["vespuq_plugin_card_md"] = run_dir / "vespuq_plugin_card.md"
        print(f"saved_vespuq_plugin: {model_path}")

    # Provenance manifest: config snapshot + SHA-256 checksums of every emitted artifact
    # and consumed input.
    write_run_manifest(
        run_dir,
        config=config,
        metrics=report.get("summary", {}),
        artifacts=artifacts,
        inputs=inputs or None,
    )

    print(markdown.encode("ascii", "replace").decode("ascii"))
    print(f"saved_vespuq_report: {run_dir / 'vespuq_report.md'}")
    return report


def main(argv: Iterable[str] | None = None) -> None:
    from vesp.common.version import package_version

    parser = argparse.ArgumentParser(description="VESP-UQ: calibration + trajectory force-risk screening.")
    parser.add_argument("--version", action="version", version=f"vesp-uq {package_version()}")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="persist the fitted layer as vespuq_plugin.pt (+ model card); overrides output.save_model",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.save_model:
        config.setdefault("output", {})["save_model"] = True
    run(config)


if __name__ == "__main__":
    main()
