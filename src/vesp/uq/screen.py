"""Serve-side trajectory risk screening with a persisted VESP-UQ layer (NO refitting).

    python -m vesp.uq.screen --model outputs/run/vespuq_plugin.pt --config configs/vespuq/vespuq_smoke.yaml --out outputs/screen
    python -m vesp.uq.screen --model outputs/run/vespuq_plugin.pt --trajectories orbits.csv --out outputs/screen

Train/serve separation: ``python -m vesp.uq.run`` is the TRAINING driver -- it fits the layer
from calibration data and (with ``output.save_model: true`` or ``--save-model``) persists the
fitted model together with its decision policy and a model card. This module is the SERVING
driver: it loads that artifact, scores a new trajectory ensemble (external CSV or a generated
Keplerian set from a config's ``uq.screening`` block), applies the packaged -- or explicitly
overridden -- decision policy, and writes the same provenance-checked artifact set. Nothing is
fitted here, so screening is cheap and repeatable.

Honesty: at serve time there is usually NO ground-truth oracle. Unless the trajectory CSV carries
surrogate/reference acceleration pairs (then the residual force error is used as a diagnostic),
the outputs are force-risk / OOD scores only -- never a position-error or accuracy claim.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable
from pathlib import Path

import torch

from vesp.common.artifacts import (
    atomic_write_json,
    atomic_write_text,
    ensure_run_layout,
    write_run_manifest,
)
from vesp.common.config import load_config
from vesp.uq.experiment import _build_trajectories, _resolve_time_weighting, _time_weights
from vesp.uq.io import load_trajectory_csv
from vesp.uq.physical_units import MODEL_UNITS, resolve_acceleration_scale
from vesp.uq.plugin import VESPUQPlugin
from vesp.uq.reporting import build_tables, csv_text, expected_error_summary, fmt
from vesp.uq.scoring import (
    aggregate_trajectory_error,
    canonical_scoring_name,
    is_absolute_scoring,
    is_relative_scoring,
)
from vesp.uq.selection import select_reruns

__all__ = ["run_screen", "main"]


def _resolve_scoring(cli_scoring: str | None, policy: dict, plugin: VESPUQPlugin) -> tuple[str, str]:
    """Scoring mode + its origin, precedence CLI > packaged policy > plugin default."""

    if cli_scoring:
        canonical_scoring_name(cli_scoring)  # validates; raises ValueError for unknown modes
        return cli_scoring, "cli"
    if policy.get("scoring"):
        return str(policy["scoring"]), "model"
    return plugin.risk_scoring, "plugin_default"


def _resolve_selection(
    *,
    cli_threshold: float | None,
    cli_rerun_fraction: float | None,
    cli_max_rerun_fraction: float | None,
    policy: dict,
    scoring: str,
) -> dict:
    """Selection policy with explicit precedence: CLI > packaged model policy > default fraction.

    A threshold PACKAGED with the model is only applied when the serve-time scoring mode is the
    same canonical mode it was calibrated for -- a threshold on one score scale is meaningless on
    another (relative supervisor scores are per-trajectory normalized; absolute modes are on a
    fixed expected-force-error scale). On a mismatch, the caller must pass an explicit
    ``--threshold`` or ``--rerun-fraction``.
    """

    max_frac = cli_max_rerun_fraction if cli_max_rerun_fraction is not None else policy.get("max_rerun_fraction")
    if cli_threshold is not None:
        return {
            "mode": "threshold",
            "origin": "cli",
            "threshold": float(cli_threshold),
            "max_rerun_fraction": float(max_frac) if max_frac is not None else None,
        }
    if cli_rerun_fraction is not None:
        return {"mode": "fraction", "origin": "cli", "rerun_fraction": float(cli_rerun_fraction)}
    if policy.get("threshold") is not None:
        packaged_scoring = str(policy.get("scoring"))
        if canonical_scoring_name(scoring) != canonical_scoring_name(packaged_scoring):
            raise ValueError(
                f"the model's packaged threshold was calibrated for scoring "
                f"{packaged_scoring!r} but this run scores with {scoring!r}; thresholds do not "
                "transfer across score scales -- pass --threshold or --rerun-fraction explicitly"
            )
        return {
            "mode": "threshold",
            "origin": "model",
            "threshold": float(policy["threshold"]),
            "threshold_source": policy.get("threshold_source"),
            "threshold_quantile": policy.get("threshold_quantile"),
            "max_rerun_fraction": float(max_frac) if max_frac is not None else None,
        }
    if policy.get("rerun_fraction") is not None:
        return {
            "mode": "fraction",
            "origin": "model",
            "rerun_fraction": float(policy["rerun_fraction"]),
            "fraction_policy": str(policy.get("fraction_policy") or "topk"),
        }
    return {"mode": "fraction", "origin": "default", "rerun_fraction": 0.20}


def _resolve_trajectories(
    *,
    trajectories_csv: str | None,
    trajectory_units: str,
    config: dict | None,
    seed: int,
    dtype: torch.dtype,
) -> dict:
    """Trajectory ensemble from an explicit CSV (preferred) or a config's screening block."""

    if trajectories_csv:
        scale = resolve_acceleration_scale(config) if config is not None else None
        ds = load_trajectory_csv(
            trajectories_csv, dtype=dtype, acceleration_scale=scale, acceleration_units=trajectory_units
        )
        return {
            "trajectories": ds.trajectories,
            "source": "csv",
            "path": str(trajectories_csv),
            "residuals": ds.residual_accelerations,
            "units": ds.metadata.get("units"),
        }
    if config is not None:
        screen_cfg = config.get("uq", {}).get("screening", {}) or {}
        return _build_trajectories(screen_cfg, seed=seed, dtype=dtype, config=config)
    raise ValueError("provide --trajectories <csv> or --config <yaml> (with a uq.screening block)")


def run_screen(
    *,
    model_path: str | Path,
    out_dir: str | Path,
    trajectories_csv: str | None = None,
    trajectory_units: str = MODEL_UNITS,
    config: dict | None = None,
    scoring: str | None = None,
    threshold: float | None = None,
    rerun_fraction: float | None = None,
    max_rerun_fraction: float | None = None,
    time_weighting: str | None = None,
    device: str = "cpu",
) -> dict:
    """Score a trajectory ensemble with a persisted plugin and write the screening artifacts."""

    model_path = Path(model_path)
    plugin = VESPUQPlugin.load(model_path, device=device)
    metadata = plugin.user_metadata or {}
    policy = metadata.get("decision_policy", {}) or {}

    scoring_used, scoring_origin = _resolve_scoring(scoring, policy, plugin)
    canonical_scoring_name(scoring_used)  # validate early with the standard error message
    selection = _resolve_selection(
        cli_threshold=threshold,
        cli_rerun_fraction=rerun_fraction,
        cli_max_rerun_fraction=max_rerun_fraction,
        policy=policy,
        scoring=scoring_used,
    )

    seed = int(config.get("seed", 0)) if config is not None else 0
    dtype = plugin.dtype
    traj_info = _resolve_trajectories(
        trajectories_csv=trajectories_csv,
        trajectory_units=trajectory_units,
        config=config,
        seed=seed,
        dtype=dtype,
    )
    trajectories = traj_info["trajectories"]
    if not trajectories:
        raise ValueError("trajectory source resolved to an empty ensemble")

    # Time weighting precedence: CLI > config screening block > packaged policy > none.
    if time_weighting is not None:
        tw = str(time_weighting).lower()
        if tw not in {"none", "kepler_r2"}:
            raise ValueError("time_weighting must be 'none' or 'kepler_r2'")
    elif config is not None:
        tw = _resolve_time_weighting(config.get("uq", {}).get("screening", {}) or {})
    else:
        tw = str(policy.get("time_weighting") or "none").lower()
    weights = [_time_weights(t) for t in trajectories] if tw == "kepler_r2" else None

    t0 = time.perf_counter()
    scores = plugin.score_ensemble(trajectories, scoring=scoring_used, weights=weights)
    score_seconds = time.perf_counter() - t0
    risk_scores = torch.tensor([s.risk_score for s in scores], dtype=torch.float64)

    # Serve-time diagnostic oracle: ONLY the CSV's own residual force error (if acceleration
    # pairs were present). There is deliberately no nearest-neighbour oracle here -- serving has
    # no calibration samples, and inventing one would blur the train/serve boundary.
    residuals = traj_info.get("residuals")
    if residuals is not None:
        true_error = torch.tensor(
            [
                aggregate_trajectory_error(torch.linalg.norm(res.to(torch.float64), dim=-1), "p95")
                for res in residuals
            ],
            dtype=torch.float64,
        )
        true_error_mode = "residual_csv"
    else:
        true_error = None
        true_error_mode = "none"

    if selection["mode"] == "threshold":
        screening = select_reruns(
            risk_scores,
            threshold=selection["threshold"],
            max_rerun_fraction=selection.get("max_rerun_fraction"),
            true_error=true_error,
            threshold_source=selection.get("threshold_source") or selection["origin"],
            threshold_quantile=selection.get("threshold_quantile"),
        )
    else:
        screening = select_reruns(
            risk_scores,
            rerun_fraction=selection["rerun_fraction"],
            fraction_policy=str(selection.get("fraction_policy") or "topk"),
            true_error=true_error,
        )

    n_traj = len(trajectories)
    n_points = sum(int(t.shape[0]) for t in trajectories)
    report = {
        "mode": "serve",
        "model": {
            "path": str(model_path),
            "state_version": metadata.get("state_version"),
            "kind": metadata.get("kind"),
            "fit": dict(plugin.fit_info),
            "provenance": metadata.get("provenance", {}),
        },
        "screening": {
            "scoring": scoring_used,
            "scoring_canonical": canonical_scoring_name(scoring_used),
            "scoring_scale": (
                "relative"
                if is_relative_scoring(scoring_used)
                else ("absolute" if is_absolute_scoring(scoring_used) else "sigma")
            ),
            "scoring_origin": scoring_origin,
            "selection_origin": selection["origin"],
            "trajectory_source": traj_info["source"],
            "trajectory_path": traj_info.get("path"),
            "trajectory_units": traj_info.get("units"),
            "n_trajectories": n_traj,
            "n_output_points_total": n_points,
            "time_weighting": tw,
            "true_error_mode": true_error_mode,
            "units": metadata.get("units", {}),
            "screen": screening.to_dict(),
            "expected_error": expected_error_summary(scores, plugin.domain_support),
        },
        "runtime": {
            "score_seconds_total": score_seconds,
            "score_ms_per_trajectory": 1.0e3 * score_seconds / max(1, n_traj),
            "score_us_per_output_point": 1.0e6 * score_seconds / max(1, n_points),
            "note": "serve mode: no fitting; VESP-UQ is evaluated at output trajectory points only.",
        },
    }

    layout = ensure_run_layout(Path(out_dir))
    run_dir = layout.run_dir
    flagged_set = set(screening.flagged_indices)
    true_error_for_tables = (
        true_error if true_error is not None else torch.full((n_traj,), float("nan"), dtype=torch.float64)
    )
    tables = build_tables(scores, screening, true_error_for_tables, flagged_set)

    atomic_write_json(run_dir / "screening_report.json", report)
    atomic_write_text(run_dir / "screening_report.md", _build_screen_md(report))
    atomic_write_text(
        run_dir / "trajectory_scores.csv", csv_text(tables["trajectory_header"], tables["trajectory_rows"])
    )
    atomic_write_text(
        run_dir / "flagged_trajectories.csv", csv_text(tables["trajectory_header"], tables["flagged_rows"])
    )

    inputs: dict[str, Path] = {"vespuq_plugin_pt": model_path}
    if traj_info.get("path"):
        inputs["trajectory_csv"] = Path(traj_info["path"])
    write_run_manifest(
        run_dir,
        config=config or {},
        metrics={
            "n_flagged": screening.n_flagged,
            "n_trajectories": n_traj,
            "zero_alarms": screening.n_flagged == 0,
        },
        artifacts={
            "screening_report_json": run_dir / "screening_report.json",
            "screening_report_md": run_dir / "screening_report.md",
            "trajectory_scores_csv": run_dir / "trajectory_scores.csv",
            "flagged_trajectories_csv": run_dir / "flagged_trajectories.csv",
        },
        inputs=inputs,
    )
    print(f"flagged {screening.n_flagged}/{n_traj} trajectories")
    print(f"saved_screening_report: {run_dir / 'screening_report.md'}")
    return report


def _build_screen_md(report: dict) -> str:
    """Compact Markdown for one serve-mode screening run."""

    model = report["model"]
    sc = report["screening"]
    screen = sc["screen"]
    rt = report["runtime"]
    fit = model.get("fit", {})
    sel_mode = screen.get("selection_mode", "fraction")
    lines = [
        "# VESP-UQ Screening Report (serve mode -- persisted model, no refit)",
        "",
        f"model: `{model['path']}` (state v{model.get('state_version')}, "
        f"{fit.get('n_sources', '?')} sources, lambda_l2={fmt(fit.get('lambda_l2'))}, "
        f"noise_model={fit.get('noise_model', '?')})",
        f"trajectories: `{sc.get('trajectory_path') or sc['trajectory_source']}` "
        f"({sc['n_trajectories']} trajectories, {sc['n_output_points_total']} output points)",
        f"scoring: `{sc['scoring']}` (scale `{sc['scoring_scale']}`, origin `{sc['scoring_origin']}`)  |  "
        f"time weighting: `{sc['time_weighting']}`",
        "",
        f"- selection: `{sel_mode}` (policy origin `{sc['selection_origin']}`)"
        + (
            f" -> absolute force-risk budget {fmt(screen.get('threshold'), '.4e')}"
            if sel_mode != "fraction"
            else f" (requested {fmt(100 * (screen.get('requested_rerun_fraction') or 0.0), '.1f')}%)"
        ),
        f"- flagged: **{screen['n_flagged']}/{sc['n_trajectories']}**"
        + (
            "  -- no trajectory exceeded the budget (zero alarms)"
            if screen["n_flagged"] == 0 and sel_mode != "fraction"
            else ""
        ),
        f"- true-error diagnostic: `{sc['true_error_mode']}`"
        + (
            " (no oracle at serve time; scores are force-risk / OOD only)"
            if sc["true_error_mode"] == "none"
            else " (residual force error from the CSV's acceleration pairs)"
        ),
        f"- runtime: {fmt(rt['score_ms_per_trajectory'], '.3f')} ms/trajectory "
        f"({fmt(rt['score_us_per_output_point'], '.2f')} us/output point)",
        "",
        "_Force-risk / OOD screening only: not a position-error prediction, not an accuracy "
        "claim. See `docs/VESP_UQ_LIMITATIONS.md`._",
        "",
    ]
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> None:
    from vesp.common.version import package_version

    parser = argparse.ArgumentParser(
        description="Screen a trajectory ensemble with a persisted VESP-UQ layer (no refitting)."
    )
    parser.add_argument("--version", action="version", version=f"vesp-uq {package_version()}")
    parser.add_argument("--model", required=True, help="path to a saved vespuq_plugin.pt")
    parser.add_argument("--out", required=True, help="output run directory")
    parser.add_argument("--trajectories", default=None, help="external trajectory CSV (Format A/B)")
    parser.add_argument(
        "--trajectory-units",
        default=MODEL_UNITS,
        help="acceleration units of the CSV (physical units need --config for the scale)",
    )
    parser.add_argument("--config", default=None, help="config YAML (screening block / units)")
    parser.add_argument("--scoring", default=None, help="override the scoring mode")
    parser.add_argument("--threshold", type=float, default=None, help="absolute risk budget override")
    parser.add_argument("--rerun-fraction", type=float, default=None, help="top-fraction override")
    parser.add_argument("--max-rerun-fraction", type=float, default=None, help="cap for threshold mode")
    parser.add_argument("--time-weighting", default=None, choices=["none", "kepler_r2"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    config = load_config(args.config) if args.config else None
    run_screen(
        model_path=args.model,
        out_dir=args.out,
        trajectories_csv=args.trajectories,
        trajectory_units=args.trajectory_units,
        config=config,
        scoring=args.scoring,
        threshold=args.threshold,
        rerun_fraction=args.rerun_fraction,
        max_rerun_fraction=args.max_rerun_fraction,
        time_weighting=args.time_weighting,
        device=args.device,
    )


if __name__ == "__main__":
    main()
