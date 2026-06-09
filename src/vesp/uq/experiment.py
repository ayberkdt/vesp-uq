"""VESP-UQ experiment pipeline: fit, calibrate, score, screen -> report dict.

This is the orchestration layer behind ``python -m vesp.uq.run``. It is deliberately free of any
file/CLI concerns (see :mod:`vesp.uq.run`) and of report formatting (see :mod:`vesp.uq.reporting`).

The trajectory ensemble can be **generated** (synthetic Keplerian orbits) or **loaded** from an
external surrogate-output CSV (``uq.screening.trajectory_source: generated | csv``). When the CSV
carries surrogate/reference acceleration pairs, the *true force error* along each trajectory can
be read directly from the residual instead of a nearest-neighbour oracle.
"""

from __future__ import annotations

import time

import torch

from vesp.common.units import UnitConfig
from vesp.data.dataset import load_csv_dataset
from vesp.uq.data import (
    UQSamples,
    load_uq_samples_from_csv,
    make_synthetic_uq_samples,
    split_uq_samples,
)
from vesp.uq.ensemble import generate_orbit_ensemble, nearest_neighbor_error_magnitude
from vesp.uq.io import load_trajectory_csv
from vesp.uq.physical_units import MODEL_UNITS, resolve_acceleration_scale
from vesp.uq.plugin import VESPUQPlugin
from vesp.uq.reporting import build_summary, build_tables, expected_error_summary
from vesp.uq.scoring import (
    aggregate_trajectory_error,
    canonical_scoring_name,
    is_absolute_scoring,
    is_relative_scoring,
)
from vesp.uq.selection import select_reruns
from vesp.uq.thresholds import resolve_threshold

# Re-exported under their historical private names for backward compatibility (see run.py).
__all__ = ["run_vespuq"]


def _load_samples(config: dict, dtype: torch.dtype) -> UQSamples:
    """Load VESP-UQ calibration samples (synthetic / unit-correct residual / generic CSV)."""

    data_cfg = config.get("data", {})
    if str(data_cfg.get("type", "csv")).lower() == "synthetic" or not data_cfg.get("path"):
        return make_synthetic_uq_samples(
            n=int(data_cfg.get("n", 512)),
            n_truth_sources=int(data_cfg.get("n_truth_sources", 24)),
            noise_std=float(data_cfg.get("noise_std", 1.0e-4)),
            seed=int(config.get("seed", 0)),
            dtype=dtype,
        )
    fmt = str(data_cfg.get("format", "residual")).lower()
    if fmt == "residual":
        # unit-correct path for the band-limited residual dataset (acceleration IS the error)
        units = UnitConfig.from_config(config)
        data = load_csv_dataset(data_cfg["path"], dtype=dtype, unit_config=units)
        return UQSamples(
            positions=data.positions,
            error=data.acceleration,
            reference=data.acceleration.clone(),
            surrogate=torch.zeros_like(data.acceleration),
            metadata={"mode": "residual", "path": str(data_cfg["path"])},
        )
    return load_uq_samples_from_csv(data_cfg["path"], dtype=dtype, mode=fmt)


def _time_weights(traj: torch.Tensor) -> torch.Tensor:
    """Approximate per-point time weights for a true-anomaly-uniform Keplerian sample (~r^2)."""

    r = torch.linalg.norm(traj, dim=-1).to(torch.float64)
    return r * r


def _resolve_time_weighting(screen_cfg: dict) -> str:
    """Resolve the time-weighting mode, honoring the legacy ``time_weighted`` boolean.

    ``uq.screening.time_weighting: none | kepler_r2`` takes precedence; otherwise the legacy
    ``time_weighted: true/false`` maps to ``kepler_r2`` / ``none``.
    """

    mode = screen_cfg.get("time_weighting")
    if mode is None:
        mode = "kepler_r2" if bool(screen_cfg.get("time_weighted", False)) else "none"
    mode = str(mode).lower()
    if mode not in {"none", "kepler_r2"}:
        raise ValueError("uq.screening.time_weighting must be 'none' or 'kepler_r2'")
    return mode


def _units_metadata(config: dict) -> dict:
    """Conservative units metadata for the report (no invented physical conversion).

    Physical acceleration conversion is reported as available only when the config supplies explicit
    metadata (``body.acceleration_scale_m_s2`` or a physical ``body.acceleration_units``); otherwise
    the report stays in model-normalized units with a clear note.
    """

    body = config.get("body", {})
    accel_units = str(body.get("acceleration_units", "model_normalized_accel"))
    scale = resolve_acceleration_scale(config)
    meta = {
        "risk_score_units": "model_normalized_accel",
        "acceleration_metric_units": accel_units,
        "position_units": str(body.get("position_units", "normalized")),
        "force_error_scale_note": (
            "Risk scores and expected force errors are in the model's normalized-acceleration units "
            "(dU/d(model coordinate)) by default. A physical conversion is applied only when explicit "
            "metadata is supplied (body.acceleration_scale_m_s2 or a physical body.acceleration_units); "
            "see the physical_conversion_* fields below. No physical scale is ever inferred."
        ),
        "physical_R_body": body.get("physical_R_body"),
        "physical_R_body_units": body.get("physical_R_body_units"),
        # Physical-budget conversion metadata (explicit only -- never inferred).
        "physical_conversion_available": scale.physical,
        "acceleration_scale_m_s2": scale.scale_m_s2,
        "acceleration_scale_source": scale.source,
        "score_units_by_scale": {
            "relative": "per-trajectory-normalized (ranking only; no physical budget)",
            "absolute": "fixed model-normalized acceleration scale (physical budget allowed)",
            "sigma": "predictive-uncertainty magnitude (model-normalized acceleration)",
        },
    }
    if not scale.physical:
        meta["physical_conversion_note"] = (
            "physical acceleration conversion unavailable; values are reported in "
            "model-normalized acceleration units."
        )
    return meta


def _build_trajectories(screen_cfg: dict, *, seed: int, dtype: torch.dtype, config: dict | None = None) -> dict:
    """Resolve the trajectory ensemble: generated Keplerian orbits or an external CSV.

    Returns a dict with ``trajectories`` (list of (T,3)), ``source``, ``path``, optional
    ``residuals`` (per-trajectory ``(T,3)`` residual force error when the CSV had accel pairs), and
    ``units`` (the loader's per-file unit metadata, or ``None`` for generated orbits).

    For a CSV source, ``uq.screening.trajectory_acceleration_units`` (default
    ``model_normalized_accel``) declares the CSV acceleration units; a physical unit is converted to
    model units via ``body.acceleration_scale_m_s2`` (requires ``config``). Generated orbits are
    always in model-normalized coordinates.
    """

    source = str(screen_cfg.get("trajectory_source", "generated")).lower()
    if source == "generated":
        ensemble = generate_orbit_ensemble(
            n_orbits=int(screen_cfg.get("n_orbits", 200)),
            n_points=int(screen_cfg.get("n_points", 48)),
            r_peri_range=tuple(screen_cfg.get("r_peri_range", (1.02, 1.30))),
            r_apo_range=tuple(screen_cfg.get("r_apo_range", (1.30, 1.60))),
            seed=seed,
            dtype=dtype,
        )
        return {"trajectories": ensemble.trajectories, "source": "generated", "path": None, "residuals": None, "units": None}
    if source == "csv":
        path = screen_cfg.get("trajectory_path")
        if not path:
            raise ValueError("uq.screening.trajectory_source=csv requires uq.screening.trajectory_path")
        accel_units = str(screen_cfg.get("trajectory_acceleration_units", MODEL_UNITS))
        scale = resolve_acceleration_scale(config) if config is not None else None
        ds = load_trajectory_csv(path, dtype=dtype, acceleration_scale=scale, acceleration_units=accel_units)
        return {
            "trajectories": ds.trajectories,
            "source": "csv",
            "path": str(path),
            "residuals": ds.residual_accelerations,  # None unless accel pairs were present
            "units": ds.metadata.get("units"),
        }
    raise ValueError("uq.screening.trajectory_source must be 'generated' or 'csv'")


def run_vespuq(config: dict) -> dict:
    from vesp.common.config import get_dtype

    dtype = get_dtype(config)
    samples = _load_samples(config, dtype)
    seed = int(config.get("seed", 0))
    train, held = split_uq_samples(
        samples, train_fraction=float(config.get("data", {}).get("train_fraction", 0.7)), seed=seed
    )

    plugin = VESPUQPlugin.from_config(config)
    t0 = time.perf_counter()
    plugin.fit(train.positions, train.surrogate, train.reference)
    fit_seconds = time.perf_counter() - t0

    bands = config.get("evaluation", {}).get("altitude_bands")
    t0 = time.perf_counter()
    calibration = plugin.evaluate_calibration(held.positions, held.error, altitude_bands=bands)
    calibration_eval_seconds = time.perf_counter() - t0

    # ---------------- Experiment 3: trajectory risk screening ----------------
    screen_cfg = config.get("uq", {}).get("screening", {})
    traj_info = _build_trajectories(screen_cfg, seed=seed, dtype=dtype, config=config)
    trajectories = traj_info["trajectories"]
    scoring = plugin.risk_scoring
    fraction_policy = str(screen_cfg.get("fraction_policy", "topk")).lower()

    # Time-weighting: orbits are sampled uniformly in true anomaly, which oversamples periapsis
    # for eccentric orbits. `kepler_r2` weights each point by ~dt (proportional to r^2).
    time_weighting = _resolve_time_weighting(screen_cfg)
    weights = [_time_weights(t) for t in trajectories] if time_weighting == "kepler_r2" else None

    t0 = time.perf_counter()
    scores = plugin.score_ensemble(trajectories, weights=weights)
    score_seconds = time.perf_counter() - t0
    risk_scores = torch.tensor([s.risk_score for s in scores], dtype=torch.float64)

    # True force error along each trajectory -- the diagnostic oracle. DECOUPLED from the risk
    # scoring mode via an explicit aggregator (default p95, robust to single-NN spikes).
    true_error_aggregator = str(screen_cfg.get("true_error_aggregator", "p95")).lower()
    oracle_source = str(screen_cfg.get("oracle_source", "heldout")).lower()
    if oracle_source not in {"heldout", "all"}:
        raise ValueError("uq.screening.oracle_source must be 'heldout' or 'all'")
    oracle = samples if oracle_source == "all" else held

    # Prefer the *direct* residual force error when an external CSV supplied accel pairs.
    use_residual = traj_info["residuals"] is not None and str(
        screen_cfg.get("true_error_source", "auto")
    ).lower() in {"auto", "residual_csv"}
    true_error = torch.empty(len(trajectories), dtype=torch.float64)
    if use_residual:
        true_error_mode = "residual_csv"
        for i, res in enumerate(traj_info["residuals"]):
            mag = torch.linalg.norm(res.to(torch.float64), dim=-1)
            true_error[i] = aggregate_trajectory_error(mag, true_error_aggregator)
    else:
        true_error_mode = f"nn_oracle:{oracle_source}"
        for i, traj in enumerate(trajectories):
            nn = nearest_neighbor_error_magnitude(traj.to(dtype), oracle.positions, oracle.error)
            true_error[i] = aggregate_trajectory_error(nn.to(torch.float64), true_error_aggregator)

    # Selection policy: an absolute force-risk budget (optionally capped) takes precedence over the
    # fixed top-fraction. Pointwise budgets are rejected for relative scoring (scale mismatch).
    max_rerun_fraction = screen_cfg.get("max_rerun_fraction")
    rerun_fraction = float(screen_cfg.get("rerun_fraction", 0.20))
    threshold, threshold_meta = resolve_threshold(
        screen_cfg, plugin, held, scoring, dtype=dtype, seed=seed, config=config
    )
    # A physical budget may carry its own optional rerun cap (uq.physical_budget.max_rerun_fraction).
    if max_rerun_fraction is None and threshold_meta["threshold_source"] == "physical_budget":
        max_rerun_fraction = config.get("uq", {}).get("physical_budget", {}).get("max_rerun_fraction")
    if threshold is not None:
        screening = select_reruns(
            risk_scores,
            threshold=float(threshold),
            max_rerun_fraction=float(max_rerun_fraction) if max_rerun_fraction is not None else None,
            true_error=true_error,
            threshold_source=threshold_meta["threshold_source"],
            threshold_quantile=threshold_meta["threshold_quantile"],
        )
    else:
        screening = select_reruns(
            risk_scores,
            rerun_fraction=rerun_fraction,
            fraction_policy=fraction_policy,
            true_error=true_error,
        )

    n_traj = len(trajectories)
    n_points_total = sum(int(t.shape[0]) for t in trajectories)
    flagged_set = set(screening.flagged_indices)
    report = {
        "dataset": str(config.get("data", {}).get("path") or samples.metadata.get("mode", "synthetic")),
        "fit": plugin.fit_info,
        "units": _units_metadata(config),
        "experiment_1_calibration": calibration,
        "experiment_3_screening": {
            "scoring": scoring,
            "scoring_canonical": canonical_scoring_name(scoring),
            "scoring_scale": "relative" if is_relative_scoring(scoring) else ("absolute" if is_absolute_scoring(scoring) else "sigma"),
            "oracle_source": oracle_source,
            "true_error_mode": true_error_mode,
            "trajectory_source": traj_info["source"],
            "trajectory_path": traj_info["path"],
            "trajectory_units": traj_info.get("units"),
            "external_trajectory_count": n_traj if traj_info["source"] == "csv" else None,
            "external_output_points_total": n_points_total if traj_info["source"] == "csv" else None,
            "n_trajectories": n_traj,
            "n_output_points_total": n_points_total,
            "true_error_aggregator": true_error_aggregator,
            "time_weighting": time_weighting,
            "time_weighted": time_weighting == "kepler_r2",  # legacy boolean, kept for readers
            "fraction_policy": fraction_policy,
            "domain_support": plugin.domain_support,
            "domain_component_weights": {
                "distance": plugin.domain_distance_weight,
                "radial": plugin.domain_radial_weight,
                "angular": plugin.domain_angular_weight,
            },
            "threshold_source": threshold_meta["threshold_source"],
            "threshold_quantile": threshold_meta["threshold_quantile"],
            "threshold_multiplier": threshold_meta["threshold_multiplier"],
            "threshold_calibration_scoring": threshold_meta["threshold_calibration_scoring"],
            "threshold_calibration_n": threshold_meta["threshold_calibration_n"],
            "threshold_compatibility_note": threshold_meta["threshold_compatibility_note"],
            "threshold_model_units": threshold_meta["threshold_model_units"],
            "threshold_physical_value": threshold_meta["threshold_physical_value"],
            "threshold_physical_units": threshold_meta["threshold_physical_units"],
            "acceleration_scale_m_s2": threshold_meta["acceleration_scale_m_s2"],
            "conformal_enabled": threshold_meta["conformal_enabled"],
            "conformal_scale": threshold_meta["conformal_scale"],
            "conformal_alpha": threshold_meta["conformal_alpha"],
            "conformal_coverage_before": threshold_meta["conformal_coverage_before"],
            "conformal_coverage_after": threshold_meta["conformal_coverage_after"],
            "threshold_model_units_raw": threshold_meta["threshold_model_units_raw"],
            "expected_error": expected_error_summary(scores, plugin.domain_support),
            "screen": screening.to_dict(),
        },
        "runtime": {
            "fit_seconds": fit_seconds,
            "calibration_eval_seconds": calibration_eval_seconds,
            "score_seconds_total": score_seconds,
            "score_ms_per_trajectory": 1.0e3 * score_seconds / max(1, n_traj),
            "score_us_per_output_point": 1.0e6 * score_seconds / max(1, n_points_total),
            "note": "VESP-UQ is evaluated at output trajectory points only, not inside every integrator RHS call.",
        },
    }
    report["summary"] = build_summary(report)
    # tables attached for CSV emission (not part of the JSON report body)
    report["_tables"] = build_tables(scores, screening, true_error, flagged_set)
    return report
