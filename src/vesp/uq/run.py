"""Driver for the VESP-UQ IAC experiments: standalone calibration + trajectory risk screening.

    python -m vesp.uq.run --config configs/vespuq_real_lunar.yaml
    python -m vesp.uq.run --config configs/vespuq_smoke.yaml

Runs the two experiments the plan calls minimal:

  * Experiment 1 -- standalone residual-error calibration (does the layer reduce low-altitude
    overconfidence?): fit the equivalent-source posterior, calibrate altitude-dependent noise,
    report per-band PICP90 / z_std / ellipsoid coverage and whether epistemic uncertainty grows
    toward the surface.
  * Experiment 3 -- trajectory risk screening: score a synthetic orbit ensemble with the fitted
    layer, flag the riskiest subset for high-fidelity rerun, and validate against a
    nearest-neighbour ground-truth error read from held-out real samples.

VESP-UQ is evaluated only at OUTPUT trajectory points (post-processing), not inside every
integrator RHS call. Outputs: JSON + Markdown reports and CSV tables for calibration and
trajectory screening.
"""

from __future__ import annotations

import argparse
import csv as _csv
import io
import time
from pathlib import Path
from typing import Iterable

import torch

from vesp.common.artifacts import atomic_write_json, atomic_write_text, ensure_run_layout
from vesp.common.config import get_dtype, load_config
from vesp.common.units import UnitConfig
from vesp.data.dataset import load_csv_dataset
from vesp.uq.data import UQSamples, load_uq_samples_from_csv, make_synthetic_uq_samples, split_uq_samples
from vesp.uq.ensemble import generate_orbit_ensemble, nearest_neighbor_error_magnitude
from vesp.uq.plugin import VESPUQPlugin
from vesp.uq.trajectory import select_reruns


def _load_samples(config: dict, dtype: torch.dtype) -> UQSamples:
    """Load VESP-UQ calibration samples (synthetic / unit-correct residual / generic CSV)."""

    data_cfg = config.get("data", {})
    dtype_ = dtype
    if str(data_cfg.get("type", "csv")).lower() == "synthetic" or not data_cfg.get("path"):
        return make_synthetic_uq_samples(
            n=int(data_cfg.get("n", 512)),
            n_truth_sources=int(data_cfg.get("n_truth_sources", 24)),
            noise_std=float(data_cfg.get("noise_std", 1.0e-4)),
            seed=int(config.get("seed", 0)),
            dtype=dtype_,
        )
    fmt = str(data_cfg.get("format", "residual")).lower()
    if fmt == "residual":
        # unit-correct path for the band-limited residual dataset (acceleration IS the error)
        units = UnitConfig.from_config(config)
        data = load_csv_dataset(data_cfg["path"], dtype=dtype_, unit_config=units)
        return UQSamples(
            positions=data.positions,
            error=data.acceleration,
            reference=data.acceleration.clone(),
            surrogate=torch.zeros_like(data.acceleration),
            metadata={"mode": "residual", "path": str(data_cfg["path"])},
        )
    return load_uq_samples_from_csv(data_cfg["path"], dtype=dtype_, mode=fmt)


def _csv_text(header: list[str], rows: list[list]) -> str:
    buf = io.StringIO()
    writer = _csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue()


def run_vespuq(config: dict) -> dict:
    dtype = get_dtype(config)
    samples = _load_samples(config, dtype)
    seed = int(config.get("seed", 0))
    train, held = split_uq_samples(samples, train_fraction=float(config.get("data", {}).get("train_fraction", 0.7)), seed=seed)

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
    ensemble = generate_orbit_ensemble(
        n_orbits=int(screen_cfg.get("n_orbits", 200)),
        n_points=int(screen_cfg.get("n_points", 48)),
        r_peri_range=tuple(screen_cfg.get("r_peri_range", (1.02, 1.30))),
        r_apo_range=tuple(screen_cfg.get("r_apo_range", (1.30, 1.60))),
        seed=seed,
        dtype=dtype,
    )
    scoring = plugin.risk_scoring
    aggregate = torch.amax if scoring in {"max", "low_alt_integral", "time_above"} else torch.mean

    t0 = time.perf_counter()
    scores = plugin.score_ensemble(ensemble.trajectories)
    score_seconds = time.perf_counter() - t0
    risk_scores = torch.tensor([s.risk_score for s in scores], dtype=torch.float64)

    # nearest-neighbour ground-truth error magnitude along each orbit. The oracle uses held-out
    # samples by default (no leakage); 'all' uses the full sample set (denser, less NN noise).
    oracle_source = str(screen_cfg.get("oracle_source", "heldout")).lower()
    if oracle_source not in {"heldout", "all"}:
        raise ValueError("uq.screening.oracle_source must be 'heldout' or 'all'")
    oracle = samples if oracle_source == "all" else held
    true_error = torch.empty(len(ensemble.trajectories), dtype=torch.float64)
    for i, traj in enumerate(ensemble.trajectories):
        nn = nearest_neighbor_error_magnitude(traj.to(dtype), oracle.positions, oracle.error)
        true_error[i] = aggregate(nn.to(torch.float64))

    rerun_fraction = float(screen_cfg.get("rerun_fraction", 0.20))
    screening = select_reruns(risk_scores, rerun_fraction=rerun_fraction, true_error=true_error)

    n_traj = len(ensemble.trajectories)
    n_points_total = sum(int(t.shape[0]) for t in ensemble.trajectories)
    flagged_set = set(screening.flagged_indices)
    report = {
        "dataset": str(config.get("data", {}).get("path") or samples.metadata.get("mode", "synthetic")),
        "fit": plugin.fit_info,
        "experiment_1_calibration": calibration,
        "experiment_3_screening": {
            "scoring": scoring,
            "oracle_source": oracle_source,
            "n_trajectories": n_traj,
            "n_output_points_total": n_points_total,
            "true_error_aggregator": "max" if aggregate is torch.amax else "mean",
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
    report["summary"] = _summary(report)
    # tables attached for CSV emission (not part of the JSON report body)
    report["_tables"] = _build_tables(scores, screening, true_error, flagged_set)
    return report


def _summary(report: dict) -> dict:
    cal = report["experiment_1_calibration"]
    screen = report["experiment_3_screening"]["screen"]
    out: dict = {}
    if "low_high_epistemic_std_ratio" in cal:
        out["low_high_epistemic_std_ratio"] = cal["low_high_epistemic_std_ratio"]
        out["epistemic_grows_at_low_altitude"] = cal["low_high_epistemic_std_ratio"] > 1.0
    if "low_high_pred_sigma_ratio" in cal:
        out["low_high_pred_sigma_ratio"] = cal["low_high_pred_sigma_ratio"]
    low = cal.get("low", {})
    if "picp_90" in low:
        out["low_band_picp_90"] = low["picp_90"]
        out["low_band_calibrated_90"] = abs(low["picp_90"] - 0.90) <= 0.1
    if "ellipsoid_picp_90" in low:
        out["low_band_ellipsoid_picp_90"] = low["ellipsoid_picp_90"]
    out["rerun_fraction"] = screen["rerun_fraction"]
    out["capture_rate"] = screen.get("capture_rate")
    out["spearman_risk_vs_error"] = screen.get("spearman_risk_vs_error")
    out["error_ratio_flagged_to_accepted"] = screen.get("error_ratio_flagged_to_accepted")
    if screen.get("error_ratio_flagged_to_accepted"):
        out["screen_concentrates_error"] = screen["error_ratio_flagged_to_accepted"] > 1.0
    return out


def _build_tables(scores, screening, true_error, flagged_set) -> dict:
    traj_header = [
        "trajectory_id", "risk_score", "max_sigma", "mean_sigma", "low_altitude_sigma_integral",
        "time_above_threshold", "combined_altitude_risk", "min_radius", "mean_radius",
        "mean_epistemic_sigma", "flagged_for_rerun", "true_error",
    ]
    traj_rows = []
    for i, s in enumerate(scores):
        traj_rows.append([
            i, s.risk_score, s.max_sigma, s.mean_sigma, s.low_altitude_sigma_integral,
            s.time_above_threshold, s.combined_altitude_risk, s.min_radius, s.mean_radius,
            s.mean_epistemic_sigma, int(i in flagged_set), float(true_error[i]),
        ])
    flagged_rows = [r for r in traj_rows if r[-2] == 1]
    return {"trajectory_header": traj_header, "trajectory_rows": traj_rows, "flagged_rows": flagged_rows}


def _calibration_table(calibration: dict) -> tuple[list[str], list[list]]:
    metric_keys = [
        "n", "mean_radius", "rmse", "mean_pred_std", "mean_epistemic_std", "z_std",
        "picp_50", "picp_68", "picp_90", "picp_95", "nll", "crps",
        "ellipsoid_picp_50", "ellipsoid_picp_68", "ellipsoid_picp_90", "ellipsoid_picp_95",
        "mean_mahalanobis_d2", "median_mahalanobis_d2",
    ]
    header = ["band"] + metric_keys
    rows = []
    for name in ("all", "low", "mid", "high"):
        m = calibration.get(name)
        if not m:
            continue
        rows.append([name] + [m.get(k, "") for k in metric_keys])
    return header, rows


def _fmt(x, spec: str = ".3g") -> str:
    if x is None:
        return "n/a"
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return str(x)


def build_report_md(report: dict) -> str:
    fit = report["fit"]
    cal = report["experiment_1_calibration"]
    screen = report["experiment_3_screening"]
    sc = screen["screen"]
    rt = report["runtime"]
    s = report["summary"]
    lines = [
        "# VESP-UQ Report - Equivalent-Source Uncertainty Calibration Layer",
        "",
        f"dataset: `{report['dataset']}`",
        f"sources: {fit['n_sources']}  |  reg: {fit['reg_method']} (lambda_l2={_fmt(fit.get('lambda_l2'))})  "
        f"|  noise_model: {fit['noise_model']}  |  covariance_mode: {fit.get('covariance_mode', 'exact')}  "
        f"|  global noise_std={_fmt(fit.get('noise_std'))}",
    ]
    if "altitude_noise_b" in fit:
        lines.append(
            f"altitude noise sigma^2(h)=a*h^(-b): a={_fmt(fit['altitude_noise_a'], '.3e')}, "
            f"b={_fmt(fit['altitude_noise_b'], '.3f')} (h=r-1; larger b = faster growth toward surface)"
        )
    lines += [
        "",
        "## Experiment 1 - Standalone residual-error calibration",
        "",
        "| band | mean_radius | rmse | mean_pred_std | mean_epi_std | z_std | picp_90 | ell_picp_90 | mean_d2 | nll |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("all", "low", "mid", "high"):
        m = cal.get(name)
        if not m:
            continue
        lines.append(
            f"| {name} | {_fmt(m.get('mean_radius'), '.3f')} | {_fmt(m.get('rmse'), '.3e')} | "
            f"{_fmt(m.get('mean_pred_std'), '.3e')} | {_fmt(m.get('mean_epistemic_std'), '.3e')} | "
            f"{_fmt(m.get('z_std'), '.2f')} | {_fmt(m.get('picp_90'), '.2f')} | "
            f"{_fmt(m.get('ellipsoid_picp_90'), '.2f')} | {_fmt(m.get('mean_mahalanobis_d2'), '.2f')} | "
            f"{_fmt(m.get('nll'), '.3f')} |"
        )
    if "low_high_epistemic_std_ratio" in cal:
        grows = s.get("epistemic_grows_at_low_altitude")
        lines += [
            "",
            f"- Epistemic uncertainty grows toward low altitude: **{'YES' if grows else 'NO'}** "
            f"(low/high epistemic std ratio = {_fmt(cal['low_high_epistemic_std_ratio'], '.2f')}, "
            f"predictive sigma ratio = {_fmt(cal.get('low_high_pred_sigma_ratio'), '.2f')}).",
        ]
    lines += [
        "",
        "## Experiment 3 - Trajectory risk screening",
        "",
        f"- ensemble: {screen['n_trajectories']} orbits, {screen['n_output_points_total']} output points "
        f"(scoring = `{screen['scoring']}`, oracle = `{screen['oracle_source']}`)",
        f"- rerun threshold (risk score): {_fmt(sc['threshold'], '.3e')}  ->  "
        f"flagged {sc['n_flagged']}/{sc['n_trajectories']} ({_fmt(100 * sc['rerun_fraction'], '.1f')}%)",
        f"- capture rate (top-decile true-error orbits flagged): **{_fmt(sc.get('capture_rate'), '.2f')}**  "
        f"| precision: {_fmt(sc.get('precision'), '.2f')}",
        f"- Spearman(risk, true error): {_fmt(sc.get('spearman_risk_vs_error'), '.2f')}",
        f"- mean true error  flagged: {_fmt(sc.get('mean_error_flagged'), '.3e')}  vs  "
        f"accepted: {_fmt(sc.get('mean_error_accepted'), '.3e')}  "
        f"(ratio {_fmt(sc.get('error_ratio_flagged_to_accepted'), '.2f')}x)",
        "",
        "## Runtime",
        "",
        f"- fit: {_fmt(rt['fit_seconds'], '.3f')} s  |  calibration eval: {_fmt(rt['calibration_eval_seconds'], '.3f')} s",
        f"- scoring: {_fmt(rt['score_ms_per_trajectory'], '.3f')} ms/trajectory "
        f"({_fmt(rt['score_us_per_output_point'], '.2f')} us/output point, {screen['n_output_points_total']} points total)",
        f"- _{rt['note']}_",
    ]
    lines += _iac_summary_md(report)
    return "\n".join(lines) + "\n"


def _iac_summary_md(report: dict) -> list[str]:
    fit = report["fit"]
    cal = report["experiment_1_calibration"]
    sc = report["experiment_3_screening"]["screen"]
    s = report["summary"]
    rt = report["runtime"]
    low, mid, high = cal.get("low", {}), cal.get("mid", {}), cal.get("high", {})
    return [
        "",
        "## IAC claim summary",
        "",
        f"- **What was fitted?** An interior equivalent-source posterior over the residual-force "
        f"error `e_a = a_reference - a_surrogate` ({fit['n_sources']} sources, "
        f"{fit['reg_method']} regularization).",
        "- **What was calibrated?** Altitude-dependent predictive uncertainty (heteroscedastic "
        "noise floor) on held-out validation residuals; the posterior mean equals the ridge point estimate.",
        f"- **Did low-altitude uncertainty increase?** "
        f"{'Yes' if s.get('epistemic_grows_at_low_altitude') else 'No'} "
        f"(low/high epistemic std ratio = {_fmt(cal.get('low_high_epistemic_std_ratio'), '.2f')}).",
        f"- **PICP90 by band (low/mid/high):** {_fmt(low.get('picp_90'), '.2f')} / "
        f"{_fmt(mid.get('picp_90'), '.2f')} / {_fmt(high.get('picp_90'), '.2f')}.",
        f"- **Fraction of trajectories flagged:** {_fmt(100 * sc['rerun_fraction'], '.1f')}% "
        f"(capture rate {_fmt(sc.get('capture_rate'), '.2f')}).",
        f"- **Did flagged trajectories carry larger true error?** "
        f"{'Yes' if s.get('screen_concentrates_error') else 'No'} "
        f"({_fmt(sc.get('error_ratio_flagged_to_accepted'), '.2f')}x the accepted-set error).",
        f"- **Runtime overhead:** {_fmt(rt['score_ms_per_trajectory'], '.3f')} ms/trajectory, "
        f"{_fmt(rt['score_us_per_output_point'], '.2f')} us/output point (post-processing only).",
        "- **What should NOT be claimed:** not a better deterministic surrogate; not true lunar "
        "density recovery; not operational orbit uncertainty propagation; not integrated with "
        "ST-LRPS. VESP-UQ is an uncertainty/risk-calibration layer at the acceleration interface.",
        "",
    ]


def run(config: dict) -> dict:
    report = run_vespuq(config)
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

    cal_header, cal_rows = _calibration_table(report["experiment_1_calibration"])
    atomic_write_text(run_dir / "calibration_by_band.csv", _csv_text(cal_header, cal_rows))
    atomic_write_text(
        run_dir / "trajectory_scores.csv", _csv_text(tables["trajectory_header"], tables["trajectory_rows"])
    )
    atomic_write_text(
        run_dir / "flagged_trajectories.csv", _csv_text(tables["trajectory_header"], tables["flagged_rows"])
    )

    print(markdown.encode("ascii", "replace").decode("ascii"))
    print(f"saved_vespuq_report: {run_dir / 'vespuq_report.md'}")
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="VESP-UQ: calibration + trajectory risk screening.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
