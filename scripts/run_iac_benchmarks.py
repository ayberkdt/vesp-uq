"""Orchestrate the VESP-UQ IAC benchmark suite into ``outputs/iac/``.

Runs the force-risk / OOD core benchmarks (which only need the calibration data) and gracefully
*skips* the optional ST-LRPS position-error diagnostic when its data files are absent -- never
failing the whole suite for a missing optional input.

    python scripts/run_iac_benchmarks.py --config configs/vespuq/vespuq_smoke.yaml

Outputs:
    outputs/iac/iac_numbers.json
    outputs/iac/iac_summary.md
    outputs/iac/calibration_report.md
    outputs/iac/force_error_benchmark.md      (+ .json + force_error_scores.csv)
    outputs/iac/absolute_threshold_screening.md
    outputs/iac/position_error_diagnostic.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from vesp.common.config import load_config
from vesp.uq.ensemble import generate_orbit_ensemble
from vesp.uq.io.run_artifacts import write_run_artifacts
from vesp.uq.reporting import calibration_table, csv_text
from vesp.uq.selection import select_reruns

if __package__:
    from scripts.run_force_error_benchmark import _benchmark_md, force_error_benchmark, prepare
else:
    from run_force_error_benchmark import _benchmark_md, force_error_benchmark, prepare


def _calibration_report(plugin, held, config) -> tuple[str, dict]:
    bands = config.get("evaluation", {}).get("altitude_bands")
    cal = plugin.evaluate_calibration(held.positions, held.error, altitude_bands=bands)
    header, rows = calibration_table(cal)
    low, high = cal.get("low", {}), cal.get("high", {})
    md = "\n".join([
        "# VESP-UQ Calibration Report",
        "",
        "Held-out residual-error calibration (force-risk uncertainty, not position error).",
        "",
        "```text",
        csv_text(header, rows).rstrip(),
        "```",
        "",
        f"- low/high epistemic std ratio: {cal.get('low_high_epistemic_std_ratio')}",
        f"- low-band PICP90: {low.get('picp_90')}  |  high-band PICP90: {high.get('picp_90')}",
        "",
    ]) + "\n"
    numbers = {
        "low_high_epistemic_std_ratio": cal.get("low_high_epistemic_std_ratio"),
        "low_band_picp_90": low.get("picp_90"),
        "high_band_picp_90": high.get("picp_90"),
    }
    return md, numbers


def _ood_altitude_sweep(plugin) -> tuple[str, dict]:
    d = torch.tensor([1.0, 0.3, 0.2], dtype=torch.float64)
    d = d / d.norm()
    radii = [1.02, 1.05, 1.10, 1.20, 1.35, 1.50]
    rows = []
    for r in radii:
        pred = plugin.predict_uncertainty((d * r).reshape(1, 3))
        ds = float(plugin.domain_support_score((d * r).reshape(1, 3))) if plugin.domain_support else float("nan")
        rows.append((r, float(pred.expected_error), float(pred.sigma), ds))
    lines = [
        "# VESP-UQ OOD Altitude Sweep (force-risk / OOD detection)",
        "",
        "Expected force error and domain-support risk along one direction at decreasing altitude.",
        "",
        "| radius | expected_force_error | sigma | domain_risk |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for r, ee, sig, ds in rows:
        lines.append(f"| {r:.2f} | {ee:.3e} | {sig:.3e} | {ds:.3f} |")
    grows = rows[0][1] > rows[-1][1]
    lines += ["", f"- expected force error grows toward low altitude: **{'YES' if grows else 'NO'}** "
              f"({rows[0][1]:.3e} at r={rows[0][0]} vs {rows[-1][1]:.3e} at r={rows[-1][0]}).", ""]
    return "\n".join(lines) + "\n", {"expected_error_grows_low_altitude": bool(grows),
                                     "expected_error_low": rows[0][1], "expected_error_high": rows[-1][1]}


def _absolute_threshold_screening(plugin, config, seed) -> tuple[str, dict]:
    screen_cfg = config.get("uq", {}).get("screening", {})
    ens = generate_orbit_ensemble(
        n_orbits=int(screen_cfg.get("n_orbits", 200)),
        n_points=int(screen_cfg.get("n_points", 48)),
        r_peri_range=tuple(screen_cfg.get("r_peri_range", (1.02, 1.30))),
        r_apo_range=tuple(screen_cfg.get("r_apo_range", (1.30, 1.60))),
        seed=seed,
        dtype=torch.float64,
    )
    abs_scores = plugin.score_ensemble(ens.trajectories, scoring="expected_abs_p95")
    risk = torch.tensor([s.risk_score for s in abs_scores], dtype=torch.float64)
    pct = {q: float(torch.quantile(risk, q)) for q in (0.50, 0.90, 0.99)}
    rmax = float(risk.max())
    n = len(ens.trajectories)
    rows = []
    for label, budget in (("above worst orbit", rmax * 1.01), ("p99", pct[0.99]), ("p90", pct[0.90])):
        rep = select_reruns(risk, threshold=budget)
        rows.append((label, budget, rep.n_above_threshold, rep.n_flagged))
    zero_capable = rows[0][3] == 0
    lines = [
        "# VESP-UQ Absolute-Threshold (Zero-Alarm) Screening",
        "",
        "Cross-trajectory-comparable absolute force-risk budget (`expected_abs_p95`). An absolute",
        "budget flags only orbits exceeding it -- possibly zero (a fixed top-fraction cannot).",
        "",
        f"- per-orbit risk: p50={pct[0.50]:.3e}  p90={pct[0.90]:.3e}  p99={pct[0.99]:.3e}  max={rmax:.3e}",
        "",
        "| budget | level | above | flagged |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, budget, above, flagged in rows:
        tag = "ZERO ALARMS" if flagged == 0 else str(flagged)
        lines.append(f"| {label} | {budget:.3e} | {above}/{n} | {tag} |")
    lines += ["", f"- zero-alarm capability demonstrated: **{'YES' if zero_capable else 'NO'}**", ""]
    return "\n".join(lines) + "\n", {"zero_alarm_capable": bool(zero_capable), "p99": pct[0.99], "max": rmax}


def _position_error_diagnostic(out_dir: Path) -> tuple[str, dict]:
    required = [
        Path("data/test_512/scenarios.csv"),
        Path("data/test_512/metrics/gpu_batch_per_scenario_metrics.csv"),
        Path("data/lunar_grail_gl0420a_L60_residual.csv"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        md = "\n".join([
            "# VESP-UQ Position-Error Diagnostic (ST-LRPS) -- SKIPPED",
            "",
            "status: skipped",
            f"reason: missing file(s): {', '.join(missing)}",
            "",
            "This is an optional *diagnostic* (does force-risk co-rank long-horizon ST-LRPS position",
            "error?), not a core VESP-UQ claim. Reading ST-LRPS metrics is NOT a propagation adapter.",
            "",
        ]) + "\n"
        return md, {"status": "skipped", "reason": f"missing {missing}"}
    md = "\n".join([
        "# VESP-UQ Position-Error Diagnostic (ST-LRPS)",
        "",
        "status: data_available",
        "Run `python scripts/analyze_512_orbits.py` for the full force-risk vs ST-LRPS position-error",
        "diagnostic (requires scipy integration; not run inline to keep the suite fast).",
        "",
        "Reminder: this is a *diagnostic* comparison only. Force-risk is not expected to rank",
        "long-horizon position error when that error is not force-model-error dominated.",
        "",
    ]) + "\n"
    return md, {"status": "data_available"}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the VESP-UQ IAC benchmark suite.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="outputs/iac")
    parser.add_argument("--rerun-fraction", type=float, default=0.10)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    out = Path(args.out_dir)
    numbers: dict = {"config": args.config, "benchmarks": {}}

    prepared = prepare(config)
    plugin, samples, train, held, dtype, seed = prepared

    cal_md, cal_n = _calibration_report(plugin, held, config)
    numbers["benchmarks"]["calibration"] = {"status": "ok", **cal_n}

    fe = force_error_benchmark(config, rerun_fraction=args.rerun_fraction, prepared=prepared)
    fe_rows = fe.pop("_scores")
    header = ["trajectory_id", "force_risk", "true_force_error", "flagged"]
    force_error_csv = csv_text(header, [[row[column] for column in header] for row in fe_rows])
    numbers["benchmarks"]["force_error"] = {"status": "ok", **{k: v for k, v in fe.items()}}

    ood_md, ood_n = _ood_altitude_sweep(plugin)
    numbers["benchmarks"]["ood_altitude_sweep"] = {"status": "ok", **ood_n}

    abs_md, abs_n = _absolute_threshold_screening(plugin, config, seed)
    numbers["benchmarks"]["absolute_threshold_screening"] = {"status": "ok", **abs_n}

    pos_md, pos_n = _position_error_diagnostic(out)
    numbers["benchmarks"]["position_error_diagnostic"] = pos_n

    summary = "\n".join([
        "# VESP-UQ IAC Benchmark Summary",
        "",
        f"config: `{args.config}`",
        "",
        "VESP-UQ is a force-risk / OOD calibration layer. The core benchmarks below test force-model",
        "risk detection and follow-up prioritization -- NOT trajectory position-error prediction.",
        "",
        f"- **force-error ranking** Spearman: {fe['spearman_force_risk_vs_true_force_error']}  "
        f"(lift {fe['lift_over_random']:.2f}x) -- the core claim.",
        f"- **OOD altitude sweep**: expected force error grows toward low altitude: "
        f"{ood_n['expected_error_grows_low_altitude']}.",
        f"- **absolute threshold**: zero-alarm capable: {abs_n['zero_alarm_capable']}.",
        f"- **calibration**: low-band PICP90 {cal_n['low_band_picp_90']}.",
        f"- **position-error diagnostic**: {pos_n['status']} "
        f"(diagnostic only; not a VESP-UQ claim).",
        "",
        "Not claimed: deterministic trajectory-accuracy improvement; position-error prediction;",
        "operational orbit covariance propagation; ST-LRPS integration.",
        "",
    ]) + "\n"
    write_run_artifacts(
        out_dir=out,
        tool="run_iac_benchmarks",
        config=config,
        config_path=args.config,
        seed=seed,
        inputs={"config": args.config},
        json_files={
            "force_error_benchmark.json": fe,
            "iac_numbers.json": numbers,
        },
        text_files={
            "calibration_report.md": cal_md,
            "force_error_benchmark.md": _benchmark_md(fe),
            "force_error_scores.csv": force_error_csv,
            "ood_altitude_sweep.md": ood_md,
            "absolute_threshold_screening.md": abs_md,
            "position_error_diagnostic.md": pos_md,
            "iac_summary.md": summary,
        },
    )
    print(summary)
    print(f"saved_iac_suite: {out}")


if __name__ == "__main__":
    main()
