#!/usr/bin/env python3
"""
N10 Exploratory Diagnostic: Dynamics-aware risk via STM dispersion.

Runs the 512-orbit ST-LRPS position-error diagnostic but uses the
linearized STM propagator to score trajectories by their predicted
position-dispersion scalar (`max(position_sigma)`).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from vesp.uq.data import load_uq_samples_from_csv
from vesp.uq.io.run_artifacts import write_run_artifacts
from vesp.uq.linear_propagation import score_stm_dispersion
from vesp.uq.plugin import VESPUQPlugin
from vesp.uq.selection import _spearman, select_reruns


def kepler_to_cartesian(a_norm, e, inc_deg, raan_deg, argp_deg, ta_deg, mu=1.0):
    ta = np.radians(ta_deg)
    inc = np.radians(inc_deg)
    raan = np.radians(raan_deg)
    argp = np.radians(argp_deg)

    r = a_norm * (1 - e**2) / (1 + e * np.cos(ta))

    x_orb = r * np.cos(ta)
    y_orb = r * np.sin(ta)

    p = a_norm * (1 - e**2)
    h = np.sqrt(mu * p)
    vx_orb = (mu / h) * -np.sin(ta)
    vy_orb = (mu / h) * (e + np.cos(ta))

    R3_W = np.array([[np.cos(raan), -np.sin(raan), 0],
                     [np.sin(raan), np.cos(raan), 0],
                     [0, 0, 1]])
    R1_i = np.array([[1, 0, 0],
                     [0, np.cos(inc), -np.sin(inc)],
                     [0, np.sin(inc), np.cos(inc)]])
    R3_w = np.array([[np.cos(argp), -np.sin(argp), 0],
                     [np.sin(argp), np.cos(argp), 0],
                     [0, 0, 1]])

    Q = R3_W @ R1_i @ R3_w
    r_vec = Q @ np.array([x_orb, y_orb, 0])
    v_vec = Q @ np.array([vx_orb, vy_orb, 0])
    return r_vec, v_vec


def main(argv=None):
    parser = argparse.ArgumentParser(description="N10: STM-dispersion vs ST-LRPS position-error diagnostic.")
    parser.add_argument("--config", default="configs/vespuq/vespuq_real_lunar.yaml")
    parser.add_argument("--data", default="data/lunar_grail_gl0420a_L60_residual.csv")
    parser.add_argument("--scenarios-dir", default="data/test_512", help="dir with scenarios.csv + metrics/")
    parser.add_argument("--out-dir", default="outputs/stm_dispersion")
    args = parser.parse_args(argv)

    cfg_path = args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    scenarios_dir = Path(args.scenarios_dir)
    scenarios_csv = scenarios_dir / "scenarios.csv"
    metrics_csv = scenarios_dir / "metrics" / "gpu_batch_per_scenario_metrics.csv"
    if not scenarios_csv.is_file() or not metrics_csv.is_file():
        raise SystemExit(
            f"missing ST-LRPS diagnostic data: expected {scenarios_csv} and {metrics_csv} "
            "(this diagnostic needs the precomputed 512-scenario set; see "
            "benchmarks/position_error_diagnostic.md)"
        )

    device = torch.device(cfg.get("device", "cpu"))
    dtype = torch.float64

    print("Loading test data & fitting VESP-UQ...")
    samples = load_uq_samples_from_csv(args.data)
    DU_km = 1738.0
    GM_km3_s2 = 4902.800066
    ACCEL_REF_KM_S2 = GM_km3_s2 / (DU_km**2)
    TU_s = np.sqrt((DU_km**3) / GM_km3_s2)

    train_pos = samples.positions.to(device)
    train_err_norm = samples.error.to(device) / ACCEL_REF_KM_S2

    plugin = VESPUQPlugin.from_config(cfg)
    plugin.fit_error(train_pos, train_err_norm)

    print("Loading 512 LUNAR test scenarios...")
    scenarios = pd.read_csv(scenarios_csv)
    metrics = pd.read_csv(metrics_csv)

    st_lrps_metrics = metrics[metrics['model'] == 'ST_LRPS_DT60'].copy()
    if len(st_lrps_metrics) == 0:
        model_name = metrics['model'].iloc[0]
        st_lrps_metrics = metrics[metrics['model'] == model_name].copy()

    st_lrps_metrics = st_lrps_metrics.sort_values('scenario_id').reset_index(drop=True)
    scenarios = scenarios.sort_values('scenario_id').reset_index(drop=True)

    true_errors = []
    y0s = []

    print("Computing initial states for STM propagation...")
    # half a day is 12 hours = 43200 s
    duration_tu = 43200.0 / TU_s
    output_dt_tu = duration_tu / 60.0

    for i in range(len(scenarios)):
        a_km = scenarios.loc[i, 'a_km']
        a_norm = a_km / DU_km
        e = scenarios.loc[i, 'e']
        inc_deg = scenarios.loc[i, 'inc_deg']
        raan_deg = scenarios.loc[i, 'raan_deg']
        argp_deg = scenarios.loc[i, 'argp_deg']
        ta_deg = scenarios.loc[i, 'ta_deg']

        r0, v0 = kepler_to_cartesian(a_norm, e, inc_deg, raan_deg, argp_deg, ta_deg)
        y0 = np.concatenate([r0, v0])
        y0s.append(y0)
        true_errors.append(st_lrps_metrics.loc[i, 'rms_pos_err_km'])

    n = len(scenarios)
    y0s_arr = np.array(y0s)
    true_err_t = torch.tensor([float(e) for e in true_errors], dtype=torch.float64)

    print("\n==================================================================")
    print("  VESP-UQ STM DISPERSION vs ST-LRPS POSITION-ERROR DIAGNOSTIC (512 orbits)")
    print("==================================================================")
    print("WARNING: this is an EXPLORATORY diagnostic, not a direct position-error claim.")

    print("\nComputing STM dispersion scores... (this may take a minute)")
    risk_scores = score_stm_dispersion(
        plugin,
        y0s_arr,
        duration_s=duration_tu,
        output_dt_s=output_dt_tu,
        device=device,
        dtype=dtype
    )

    rerun_fraction = 0.10
    k = int(np.ceil(rerun_fraction * n))

    rep = select_reruns(risk_scores, rerun_fraction=rerun_fraction, fraction_policy="topk")
    spearman_corr = _spearman(risk_scores, true_err_t)

    high_thr = float(torch.quantile(true_err_t, 1.0 - rerun_fraction))
    truly_high = true_err_t >= high_thr
    n_high = int(truly_high.sum())

    flagged_mask = torch.zeros(n, dtype=torch.bool)
    flagged_mask[rep.flagged_indices] = True

    capture_rate = float((flagged_mask & truly_high).sum()) / max(1, n_high)
    precision = float((flagged_mask & truly_high).sum()) / max(1, rep.n_flagged)
    lift = capture_rate / rep.rerun_fraction if rep.rerun_fraction > 0 else float("nan")

    mean_err_flagged = float(true_err_t[flagged_mask].mean()) if rep.n_flagged > 0 else float("nan")
    mean_err_accepted = float(true_err_t[~flagged_mask].mean()) if rep.n_flagged < n else float("nan")
    ratio = mean_err_flagged / mean_err_accepted if mean_err_accepted > 0 else float("nan")

    print(f"\n--- RELATIVE RANKING (scoring=stm_dispersion, top {rerun_fraction:.0%}) ---")
    print(f"Total Trajectories: {n}")
    print(f"Spearman (STM dispersion vs ST-LRPS position error): {spearman_corr:.4f}")
    print(f"Capture Rate (top-risk catching top-{rerun_fraction:.0%} error): {capture_rate*100:.1f}%")
    print(f"Precision: {precision*100:.1f}%")
    print(f"Lift over random (capture / rerun fraction): {lift:.2f}x")
    print(f"Mean true error flagged: {mean_err_flagged:.3f} km  vs  "
          f"accepted: {mean_err_accepted:.3f} km  (ratio {ratio:.2f}x)")

    # random baseline
    g = torch.Generator().manual_seed(0)
    rand_caps = []
    for _ in range(100):
        idx = torch.randperm(n, generator=g)[:k]
        mask = torch.zeros(n, dtype=torch.bool)
        mask[idx] = True
        rand_caps.append(float((mask & truly_high).sum()) / max(1, n_high))
    rand_caps = np.array(rand_caps)
    print(f"Random baseline capture (100 masks): mean={rand_caps.mean()*100:.1f}% +/- {rand_caps.std()*100:.1f}%")

    md_lines = [
        "# VESP-UQ STM Dispersion Diagnostic",
        "",
        "Exploratory diagnostic derived from the force-error posterior. This tests whether",
        "weighting the force-error posterior by trajectory dynamics (using `linear_propagation`)",
        "better correlates with long-horizon ST-LRPS position drift.",
        "",
        "**WARNING**: This is a diagnostic evaluation, NOT a validated position-error prediction",
        "claim. VESP-UQ remains a force-risk calibration layer.",
        "",
        "- **Scoring mode:** `stm_dispersion` (`max(sqrt(trace(P_rr)))` along integrated trajectory)",
        f"- **Trajectories:** {n} Keplerian orbits",
        f"- **Rerun fraction:** {rerun_fraction:.0%}",
        "",
        "### Results",
        "",
        f"- **Spearman correlation:** {spearman_corr:.4f}",
        f"- **Capture Rate:** {capture_rate*100:.1f}%",
        f"- **Precision:** {precision*100:.1f}%",
        f"- **Lift over random:** {lift:.2f}x",
        f"- **Mean true error flagged:** {mean_err_flagged:.3f} km vs accepted: {mean_err_accepted:.3f} km (ratio {ratio:.2f}x)",
        f"- **Random baseline capture:** {rand_caps.mean()*100:.1f}% +/- {rand_caps.std()*100:.1f}%",
        ""
    ]

    out_md = "\n".join(md_lines)

    # Route the run outputs through the provenance/artifact layer (N1 convention): JSON results
    # + Markdown + run_manifest.json with checksums, inputs included. The curated copy under
    # benchmarks/stm_dispersion_diagnostic.md is a committed snapshot of these numbers.
    results = {
        "diagnostic": "stm_dispersion_vs_stlrps_position_error",
        "is_position_error_benchmark": False,
        "note": (
            "exploratory diagnostic: rank agreement between the STM position-dispersion score "
            "and precomputed ST-LRPS position error; NOT a position-error prediction claim"
        ),
        "n_trajectories": n,
        "rerun_fraction": rerun_fraction,
        "spearman": float(spearman_corr),
        "capture_rate": capture_rate,
        "precision": precision,
        "lift_over_random": lift,
        "mean_true_error_flagged_km": mean_err_flagged,
        "mean_true_error_accepted_km": mean_err_accepted,
        "error_ratio_flagged_to_accepted": ratio,
        "random_baseline_capture_mean": float(rand_caps.mean()),
        "random_baseline_capture_std": float(rand_caps.std()),
    }
    write_run_artifacts(
        out_dir=args.out_dir,
        tool="benchmark_stm_dispersion",
        json_files={"stm_dispersion_diagnostic.json": results},
        text_files={"stm_dispersion_diagnostic.md": out_md},
        config={"_config_path": cfg_path, **cfg},
        inputs={
            "calibration_data": args.data,
            "scenarios_csv": scenarios_csv,
            "metrics_csv": metrics_csv,
        },
        seed=cfg.get("seed"),
    )
    print(f"\nWrote benchmark diagnostic artifacts to {args.out_dir}")

if __name__ == '__main__':
    main()
