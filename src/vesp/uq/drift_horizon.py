"""Force-risk vs trajectory-drift multi-horizon diagnostic for VESP-UQ (WP10).

VESP-UQ's primary claim is that it ranks trajectory FORCE-model error (10.1). A separate, weaker
question is whether that force-risk also ranks long-horizon position drift (10.2). The ST-LRPS
appendix found a null long-horizon correlation; this module makes that a controlled *horizon
sweep*: for each trajectory it propagates the linearized force-error covariance (the established
:class:`vesp.uq.linear_propagation.LinearForceErrorCovariancePropagator`) and reads the model-
implied position dispersion at several horizons (1, 6, 12, 60 orbital periods -- roughly one orbit
to a few days for a low lunar orbit), then correlates force-risk and true force error with that
dispersion at each horizon.

Horizons are expressed in orbital periods (unit-free); no physical time scale is invented. The
position dispersion is the force-error-posterior-implied sigma, the same quantity the ST-LRPS
appendix reports -- this is a diagnostic, not a validated position-error propagation.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch

from vesp.uq.altitude_controlled import spearman
from vesp.uq.benchmarking import evaluate_score_against_true_error
from vesp.uq.ensemble import nearest_neighbor_error_magnitude
from vesp.uq.io.run_artifacts import write_run_artifacts
from vesp.uq.linear_propagation import LinearForceErrorCovariancePropagator
from vesp.uq.risk_baselines import prepare
from vesp.uq.scoring import aggregate_trajectory_error
from vesp.uq.suite import _csv, _fmt, _pm, band_label, git_commit_hash, mean_std

HORIZON_PERIODS = (1, 6, 12, 60)  # ~1 orbit, ~half day, ~1 day, ~5 days for a ~2 h low lunar orbit
_MU = 1.0
_SNAPS_PER_PERIOD = 20


def _random_rotations(n, generator, dtype):
    mats = torch.randn(n, 3, 3, generator=generator, dtype=dtype)
    q, r = torch.linalg.qr(mats)
    sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return q * sign.unsqueeze(-2)


def _build_initial_ensemble(n_orbits, n_points, seed, dtype):
    """Random-orientation orbits: return (paths, y0 states (N,6), periods, semi-major axis)."""

    g = torch.Generator().manual_seed(int(seed) + 5)
    r_peri = 1.04 + 0.20 * torch.rand(n_orbits, generator=g, dtype=dtype)
    r_apo = r_peri + 0.40 * torch.rand(n_orbits, generator=g, dtype=dtype)
    a = 0.5 * (r_peri + r_apo)
    e = (r_apo - r_peri) / (r_apo + r_peri).clamp_min(torch.finfo(dtype).tiny)
    p = a * (1.0 - e * e)
    period = 2.0 * math.pi * torch.sqrt(a ** 3 / _MU)
    v_peri = torch.sqrt(_MU * (2.0 / r_peri - 1.0 / a).clamp_min(0.0))

    theta = torch.linspace(0.0, 2.0 * math.pi, n_points + 1, dtype=dtype)[:-1]
    rot = _random_rotations(n_orbits, g, dtype)
    paths, states = [], torch.empty(n_orbits, 6, dtype=dtype)
    for k in range(n_orbits):
        r = p[k] / (1.0 + e[k] * torch.cos(theta))
        plane = torch.stack([r * torch.cos(theta), r * torch.sin(theta), torch.zeros_like(theta)], dim=-1)
        paths.append(plane @ rot[k].transpose(0, 1))
        r0 = rot[k] @ torch.tensor([r_peri[k], 0.0, 0.0], dtype=dtype)
        v0 = rot[k] @ torch.tensor([0.0, v_peri[k], 0.0], dtype=dtype)
        states[k] = torch.cat([r0, v0])
    return paths, states, period, a


def drift_horizon_run(config: dict, *, seed: int, n_orbits: int = 80, n_points: int = 120) -> dict:
    """One seed: force-risk, true force error, and position dispersion at each horizon per trajectory."""

    cfg = copy.deepcopy(config)
    cfg["seed"] = int(seed)
    plugin, _samples, _train, held, dtype, _ = prepare(cfg)
    band = band_label(cfg)
    aggregator = str(cfg.get("uq", {}).get("screening", {}).get("true_error_aggregator", "p95")).lower()

    paths, states, period, _a = _build_initial_ensemble(n_orbits, n_points, seed, dtype)

    # VESP-UQ supervisor force-risk + true force error per trajectory
    scored = plugin.score_ensemble(paths, scoring="supervisor_rel_p95")
    force_risk = torch.tensor([s.risk_score for s in scored], dtype=torch.float64)
    true_error = torch.empty(n_orbits, dtype=torch.float64)
    for i, traj in enumerate(paths):
        nn = nearest_neighbor_error_magnitude(traj.to(dtype), held.positions, held.error)
        true_error[i] = aggregate_trajectory_error(nn.to(torch.float64), aggregator)

    # one propagation per trajectory to the longest horizon; snapshot dispersion at each horizon
    max_k = max(HORIZON_PERIODS)
    prop = LinearForceErrorCovariancePropagator(plugin, dt_s=60.0, mu=_MU, dtype=dtype)
    dispersion = {k: torch.empty(n_orbits, dtype=torch.float64) for k in HORIZON_PERIODS}
    for i in range(n_orbits):
        T = float(period[i])
        out_dt = T / _SNAPS_PER_PERIOD
        res = prop.propagate(states[i].numpy(), duration_s=max_k * T, output_dt_s=out_dt)
        times = res.times
        for k in HORIZON_PERIODS:
            idx = int(np.argmin(np.abs(times - k * T)))
            dispersion[k][i] = float(res.position_sigma[idx])

    return {"band": band, "seed": int(seed), "n_orbits": n_orbits,
            "force_risk": force_risk, "true_error": true_error, "dispersion": dispersion}


def _horizon_metrics(run: dict) -> list[dict]:
    fr, te = run["force_risk"], run["true_error"]
    rows = []
    # 10.1 force-error ranking (the main claim), horizon-independent
    rows.append({"band": run["band"], "seed": run["seed"], "horizon_periods": 0,
                 "diagnostic": "force_error_ranking",
                 "spearman_forcerisk_vs_target": spearman(fr, te),
                 "spearman_trueerror_vs_target": float("nan"),
                 "capture10_forcerisk": float(
                     evaluate_score_against_true_error(fr, te, rerun_fraction=0.10)["capture_rate"])})
    # 10.2 drift ranking at each horizon
    for k in HORIZON_PERIODS:
        disp = run["dispersion"][k]
        rows.append({"band": run["band"], "seed": run["seed"], "horizon_periods": k,
                     "diagnostic": "drift_ranking",
                     "spearman_forcerisk_vs_target": spearman(fr, disp),
                     "spearman_trueerror_vs_target": spearman(te, disp),
                     "capture10_forcerisk": float(
                         evaluate_score_against_true_error(fr, disp, rerun_fraction=0.10)["capture_rate"])})
    return rows


def _aggregate(rows) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["band"], r["diagnostic"], r["horizon_periods"]), []).append(r)
    out = {}
    for key, rs in groups.items():
        out[key] = {
            "spearman_forcerisk_vs_target": mean_std([r["spearman_forcerisk_vs_target"] for r in rs]),
            "spearman_trueerror_vs_target": mean_std([r["spearman_trueerror_vs_target"] for r in rs]),
            "capture10_forcerisk": mean_std([r["capture10_forcerisk"] for r in rs]),
            "n_seeds": len(rs),
        }
    return out


def _csv_table(agg) -> str:
    cols = ["band", "diagnostic", "horizon_periods", "n_seeds",
            "spearman_forcerisk_vs_target_mean", "spearman_forcerisk_vs_target_std",
            "spearman_trueerror_vs_target_mean", "capture10_forcerisk_mean"]
    rows = [cols]
    for (band, diag, k), a in sorted(agg.items()):
        rows.append([band, diag, k, a["n_seeds"],
                     a["spearman_forcerisk_vs_target"]["mean"], a["spearman_forcerisk_vs_target"]["std"],
                     a["spearman_trueerror_vs_target"]["mean"], a["capture10_forcerisk"]["mean"]])
    return _csv(rows)


def _md(agg) -> str:
    lines = [
        "# VESP-UQ Force-Risk vs Trajectory-Drift, Multi-Horizon (WP10)",
        "",
        "Two diagnostics on the same ensemble. (10.1) Force-error ranking is the main VESP-UQ claim. "
        "(10.2) Drift ranking asks whether the force-risk also ranks the model-implied position "
        "dispersion at horizons of 1, 6, 12, 60 orbital periods (~1 orbit to ~5 days for a low lunar "
        "orbit). Horizons are in orbital periods; the dispersion is the force-error-posterior sigma "
        "(diagnostic, not a validated position-error propagation). Mean +/- std across seeds.",
        "",
        "| band | diagnostic | horizon (periods) | spearman(force-risk, target) | spearman(true err, target) | capture@10 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for (band, diag, k), a in sorted(agg.items()):
        hz = "force error" if diag == "force_error_ranking" else f"{k}"
        te_sp = a["spearman_trueerror_vs_target"]["mean"]
        lines.append(
            f"| {band} | {diag} | {hz} | {_pm(a['spearman_forcerisk_vs_target'], '.3f')} | "
            f"{_fmt(te_sp, '.3f') if te_sp == te_sp else 'n/a'} | {_pm(a['capture10_forcerisk'], '.3f')} |"
        )
    lines += [
        "",
        "Interpretation: VESP-UQ ranks force-model error (10.1). If the drift-ranking correlation "
        "(10.2) is strong at short horizons and decays with horizon, the loss of correlation is a "
        "horizon/dynamics effect (error accumulation, phasing, cancellation), not a failure of the "
        "force-risk signal. A null long-horizon correlation supports the scope boundary that VESP-UQ "
        "ranks force-model risk, not long-horizon position error.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _plot(agg, out_path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    bands = sorted({b for (b, _, _) in agg})
    fig, ax = plt.subplots(figsize=(7, 5))
    for band in bands:
        pts = sorted([(k, a["spearman_forcerisk_vs_target"]["mean"])
                      for (b, d, k), a in agg.items() if b == band and d == "drift_ranking"],
                     key=lambda t: t[0])
        xs = [p[0] for p in pts]
        ys = [p[1] if p[1] is not None else 0.0 for p in pts]
        ax.plot(xs, ys, marker="o", label=f"{band}: force-risk vs drift")
    ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("horizon (orbital periods)")
    ax.set_ylabel("Spearman(force-risk, position dispersion)")
    ax.set_title("Force-risk vs drift ranking across horizons")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def run_drift_horizon(
    configs, *, seeds=(0, 1, 2), n_orbits=80, n_points=120,
    out_dir="outputs/drift_horizon/", make_plots=True,
) -> dict:
    """Run the multi-horizon force-risk-vs-drift diagnostic over configs x seeds; write artifacts."""

    out_dir = Path(out_dir)
    runs = [drift_horizon_run(cfg, seed=s, n_orbits=n_orbits, n_points=n_points)
            for cfg in configs for s in seeds]
    rows = [r for run in runs for r in _horizon_metrics(run)]
    agg = _aggregate(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_files = {}
    if make_plots and _plot(agg, out_dir / "drift_horizon_curve.png"):
        artifact_files["drift_horizon_curve.png"] = out_dir / "drift_horizon_curve.png"

    write_run_artifacts(
        out_dir,
        tool="run_drift_horizon",
        config=configs[0],
        json_files={"drift_horizon_meta.json": {
            "git_commit": git_commit_hash(), "seeds": list(seeds),
            "horizon_periods": list(HORIZON_PERIODS), "n_orbits": n_orbits, "mu": _MU,
        }},
        text_files={
            "drift_horizon.csv": _csv_table(agg),
            "drift_horizon.md": _md(agg),
        },
        artifact_files=artifact_files,
        manifest_name="manifest.json",
    )
    return {"out_dir": str(out_dir), "agg": agg}
