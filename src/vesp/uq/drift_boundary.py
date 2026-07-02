"""Drift-boundary characterization: where does force-risk predict position drift? (B)

WP10 showed force-risk ranks the model-implied position dispersion at short horizons and decays at
long ones; WP9 showed VESP-UQ's ranking value is family-dependent. This module combines them into a
**(family x horizon) grid**: for each controlled orbit family and each horizon it correlates the
VESP-UQ force-risk (and the true force error) with the propagated position dispersion. The result is
a map of the regime where force-risk predicts drift -- a *characterization*, never a validated
position-error propagation (the dispersion is the force-error-posterior sigma, per the ST-LRPS
caveat in VESP_UQ_LIMITATIONS.md).

Horizons are in orbital periods (unit-free). Claims-safe: a positive result is reported as a rank
correlation in a named regime; a null is reported as a scope boundary, never as a failure.

Relationship to :mod:`vesp.uq.drift_horizon`: ``drift_horizon`` is the generic random-orbit
multi-horizon sweep; this module is the family-conditioned boundary map that reuses the same
linear-propagation primitive but intentionally fixes the trajectory families.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from vesp.uq.altitude_controlled import spearman
from vesp.uq.baselines import prepare
from vesp.uq.benchmarking import evaluate_score_against_true_error
from vesp.uq.ensemble import nearest_neighbor_error_magnitude
from vesp.uq.io.run_artifacts import write_run_artifacts
from vesp.uq.linear_propagation import LinearForceErrorCovariancePropagator
from vesp.uq.scoring import aggregate_trajectory_error
from vesp.uq.suite import _csv, _fmt, _pm, band_label, git_commit_hash, mean_std
from vesp.uq.trajectory_families import generate_family

HORIZON_PERIODS = (1, 3, 6, 12)  # short-horizon focus: where force-risk should predict local drift
# Families that span the drift-relevant regimes (low/eccentric/polar/high); a tractable subset.
DEFAULT_FAMILIES = ("low_alt_near_circular", "eccentric_perilune", "polar", "high_alt_transfer")
_MU = 1.0
_SNAPS_PER_PERIOD = 12
_MEANINGFUL_SPEARMAN = 0.20  # boundary threshold for "force-risk predicts drift"


def drift_boundary_run(config: dict, *, seed: int, families, n_orbits: int = 20, n_points: int = 120) -> dict:
    """One seed: per-family, per-horizon correlation of force-risk / true error with drift."""

    cfg = copy.deepcopy(config)
    cfg["seed"] = int(seed)
    plugin, _samples, _train, held, dtype, _ = prepare(cfg)
    band = band_label(cfg)
    aggregator = str(cfg.get("uq", {}).get("screening", {}).get("true_error_aggregator", "p95")).lower()
    prop = LinearForceErrorCovariancePropagator(plugin, dt_s=60.0, mu=_MU, dtype=dtype)
    max_k = max(HORIZON_PERIODS)

    rows = []
    for fam_name in families:
        fam = generate_family(fam_name, n_orbits=n_orbits, n_points=n_points, seed=int(seed), dtype=dtype)
        paths = fam.trajectories
        scored = plugin.score_ensemble(paths, scoring="supervisor_rel_p95")
        force_risk = torch.tensor([s.risk_score for s in scored], dtype=torch.float64)
        true_error = torch.empty(len(paths), dtype=torch.float64)
        for i, traj in enumerate(paths):
            nn = nearest_neighbor_error_magnitude(traj.to(dtype), held.positions, held.error)
            true_error[i] = aggregate_trajectory_error(nn.to(torch.float64), aggregator)

        dispersion = {k: torch.empty(len(paths), dtype=torch.float64) for k in HORIZON_PERIODS}
        assert fam.period is not None and fam.initial_states is not None
        for i in range(len(paths)):
            T = float(fam.period[i])
            res = prop.propagate(fam.initial_states[i].numpy(), duration_s=max_k * T,
                                 output_dt_s=T / _SNAPS_PER_PERIOD)
            for k in HORIZON_PERIODS:
                idx = int(np.argmin(np.abs(res.times - k * T)))
                dispersion[k][i] = float(res.position_sigma[idx])

        for k in HORIZON_PERIODS:
            disp = dispersion[k]
            rows.append({
                "band": band, "seed": int(seed), "family": fam_name, "horizon_periods": k,
                "spearman_forcerisk_vs_drift": spearman(force_risk, disp),
                "spearman_trueerror_vs_drift": spearman(true_error, disp),
                "capture10_forcerisk_vs_drift": float(
                    evaluate_score_against_true_error(force_risk, disp, rerun_fraction=0.10)["capture_rate"]),
            })
    return {"band": band, "seed": int(seed), "rows": rows}


def _aggregate(rows) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["band"], r["family"], r["horizon_periods"]), []).append(r)
    out = {}
    for key, rs in groups.items():
        out[key] = {
            "spearman_forcerisk_vs_drift": mean_std([r["spearman_forcerisk_vs_drift"] for r in rs]),
            "spearman_trueerror_vs_drift": mean_std([r["spearman_trueerror_vs_drift"] for r in rs]),
            "capture10_forcerisk_vs_drift": mean_std([r["capture10_forcerisk_vs_drift"] for r in rs]),
            "n_seeds": len(rs),
        }
    return out


def _verdict(agg: dict) -> dict:
    """Per (band, family): the horizon regime in which force-risk predicts drift (peak + range).

    The correlation typically peaks at an *intermediate* horizon -- at 1 period the dispersion is
    too small/noisy, and at long horizons phasing/accumulation decouple it -- so the regime is the
    set of horizons clearing the threshold, not a band anchored at period 1.
    """

    verdict = {}
    keys = sorted({(b, f) for (b, f, _) in agg})
    for band, fam in keys:
        scored = []
        for k in sorted(HORIZON_PERIODS):
            m = agg.get((band, fam, k), {}).get("spearman_forcerisk_vs_drift", {}).get("mean")
            if m is not None:
                scored.append((k, m))
        predicting = [k for k, m in scored if m >= _MEANINGFUL_SPEARMAN]
        peak_k, peak_v = (max(scored, key=lambda t: t[1]) if scored else (None, None))
        verdict[(band, fam)] = {
            "predicts_drift": bool(predicting),
            "predicting_horizons": predicting,
            "predicts_drift_from_periods": (min(predicting) if predicting else 0),
            "predicts_drift_up_to_periods": (max(predicting) if predicting else 0),
            "peak_horizon_periods": peak_k,
            "peak_spearman": peak_v,
        }
    return verdict


def _csv_table(agg: dict) -> str:
    cols = ["band", "family", "horizon_periods", "n_seeds",
            "spearman_forcerisk_vs_drift_mean", "spearman_forcerisk_vs_drift_std",
            "spearman_trueerror_vs_drift_mean", "capture10_forcerisk_vs_drift_mean"]
    rows = [cols]
    for (band, fam, k), a in sorted(agg.items()):
        rows.append([band, fam, k, a["n_seeds"],
                     a["spearman_forcerisk_vs_drift"]["mean"], a["spearman_forcerisk_vs_drift"]["std"],
                     a["spearman_trueerror_vs_drift"]["mean"], a["capture10_forcerisk_vs_drift"]["mean"]])
    return _csv(rows)


def _md(agg: dict, verdict: dict) -> str:
    lines = [
        "# VESP-UQ Drift-Boundary Characterization (B)",
        "",
        "Per orbit family x horizon: Spearman of the VESP-UQ force-risk with the propagated position "
        "dispersion (the force-error-posterior sigma). This maps *where* force-risk predicts drift. "
        "Horizons are in orbital periods; the dispersion is a diagnostic, not a validated position-"
        "error propagation. Mean +/- std across seeds.",
        "",
        "| band | family | horizon (periods) | spearman(force-risk, drift) | capture@10 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for (band, fam, k), a in sorted(agg.items()):
        lines.append(
            f"| {band} | {fam} | {k} | {_pm(a['spearman_forcerisk_vs_drift'], '.3f')} | "
            f"{_pm(a['capture10_forcerisk_vs_drift'], '.3f')} |"
        )
    lines += ["", "## Boundary verdict", ""]
    for (band, fam), v in sorted(verdict.items()):
        if v["predicts_drift"]:
            lo, hi = v["predicts_drift_from_periods"], v["predicts_drift_up_to_periods"]
            rng = f"~{lo} period" if lo == hi else f"~{lo}-{hi} orbital periods"
            lines.append(f"- **{band} / {fam}**: force-risk predicts position drift at {rng} "
                         f"(peak Spearman {_fmt(v['peak_spearman'], '.3f')} at "
                         f"{v['peak_horizon_periods']} periods; threshold {_MEANINGFUL_SPEARMAN}).")
        else:
            lines.append(f"- **{band} / {fam}**: force-risk does NOT predict position drift at any "
                         "tested horizon (scope boundary).")
    lines += [
        "",
        "Interpretation: families/horizons where force-risk tracks drift are the regime in which the "
        "force-model-error signal dominates the dispersion; elsewhere phasing, accumulation, and "
        "cancellation decouple them. This characterizes the operational reach of the force-risk score "
        "without claiming a validated position-error or orbit-covariance product.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _plot_heatmap(agg: dict, out_path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    bands = sorted({b for (b, _, _) in agg})
    horizons = sorted({k for (_, _, k) in agg})
    fig, axes = plt.subplots(1, len(bands), figsize=(6 * max(1, len(bands)), 5), squeeze=False)
    for bi, band in enumerate(bands):
        ax = axes[0][bi]
        families = sorted({f for (b, f, _) in agg if b == band})
        grid = np.full((len(families), len(horizons)), np.nan)
        for fi, fam in enumerate(families):
            for ki, k in enumerate(horizons):
                v = agg.get((band, fam, k), {}).get("spearman_forcerisk_vs_drift", {}).get("mean")
                if v is not None:
                    grid[fi, ki] = v
        im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6)
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels([str(h) for h in horizons])
        ax.set_yticks(range(len(families)))
        ax.set_yticklabels(families, fontsize=8)
        ax.set_xlabel("horizon (orbital periods)")
        ax.set_title(f"{band}: Spearman(force-risk, drift)")
        for fi in range(len(families)):
            for ki in range(len(horizons)):
                if not np.isnan(grid[fi, ki]):
                    ax.text(ki, fi, f"{grid[fi, ki]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def run_drift_boundary(
    configs, *, seeds=(0, 1), families=None, n_orbits=20, n_points=120,
    out_dir="outputs/drift_boundary/", make_plots=True,
) -> dict:
    """Run the (family x horizon) drift-boundary characterization; write grid + heatmap + verdict."""

    out_dir = Path(out_dir)
    families = list(families) if families else list(DEFAULT_FAMILIES)
    runs = [drift_boundary_run(cfg, seed=s, families=families, n_orbits=n_orbits, n_points=n_points)
            for cfg in configs for s in seeds]
    rows = [r for run in runs for r in run["rows"]]
    agg = _aggregate(rows)
    verdict = _verdict(agg)

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_files = {}
    if make_plots and _plot_heatmap(agg, out_dir / "drift_boundary_heatmap.png"):
        artifact_files["drift_boundary_heatmap.png"] = out_dir / "drift_boundary_heatmap.png"

    write_run_artifacts(
        out_dir,
        tool="run_drift_boundary",
        config=configs[0],
        json_files={"drift_boundary_meta.json": {
            "git_commit": git_commit_hash(), "seeds": list(seeds), "families": families,
            "horizon_periods": list(HORIZON_PERIODS), "n_orbits": n_orbits,
            "meaningful_spearman": _MEANINGFUL_SPEARMAN,
            "verdict": {f"{b}/{f}": v for (b, f), v in verdict.items()},
        }},
        text_files={
            "drift_boundary.csv": _csv_table(agg),
            "drift_boundary.md": _md(agg, verdict),
        },
        artifact_files=artifact_files,
        manifest_name="manifest.json",
    )
    return {"out_dir": str(out_dir), "agg": agg, "verdict": verdict}
