"""Trajectory-family diversity study driver for VESP-UQ (WP9).

For each controlled orbit family (:mod:`vesp.uq.trajectory_families`) this scores VESP-UQ and the
baseline selectors against trajectory true FORCE-model error and reports, per family: the altitude
/ inclination / eccentricity ranges, the VESP-UQ supervisor ranking metrics, the best baseline, and
whether VESP-UQ adds ranking value beyond altitude (partial correlation given min-radius). Reuses
the fit / true-error / score-assembly core so the family study and the main suite agree.

Targets force-model error, never position error.
"""

from __future__ import annotations

import copy
from pathlib import Path

from vesp.uq.altitude_controlled import min_radius_scores, partial_pearson_given_altitude
from vesp.uq.baselines import random_scores
from vesp.uq.benchmarking import compare_baselines
from vesp.uq.io.run_artifacts import write_run_artifacts
from vesp.uq.risk_baselines import assemble_baseline_scores, prepare, true_force_error
from vesp.uq.suite import _csv, _fmt, _pm, band_label, git_commit_hash, mean_std
from vesp.uq.trajectory_families import FAMILIES, family_descriptor, generate_family

DEFAULT_FRACTIONS = (0.05, 0.10, 0.20, 0.30)
PRIMARY_FRACTION = 0.20
_SELECTORS = ("random", "min_altitude", "low_altitude_exposure", "uncertainty_only",
              "knn_p95", "domain_support", "supervisor")
_BASELINE_POOL = ("min_altitude", "low_altitude_exposure", "uncertainty_only", "knn_p95",
                  "domain_support")
_RANK_METRICS = ("spearman", "capture_rate", "precision", "lift_over_random",
                 "force_error_ratio_flagged_to_accepted")


def family_run(config: dict, *, seed: int, families, n_orbits: int, n_points: int,
               fractions=DEFAULT_FRACTIONS) -> dict:
    """Score every family at one seed; return per-(family, selector, fraction) ranking rows."""

    cfg = copy.deepcopy(config)
    cfg["seed"] = int(seed)
    plugin, _samples, train, held, dtype, _ = prepare(cfg)
    band = band_label(cfg)
    aggregator = str(cfg.get("uq", {}).get("screening", {}).get("true_error_aggregator", "p95")).lower()

    rows, descriptors, partial = [], {}, {}
    for fam_name in families:
        fam = generate_family(fam_name, n_orbits=n_orbits, n_points=n_points, seed=int(seed), dtype=dtype)
        trajectories = fam.trajectories
        descriptors[fam_name] = family_descriptor(fam)
        te, _ = true_force_error(trajectories, residuals=None, held=held, aggregator=aggregator, dtype=dtype)
        scores = assemble_baseline_scores(cfg, plugin, trajectories, train.positions)
        scores["random"] = random_scores(len(trajectories), seed=int(seed))
        resolved = {name: scores[name] for name in _SELECTORS if name in scores}

        min_radius = min_radius_scores(trajectories)
        partial[fam_name] = {
            name: partial_pearson_given_altitude(resolved[name], te, min_radius)
            for name in resolved
        }
        for frac in fractions:
            results = compare_baselines(resolved, te, rerun_fraction=float(frac))
            for selector, m in results.items():
                rows.append({"band": band, "seed": int(seed), "family": fam_name, "selector": selector,
                             "rerun_fraction": float(frac), **{k: m.get(k) for k in _RANK_METRICS}})
    return {"band": band, "seed": int(seed), "rows": rows, "descriptors": descriptors, "partial": partial}


def _aggregate(rows) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["band"], r["family"], r["selector"], round(float(r["rerun_fraction"]), 6))
        groups.setdefault(key, []).append(r)
    out = {}
    for key, rs in groups.items():
        agg = {m: mean_std([r.get(m) for r in rs]) for m in _RANK_METRICS}
        agg["n_seeds"] = len(rs)  # type: ignore[assignment]
        out[key] = agg
    return out


def _aggregate_partial(runs) -> dict:
    out: dict[tuple, dict] = {}
    bands = sorted({r["band"] for r in runs})
    for band in bands:
        for fam in sorted({f for r in runs if r["band"] == band for f in r["partial"]}):
            sels = sorted({s for r in runs if r["band"] == band for s in r["partial"].get(fam, {})})
            out[(band, fam)] = {
                s: mean_std([r["partial"].get(fam, {}).get(s) for r in runs if r["band"] == band])
                for s in sels
            }
    return out


def _best_baseline(agg: dict, band: str, fam: str, frac: float) -> tuple[str | None, float | None]:
    best, best_val = None, -2.0
    for name in _BASELINE_POOL:
        a = agg.get((band, fam, name, round(frac, 6)))
        if a and a["spearman"]["mean"] is not None and a["spearman"]["mean"] > best_val:
            best, best_val = name, a["spearman"]["mean"]
    return best, (best_val if best is not None else None)


def _summary_csv(agg: dict, partial_agg: dict, descriptors: dict, primary: float) -> str:
    cols = ["band", "family", "n_trajectories", "min_radius_low", "min_radius_high",
            "inclination_deg_low", "inclination_deg_high", "eccentricity_low", "eccentricity_high",
            "supervisor_spearman_mean", "supervisor_capture_mean", "supervisor_lift_mean",
            "supervisor_partial_given_alt_mean", "best_baseline", "best_baseline_spearman",
            "vespuq_adds_value_beyond_altitude"]
    rows = [cols]
    keys = sorted({(b, f) for (b, f, _, _) in agg})
    for band, fam in keys:
        d = descriptors.get((band, fam), {})
        sup = agg.get((band, fam, "supervisor", round(primary, 6)), {})
        alt = agg.get((band, fam, "min_altitude", round(primary, 6)), {})
        sup_partial = partial_agg.get((band, fam), {}).get("supervisor", {})
        best, best_val = _best_baseline(agg, band, fam, primary)
        sup_sp = sup.get("spearman", {}).get("mean") if sup else None
        alt_sp = alt.get("spearman", {}).get("mean") if alt else None
        adds = bool(
            (sup_partial.get("mean") is not None and sup_partial["mean"] > 0.05)
            or (sup_sp is not None and alt_sp is not None and sup_sp > alt_sp + 0.02)
        )
        row_vals = [
            band, fam, d.get("n_trajectories"), d.get("min_radius_low"), d.get("min_radius_high"),
            d.get("inclination_deg_low"), d.get("inclination_deg_high"),
            d.get("eccentricity_low"), d.get("eccentricity_high"),
            sup_sp, sup.get("capture_rate", {}).get("mean") if sup else None,
            sup.get("lift_over_random", {}).get("mean") if sup else None,
            sup_partial.get("mean"), best, best_val, adds,
        ]
        rows.append([str(v) if v is not None else "" for v in row_vals])
    return _csv(rows)


def _summary_md(agg: dict, partial_agg: dict, descriptors: dict, primary: float) -> str:
    lines = [
        "# VESP-UQ Trajectory-Family Diversity (WP9)",
        "",
        f"Per controlled orbit family: altitude / inclination / eccentricity ranges, the VESP-UQ "
        f"supervisor ranking at the {primary:.0%} budget, the best baseline, and whether VESP-UQ adds "
        "ranking value beyond altitude (partial correlation given min-radius). Mean +/- std across seeds.",
        "",
        "| band | family | n | min-alt range | incl (deg) | ecc | sup spearman | sup capture | "
        "partial(given alt) | best baseline | adds value? |",
        "| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- | :---: |",
    ]
    keys = sorted({(b, f) for (b, f, _, _) in agg})
    for band, fam in keys:
        d = descriptors.get((band, fam), {})
        sup = agg.get((band, fam, "supervisor", round(primary, 6)), {})
        sup_partial = partial_agg.get((band, fam), {}).get("supervisor", {})
        best, _ = _best_baseline(agg, band, fam, primary)
        sup_sp = sup.get("spearman", {}).get("mean") if sup else None
        alt = agg.get((band, fam, "min_altitude", round(primary, 6)), {})
        alt_sp = alt.get("spearman", {}).get("mean") if alt else None
        adds = (sup_partial.get("mean") is not None and sup_partial["mean"] > 0.05) or \
               (sup_sp is not None and alt_sp is not None and sup_sp > alt_sp + 0.02)
        rng = f"{_fmt(d.get('min_radius_low'), '.3f')}-{_fmt(d.get('min_radius_high'), '.3f')}"
        incl = f"{_fmt(d.get('inclination_deg_low'), '.0f')}-{_fmt(d.get('inclination_deg_high'), '.0f')}"
        ecc = f"{_fmt(d.get('eccentricity_low'), '.2f')}-{_fmt(d.get('eccentricity_high'), '.2f')}"
        lines.append(
            f"| {band} | {fam} | {d.get('n_trajectories')} | {rng} | {incl} | {ecc} | "
            f"{_pm(sup.get('spearman'), '.3f') if sup else 'n/a'} | "
            f"{_pm(sup.get('capture_rate'), '.3f') if sup else 'n/a'} | "
            f"{_pm(sup_partial, '.3f')} | {best} | {'yes' if adds else 'no'} |"
        )
    lines += [
        "",
        "Interpretation: families where `adds value? = yes` are those in which the VESP-UQ supervisor "
        "ranks force error beyond the altitude trend; families marked `no` are altitude-dominated and "
        "VESP-UQ's contribution there is the calibrated local covariance, not scalar ranking. The "
        "`ood_low_alt` family probes periapses at/below the training-support edge.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _plot_budget(agg: dict, primary: float, out_path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    bands = sorted({b for (b, _, _, _) in agg})
    fig, axes = plt.subplots(1, len(bands), figsize=(8 * max(1, len(bands)), 5.5), squeeze=False)
    for bi, band in enumerate(bands):
        ax = axes[0][bi]
        families = sorted({f for (b, f, _, _) in agg if b == band})
        for fam in families:
            pts = sorted([(frac, agg[(band, fam, "supervisor", frac)]["capture_rate"]["mean"])
                          for (b, f, s, frac) in agg if b == band and f == fam and s == "supervisor"],
                         key=lambda t: t[0])
            xs = [p[0] for p in pts]
            ys = [p[1] if p[1] is not None else 0.0 for p in pts]
            ax.plot(xs, ys, marker="o", label=fam, linewidth=1.2)
        ax.set_title(f"{band}: VESP-UQ supervisor capture by family")
        ax.set_xlabel("rerun fraction")
        ax.set_ylabel("capture rate")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def run_trajectory_families(
    configs, *, seeds=(0, 1, 2), families=None, n_orbits=2000, n_points=120,
    fractions=DEFAULT_FRACTIONS, primary_fraction=PRIMARY_FRACTION,
    out_dir="outputs/trajectory_families/", make_plots=True,
) -> dict:
    """Run the trajectory-family study over configs x seeds; write summary + budget curves + manifest."""

    out_dir = Path(out_dir)
    families = list(families) if families else list(FAMILIES)
    runs = [family_run(cfg, seed=s, families=families, n_orbits=n_orbits, n_points=n_points, fractions=fractions)
            for cfg in configs for s in seeds]
    rows = [r for run in runs for r in run["rows"]]
    agg = _aggregate(rows)
    partial_agg = _aggregate_partial(runs)
    descriptors = {}
    for run in runs:
        for fam, d in run["descriptors"].items():
            descriptors[(run["band"], fam)] = d

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_files = {}
    if make_plots and _plot_budget(agg, primary_fraction, out_dir / "trajectory_family_budget_curves.png"):
        artifact_files["trajectory_family_budget_curves.png"] = out_dir / "trajectory_family_budget_curves.png"

    runs_cols = ["band", "seed", "family", "selector", "rerun_fraction", *_RANK_METRICS]
    write_run_artifacts(
        out_dir,
        tool="run_trajectory_families",
        config=configs[0],
        json_files={"trajectory_family_meta.json": {
            "git_commit": git_commit_hash(), "seeds": list(seeds), "families": families,
            "n_orbits": n_orbits, "n_points": n_points, "primary_fraction": primary_fraction,
        }},
        text_files={
            "trajectory_family_summary.csv": _summary_csv(agg, partial_agg, descriptors, primary_fraction),
            "trajectory_family_summary.md": _summary_md(agg, partial_agg, descriptors, primary_fraction),
            "trajectory_family_runs.csv": _csv([runs_cols] + [[r.get(c) for c in runs_cols] for r in rows]),
        },
        artifact_files=artifact_files,
        manifest_name="manifest.json",
    )
    return {"out_dir": str(out_dir), "agg": agg, "descriptors": descriptors}
