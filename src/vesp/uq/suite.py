"""VESP-UQ journal benchmark suite: multi-seed, multi-fraction, multi-selector evaluation.

This module fans a VESP-UQ config out over ``seeds`` x ``rerun_fractions`` x ``selectors`` and
aggregates the results into journal-ready tables and curves. It answers the reviewer question
"does VESP-UQ add useful information beyond simple altitude heuristics, and under what
conditions?" by combining four diagnostics on a single fitted layer per (config, seed):

* **Calibration robustness** (Table A): per-altitude-band PICP / z-std / RMSE, mean +/- std across
  seeds (via :meth:`VESPUQPlugin.evaluate_calibration`).
* **Ranking robustness** (Table B): Spearman / capture / precision / lift / error-ratio per
  selector at the primary rerun budget, mean +/- std across seeds.
* **Rerun-budget curves**: the same ranking metrics swept over the full fraction grid.
* **Altitude-controlled incremental value**: within-altitude-bin Spearman, partial correlation
  given min-radius, and a matched-altitude paired sign test (:mod:`vesp.uq.altitude_controlled`).

Everything targets trajectory-level true FORCE-MODEL error (``a_reference - a_surrogate``), never
position error. Runs are deterministic given a seed. Reuses the fit / true-error / score-assembly
core in :mod:`vesp.uq.risk_baselines` rather than re-implementing it.
"""

from __future__ import annotations

import copy
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from vesp.uq.altitude_controlled import (
    matched_altitude_pairs,
    min_radius_scores,
    partial_correlations,
    within_altitude_bin_ranking,
)
from vesp.uq.baselines import label_shuffled_scores, random_scores
from vesp.uq.benchmarking import (
    DECISION_METRIC_KEYS,
    compare_baselines,
    decision_quality_metrics,
)
from vesp.uq.experiment import _build_trajectories, _resolve_time_weighting, _time_weights
from vesp.uq.integrity.metric_invariants import validate_metric, validate_row
from vesp.uq.integrity.split_guard import Split, assert_no_test_access, reveal, tag
from vesp.uq.io.run_artifacts import write_run_artifacts
from vesp.uq.risk_baselines import assemble_baseline_scores, prepare, true_force_error
from vesp.uq.significance import (
    metric_auroc,
    metric_capture,
    metric_spearman,
    paired_bootstrap_ci,
    seed_paired_test,
)

# Canonical CLI selector name -> internal score key from assemble_baseline_scores (+ "random").
SELECTOR_ALIASES = {
    "random": "random",
    "min_altitude": "min_altitude",
    "low_altitude_exposure": "low_altitude_exposure",
    "uncertainty_only": "uncertainty_only",
    "domain_support": "domain_support",
    "knn_p95": "knn_p95",
    "vespuq_supervisor": "supervisor",
    "supervisor": "supervisor",
    "altitude_residual_expected_ratio": "altitude_residual_expected_ratio",
    "altitude_residual_expected_delta": "altitude_residual_expected_delta",
    "calibrated_supervisor": "calibrated_supervisor",
    "label_shuffled": "label_shuffled",
}
DEFAULT_SELECTORS = (
    "random",
    "label_shuffled",
    "min_altitude",
    "low_altitude_exposure",
    "uncertainty_only",
    "domain_support",
    "knn_p95",
    "vespuq_supervisor",
)
# Negative controls (G4): these MUST score at chance. A violation trips the suite's placebo
# assertion -- a continuous, built-in leakage / fabrication detector.
PLACEBO_SELECTORS = ("random", "label_shuffled")
DEFAULT_FRACTIONS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)

# Metrics aggregated mean +/- std across seeds.
RANKING_AGG_METRICS = (
    "spearman",
    "capture_rate",
    "precision",
    "lift_over_random",
    "force_error_ratio_flagged_to_accepted",
    "mean_true_error_flagged",
    "mean_true_error_accepted",
)
CALIB_AGG_METRICS = (
    "n",
    "rmse",
    "mean_pred_std",
    "z_std",
    "z_mean",
    "picp_50",
    "picp_68",
    "picp_90",
    "picp_95",
    "ellipsoid_picp_90",
    "mean_epistemic_std",
    "mean_radius",
    "nll",
    "crps",
    # component-wise (radial vs tangential) calibration (WP-C)
    "radial_z_std",
    "tangential_z_std",
    "radial_picp_90",
    "tangential_picp_90",
    "radial_winkler_90",
    "tangential_winkler_90",
    "calibration_error_90",
)
CALIB_REGIONS = ("all", "low", "mid", "high")
# Altitude-controlled diagnostics are reported for these (alias) selectors when present.
ALTITUDE_CONTROL_SELECTORS = ("vespuq_supervisor", "uncertainty_only", "domain_support", "min_altitude")
MATCHED_PAIR_SELECTORS = ("vespuq_supervisor", "uncertainty_only", "domain_support")

# Decision-quality metrics (WP-B) aggregated mean +/- std across seeds.
DECISION_AGG_METRICS = DECISION_METRIC_KEYS
# Significance (WP-A): test each candidate selector against the strong altitude comparator.
SIGNIFICANCE_COMPARATOR = "min_altitude"
SIGNIFICANCE_CANDIDATES = ("vespuq_supervisor", "uncertainty_only")
SIGNIFICANCE_METRICS = ("spearman", "capture", "auroc")
HIGH_ERROR_QUANTILE = 0.90


# --------------------------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------------------------- #
def git_commit_hash() -> str | None:
    """Best-effort short git commit hash for provenance (``None`` outside a repo)."""

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    h = out.stdout.strip()
    return h or None


def band_label(config: dict) -> str:
    """Residual-band label (``L60`` / ``L90`` / dataset stem) from the config data path."""

    path = str(config.get("data", {}).get("path") or "")
    for tag_name in ("L120", "L90", "L60"):
        if tag_name in path:
            return tag_name
    run_name = str(config.get("output", {}).get("run_name") or "")
    if "L90" in run_name:
        return "L90"
    if "L60" in run_name:
        return "L60"
    return Path(path).stem or run_name or "dataset"


def _finite(values):
    out = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def mean_std(values) -> dict:
    """Mean / std / n over the finite entries of ``values`` (population std, like the random loop)."""

    finite = _finite(values)
    n = len(finite)
    if n == 0:
        return {"mean": None, "std": None, "n": 0}
    mean = sum(finite) / n
    var = sum((x - mean) ** 2 for x in finite) / n if n > 1 else 0.0
    return {"mean": mean, "std": math.sqrt(var), "n": n}


def _csv(rows) -> str:
    def cell(v):
        if v is None:
            return ""
        if isinstance(v, float):
            if math.isnan(v):
                return ""
            return repr(v)
        return str(v)

    return "\n".join(",".join(cell(v) for v in row) for row in rows) + "\n"


def _fmt(x, spec=".4f") -> str:
    if x is None:
        return "n/a"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(f):
        return "n/a"
    return format(f, spec)


def _pm(agg: dict, spec=".4f") -> str:
    """Render a mean_std dict as ``mean +/- std``."""

    if not agg or agg.get("mean") is None:
        return "n/a"
    return f"{_fmt(agg['mean'], spec)} +/- {_fmt(agg.get('std'), spec)}"


def _nearest_fraction(fractions, target: float) -> float:
    return min(fractions, key=lambda f: abs(float(f) - float(target)))


# --------------------------------------------------------------------------------------------- #
# per-(config, seed) compute
# --------------------------------------------------------------------------------------------- #
def compute_run(config: dict, *, seed: int, rerun_fractions, selectors,
                reference_fraction: float = 0.20) -> dict:
    """Fit one VESP-UQ layer at ``seed`` and evaluate calibration, ranking, budget, altitude control."""

    cfg = copy.deepcopy(config)
    cfg["seed"] = int(seed)
    band = band_label(cfg)

    t0 = time.perf_counter()
    plugin, samples, train, held, dtype, _ = prepare(cfg)
    fit_seconds = time.perf_counter() - t0

    # ---- calibration (Experiment 1) ----
    bands = cfg.get("evaluation", {}).get("altitude_bands")
    t0 = time.perf_counter()
    calibration = plugin.evaluate_calibration(held.positions, held.error, altitude_bands=bands)
    calib_seconds = time.perf_counter() - t0

    # ---- trajectories + true force error ----
    screen_cfg = cfg.get("uq", {}).get("screening", {})
    traj_info = _build_trajectories(screen_cfg, seed=int(seed), dtype=dtype, config=cfg)
    trajectories = traj_info["trajectories"]
    aggregator = str(screen_cfg.get("true_error_aggregator", "p95")).lower()
    time_weighting = _resolve_time_weighting(screen_cfg)
    weights = [_time_weights(t) for t in trajectories] if time_weighting == "kepler_r2" else None
    te, te_source = true_force_error(
        trajectories, residuals=traj_info["residuals"], held=held,
        aggregator=aggregator, dtype=dtype, weights=weights,
    )
    # G2: tag the true-error oracle (the held-out TEST labels) so any read of it during the
    # score-assembly region below trips a SplitLeakageError. Scores must rank trajectories WITHOUT
    # consulting the very error they are evaluated against.
    te_tagged = tag(te, split=Split.TEST, role="oracle")

    # ---- assemble selector scores (guarded: no oracle / test-label access) ----
    t0 = time.perf_counter()
    with assert_no_test_access():
        all_scores = assemble_baseline_scores(cfg, plugin, trajectories, train.positions, weights=weights)
        all_scores["random"] = random_scores(len(trajectories), seed=int(seed))
    score_seconds = time.perf_counter() - t0
    # Reveal the oracle only now that scoring is done -- the remaining use (metric evaluation and the
    # label-shuffled placebo) is the legitimate, post-selection consumption of the test labels.
    te = reveal(te_tagged, allow_test=True)
    # G4: the label-shuffled negative control -- the true error permuted (right marginal, no
    # alignment). Built outside the guarded block precisely because it deliberately uses the oracle.
    all_scores["label_shuffled"] = label_shuffled_scores(te, seed=int(seed))

    resolved: dict[str, torch.Tensor] = {}
    missing = []
    for name in selectors:
        key = SELECTOR_ALIASES.get(name, name)
        if key in all_scores:
            resolved[name] = all_scores[key]
        else:
            missing.append(name)

    n_traj = len(trajectories)
    n_points = sum(int(t.shape[0]) for t in trajectories)
    runtime_ms_per_traj = 1000.0 * score_seconds / max(1, n_traj)
    runtime_us_per_point = 1.0e6 * score_seconds / max(1, n_points)

    # ---- ranking metrics over the fraction grid ----
    ranking_rows = []
    for frac in rerun_fractions:
        results = compare_baselines(resolved, te, rerun_fraction=float(frac))
        for selector, metrics in results.items():
            row = {
                "band": band,
                "seed": int(seed),
                "selector": selector,
                "rerun_fraction": float(frac),
                "runtime_ms_per_traj": runtime_ms_per_traj,
                "runtime_us_per_point": runtime_us_per_point,
            }
            row.update({k: metrics.get(k) for k in RANKING_AGG_METRICS})
            validate_row(row, where=f"ranking[{band} seed={seed} {selector} f={frac}]")  # G3
            ranking_rows.append(row)

    # ---- decision-quality metrics (WP-B): one row per selector, over the full budget grid ----
    decision_rows = []
    for selector, scores in resolved.items():
        dq = decision_quality_metrics(
            scores, te, fractions=rerun_fractions,
            high_quantile=HIGH_ERROR_QUANTILE, reference_fraction=float(reference_fraction),
        )
        row = {"band": band, "seed": int(seed), "selector": selector}
        row.update({k: dq.get(k) for k in DECISION_AGG_METRICS})
        validate_row(row, where=f"decision[{band} seed={seed} {selector}]")  # G3
        decision_rows.append(row)

    # ---- calibration rows ----
    calibration_rows = []
    for region in CALIB_REGIONS:
        band_metrics = calibration.get(region)
        if not isinstance(band_metrics, dict):
            continue
        row = {"band": band, "seed": int(seed), "region": region}
        row.update({k: band_metrics.get(k) for k in CALIB_AGG_METRICS})
        validate_row(row, where=f"calibration[{band} seed={seed} {region}]")  # G3
        calibration_rows.append(row)
    calibration_scalars = {
        "band": band,
        "seed": int(seed),
        "low_high_epistemic_std_ratio": calibration.get("low_high_epistemic_std_ratio"),
        "low_high_pred_sigma_ratio": calibration.get("low_high_pred_sigma_ratio"),
    }

    # ---- altitude-controlled incremental value ----
    min_radius = min_radius_scores(trajectories)
    ac_scores = {n: resolved[n] for n in ALTITUDE_CONTROL_SELECTORS if n in resolved}
    within = within_altitude_bin_ranking(ac_scores, te, min_radius)
    partial = partial_correlations(ac_scores, te, min_radius)
    matched = {
        n: matched_altitude_pairs(resolved[n], te, min_radius)
        for n in MATCHED_PAIR_SELECTORS
        if n in resolved
    }

    return {
        "band": band,
        "seed": int(seed),
        "config_path": cfg.get("_config_path"),
        "dataset": str(cfg.get("data", {}).get("path") or (samples.metadata or {}).get("mode", "synthetic")),
        "n_trajectories": n_traj,
        "n_points": n_points,
        "true_force_error_source": te_source,
        "true_force_error_aggregator": aggregator,
        "time_weighting": time_weighting,
        "missing_selectors": missing,
        "timings": {
            "fit_seconds": fit_seconds,
            "calibration_seconds": calib_seconds,
            "score_seconds": score_seconds,
            "runtime_ms_per_traj": runtime_ms_per_traj,
            "runtime_us_per_point": runtime_us_per_point,
        },
        "ranking_rows": ranking_rows,
        "decision_rows": decision_rows,
        "calibration_rows": calibration_rows,
        "calibration_scalars": calibration_scalars,
        "altitude_controlled": {"within_bin": within, "partial": partial, "matched": matched},
        # raw per-trajectory scores + true error, kept in memory for the WP-A trajectory bootstrap
        # (never written to disk).
        "scores": resolved,
        "true_error": te,
    }


# --------------------------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------------------------- #
def aggregate_ranking(ranking_rows) -> dict:
    """Group ranking rows by (band, selector, fraction); aggregate each metric mean +/- std."""

    groups: dict[tuple, list[dict]] = {}
    for row in ranking_rows:
        key = (row["band"], row["selector"], round(float(row["rerun_fraction"]), 6))
        groups.setdefault(key, []).append(row)
    out = {}
    for key, rows in groups.items():
        agg: dict[str, Any] = {m: mean_std([r.get(m) for r in rows]) for m in RANKING_AGG_METRICS}
        agg["runtime_ms_per_traj"] = mean_std([r.get("runtime_ms_per_traj") for r in rows])
        agg["n_seeds"] = len(rows)
        for m in RANKING_AGG_METRICS:  # G3: aggregated means stay in-domain too
            validate_metric(m, agg[m]["mean"], where=f"ranking_agg{key}")
        out[key] = agg
    return out


def aggregate_decision(decision_rows) -> dict:
    """Group decision-quality rows by (band, selector); aggregate each metric mean +/- std."""

    groups: dict[tuple, list[dict]] = {}
    for row in decision_rows:
        groups.setdefault((row["band"], row["selector"]), []).append(row)
    out = {}
    for key, rows in groups.items():
        agg: dict[str, Any] = {m: mean_std([r.get(m) for r in rows]) for m in DECISION_AGG_METRICS}
        agg["n_seeds"] = len(rows)
        for m in DECISION_AGG_METRICS:  # G3
            validate_metric(m, agg[m]["mean"], where=f"decision_agg{key}")
        out[key] = agg
    return out


def _per_seed_metric_series(runs, primary_fraction: float) -> dict:
    """``{(band, selector): {metric: [value per seed, seed-ordered]}}`` for the significance tests.

    Spearman and capture come from the ranking rows at the primary budget; AUROC from the decision
    rows. Seeds are ordered so the candidate and comparator series align pairwise.
    """

    series: dict[tuple, dict[str, list]] = {}
    for run in sorted(runs, key=lambda r: (r["band"], r["seed"])):
        band = run["band"]
        spearman_at = {}
        capture_at = {}
        for r in run["ranking_rows"]:
            if abs(float(r["rerun_fraction"]) - primary_fraction) <= 1e-9:
                spearman_at[r["selector"]] = r.get("spearman")
                capture_at[r["selector"]] = r.get("capture_rate")
        auroc_at = {r["selector"]: r.get("auroc") for r in run["decision_rows"]}
        selectors = set(spearman_at) | set(auroc_at)
        for sel in selectors:
            d = series.setdefault((band, sel), {"spearman": [], "capture": [], "auroc": []})
            d["spearman"].append(spearman_at.get(sel))
            d["capture"].append(capture_at.get(sel))
            d["auroc"].append(auroc_at.get(sel))
    return series


def compute_significance(runs, *, primary_fraction: float, n_boot: int = 2000, seed: int = 0) -> list[dict]:
    """Test each candidate selector against the altitude comparator per band (WP-A).

    For each (band, candidate, metric) it reports the across-seed Wilcoxon signed-rank test on the
    per-seed metric differences AND a trajectory-level paired bootstrap (CI + p-value) on the first
    seed of that band. Returns a flat list of result rows (one per band/candidate/metric).
    """

    series = _per_seed_metric_series(runs, primary_fraction)
    first_run_by_band: dict[str, dict] = {}
    for run in sorted(runs, key=lambda r: (r["band"], r["seed"])):
        first_run_by_band.setdefault(run["band"], run)

    metric_fns = {"spearman": metric_spearman,
                  "capture": metric_capture(primary_fraction),
                  "auroc": metric_auroc(HIGH_ERROR_QUANTILE)}

    rows = []
    bands = sorted({b for (b, _) in series})
    for band in bands:
        comparator_present = (band, SIGNIFICANCE_COMPARATOR) in series
        for cand in SIGNIFICANCE_CANDIDATES:
            if (band, cand) not in series or not comparator_present:
                continue
            for metric in SIGNIFICANCE_METRICS:
                seed_test = seed_paired_test(
                    series[(band, cand)][metric], series[(band, SIGNIFICANCE_COMPARATOR)][metric])
                run0 = first_run_by_band[band]
                boot = {"delta": None, "ci_low": None, "ci_high": None,
                        "p_value": None, "significant": None}
                if cand in run0["scores"] and SIGNIFICANCE_COMPARATOR in run0["scores"]:
                    boot = paired_bootstrap_ci(
                        run0["scores"][cand], run0["scores"][SIGNIFICANCE_COMPARATOR],
                        run0["true_error"], metric=metric_fns[metric], n_boot=n_boot, seed=seed,
                    )
                rows.append({
                    "band": band,
                    "candidate": cand,
                    "comparator": SIGNIFICANCE_COMPARATOR,
                    "metric": metric,
                    "seed_mean_delta": seed_test["mean_delta"],
                    "seed_wilcoxon_p": seed_test["p_value"],
                    "n_seeds": seed_test["n_seeds"],
                    "boot_delta": boot.get("delta"),
                    "boot_ci_low": boot.get("ci_low"),
                    "boot_ci_high": boot.get("ci_high"),
                    "boot_p": boot.get("p_value"),
                    "boot_significant": boot.get("significant"),
                })
    return rows


def aggregate_calibration(calibration_rows) -> dict:
    """Group calibration rows by (band, region); aggregate each metric mean +/- std."""

    groups: dict[tuple, list[dict]] = {}
    for row in calibration_rows:
        groups.setdefault((row["band"], row["region"]), []).append(row)
    out = {}
    for key, rows in groups.items():
        agg: dict[str, Any] = {m: mean_std([r.get(m) for r in rows]) for m in CALIB_AGG_METRICS}
        agg["n_seeds"] = len(rows)
        for m in CALIB_AGG_METRICS:  # G3
            validate_metric(m, agg[m]["mean"], where=f"calibration_agg{key}")
        out[key] = agg
    return out


def aggregate_altitude_controlled(runs) -> dict:
    """Aggregate within-bin Spearman, partial correlation, and matched concordance across seeds."""

    bands = sorted({r["band"] for r in runs})
    out = {}
    for band in bands:
        band_runs = [r for r in runs if r["band"] == band]
        selectors = sorted({s for r in band_runs for s in r["altitude_controlled"]["partial"]})
        partial = {}
        within = {}
        for sel in selectors:
            partial[sel] = {
                "partial_pearson_given_min_radius": mean_std(
                    [r["altitude_controlled"]["partial"].get(sel, {}).get("partial_pearson_given_min_radius")
                     for r in band_runs]
                ),
                "pearson": mean_std(
                    [r["altitude_controlled"]["partial"].get(sel, {}).get("pearson") for r in band_runs]
                ),
                "spearman": mean_std(
                    [r["altitude_controlled"]["partial"].get(sel, {}).get("spearman") for r in band_runs]
                ),
            }
            within[sel] = mean_std(
                [r["altitude_controlled"]["within_bin"]["weighted_within_bin_spearman"].get(sel)
                 for r in band_runs]
            )
        matched_selectors = sorted({s for r in band_runs for s in r["altitude_controlled"]["matched"]})
        matched = {}
        for sel in matched_selectors:
            matched[sel] = {
                "concordance_rate": mean_std(
                    [r["altitude_controlled"]["matched"].get(sel, {}).get("concordance_rate")
                     for r in band_runs]
                ),
                "mean_delta_true_error": mean_std(
                    [r["altitude_controlled"]["matched"].get(sel, {}).get("mean_delta_true_error")
                     for r in band_runs]
                ),
                "sign_test_p_value_one_sided": mean_std(
                    [r["altitude_controlled"]["matched"].get(sel, {}).get("sign_test_p_value_one_sided")
                     for r in band_runs]
                ),
                "n_pairs": mean_std(
                    [r["altitude_controlled"]["matched"].get(sel, {}).get("n_pairs") for r in band_runs]
                ),
            }
        out[band] = {"partial": partial, "within_bin": within, "matched": matched}
    return out


# --------------------------------------------------------------------------------------------- #
# G4 -- negative-control (placebo) assertion
# --------------------------------------------------------------------------------------------- #
class PlaceboLeakageError(RuntimeError):
    """Raised when a negative-control selector scores above chance (a G4 violation)."""


# Chance-band half-width = k / sqrt(n_effective), floored so small smoke runs do not trip on
# ordinary sampling noise. k ~ a few standard deviations of a null Spearman / capture estimate.
PLACEBO_TOL_K = 3.5
PLACEBO_TOL_FLOOR = 0.20


def placebo_tolerance(n_effective: int, *, k: float = PLACEBO_TOL_K,
                      floor: float = PLACEBO_TOL_FLOOR) -> float:
    """Half-width of the at-chance band for a placebo metric at effective sample size ``n_effective``."""

    n = max(1, int(n_effective))
    return max(float(floor), float(k) / math.sqrt(n))


def assert_placebos_at_chance(ranking_agg, *, primary_fraction: float, n_trajectories: int,
                              n_seeds: int, placebos=PLACEBO_SELECTORS) -> dict:
    """Assert every negative control scores at chance at the primary budget, per band (G4).

    A placebo (``random``, ``label_shuffled``) has no real alignment with the true error, so its
    aggregated Spearman must be ~0 and its aggregated capture rate must be ~``primary_fraction``
    (the chance capture). A placebo that beats chance means scores are leaking alignment with the
    oracle -- so a violation raises :class:`PlaceboLeakageError` and fails the run. Returns the
    per-(band, placebo) checks (so the caller can log them); ``None``/non-finite aggregates are
    treated as legitimately-missing and skipped.
    """

    tol = placebo_tolerance(int(n_trajectories) * max(1, int(n_seeds)))
    checks: dict[tuple, dict] = {}
    for (band, selector, frac), agg in ranking_agg.items():
        if selector not in placebos or abs(float(frac) - float(primary_fraction)) > 1e-9:
            continue
        sp = agg["spearman"]["mean"]
        cap = agg["capture_rate"]["mean"]
        sp_ok = sp is None or not math.isfinite(float(sp)) or abs(float(sp)) <= tol
        cap_ok = (cap is None or not math.isfinite(float(cap))
                  or abs(float(cap) - float(primary_fraction)) <= tol)
        checks[(band, selector)] = {"spearman": sp, "capture_rate": cap, "tol": tol,
                                    "ok": bool(sp_ok and cap_ok)}
        if not sp_ok:
            raise PlaceboLeakageError(
                f"placebo {selector!r} on {band} has Spearman {float(sp):.3f} (|.| > tol {tol:.3f}) "
                "at the primary budget -- scores are leaking alignment with the true-error oracle")
        if not cap_ok:
            raise PlaceboLeakageError(
                f"placebo {selector!r} on {band} captures {float(cap):.3f} vs chance "
                f"{float(primary_fraction):.3f} (gap > tol {tol:.3f}) -- a negative control must "
                "not beat chance")
    return checks


# --------------------------------------------------------------------------------------------- #
# table / curve writers
# --------------------------------------------------------------------------------------------- #
def _benchmark_runs_csv(ranking_rows) -> str:
    cols = ["band", "seed", "selector", "rerun_fraction", *RANKING_AGG_METRICS,
            "runtime_ms_per_traj", "runtime_us_per_point"]
    rows = [cols] + [[r.get(c) for c in cols] for r in ranking_rows]
    return _csv(rows)


def _benchmark_summary_csv(ranking_agg, primary_fraction) -> str:
    cols = ["band", "selector", "rerun_fraction", "n_seeds"]
    for m in RANKING_AGG_METRICS:
        cols += [f"{m}_mean", f"{m}_std"]
    rows = [cols]
    for (band, selector, frac), agg in sorted(ranking_agg.items()):
        if abs(frac - primary_fraction) > 1e-9:
            continue
        row = [band, selector, frac, agg["n_seeds"]]
        for m in RANKING_AGG_METRICS:
            row += [agg[m]["mean"], agg[m]["std"]]
        rows.append(row)
    return _csv(rows)


def _rerun_budget_csv(ranking_agg) -> str:
    cols = ["band", "selector", "rerun_fraction", "n_seeds",
            "capture_rate_mean", "capture_rate_std",
            "precision_mean", "precision_std",
            "lift_over_random_mean", "lift_over_random_std",
            "force_error_ratio_flagged_to_accepted_mean", "force_error_ratio_flagged_to_accepted_std",
            "spearman_mean", "spearman_std"]
    rows = [cols]
    for (band, selector, frac), agg in sorted(ranking_agg.items()):
        rows.append([
            band, selector, frac, agg["n_seeds"],
            agg["capture_rate"]["mean"], agg["capture_rate"]["std"],
            agg["precision"]["mean"], agg["precision"]["std"],
            agg["lift_over_random"]["mean"], agg["lift_over_random"]["std"],
            agg["force_error_ratio_flagged_to_accepted"]["mean"],
            agg["force_error_ratio_flagged_to_accepted"]["std"],
            agg["spearman"]["mean"], agg["spearman"]["std"],
        ])
    return _csv(rows)


def _calibration_summary_csv(calib_agg) -> str:
    cols = ["band", "region", "n_seeds"]
    for m in CALIB_AGG_METRICS:
        cols += [f"{m}_mean", f"{m}_std"]
    rows = [cols]
    for (band, region), agg in sorted(calib_agg.items()):
        row = [band, region, agg["n_seeds"]]
        for m in CALIB_AGG_METRICS:
            row += [agg[m]["mean"], agg[m]["std"]]
        rows.append(row)
    return _csv(rows)


def _calibration_md(calib_agg, primary_fraction) -> str:
    lines = [
        "# VESP-UQ Calibration Robustness (Table A)",
        "",
        "Per-altitude-band held-out force-error calibration, mean +/- std across seeds. "
        "`z_std` ~ 1 is well-calibrated (>1 overconfident); `picp_90` should approach 0.90.",
        "",
        "| band | region | n_seeds | rmse | mean_pred_std | z_std | picp_68 | picp_90 | ellipsoid_picp_90 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (band, region), agg in sorted(calib_agg.items()):
        lines.append(
            f"| {band} | {region} | {agg['n_seeds']} | {_pm(agg['rmse'], '.3e')} | "
            f"{_pm(agg['mean_pred_std'], '.3e')} | {_pm(agg['z_std'], '.3f')} | "
            f"{_pm(agg['picp_68'], '.3f')} | {_pm(agg['picp_90'], '.3f')} | "
            f"{_pm(agg['ellipsoid_picp_90'], '.3f')} |"
        )
    lines += [
        "",
        "## Component-wise calibration (radial vs tangential, WP-C)",
        "",
        "The predictive error covariance split into the local radial vs tangential frame. The radial "
        "component carries the altitude-dependent force-model error; a band mis-calibrated mainly in "
        "the radial axis points to the noise law / geometry rather than an isotropic scale error. "
        "`z_std` ~ 1 calibrated; Winkler is the interval score (lower is better).",
        "",
        "| band | region | radial z_std | tangential z_std | radial PICP90 | tangential PICP90 | "
        "calib_err_90 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (band, region), agg in sorted(calib_agg.items()):
        lines.append(
            f"| {band} | {region} | {_pm(agg['radial_z_std'], '.3f')} | "
            f"{_pm(agg['tangential_z_std'], '.3f')} | {_pm(agg['radial_picp_90'], '.3f')} | "
            f"{_pm(agg['tangential_picp_90'], '.3f')} | {_pm(agg['calibration_error_90'], '.3f')} |"
        )
    lines += ["", f"Primary rerun budget for ranking tables: {primary_fraction:.0%}.", ""]
    return "\n".join(lines) + "\n"


def _decision_summary_csv(decision_agg) -> str:
    cols = ["band", "selector", "n_seeds"]
    for m in DECISION_AGG_METRICS:
        cols += [f"{m}_mean", f"{m}_std"]
    rows = [cols]
    for (band, selector), agg in sorted(decision_agg.items()):
        row = [band, selector, agg["n_seeds"]]
        for m in DECISION_AGG_METRICS:
            row += [agg[m]["mean"], agg[m]["std"]]
        rows.append(row)
    return _csv(rows)


def _decision_md(decision_agg, reference_fraction) -> str:
    lines = [
        "# VESP-UQ Decision Quality (Table C)",
        "",
        "Risk scores judged as the screening tool they are, against trajectory true FORCE-model "
        f"error. AUROC/AUPRC detect the top-decile-error class (chance AUROC 0.5); capture-AUC is the "
        "budget-integrated capture (normalized: 1.0 = matches the oracle at every budget); "
        f"oracle-regret is the captured-error gap to the oracle at the {reference_fraction:.0%} budget "
        "(0 = optimal, 1 = random). Mean +/- std across seeds.",
        "",
        "| band | selector | AUROC | AUPRC | capture-AUC (norm) | oracle-regret |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for (band, selector), agg in sorted(decision_agg.items()):
        lines.append(
            f"| {band} | {selector} | {_pm(agg['auroc'], '.3f')} | {_pm(agg['auprc'], '.3f')} | "
            f"{_pm(agg['capture_auc_normalized'], '.3f')} | {_pm(agg['oracle_regret'], '.3f')} |"
        )
    lines += ["", ""]
    return "\n".join(lines) + "\n"


def _significance_csv(sig_rows) -> str:
    cols = ["band", "candidate", "comparator", "metric", "n_seeds",
            "seed_mean_delta", "seed_wilcoxon_p",
            "boot_delta", "boot_ci_low", "boot_ci_high", "boot_p", "boot_significant"]
    rows = [cols] + [[r.get(c) for c in cols] for r in sig_rows]
    return _csv(rows)


def _significance_md(sig_rows) -> str:
    lines = [
        "# VESP-UQ Significance Tests (WP-A)",
        "",
        "Each candidate selector vs the strong altitude comparator (`min_altitude`). `delta > 0` means "
        "the candidate beats altitude on that metric. Two complementary tests: a Wilcoxon signed-rank "
        "across seeds (exact), and a trajectory-level paired bootstrap (95% CI + p) on the first seed. "
        "A claim of 'beats altitude' requires the bootstrap CI to exclude 0; otherwise the verdict is "
        "'indistinguishable from altitude' and the contribution is the calibrated covariance (Table A).",
        "",
        "| band | candidate | metric | seed dDelta | Wilcoxon p | boot delta [95% CI] | boot p | verdict |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in sig_rows:
        ci = (f"{_fmt(r['boot_delta'], '.3f')} "
              f"[{_fmt(r['boot_ci_low'], '.3f')}, {_fmt(r['boot_ci_high'], '.3f')}]")
        verdict = ("beats altitude" if r.get("boot_significant") and (r.get("boot_delta") or 0) > 0
                   else ("altitude beats" if r.get("boot_significant") else "indistinguishable"))
        lines.append(
            f"| {r['band']} | {r['candidate']} | {r['metric']} | {_fmt(r['seed_mean_delta'], '.3f')} | "
            f"{_fmt(r['seed_wilcoxon_p'], '.3f')} | {ci} | {_fmt(r['boot_p'], '.3f')} | {verdict} |"
        )
    lines += [
        "",
        "The Wilcoxon p across only a few seeds has low power; the bootstrap CI on a full trajectory "
        "ensemble is the primary significance evidence. Both target force-model error, not position error.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _summary_md(ranking_agg, ac_agg, primary_fraction, meta) -> str:
    lines = [
        "# VESP-UQ Benchmark Suite Summary",
        "",
        f"- configs: {', '.join(meta['bands'])}  |  seeds: {meta['seeds']}  |  "
        f"primary rerun budget: {primary_fraction:.0%}",
        f"- git commit: `{meta.get('git_commit')}`  |  trajectories/run: {meta.get('n_trajectories')}",
        f"- true force error: `{meta.get('true_force_error_source')}` "
        f"(aggregator `{meta.get('true_force_error_aggregator')}`)",
        "",
        "## Table B -- Ranking robustness (primary budget)",
        "",
        f"Mean +/- std across seeds at the {primary_fraction:.0%} rerun budget. Target: trajectory "
        "true FORCE-model error (not position error).",
        "",
        "| band | selector | capture | precision | lift | spearman | err_ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (band, selector, frac), agg in sorted(ranking_agg.items()):
        if abs(frac - primary_fraction) > 1e-9:
            continue
        lines.append(
            f"| {band} | {selector} | {_pm(agg['capture_rate'], '.3f')} | "
            f"{_pm(agg['precision'], '.3f')} | {_pm(agg['lift_over_random'], '.2f')} | "
            f"{_pm(agg['spearman'], '.3f')} | "
            f"{_pm(agg['force_error_ratio_flagged_to_accepted'], '.2f')} |"
        )
    lines += [
        "",
        "## Altitude-controlled incremental value",
        "",
        "Partial correlation of each score with true force error after regressing out min-radius, "
        "and the matched-altitude paired sign test (concordance > 0.5 => signal beyond altitude).",
        "",
        "| band | selector | partial_corr(given alt) | within-bin spearman | matched concordance |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for band, payload in sorted(ac_agg.items()):
        for sel in sorted(payload["partial"]):
            matched = payload["matched"].get(sel, {})
            lines.append(
                f"| {band} | {sel} | "
                f"{_pm(payload['partial'][sel]['partial_pearson_given_min_radius'], '.3f')} | "
                f"{_pm(payload['within_bin'][sel], '.3f')} | "
                f"{_pm(matched.get('concordance_rate'), '.3f') if matched else 'n/a'} |"
            )
    lines += [
        "",
        "Interpretation: a positive partial correlation and a matched concordance above 0.5 mean the "
        "selector carries force-error ranking signal that is NOT explained by altitude alone. If "
        "altitude-only matches or beats VESP-UQ, the added value is the calibrated local covariance "
        "(Table A), not a superior scalar ranking. This is a force-risk diagnostic only.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _selector_ablation_csv(ranking_agg, primary_fraction) -> str:
    cols = ["band", "selector", "rerun_fraction", "n_seeds",
            "capture_rate_mean", "capture_rate_std",
            "lift_over_random_mean", "lift_over_random_std",
            "spearman_mean", "spearman_std",
            "force_error_ratio_flagged_to_accepted_mean", "force_error_ratio_flagged_to_accepted_std"]
    rows = [cols]
    for (band, selector, frac), agg in sorted(ranking_agg.items()):
        if abs(frac - primary_fraction) > 1e-9:
            continue
        rows.append([
            band, selector, frac, agg["n_seeds"],
            agg["capture_rate"]["mean"], agg["capture_rate"]["std"],
            agg["lift_over_random"]["mean"], agg["lift_over_random"]["std"],
            agg["spearman"]["mean"], agg["spearman"]["std"],
            agg["force_error_ratio_flagged_to_accepted"]["mean"],
            agg["force_error_ratio_flagged_to_accepted"]["std"],
        ])
    return _csv(rows)


def _altitude_bin_csv(runs) -> str:
    cols = ["band", "seed", "bin", "n", "min_radius_low", "min_radius_high",
            "true_error_var", "selector", "within_bin_spearman", "within_bin_capture"]
    rows = [cols]
    for r in runs:
        for b in r["altitude_controlled"]["within_bin"]["bins"]:
            for sel, m in b["methods"].items():
                rows.append([
                    r["band"], r["seed"], b["bin"], b["n"], b["min_radius_low"],
                    b["min_radius_high"], b["true_error_var"], sel,
                    m["spearman"], m["capture"],
                ])
    return _csv(rows)


def _partial_correlation_csv(ac_agg) -> str:
    cols = ["band", "selector", "n_seeds",
            "pearson_mean", "spearman_mean",
            "partial_pearson_given_min_radius_mean", "partial_pearson_given_min_radius_std"]
    rows = [cols]
    for band, payload in sorted(ac_agg.items()):
        for sel, m in sorted(payload["partial"].items()):
            pp = m["partial_pearson_given_min_radius"]
            rows.append([
                band, sel, pp["n"], m["pearson"]["mean"], m["spearman"]["mean"],
                pp["mean"], pp["std"],
            ])
    return _csv(rows)


def _matched_pairs_csv(runs) -> str:
    """Per-pair rows for the first seed of each band (full multi-seed set summarized separately)."""

    cols = ["band", "seed", "selector", "traj_high_score", "traj_low_score",
            "min_radius_high_score", "min_radius_low_score", "min_radius_gap",
            "score_high", "score_low", "true_error_high_score", "true_error_low_score",
            "delta_true_error", "concordant"]
    rows = [cols]
    seen_bands = set()
    for r in sorted(runs, key=lambda x: (x["band"], x["seed"])):
        if r["band"] in seen_bands:
            continue
        seen_bands.add(r["band"])
        for sel, res in r["altitude_controlled"]["matched"].items():
            for pr in res["rows"]:
                rows.append([
                    r["band"], r["seed"], sel, pr["traj_high_score"], pr["traj_low_score"],
                    pr["min_radius_high_score"], pr["min_radius_low_score"], pr["min_radius_gap"],
                    pr["score_high"], pr["score_low"], pr["true_error_high_score"],
                    pr["true_error_low_score"], pr["delta_true_error"], pr["concordant"],
                ])
    return _csv(rows)


def _matched_summary_md(ac_agg) -> str:
    lines = [
        "# Matched-Altitude Pair Test (incremental value beyond altitude)",
        "",
        "Trajectories are matched to within a min-radius caliper; for each matched pair the "
        "higher-scoring trajectory is compared to the lower-scoring one. A concordance rate above "
        "0.50 means higher score tracks higher true FORCE error *at the same altitude*. Values are "
        "mean +/- std across seeds; the one-sided sign-test p-value tests concordance > 0.5.",
        "",
        "| band | selector | n_pairs | concordance | mean d(true error) | sign-test p |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for band, payload in sorted(ac_agg.items()):
        for sel, m in sorted(payload["matched"].items()):
            lines.append(
                f"| {band} | {sel} | {_pm(m['n_pairs'], '.0f')} | "
                f"{_pm(m['concordance_rate'], '.3f')} | {_pm(m['mean_delta_true_error'], '.3e')} | "
                f"{_pm(m['sign_test_p_value_one_sided'], '.3f')} |"
            )
    lines += [
        "",
        "A concordance indistinguishable from 0.50 means the score adds no altitude-independent "
        "ranking signal in that band; the calibrated local covariance remains the contribution.",
        "",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------- #
# plots (best-effort; never abort the data outputs)
# --------------------------------------------------------------------------------------------- #
def _plot_budget_curves(ranking_agg, out_path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    bands = sorted({b for (b, _, _) in ranking_agg})
    panels = [("capture_rate", "Capture rate"), ("lift_over_random", "Lift over random"),
              ("force_error_ratio_flagged_to_accepted", "Flagged/accepted error ratio")]
    fig, axes = plt.subplots(len(bands), 3, figsize=(15, 4 * max(1, len(bands))), squeeze=False)
    for bi, band in enumerate(bands):
        selectors = sorted({s for (b, s, _) in ranking_agg if b == band})
        for pi, (metric, title) in enumerate(panels):
            ax = axes[bi][pi]
            for sel in selectors:
                pts = sorted(
                    [(frac, ranking_agg[(band, sel, frac)][metric]["mean"],
                      ranking_agg[(band, sel, frac)][metric]["std"])
                     for (b, s, frac) in ranking_agg if b == band and s == sel],
                    key=lambda t: t[0],
                )
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                es = [p[2] if p[2] is not None else 0.0 for p in pts]
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=2, label=sel, linewidth=1.2)
            ax.set_title(f"{band}: {title}")
            ax.set_xlabel("rerun fraction")
            ax.grid(True, alpha=0.3)
            if pi == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def _plot_selector_ablation(ranking_agg, primary_fraction, out_path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    bands = sorted({b for (b, _, _) in ranking_agg})
    fig, axes = plt.subplots(1, len(bands), figsize=(7 * max(1, len(bands)), 4.5), squeeze=False)
    for bi, band in enumerate(bands):
        ax = axes[0][bi]
        items = sorted(
            [(s, ranking_agg[(b, s, f)]["capture_rate"]["mean"],
              ranking_agg[(b, s, f)]["capture_rate"]["std"])
             for (b, s, f) in ranking_agg if b == band and abs(f - primary_fraction) <= 1e-9],
            key=lambda t: (t[1] if t[1] is not None else -1),
        )
        labels = [i[0] for i in items]
        vals = [i[1] if i[1] is not None else 0.0 for i in items]
        errs = [i[2] if i[2] is not None else 0.0 for i in items]
        ax.barh(labels, vals, xerr=errs, capsize=3, color="#3b7dd8")
        ax.set_title(f"{band}: capture @ {primary_fraction:.0%}")
        ax.set_xlabel("capture rate")
        ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def _plot_altitude_bins(runs, out_path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    bands = sorted({r["band"] for r in runs})
    fig, axes = plt.subplots(1, len(bands), figsize=(7 * max(1, len(bands)), 4.5), squeeze=False)
    for bi, band in enumerate(bands):
        ax = axes[0][bi]
        band_runs = [r for r in runs if r["band"] == band]
        selectors = sorted({s for r in band_runs for s in
                            r["altitude_controlled"]["within_bin"]["bins"][0]["methods"]}) \
            if band_runs and band_runs[0]["altitude_controlled"]["within_bin"]["bins"] else []
        for sel in selectors:
            # average within-bin spearman per bin index across seeds
            per_bin: dict[int, list[float]] = {}
            for r in band_runs:
                for b in r["altitude_controlled"]["within_bin"]["bins"]:
                    v = b["methods"].get(sel, {}).get("spearman")
                    if v is not None and math.isfinite(float(v)):
                        per_bin.setdefault(b["bin"], []).append(float(v))
            xs = sorted(per_bin)
            ys = [sum(per_bin[x]) / len(per_bin[x]) for x in xs]
            ax.plot(xs, ys, marker="o", label=sel, linewidth=1.3)
        ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{band}: within-bin Spearman")
        ax.set_xlabel("altitude bin (low -> high radius)")
        ax.set_ylabel("Spearman(score, true error)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------------------------- #
def plan_matrix(configs, seeds, rerun_fractions, selectors) -> list[dict]:
    """Enumerate the run matrix (for ``--dry-run`` listing) without doing any compute."""

    plan = []
    for cfg in configs:
        band = band_label(cfg)
        for seed in seeds:
            plan.append({
                "band": band,
                "config_path": cfg.get("_config_path"),
                "seed": int(seed),
                "rerun_fractions": [float(f) for f in rerun_fractions],
                "selectors": list(selectors),
            })
    return plan


def run_suite(
    configs,
    *,
    seeds=(0, 1, 2, 3, 4),
    rerun_fractions=DEFAULT_FRACTIONS,
    selectors=DEFAULT_SELECTORS,
    out_dir="outputs/benchmark_suite",
    primary_fraction: float = 0.20,
    make_plots: bool = True,
    n_orbits: int | None = None,
    progress: bool = True,
) -> dict:
    """Run the full suite over configs x seeds and write all journal artifacts under ``out_dir``.

    ``n_orbits`` overrides ``uq.screening.n_orbits`` for every config (e.g. to trade a little
    statistical power for speed). ``progress`` prints a line as each (config, seed) run completes
    so a long real-data sweep is observable instead of silent until the final write.
    """

    out_dir = Path(out_dir)
    seeds = [int(s) for s in seeds]
    rerun_fractions = sorted({round(float(f), 6) for f in rerun_fractions})
    primary_fraction = _nearest_fraction(rerun_fractions, primary_fraction)
    if n_orbits is not None:
        for cfg in configs:
            cfg.setdefault("uq", {}).setdefault("screening", {})["n_orbits"] = int(n_orbits)

    total = len(configs) * len(seeds)
    runs = []
    done = 0
    suite_t0 = time.perf_counter()
    for cfg in configs:
        for seed in seeds:
            t0 = time.perf_counter()
            runs.append(compute_run(cfg, seed=seed, rerun_fractions=rerun_fractions,
                                    selectors=selectors, reference_fraction=primary_fraction))
            done += 1
            if progress:
                elapsed = time.perf_counter() - suite_t0
                eta = elapsed / done * (total - done)
                print(f"[suite] {done}/{total} done ({band_label(cfg)} seed={seed}, "
                      f"{time.perf_counter() - t0:.0f}s) | elapsed {elapsed:.0f}s | eta {eta:.0f}s",
                      flush=True)

    ranking_rows = [row for r in runs for row in r["ranking_rows"]]
    decision_rows = [row for r in runs for row in r["decision_rows"]]
    calibration_rows = [row for r in runs for row in r["calibration_rows"]]
    ranking_agg = aggregate_ranking(ranking_rows)
    decision_agg = aggregate_decision(decision_rows)
    calib_agg = aggregate_calibration(calibration_rows)
    ac_agg = aggregate_altitude_controlled(runs)
    significance_rows = compute_significance(runs, primary_fraction=primary_fraction)

    first = runs[0]
    # G4: every negative control must score at chance, or the run fails loudly (continuous
    # leakage / fabrication detector). Raises PlaceboLeakageError on a violation.
    placebo_checks = assert_placebos_at_chance(
        ranking_agg, primary_fraction=primary_fraction,
        n_trajectories=first["n_trajectories"], n_seeds=len(seeds),
    )
    meta = {
        "bands": sorted({r["band"] for r in runs}),
        "seeds": seeds,
        "selectors": list(selectors),
        "rerun_fractions": rerun_fractions,
        "primary_fraction": primary_fraction,
        "git_commit": git_commit_hash(),
        "n_trajectories": first["n_trajectories"],
        "true_force_error_source": first["true_force_error_source"],
        "true_force_error_aggregator": first["true_force_error_aggregator"],
        "missing_selectors": sorted({m for r in runs for m in r["missing_selectors"]}),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "placebo_checks": [
            {"band": band, "selector": selector, **payload}
            for (band, selector), payload in sorted(placebo_checks.items())
        ],
    }

    # plots first, so their paths can be registered as prewritten artifacts in the manifest.
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_files = {}
    if make_plots:
        if _plot_budget_curves(ranking_agg, out_dir / "rerun_budget_curves.png"):
            artifact_files["rerun_budget_curves.png"] = out_dir / "rerun_budget_curves.png"
        if _plot_selector_ablation(ranking_agg, primary_fraction, out_dir / "selector_ablation.png"):
            artifact_files["selector_ablation.png"] = out_dir / "selector_ablation.png"
        if _plot_altitude_bins(runs, out_dir / "altitude_bin_ranking.png"):
            artifact_files["altitude_bin_ranking.png"] = out_dir / "altitude_bin_ranking.png"

    manifest_payload = {
        "_provenance": {"tool": "run_vespuq_benchmark_suite", "git_commit": meta["git_commit"]},
        "meta": meta,
        "timings": [{"band": r["band"], "seed": r["seed"], **r["timings"]} for r in runs],
    }
    write_run_artifacts(
        out_dir,
        tool="run_vespuq_benchmark_suite",
        config=configs[0],
        json_files={"benchmark_meta.json": manifest_payload},
        text_files={
            "benchmark_runs.csv": _benchmark_runs_csv(ranking_rows),
            "benchmark_summary.csv": _benchmark_summary_csv(ranking_agg, primary_fraction),
            "benchmark_summary.md": _summary_md(ranking_agg, ac_agg, primary_fraction, meta),
            "decision_quality.csv": _decision_summary_csv(decision_agg),
            "decision_quality.md": _decision_md(decision_agg, primary_fraction),
            "significance_summary.csv": _significance_csv(significance_rows),
            "significance_summary.md": _significance_md(significance_rows),
            "rerun_budget_curves.csv": _rerun_budget_csv(ranking_agg),
            "selector_ablation.csv": _selector_ablation_csv(ranking_agg, primary_fraction),
            "calibration_summary.csv": _calibration_summary_csv(calib_agg),
            "calibration_summary.md": _calibration_md(calib_agg, primary_fraction),
            "altitude_bin_ranking.csv": _altitude_bin_csv(runs),
            "partial_correlation_summary.csv": _partial_correlation_csv(ac_agg),
            "matched_altitude_pairs.csv": _matched_pairs_csv(runs),
            "matched_altitude_summary.md": _matched_summary_md(ac_agg),
        },
        artifact_files=artifact_files,
        manifest_name="manifest.json",
    )
    return {
        "out_dir": str(out_dir),
        "meta": meta,
        "ranking_agg": ranking_agg,
        "decision_agg": decision_agg,
        "calib_agg": calib_agg,
        "altitude_controlled": ac_agg,
        "significance": significance_rows,
        "placebo_checks": placebo_checks,
    }
