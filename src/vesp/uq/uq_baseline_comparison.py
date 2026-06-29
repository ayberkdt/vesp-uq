"""Head-to-head: VESP-UQ vs a Gaussian-process UQ baseline (WP-D).

Fits the VESP-UQ plugin and a :class:`~vesp.uq.baselines_uq.GPResidualUQ` on the *same* train/held
split and scores them on identical metrics: per-band calibration (z_std, PICP90, ellipsoid PICP90,
radial/tangential z_std, calibration error, Winkler), trajectory-screening decision quality
(AUROC / capture-AUC / oracle-regret), and fit/predict runtime. The point is not to crown a winner
on in-support error but to show, with numbers, what each method buys -- the GP is a strong,
well-understood spatial UQ baseline; VESP-UQ adds physics-structured, altitude-aware covariance.

Everything targets trajectory-level true FORCE-model error; no position-error or density claim.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

from vesp.uq.baselines import min_altitude_scores, vespuq_scores
from vesp.uq.baselines_uq import GPResidualUQ
from vesp.uq.benchmarking import decision_quality_metrics
from vesp.uq.experiment import _build_trajectories, _resolve_time_weighting, _time_weights
from vesp.uq.integrity.metric_invariants import validate_row
from vesp.uq.io.run_artifacts import write_run_artifacts
from vesp.uq.plugin import VESPUQPlugin
from vesp.uq.risk_baselines import SUPERVISOR_SCORING, prepare, true_force_error
from vesp.uq.suite import _csv, _pm, band_label, git_commit_hash, mean_std

CALIB_REGIONS = ("all", "low", "mid", "high")
CALIB_KEYS = ("z_std", "picp_90", "ellipsoid_picp_90", "radial_z_std", "tangential_z_std",
              "calibration_error_90", "radial_winkler_90", "rmse", "mean_pred_std")
DECISION_KEYS = ("auroc", "auprc", "capture_auc_normalized", "oracle_regret")
DEFAULT_FRACTIONS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)


def comparison_run(config: dict, *, seed: int, rerun_fractions=DEFAULT_FRACTIONS,
                   reference_fraction: float = 0.20, n_orbits: int | None = None) -> dict:
    """One (config, seed): fit both models, return calibration + decision + runtime rows."""

    cfg = copy.deepcopy(config)
    cfg["seed"] = int(seed)
    if n_orbits is not None:
        cfg.setdefault("uq", {}).setdefault("screening", {})["n_orbits"] = int(n_orbits)
    band = band_label(cfg)
    plugin0, samples, train, held, dtype, _ = prepare(cfg)
    bands = cfg.get("evaluation", {}).get("altitude_bands")

    # timed fits (fresh instances, so the fit cost is measured fairly for both)
    plugin = VESPUQPlugin.from_config(cfg)
    t0 = time.perf_counter()
    plugin.fit(train.positions, train.surrogate, train.reference)
    plugin_fit_s = time.perf_counter() - t0

    gp = GPResidualUQ(seed=int(seed), dtype=dtype)
    t0 = time.perf_counter()
    gp.fit(train.positions, train.error)
    gp_fit_s = time.perf_counter() - t0

    # calibration (same metrics for both)
    t0 = time.perf_counter()
    plugin_cal = plugin.evaluate_calibration(held.positions, held.error, altitude_bands=bands)
    plugin_pred_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    gp_cal = gp.evaluate_calibration(held.positions, held.error, altitude_bands=bands)
    gp_pred_s = time.perf_counter() - t0
    n_held = int(held.positions.shape[0])

    calib_rows = []
    for model, cal in (("vespuq", plugin_cal), ("gp", gp_cal)):
        for region in CALIB_REGIONS:
            bm = cal.get(region)
            if not isinstance(bm, dict):
                continue
            row = {"band": band, "seed": int(seed), "model": model, "region": region}
            row.update({k: bm.get(k) for k in CALIB_KEYS})
            validate_row(row, where=f"uq_baseline_calib[{band} seed={seed} {model} {region}]")  # G3
            calib_rows.append(row)

    # decision quality on a shared trajectory ensemble + true force error
    screen_cfg = cfg.get("uq", {}).get("screening", {})
    traj_info = _build_trajectories(screen_cfg, seed=int(seed), dtype=dtype, config=cfg)
    trajectories = traj_info["trajectories"]
    aggregator = str(screen_cfg.get("true_error_aggregator", "p95")).lower()
    time_weighting = _resolve_time_weighting(screen_cfg)
    weights = [_time_weights(t) for t in trajectories] if time_weighting == "kepler_r2" else None
    te, te_source = true_force_error(trajectories, residuals=traj_info["residuals"], held=held,
                                     aggregator=aggregator, dtype=dtype, weights=weights)

    scores = {
        "vespuq": vespuq_scores(plugin, trajectories, SUPERVISOR_SCORING, weights=weights),
        "gp": gp.score_trajectories(trajectories, aggregator=aggregator),
        "min_altitude": min_altitude_scores(trajectories),
    }
    decision_rows = []
    for model, sc in scores.items():
        dq = decision_quality_metrics(sc, te, fractions=rerun_fractions,
                                      reference_fraction=reference_fraction)
        row = {"band": band, "seed": int(seed), "model": model}
        row.update({k: dq.get(k) for k in DECISION_KEYS})
        validate_row(row, where=f"uq_baseline_decision[{band} seed={seed} {model}]")  # G3
        decision_rows.append(row)

    runtime_rows = [
        {"band": band, "seed": int(seed), "model": "vespuq", "fit_seconds": plugin_fit_s,
         "predict_seconds_held": plugin_pred_s, "n_held": n_held},
        {"band": band, "seed": int(seed), "model": "gp", "fit_seconds": gp_fit_s,
         "predict_seconds_held": gp_pred_s, "n_held": n_held,
         "gp_lengthscale": gp.lengthscale_, "gp_n_train": int(gp.train_x_.shape[0])},
    ]
    return {"band": band, "seed": int(seed), "calib_rows": calib_rows,
            "decision_rows": decision_rows, "runtime_rows": runtime_rows,
            "true_force_error_source": te_source}


def _aggregate(rows, group_keys, metric_keys) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in group_keys), []).append(r)
    return {key: {m: mean_std([r.get(m) for r in rs]) for m in metric_keys} | {"n_seeds": len(rs)}
            for key, rs in groups.items()}


def _calib_csv(agg) -> str:
    cols = ["band", "region", "model", "n_seeds"] + [f"{k}_mean" for k in CALIB_KEYS] \
        + [f"{k}_std" for k in CALIB_KEYS]
    rows = [cols]
    for (band, region, model), a in sorted(agg.items()):
        rows.append([band, region, model, a["n_seeds"]]
                    + [a[k]["mean"] for k in CALIB_KEYS] + [a[k]["std"] for k in CALIB_KEYS])
    return _csv(rows)


def _decision_csv(agg) -> str:
    cols = ["band", "model", "n_seeds"] + [f"{k}_mean" for k in DECISION_KEYS] \
        + [f"{k}_std" for k in DECISION_KEYS]
    rows = [cols]
    for (band, model), a in sorted(agg.items()):
        rows.append([band, model, a["n_seeds"]]
                    + [a[k]["mean"] for k in DECISION_KEYS] + [a[k]["std"] for k in DECISION_KEYS])
    return _csv(rows)


def _md(calib_agg, decision_agg, runtime_agg) -> str:
    lines = [
        "# VESP-UQ vs Gaussian-Process UQ Baseline (WP-D)",
        "",
        "Both models are fit on the same train split and evaluated on identical held-out calibration "
        "and trajectory-screening metrics. The GP is a strong, well-understood spatial UQ baseline; "
        "VESP-UQ adds physics-structured, altitude-aware covariance. Mean +/- std across seeds.",
        "",
        "## Calibration (per band)",
        "",
        "| band | region | model | z_std | PICP90 | ellipsoid PICP90 | radial z_std | calib_err_90 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (band, region, model), a in sorted(calib_agg.items()):
        lines.append(
            f"| {band} | {region} | {model} | {_pm(a['z_std'], '.3f')} | {_pm(a['picp_90'], '.3f')} | "
            f"{_pm(a['ellipsoid_picp_90'], '.3f')} | {_pm(a['radial_z_std'], '.3f')} | "
            f"{_pm(a['calibration_error_90'], '.3f')} |"
        )
    lines += [
        "",
        "## Decision quality (trajectory screening)",
        "",
        "| band | model | AUROC | AUPRC | capture-AUC (norm) | oracle-regret |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for (band, model), a in sorted(decision_agg.items()):
        lines.append(
            f"| {band} | {model} | {_pm(a['auroc'], '.3f')} | {_pm(a['auprc'], '.3f')} | "
            f"{_pm(a['capture_auc_normalized'], '.3f')} | {_pm(a['oracle_regret'], '.3f')} |"
        )
    lines += [
        "",
        "## Runtime",
        "",
        "| band | model | fit (s) | predict-held (s) |",
        "| --- | --- | ---: | ---: |",
    ]
    for (band, model), a in sorted(runtime_agg.items()):
        lines.append(f"| {band} | {model} | {_pm(a['fit_seconds'], '.3f')} | "
                     f"{_pm(a['predict_seconds_held'], '.4f')} |")
    lines += [
        "",
        "Interpretation: where the GP matches or beats VESP-UQ on in-support calibration, VESP-UQ's "
        "contribution is its physics-structured covariance and altitude extrapolation, not a lower "
        "in-support error. Reported honestly either way; force-model error only.",
        "",
    ]
    return "\n".join(lines) + "\n"


def run_uq_baseline_comparison(configs, *, seeds=(0, 1, 2), rerun_fractions=DEFAULT_FRACTIONS,
                               reference_fraction: float = 0.20, n_orbits: int | None = None,
                               out_dir="outputs/uq_baseline_comparison/") -> dict:
    """Run the VESP-UQ vs GP comparison over configs x seeds; write tables + manifest.

    ``n_orbits`` overrides each config's screening orbit count (caps the decision-quality screening
    cost); ``None`` uses the config value.
    """

    out_dir = Path(out_dir)
    runs = [comparison_run(cfg, seed=s, rerun_fractions=rerun_fractions,
                           reference_fraction=reference_fraction, n_orbits=n_orbits)
            for cfg in configs for s in seeds]
    calib_rows = [r for run in runs for r in run["calib_rows"]]
    decision_rows = [r for run in runs for r in run["decision_rows"]]
    runtime_rows = [r for run in runs for r in run["runtime_rows"]]
    calib_agg = _aggregate(calib_rows, ("band", "region", "model"), CALIB_KEYS)
    decision_agg = _aggregate(decision_rows, ("band", "model"), DECISION_KEYS)
    runtime_agg = _aggregate(runtime_rows, ("band", "model"), ("fit_seconds", "predict_seconds_held"))

    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_artifacts(
        out_dir,
        tool="run_uq_baseline_comparison",
        config=configs[0],
        json_files={"uq_baseline_comparison_meta.json": {
            "git_commit": git_commit_hash(), "seeds": list(seeds),
            "true_force_error_source": runs[0]["true_force_error_source"],
        }},
        text_files={
            "uq_baseline_comparison.csv": _calib_csv(calib_agg),
            "uq_baseline_decision.csv": _decision_csv(decision_agg),
            "uq_baseline_comparison.md": _md(calib_agg, decision_agg, runtime_agg),
        },
        manifest_name="manifest.json",
    )
    return {"out_dir": str(out_dir), "calib_agg": calib_agg, "decision_agg": decision_agg,
            "runtime_agg": runtime_agg}
