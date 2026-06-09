"""Conformal calibration + sentinel false-negative audit for VESP-UQ.

Wraps a fitted :class:`~vesp.uq.plugin.VESPUQPlugin` with two post-hoc reliability checks:

1. **Conformal calibration** of the predictive *force-error* uncertainty: on held-out residual
   samples, learn a single multiplicative scale so the predictive band empirically covers the true
   force error at the ``1 - alpha`` level, and report coverage before vs after (see
   :mod:`vesp.uq.conformal`). VESP-UQ is itself a fitted uncertainty model, so its nominal intervals
   are not assumed correct -- coverage is measured, not guaranteed.

2. **Sentinel audit** of the accepted (low-risk) trajectories: after the configured risk screening
   flags a rerun budget, draw a small deterministic random sample from the accepted set and estimate
   how many are genuinely high *force* error -- the false negatives (see :mod:`vesp.uq.audit`).

    python scripts/run_calibration_audit.py --config configs/vespuq/vespuq_smoke.yaml

Outputs (under --out-dir, default outputs/audit):
    calibration_audit.json, calibration_audit.md, sentinel_audit.csv

Everything measured here is force-model error (``a_reference - a_surrogate``); none of it is a
position-error or orbit-covariance diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vesp.common.config import get_dtype, load_config
from vesp.uq.audit import audit_summary_dict, evaluate_false_negatives, select_sentinel_audit
from vesp.uq.conformal import coverage_before_after, fit_conformal_scale
from vesp.uq.data import split_uq_samples
from vesp.uq.ensemble import nearest_neighbor_error_magnitude
from vesp.uq.experiment import (
    _build_trajectories,
    _load_samples,
    _resolve_time_weighting,
    _time_weights,
)
from vesp.uq.metrics import mahalanobis_squared
from vesp.uq.plugin import VESPUQPlugin
from vesp.uq.scoring import aggregate_trajectory_error
from vesp.uq.selection import select_reruns
from vesp.uq.thresholds import resolve_threshold


def prepare(config: dict):
    """Fit VESP-UQ on the train split; return (plugin, samples, train, held, dtype, seed)."""

    dtype = get_dtype(config)
    samples = _load_samples(config, dtype)
    seed = int(config.get("seed", 0))
    train, held = split_uq_samples(
        samples, train_fraction=float(config.get("data", {}).get("train_fraction", 0.7)), seed=seed
    )
    plugin = VESPUQPlugin.from_config(config)
    plugin.fit(train.positions, train.surrogate, train.reference)
    return plugin, samples, train, held, dtype, seed


def heldout_force_error_pair(plugin: VESPUQPlugin, held, mode: str):
    """Build (predicted_error, true_error) arrays for conformal calibration on held-out samples.

    The true error is the predictive residual ``observed_error - posterior_mean_error``; the
    predicted error is the matching predictive uncertainty for ``mode``:

    - ``norm``: predicted = total predictive ``sigma`` (per point), true = residual vector.
    - ``component_max``: predicted = per-component std, true = residual vector.
    - ``mahalanobis``: predicted = the nominal 3-DOF radius ``sqrt(3)``, true = the realized
      Mahalanobis distance of the residual under the predictive 3x3 covariance.
    """

    cov = plugin.predict_covariance_3x3(held.positions)
    residual = held.error - cov.mean_error  # (N, 3) predictive residual force error
    if mode == "mahalanobis":
        d = torch.sqrt(mahalanobis_squared(residual, cov.covariance).clamp_min(0.0))
        predicted = torch.full_like(d, float(3.0) ** 0.5)  # E[d] reference for chi-square(3)
        return predicted, d
    if mode == "component_max":
        return cov.std_components, residual
    return cov.sigma, residual  # norm (default)


def screen_trajectories(plugin: VESPUQPlugin, config: dict, samples, held, dtype, seed):
    """Replicate the configured VESP-UQ screening; return (risk, true_error, screening report).

    Mirrors :func:`vesp.uq.experiment.run_vespuq` (score the ensemble, read the true force error,
    apply the threshold/fraction selection policy) so the audit screens exactly as the main run.
    """

    screen_cfg = config.get("uq", {}).get("screening", {})
    traj_info = _build_trajectories(screen_cfg, seed=seed, dtype=dtype)
    trajectories = traj_info["trajectories"]
    scoring = plugin.risk_scoring

    time_weighting = _resolve_time_weighting(screen_cfg)
    weights = [_time_weights(t) for t in trajectories] if time_weighting == "kepler_r2" else None
    scores = plugin.score_ensemble(trajectories, weights=weights)
    risk_scores = torch.tensor([s.risk_score for s in scores], dtype=torch.float64)

    # True force error per trajectory (NOT position error): direct residual if a CSV supplied accel
    # pairs, else nearest-neighbour read from held-out residual samples.
    aggregator = str(screen_cfg.get("true_error_aggregator", "p95")).lower()
    oracle_source = str(screen_cfg.get("oracle_source", "heldout")).lower()
    if oracle_source not in {"heldout", "all"}:
        raise ValueError("uq.screening.oracle_source must be 'heldout' or 'all'")
    oracle = samples if oracle_source == "all" else held
    use_residual = traj_info["residuals"] is not None and str(
        screen_cfg.get("true_error_source", "auto")
    ).lower() in {"auto", "residual_csv"}
    true_error = torch.empty(len(trajectories), dtype=torch.float64)
    if use_residual:
        for i, res in enumerate(traj_info["residuals"]):
            mag = torch.linalg.norm(res.to(torch.float64), dim=-1)
            true_error[i] = aggregate_trajectory_error(mag, aggregator)
    else:
        for i, traj in enumerate(trajectories):
            nn = nearest_neighbor_error_magnitude(traj.to(dtype), oracle.positions, oracle.error)
            true_error[i] = aggregate_trajectory_error(nn.to(torch.float64), aggregator)

    max_rerun_fraction = screen_cfg.get("max_rerun_fraction")
    rerun_fraction = float(screen_cfg.get("rerun_fraction", 0.20))
    fraction_policy = str(screen_cfg.get("fraction_policy", "topk")).lower()
    threshold, threshold_meta = resolve_threshold(screen_cfg, plugin, held, scoring, dtype=dtype, seed=seed)
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
    return {
        "scoring": scoring,
        "trajectory_source": traj_info["source"],
        "trajectory_path": traj_info["path"],
        "true_error_aggregator": aggregator,
        "risk_scores": risk_scores,
        "true_error": true_error,
        "screening": screening,
    }


def run_calibration_audit(config: dict, *, prepared=None) -> dict:
    """Fit, conformally calibrate force-error uncertainty, screen, and sentinel-audit; return a dict."""

    plugin, samples, train, held, dtype, seed = prepared or prepare(config)
    uq_cfg = config.get("uq", {})
    conformal_cfg = uq_cfg.get("conformal", {}) or {}
    audit_cfg = uq_cfg.get("audit", {}) or {}

    alpha = float(conformal_cfg.get("alpha", 0.10))
    mode = str(conformal_cfg.get("mode", "norm")).lower()

    # ---- 1. held-out force-error conformal calibration ----
    predicted, true = heldout_force_error_pair(plugin, held, mode)
    calibrator = fit_conformal_scale(predicted, true, alpha=alpha, mode=mode)
    calibrated = calibrator.apply(predicted)
    coverage = coverage_before_after(predicted, true, calibrated, alpha=alpha, mode=mode)

    # ---- 2. configured risk screening ----
    screen = screen_trajectories(plugin, config, samples, held, dtype, seed)
    screening = screen["screening"]
    n_trajectories = int(screen["risk_scores"].numel())

    # ---- 3. sentinel audit of the accepted (low-risk) trajectories ----
    audit_fraction = float(audit_cfg.get("audit_fraction", 0.02))
    min_audit = int(audit_cfg.get("min_audit", 5))
    high_error_quantile = float(audit_cfg.get("high_error_quantile", 0.90))
    audit_seed = int(audit_cfg.get("seed", seed))
    sentinel = select_sentinel_audit(
        screening.flagged_indices,
        n_trajectories,
        audit_fraction=audit_fraction,
        min_audit=min_audit,
        seed=audit_seed,
    )
    false_negatives = evaluate_false_negatives(
        screening.flagged_indices,
        sentinel,
        screen["true_error"],
        high_error_quantile=high_error_quantile,
    )
    audit = audit_summary_dict(
        n_trajectories,
        screening.flagged_indices,
        sentinel,
        false_negatives,
        audit_fraction=audit_fraction,
        min_audit=min_audit,
        high_error_quantile=high_error_quantile,
        seed=audit_seed,
    )

    return {
        "config_path": config.get("_config_path"),
        "error_basis": "true_force_model_error",
        "scope_note": (
            "Conformal calibration and sentinel auditing concern force-model error "
            "(a_reference - a_surrogate). They are NOT position-error, trajectory-accuracy, or "
            "orbit-covariance diagnostics."
        ),
        "fit": plugin.fit_info,
        "conformal": {
            "alpha": alpha,
            "mode": mode,
            "n_calibration_samples": calibrator.n_calibration,
            "calibrator": calibrator.to_dict(),
            "coverage": coverage,
        },
        "screening": {
            "scoring": screen["scoring"],
            "trajectory_source": screen["trajectory_source"],
            "trajectory_path": screen["trajectory_path"],
            "true_error_aggregator": screen["true_error_aggregator"],
            "n_trajectories": n_trajectories,
            "n_flagged": screening.n_flagged,
            "screen": screening.to_dict(),
        },
        "audit": audit,
        "_sentinel_rows": [
            {
                "trajectory_id": int(i),
                "risk_score": float(screen["risk_scores"][i]),
                "true_force_error": float(screen["true_error"][i]),
                "is_high_force_error": int(i in set(false_negatives["sentinel_high_error_indices"])),
                "flagged": 0,
            }
            for i in sentinel
        ],
    }


def _audit_md(report: dict) -> str:
    def f(x, s=".4f"):
        return "n/a" if x is None else format(float(x), s)

    conf = report["conformal"]
    cov = conf["coverage"]
    scr = report["screening"]
    fn = report["audit"]["false_negatives"]
    return "\n".join([
        "# VESP-UQ Conformal Calibration + Sentinel Audit",
        "",
        "**This report concerns FORCE-MODEL error (`a_reference - a_surrogate`), not position "
        "error.** Conformal calibration measures and improves empirical held-out force-error "
        "coverage; the sentinel audit estimates force-error false negatives among accepted "
        "low-risk trajectories. Neither is a position-error or orbit-covariance diagnostic, and "
        "VESP-UQ's intervals are not assumed correct -- coverage is measured, not guaranteed.",
        "",
        "## Conformal calibration",
        f"- config: `{report.get('config_path')}`",
        f"- calibration samples: {conf['n_calibration_samples']}",
        f"- mode: `{conf['mode']}`  |  alpha: {conf['alpha']:.3f}  (target coverage "
        f"{cov['target_coverage']:.3f})",
        f"- learned conformal scale: {f(conf['calibrator']['scale'])}",
        f"- **coverage before: {f(cov['coverage_before'])}  ->  after: {f(cov['coverage_after'])}**"
        f"  (improvement {f(cov['coverage_improvement'])}; reaches target: {cov['covers_target_after']})",
        "",
        "## Risk screening",
        f"- scoring: `{scr['scoring']}`  |  trajectory source: `{scr['trajectory_source']}`",
        f"- trajectories: {scr['n_trajectories']}  |  flagged by risk: {scr['n_flagged']}",
        "",
        "## Sentinel audit (accepted low-risk trajectories)",
        f"- accepted: {report['audit']['n_accepted']}  |  selected for sentinel audit: "
        f"{report['audit']['n_sentinel']}  (fraction {report['audit']['audit_fraction']:.3f}, "
        f"min {report['audit']['min_audit']}, seed {report['audit']['seed']})",
        f"- high-error definition: true force error at/above the "
        f"{fn['high_error_quantile']:.2f} quantile (threshold {f(fn['high_error_threshold'], '.3e')})",
        f"- high-error trajectories: {fn['n_high_error']}  |  captured by rerun budget: "
        f"{fn['n_high_error_flagged']}  |  **false negatives (accepted & high-error): "
        f"{fn['n_false_negatives']}** (rate {f(fn['false_negative_rate'])})",
        f"- sentinel high-error hits: {fn['n_sentinel_high_error']} / {fn['n_sentinel']}  "
        f"(sentinel false-negative estimate {f(fn['sentinel_false_negative_rate'])})",
        "",
        "Interpretation: a false negative is an accepted trajectory whose true FORCE-MODEL error is "
        "in the high-error tail. The sentinel sample lets this rate be audited empirically instead "
        "of assumed; it says nothing about long-horizon position accuracy.",
        "",
    ]) + "\n"


def run_and_write(config: dict, *, out_dir: Path) -> dict:
    report = run_calibration_audit(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = report.pop("_sentinel_rows")

    (out_dir / "calibration_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "calibration_audit.md").write_text(_audit_md(report), encoding="utf-8")
    header = ["trajectory_id", "risk_score", "true_force_error", "is_high_force_error", "flagged"]
    lines = [",".join(header)] + [",".join(str(r[h]) for h in header) for r in rows]
    (out_dir / "sentinel_audit.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="VESP-UQ conformal calibration + sentinel audit.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="outputs/audit")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config.setdefault("_config_path", args.config)
    report = run_and_write(config, out_dir=Path(args.out_dir))
    print(_audit_md(report).encode("ascii", "replace").decode("ascii"))
    print(f"saved_calibration_audit: {Path(args.out_dir) / 'calibration_audit.md'}")


if __name__ == "__main__":
    main()
