"""Compare VESP-UQ trajectory force-risk scores against simple baseline selectors.

Target: trajectory-level true FORCE-MODEL error (never position error). Baselines:

    random | min_altitude | low_altitude_exposure | knn_p95 | domain_support (if enabled)
    | uncertainty_only (mean sigma) | altitude_residual_expected_* | supervisor (supervisor_rel_p95)

    python scripts/compare_risk_baselines.py --config configs/vespuq/vespuq_smoke.yaml

Outputs (under --out-dir, default outputs/baselines):
    baseline_comparison.json/csv/md, baseline_comparison_paper.csv,
    altitude_incremental_value.csv, altitude_incremental_sweep.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from vesp.common.config import get_dtype, load_config
from vesp.uq.baselines import (
    altitude_residual_expected_scores,
    domain_support_scores,
    fit_altitude_expected_curve,
    knn_p95_scores,
    low_altitude_exposure_scores,
    min_altitude_scores,
    random_scores,
    vespuq_scores,
)
from vesp.uq.benchmarking import METRIC_KEYS, _best_by, compare_baselines, evaluate_score_against_true_error
from vesp.uq.data import split_uq_samples
from vesp.uq.ensemble import nearest_neighbor_error_magnitude
from vesp.uq.experiment import _build_trajectories, _load_samples
from vesp.uq.io.run_artifacts import write_run_artifacts
from vesp.uq.plugin import VESPUQPlugin
from vesp.uq.scoring import aggregate_trajectory_error

# Baseline scoring modes used for the two VESP-UQ entries.
_UNCERTAINTY_SCORING = "mean"  # mean predictive sigma -- uncertainty-only (no bias / altitude)
_SUPERVISOR_SCORING = "supervisor_rel_p95"  # full supervisor (expected error * altitude * domain)
_ALTITUDE_RESIDUAL_AGGREGATOR = "p95"
_ALTITUDE_INCREMENTAL_FRACTIONS = (0.05, 0.10, 0.20)
_ALTITUDE_INCREMENTAL_BOOTSTRAP = 100
_ALTITUDE_DELTA_SPEARMAN_GATE = 0.05
_ALTITUDE_DELTA_LIFT_GATE = 0.30


def _as_float_tensor(values) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float64).reshape(-1)


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    """Average ranks for Spearman diagnostics; deterministic and dependency-free."""

    v = _as_float_tensor(values)
    order = torch.argsort(v, stable=True)
    sorted_v = v[order]
    ranks_sorted = torch.empty_like(sorted_v)
    n = int(v.numel())
    start = 0
    while start < n:
        end = start + 1
        while end < n and bool(sorted_v[end] == sorted_v[start]):
            end += 1
        ranks_sorted[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end
    ranks = torch.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    x = _as_float_tensor(a)
    y = _as_float_tensor(b)
    if x.numel() != y.numel():
        raise ValueError("correlation inputs must have the same length")
    if x.numel() < 2 or not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    denom = torch.linalg.norm(xc) * torch.linalg.norm(yc)
    if float(denom) <= 0.0:
        return float("nan")
    return float((xc @ yc) / denom)


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    return _pearson(_rankdata(a), _rankdata(b))


def _residualize(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    yy = _as_float_tensor(y)
    xx = _as_float_tensor(x)
    xc = xx - xx.mean()
    denom = float(xc @ xc)
    if denom <= 0.0:
        return yy - yy.mean()
    beta = float((xc @ (yy - yy.mean())) / denom)
    alpha = float(yy.mean()) - beta * float(xx.mean())
    return yy - (alpha + beta * xx)


def _partial_pearson_given_altitude(score: torch.Tensor, true_error: torch.Tensor, min_radius: torch.Tensor) -> float:
    return _pearson(_residualize(score, min_radius), _residualize(true_error, min_radius))


def _within_altitude_bins(
    score: torch.Tensor,
    true_error: torch.Tensor,
    min_radius: torch.Tensor,
    *,
    n_bins: int = 3,
) -> dict:
    """Spearman(score, true_error) inside min-radius quantile bins."""

    s = _as_float_tensor(score)
    e = _as_float_tensor(true_error)
    r = _as_float_tensor(min_radius)
    order = torch.argsort(r)
    chunks = torch.chunk(order, min(max(1, int(n_bins)), int(order.numel())))
    bins = []
    weighted = 0.0
    total = 0
    for idx in chunks:
        if idx.numel() < 3:
            continue
        rho = _spearman(s[idx], e[idx])
        item = {
            "n": int(idx.numel()),
            "min_radius_low": float(r[idx].min()),
            "min_radius_high": float(r[idx].max()),
            "spearman": rho,
        }
        bins.append(item)
        if math.isfinite(rho):
            weighted += rho * int(idx.numel())
            total += int(idx.numel())
    return {"weighted_spearman": (weighted / total) if total else float("nan"), "bins": bins}


def _ci(values: list[float]) -> list[float | None]:
    finite = torch.tensor([v for v in values if math.isfinite(float(v))], dtype=torch.float64)
    if finite.numel() == 0:
        return [None, None]
    return [float(torch.quantile(finite, 0.025)), float(torch.quantile(finite, 0.975))]


def _bootstrap_delta_ci(
    score: torch.Tensor,
    altitude_score: torch.Tensor,
    true_error: torch.Tensor,
    *,
    rerun_fraction: float,
    seed: int,
    n_bootstrap: int,
) -> dict:
    if int(n_bootstrap) <= 0:
        return {
            "delta_spearman_ci95": [None, None],
            "delta_lift_ci95": [None, None],
            "n_bootstrap": 0,
        }
    s = _as_float_tensor(score)
    a = _as_float_tensor(altitude_score)
    e = _as_float_tensor(true_error)
    n = int(s.numel())
    g = torch.Generator().manual_seed(int(seed))
    d_spearman: list[float] = []
    d_lift: list[float] = []
    for _ in range(int(n_bootstrap)):
        idx = torch.randint(0, n, (n,), generator=g)
        m = evaluate_score_against_true_error(s[idx], e[idx], rerun_fraction=rerun_fraction)
        b = evaluate_score_against_true_error(a[idx], e[idx], rerun_fraction=rerun_fraction)
        d_spearman.append(float(m["spearman"]) - float(b["spearman"]))
        d_lift.append(float(m["lift_over_random"]) - float(b["lift_over_random"]))
    return {
        "delta_spearman_ci95": _ci(d_spearman),
        "delta_lift_ci95": _ci(d_lift),
        "n_bootstrap": int(n_bootstrap),
    }


def _min_radius_scores(trajectories) -> torch.Tensor:
    out = torch.empty(len(trajectories), dtype=torch.float64)
    for i, traj in enumerate(trajectories):
        out[i] = float(torch.linalg.norm(torch.as_tensor(traj, dtype=torch.float64), dim=-1).min())
    return out


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


def _true_force_error(trajectories, *, residuals, held, aggregator, dtype):
    """Trajectory-level true FORCE error: direct residual pairs if present, else held-out NN oracle."""

    true_error = torch.empty(len(trajectories), dtype=torch.float64)
    if residuals is not None:
        for i, res in enumerate(residuals):
            mag = torch.linalg.norm(torch.as_tensor(res, dtype=torch.float64), dim=-1)
            true_error[i] = aggregate_trajectory_error(mag, aggregator)
        return true_error, "residual_csv"
    for i, traj in enumerate(trajectories):
        nn = nearest_neighbor_error_magnitude(traj.to(dtype), held.positions, held.error)
        true_error[i] = aggregate_trajectory_error(nn.to(torch.float64), aggregator)
    return true_error, "nn_oracle_heldout"


def baseline_scores_for(config: dict, plugin, trajectories, train_positions, *, seed: int):
    """Assemble the baseline -> per-trajectory-score mapping for a fitted plugin + trajectories."""

    low_alt = float(config.get("uq", {}).get("risk", {}).get("low_altitude_radius", 1.15))
    curve_positions = getattr(plugin, "val_positions", None)
    if curve_positions is None:
        curve_positions = train_positions
    curve = fit_altitude_expected_curve(plugin, curve_positions)
    scores = {
        "min_altitude": min_altitude_scores(trajectories),
        "low_altitude_exposure": low_altitude_exposure_scores(trajectories, low_altitude_radius=low_alt),
        "knn_p95": knn_p95_scores(train_positions, trajectories),
        "uncertainty_only": vespuq_scores(plugin, trajectories, _UNCERTAINTY_SCORING),
        "altitude_residual_expected_ratio": altitude_residual_expected_scores(
            plugin,
            trajectories,
            curve=curve,
            mode="ratio",
            aggregator=_ALTITUDE_RESIDUAL_AGGREGATOR,
        ),
        "altitude_residual_expected_delta": altitude_residual_expected_scores(
            plugin,
            trajectories,
            curve=curve,
            mode="delta",
            aggregator=_ALTITUDE_RESIDUAL_AGGREGATOR,
        ),
        "supervisor": vespuq_scores(plugin, trajectories, _SUPERVISOR_SCORING),
    }
    if getattr(plugin, "domain_support", False):
        scores["domain_support"] = domain_support_scores(plugin, trajectories)
    return scores


def altitude_incremental_value_report(
    scores: dict[str, torch.Tensor],
    true_error: torch.Tensor,
    trajectories,
    *,
    seed: int,
    rerun_fraction: float,
    rerun_fractions: tuple[float, ...] = _ALTITUDE_INCREMENTAL_FRACTIONS,
    bootstrap_samples: int = _ALTITUDE_INCREMENTAL_BOOTSTRAP,
    altitude_baseline: str = "min_altitude",
) -> dict:
    """P0 diagnostic: how much value each score adds beyond altitude-only."""

    if altitude_baseline not in scores:
        raise ValueError(f"altitude baseline {altitude_baseline!r} is missing")
    min_radius = _min_radius_scores(trajectories)
    baseline_score = scores[altitude_baseline]
    fraction_sweep = {
        f"{float(frac):.3f}": compare_baselines(scores, true_error, rerun_fraction=float(frac))
        for frac in rerun_fractions
    }
    primary = fraction_sweep[f"{float(rerun_fraction):.3f}"] if f"{float(rerun_fraction):.3f}" in fraction_sweep else compare_baselines(
        scores, true_error, rerun_fraction=float(rerun_fraction)
    )
    baseline_primary = primary[altitude_baseline]

    summary = {}
    for offset, (name, score) in enumerate(scores.items()):
        metrics = primary[name]
        within = _within_altitude_bins(score, true_error, min_radius)
        delta_spearman = float(metrics["spearman"]) - float(baseline_primary["spearman"])
        delta_lift = float(metrics["lift_over_random"]) - float(baseline_primary["lift_over_random"])
        ci = (
            {
                "delta_spearman_ci95": [0.0, 0.0],
                "delta_lift_ci95": [0.0, 0.0],
                "n_bootstrap": int(bootstrap_samples),
            }
            if name == altitude_baseline
            else _bootstrap_delta_ci(
                score,
                baseline_score,
                true_error,
                rerun_fraction=float(rerun_fraction),
                seed=int(seed) + 1009 + offset,
                n_bootstrap=int(bootstrap_samples),
            )
        )
        summary[name] = {
            "spearman": metrics["spearman"],
            "capture_rate": metrics["capture_rate"],
            "precision": metrics["precision"],
            "lift_over_random": metrics["lift_over_random"],
            "delta_spearman_vs_min_altitude": delta_spearman,
            "delta_lift_vs_min_altitude": delta_lift,
            "partial_pearson_given_min_radius": _partial_pearson_given_altitude(score, true_error, min_radius),
            "within_altitude_bin_spearman": within["weighted_spearman"],
            "within_altitude_bins": within["bins"],
            "bootstrap_delta_spearman_ci95": ci["delta_spearman_ci95"],
            "bootstrap_delta_lift_ci95": ci["delta_lift_ci95"],
            "beats_altitude_gate": bool(
                delta_spearman >= _ALTITUDE_DELTA_SPEARMAN_GATE or delta_lift >= _ALTITUDE_DELTA_LIFT_GATE
            ),
        }

    return {
        "purpose": "Quantifies score value beyond the min-altitude heuristic.",
        "altitude_baseline": altitude_baseline,
        "primary_rerun_fraction": float(rerun_fraction),
        "rerun_fractions": [float(f) for f in rerun_fractions],
        "bootstrap_samples": int(bootstrap_samples),
        "gates": {
            "delta_spearman": _ALTITUDE_DELTA_SPEARMAN_GATE,
            "delta_lift": _ALTITUDE_DELTA_LIFT_GATE,
            "rule": "pass if either delta_spearman or delta_lift meets the gate",
        },
        "summary": summary,
        "fraction_sweep": fraction_sweep,
    }


def run_baseline_comparison(
    config: dict,
    *,
    rerun_fraction: float = 0.10,
    prepared=None,
    bootstrap_samples: int = _ALTITUDE_INCREMENTAL_BOOTSTRAP,
) -> dict:
    """Run the full baseline comparison; return a payload dict (no file I/O)."""

    plugin, samples, train, held, dtype, seed = prepared or prepare(config)
    screen_cfg = config.get("uq", {}).get("screening", {})
    aggregator = str(screen_cfg.get("true_error_aggregator", "p95")).lower()

    traj_info = _build_trajectories(screen_cfg, seed=seed, dtype=dtype)
    trajectories = traj_info["trajectories"]
    true_error, te_source = _true_force_error(
        trajectories, residuals=traj_info["residuals"], held=held, aggregator=aggregator, dtype=dtype
    )

    scores = baseline_scores_for(config, plugin, trajectories, train.positions, seed=seed)
    results = compare_baselines(scores, true_error, rerun_fraction=rerun_fraction)

    # Run Random baseline 100 times
    import numpy as np

    random_metrics = {k: [] for k in METRIC_KEYS}
    for s in range(100):
        r_scores = random_scores(len(trajectories), seed=seed + s)

        r_eval = evaluate_score_against_true_error(r_scores, true_error, rerun_fraction=rerun_fraction)
        for k in METRIC_KEYS:
            random_metrics[k].append(r_eval.get(k))

    random_summary = {}
    for k in METRIC_KEYS:
        vals = [v for v in random_metrics[k] if v is not None and not math.isnan(float(v))]
        if vals:
            random_summary[k] = float(np.mean(vals))
            random_summary[k + "_std"] = float(np.std(vals))
        else:
            random_summary[k] = None
            random_summary[k + "_std"] = None

    results["random"] = random_summary
    incremental = altitude_incremental_value_report(
        scores,
        true_error,
        trajectories,
        seed=seed,
        rerun_fraction=rerun_fraction,
        bootstrap_samples=int(bootstrap_samples),
    )
    return {
        "config_dataset": str(config.get("data", {}).get("path") or samples.metadata.get("mode", "synthetic")),
        "n_trajectories": len(trajectories),
        "trajectory_source": traj_info["source"],
        "true_force_error_source": te_source,
        "true_force_error_aggregator": aggregator,
        "rerun_fraction": rerun_fraction,
        "uncertainty_scoring": _UNCERTAINTY_SCORING,
        "supervisor_scoring": _SUPERVISOR_SCORING,
        "baselines": results,
        "best_by_spearman": _best_by(results, "spearman"),
        "best_by_lift": _best_by(results, "lift_over_random"),
        "altitude_incremental_value": incremental,
    }


def _fmt(x, spec=".4f"):
    if x is None:
        return "n/a"
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return str(x)


def _csv_value(x) -> str:
    if x is None:
        return ""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v):
        return ""
    return str(v)


def _ci_text(ci) -> str:
    if not ci or ci[0] is None or ci[1] is None:
        return "n/a"
    return f"[{_fmt(ci[0])}, {_fmt(ci[1])}]"


def _comparison_md(p: dict) -> str:
    results = p["baselines"]
    cols = ["spearman", "capture_rate", "precision", "lift_over_random",
            "mean_true_error_flagged", "mean_true_error_accepted", "force_error_ratio_flagged_to_accepted"]
    short = {"spearman": "spearman", "capture_rate": "capture", "precision": "precision",
             "lift_over_random": "lift", "mean_true_error_flagged": "err_flag",
             "mean_true_error_accepted": "err_acc", "force_error_ratio_flagged_to_accepted": "ratio"}
    lines = [
        "# VESP-UQ Baseline Comparison (trajectory force-risk screening)",
        "",
        "Target: trajectory-level true **force-model** error (NOT position error). Each selector",
        "flags the top trajectories; higher score = higher risk.",
        "",
        f"- dataset: `{p['config_dataset']}`  |  trajectories: {p['n_trajectories']} "
        f"({p['trajectory_source']})",
        f"- true force error: `{p['true_force_error_source']}` "
        f"(aggregator `{p['true_force_error_aggregator']}`)  |  rerun fraction: {p['rerun_fraction']:.0%}",
        f"- uncertainty_only scoring = `{p['uncertainty_scoring']}`  |  supervisor scoring = `{p['supervisor_scoring']}`",
        "",
        "| baseline | " + " | ".join(short[c] for c in cols) + " |",
        "| --- | " + " | ".join("---:" for _ in cols) + " |",
    ]
    for name, m in results.items():
        row = " | ".join(
            _fmt(m.get(c), ".3e" if c.startswith("mean_true") else (".2f" if c in ("lift_over_random", "force_error_ratio_flagged_to_accepted") else ".4f"))
            for c in cols
        )
        lines.append(f"| `{name}` | {row} |")
    lines += [
        "",
        f"- best by Spearman: **{p['best_by_spearman']}**",
        f"- best by lift over random: **{p['best_by_lift']}**",
        "",
    ]
    inc = p.get("altitude_incremental_value") or {}
    inc_summary = inc.get("summary") or {}
    if inc_summary:
        lines += [
            "## Altitude Incremental Value",
            "",
            f"Altitude-only baseline: `{inc.get('altitude_baseline', 'min_altitude')}`. "
            "Positive deltas mean the score adds ranking value beyond minimum altitude.",
            "",
            "| method | spearman | d_spearman | lift | d_lift | partial | within-bin | CI d_spearman | CI d_lift | gate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
        for name, m in inc_summary.items():
            lines.append(
                "| "
                f"`{name}` | "
                f"{_fmt(m.get('spearman'))} | "
                f"{_fmt(m.get('delta_spearman_vs_min_altitude'))} | "
                f"{_fmt(m.get('lift_over_random'), '.2f')} | "
                f"{_fmt(m.get('delta_lift_vs_min_altitude'), '.2f')} | "
                f"{_fmt(m.get('partial_pearson_given_min_radius'))} | "
                f"{_fmt(m.get('within_altitude_bin_spearman'))} | "
                f"{_ci_text(m.get('bootstrap_delta_spearman_ci95'))} | "
                f"{_ci_text(m.get('bootstrap_delta_lift_ci95'))} | "
                f"{'pass' if m.get('beats_altitude_gate') else 'no'} |"
            )
        lines += [
            "",
            "`altitude_residual_expected_ratio` and `altitude_residual_expected_delta` are P1 scores: "
            "VESP-UQ expected error after removing an altitude-only expected-error curve fit on the "
            "plugin calibration geometry.",
            "",
        ]
    lines += [
        "Interpretation: a higher Spearman / lift means the selector concentrates the surrogate's",
        "true force-model error better. `min_altitude` and `low_altitude_exposure` are strong",
        "trivial baselines because force error usually grows toward low altitude; VESP-UQ adds value",
        "when its score (especially `supervisor`, which folds in expected bias and OOD risk) beats",
        "them, and adds none if it does not. This is a force-risk ranking comparison only -- it says",
        "nothing about long-horizon trajectory position error.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _baseline_csv(payload: dict) -> str:
    rows = [["baseline", *METRIC_KEYS]]
    for name, m in payload["baselines"].items():
        rows.append([name, *[m.get(k) for k in METRIC_KEYS]])
    return "\n".join(",".join(_csv_value(v) for v in row) for row in rows) + "\n"


def _paper_csv(payload: dict) -> str:
    band = "L90" if "L90" in payload["config_dataset"] else "L60"
    cols = [
        "band", "method", "flagged_fraction", "capture_rate", "capture_rate_std", 
        "precision", "precision_std", "lift", "lift_std", "spearman", 
        "flagged_accepted_error_ratio", "flagged_accepted_error_ratio_std", "runtime_ms_per_traj"
    ]
    rows = [cols]
    for name, m in payload["baselines"].items():
        rows.append(
            [
                band,
                name,
                payload["rerun_fraction"],
                m.get("capture_rate"),
                m.get("capture_rate_std", ""),
                m.get("precision"),
                m.get("precision_std", ""),
                m.get("lift_over_random"),
                m.get("lift_over_random_std", ""),
                m.get("spearman"),
                m.get("force_error_ratio_flagged_to_accepted"),
                m.get("force_error_ratio_flagged_to_accepted_std", ""),
                "",
            ]
        )
    return "\n".join(",".join(_csv_value(v) for v in row) for row in rows) + "\n"


def _incremental_csv(payload: dict) -> str:
    inc = payload.get("altitude_incremental_value") or {}
    rows = [
        [
            "method",
            "spearman",
            "delta_spearman_vs_min_altitude",
            "capture_rate",
            "precision",
            "lift_over_random",
            "delta_lift_vs_min_altitude",
            "partial_pearson_given_min_radius",
            "within_altitude_bin_spearman",
            "bootstrap_delta_spearman_low",
            "bootstrap_delta_spearman_high",
            "bootstrap_delta_lift_low",
            "bootstrap_delta_lift_high",
            "beats_altitude_gate",
        ]
    ]
    for name, m in (inc.get("summary") or {}).items():
        ds = m.get("bootstrap_delta_spearman_ci95") or [None, None]
        dl = m.get("bootstrap_delta_lift_ci95") or [None, None]
        rows.append(
            [
                name,
                m.get("spearman"),
                m.get("delta_spearman_vs_min_altitude"),
                m.get("capture_rate"),
                m.get("precision"),
                m.get("lift_over_random"),
                m.get("delta_lift_vs_min_altitude"),
                m.get("partial_pearson_given_min_radius"),
                m.get("within_altitude_bin_spearman"),
                ds[0],
                ds[1],
                dl[0],
                dl[1],
                m.get("beats_altitude_gate"),
            ]
        )
    return "\n".join(",".join(_csv_value(v) for v in row) for row in rows) + "\n"


def _incremental_sweep_csv(payload: dict) -> str:
    inc = payload.get("altitude_incremental_value") or {}
    rows = [["rerun_fraction", "method", "capture_rate", "precision", "lift_over_random", "spearman"]]
    for frac, methods in (inc.get("fraction_sweep") or {}).items():
        for name, m in methods.items():
            rows.append([frac, name, m.get("capture_rate"), m.get("precision"), m.get("lift_over_random"), m.get("spearman")])
    return "\n".join(",".join(_csv_value(v) for v in row) for row in rows) + "\n"


def write_outputs(payload: dict, out_dir: Path, *, config: dict | None = None) -> None:
    write_run_artifacts(
        out_dir,
        tool="compare_risk_baselines",
        config=config,
        json_files={"baseline_comparison.json": payload},
        text_files={
            "baseline_comparison.csv": _baseline_csv(payload),
            "baseline_comparison_paper.csv": _paper_csv(payload),
            "altitude_incremental_value.csv": _incremental_csv(payload),
            "altitude_incremental_sweep.csv": _incremental_sweep_csv(payload),
            "baseline_comparison.md": _comparison_md(payload),
        },
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Compare VESP-UQ force-risk scores against simple baselines.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="outputs/baselines")
    parser.add_argument("--rerun-fraction", type=float, default=0.10)
    parser.add_argument("--bootstrap-samples", type=int, default=_ALTITUDE_INCREMENTAL_BOOTSTRAP)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config.setdefault("_config_path", args.config)
    payload = run_baseline_comparison(
        config,
        rerun_fraction=args.rerun_fraction,
        bootstrap_samples=args.bootstrap_samples,
    )
    out_dir = Path(args.out_dir)
    write_outputs(payload, out_dir, config=config)
    print(_comparison_md(payload))
    print(f"saved_baseline_comparison: {out_dir / 'baseline_comparison.md'}")


if __name__ == "__main__":
    main()
