"""White-box trajectory attribution for VESP-UQ supervisor risk.

This is intentionally not SHAP/LIME. VESP-UQ's supervisor point risk is already white-box and
multiplicative:

``point_risk = expected_error * W_alt(radius) * (1 + domain_weight * d_OOD)``.

Taking logs gives an exact additive decomposition for every trajectory point:

``log(point_risk) = log(expected_error) + log(W_alt) + log(OOD factor)``.

The helpers below aggregate those exact point terms per trajectory and validate the segment-level
story with a masking test: remove the top-attribution points and check whether the held-out
nearest-neighbour true force-error profile drops more than a random mask of the same size.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import torch

from vesp.uq.cli import csv_text, fmt_float
from vesp.uq.ensemble import nearest_neighbor_error_magnitude
from vesp.uq.experiment import _build_trajectories, _resolve_time_weighting, _time_weights
from vesp.uq.io.run_artifacts import write_run_artifacts
from vesp.uq.metrics import safe_log
from vesp.uq.risk_baselines import prepare
from vesp.uq.scoring import _weighted_quantile
from vesp.uq.suite import band_label

_TERMS = ("log_expected_error", "log_altitude_weight", "log_ood_factor")


def _weights_or_none(weights, n: int) -> torch.Tensor | None:
    if weights is None:
        return None
    w = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
    if w.numel() != n:
        raise ValueError("weights must match trajectory length")
    if bool((w < 0).any()):
        raise ValueError("weights must be nonnegative")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return w / total


def _wmean(values: torch.Tensor, weights: torch.Tensor | None = None) -> float:
    v = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    w = _weights_or_none(weights, int(v.numel()))
    return float((v * w).sum()) if w is not None else float(v.mean())


def _aggregate(values: torch.Tensor, mode: str, weights: torch.Tensor | None = None) -> float:
    if mode not in {"mean", "p95", "max"}:
        raise ValueError("mode must be 'mean', 'p95', or 'max'")
    v = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    if v.numel() == 0:
        return float("nan")
    if mode == "max":
        return float(v.max())
    if mode == "mean":
        return _wmean(v, weights)
    return _weighted_quantile(v, 0.95, _weights_or_none(weights, int(v.numel())))


def _altitude_weight(radius: torch.Tensor, *, mode: str, low_altitude_radius: float) -> torch.Tensor:
    h_floor = 1.0e-3
    raw = 1.0 / (radius - 1.0).clamp_min(h_floor)
    if mode == "relative":
        return raw / torch.median(raw).clamp_min(torch.finfo(torch.float64).tiny)
    reference_h = max(float(low_altitude_radius) - 1.0, h_floor)
    return raw / (1.0 / reference_h)


def _score_scale_from_scoring(scoring: str) -> str:
    return "absolute" if "_abs" in str(scoring) else "relative"


def supervisor_log_profiles(
    plugin,
    trajectory: torch.Tensor,
    *,
    scoring: str = "supervisor_rel_p95",
    weights=None,
) -> dict:
    """Return exact per-point supervisor log terms for one trajectory."""

    pos = plugin._prep_positions(trajectory).detach().cpu()
    pred = plugin.predict_uncertainty(pos)
    radius = pred.radius.detach().cpu().to(torch.float64)
    expected = pred.expected_error.detach().cpu().to(torch.float64)
    if getattr(plugin, "domain_support", False):
        domain_risk = plugin.domain_support_score(pos).detach().cpu().to(torch.float64)
    else:
        domain_risk = torch.zeros_like(radius)
    domain_factor = 1.0 + float(getattr(plugin, "domain_weight", 1.0)) * domain_risk.clamp_min(0.0)
    altitude_mode = _score_scale_from_scoring(scoring)
    altitude_weight = _altitude_weight(
        radius, mode=altitude_mode, low_altitude_radius=float(plugin.low_altitude_radius)
    )
    terms = {
        "log_expected_error": safe_log(expected),
        "log_altitude_weight": safe_log(altitude_weight),
        "log_ood_factor": safe_log(domain_factor),
    }
    log_point_risk = terms["log_expected_error"] + terms["log_altitude_weight"] + terms["log_ood_factor"]
    point_risk = torch.exp(log_point_risk)
    return {
        "positions": pos,
        "radius": radius,
        "expected_error": expected,
        "altitude_weight": altitude_weight,
        "domain_risk": domain_risk,
        "domain_factor": domain_factor,
        "terms": terms,
        "log_point_risk": log_point_risk,
        "point_risk": point_risk,
        "weights": None if weights is None else torch.as_tensor(weights, dtype=torch.float64),
        "altitude_mode": altitude_mode,
    }


def summarize_log_attribution(profile: dict) -> dict:
    """Aggregate exact log terms and return additive contributions + absolute-share diagnostics."""

    weights = profile.get("weights")
    contributions = {name: _wmean(profile["terms"][name], weights) for name in _TERMS}
    total = sum(contributions.values())
    abs_total = sum(abs(v) for v in contributions.values())
    shares = {
        name.replace("log_", "share_abs_"): (abs(value) / abs_total if abs_total > 0.0 else float("nan"))
        for name, value in contributions.items()
    }
    dominant = max(contributions, key=lambda name: abs(contributions[name]))
    return {
        **contributions,
        "log_point_risk_mean": _wmean(profile["log_point_risk"], weights),
        "log_point_risk_additive_sum": total,
        "point_risk_p95": _aggregate(profile["point_risk"], "p95", weights),
        "point_risk_mean": _aggregate(profile["point_risk"], "mean", weights),
        "dominant_log_term": dominant,
        **shares,
    }


def _mask_top_indices(profile: dict, mask_fraction: float) -> torch.Tensor:
    n = int(profile["point_risk"].numel())
    k = min(n - 1, max(1, int(math.ceil(float(mask_fraction) * n))))
    return torch.argsort(profile["point_risk"], descending=True)[:k]


def masking_validation(
    profile: dict,
    true_error_profile: torch.Tensor,
    *,
    mask_fraction: float = 0.10,
    aggregator: str = "p95",
    seed: int = 0,
) -> dict:
    """Validate attribution by masking high-attribution points vs a random mask."""

    err = torch.as_tensor(true_error_profile, dtype=torch.float64).reshape(-1)
    if err.numel() != profile["point_risk"].numel():
        raise ValueError("true_error_profile must align with the attribution profile")
    weights = profile.get("weights")
    high_idx = _mask_top_indices(profile, mask_fraction)
    n = int(err.numel())
    k = int(high_idx.numel())
    keep_high = torch.ones(n, dtype=torch.bool)
    keep_high[high_idx] = False
    g = torch.Generator().manual_seed(int(seed))
    random_idx = torch.randperm(n, generator=g)[:k]
    keep_random = torch.ones(n, dtype=torch.bool)
    keep_random[random_idx] = False

    def _kept(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if weights is None:
            return err[mask], None
        w = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
        return err[mask], w[mask]

    base = _aggregate(err, aggregator, weights)
    high_values, high_weights = _kept(keep_high)
    random_values, random_weights = _kept(keep_random)
    masked_high = _aggregate(high_values, aggregator, high_weights)
    masked_random = _aggregate(random_values, aggregator, random_weights)
    denom = max(abs(base), torch.finfo(torch.float64).tiny)
    return {
        "mask_fraction": float(k / n),
        "n_points": n,
        "n_masked": k,
        "aggregator": aggregator,
        "true_error_base": base,
        "true_error_after_top_mask": masked_high,
        "true_error_after_random_mask": masked_random,
        "top_mask_drop_fraction": (base - masked_high) / denom,
        "random_mask_drop_fraction": (base - masked_random) / denom,
        "top_mask_beats_random": (base - masked_high) > (base - masked_random),
    }


def _summary_md(summary_rows: list[dict[str, Any]], masking_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# VESP-UQ Log-Factor Attribution",
        "",
        "White-box attribution for the multiplicative supervisor. No SHAP/LIME or black-box sampling is "
        "used: each row decomposes `log(point_risk)` into exact expected-error, altitude, and "
        "domain-support terms.",
        "",
        "## Summary",
        "",
        "| band | n | dominant expected | dominant altitude | dominant OOD | top-mask win-rate | median top-drop |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    bands = sorted({row["band"] for row in summary_rows})
    for band in bands:
        rows = [r for r in summary_rows if r["band"] == band]
        masks = [r for r in masking_rows if r["band"] == band]
        n = len(rows)
        dom_expected = sum(1 for r in rows if r["dominant_log_term"] == "log_expected_error")
        dom_alt = sum(1 for r in rows if r["dominant_log_term"] == "log_altitude_weight")
        dom_ood = sum(1 for r in rows if r["dominant_log_term"] == "log_ood_factor")
        wins = sum(1 for r in masks if str(r.get("top_mask_beats_random")).lower() == "true")
        drops = sorted(float(r["top_mask_drop_fraction"]) for r in masks if math.isfinite(float(r["top_mask_drop_fraction"])))
        median_drop = drops[len(drops) // 2] if drops else float("nan")
        lines.append(
            f"| {band} | {n} | {dom_expected / max(1, n):.2f} | {dom_alt / max(1, n):.2f} | "
            f"{dom_ood / max(1, n):.2f} | {wins / max(1, len(masks)):.2f} | {fmt_float(median_drop, '.3f')} |"
        )
    lines += [
        "",
        "## Honesty Notes",
        "",
        "- `d_OOD` is calibration-support distance, not dynamical instability.",
        "- The masking test validates whether high-attribution trajectory segments also carry held-out "
        "nearest-neighbour force error. Without that test, attribution is only a score explanation.",
        "- This layer explains existing risk scores; it does not change rerun selection by itself.",
        "",
    ]
    return "\n".join(lines)


def run_log_attribution(
    configs: list[dict],
    *,
    out_dir: str | Path = "outputs/log_attribution",
    n_orbits: int | None = None,
    n_points: int | None = None,
    top_n: int = 25,
    mask_fraction: float = 0.10,
    scoring: str = "supervisor_rel_p95",
    aggregator: str = "p95",
) -> dict:
    """Run exact log-factor attribution and masking validation for each config."""

    summary_rows: list[dict[str, Any]] = []
    masking_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    case_meta: list[dict[str, Any]] = []
    inputs: dict[str, str | Path] = {}

    for ci, config in enumerate(configs):
        cfg = copy.deepcopy(config)
        if n_orbits is not None:
            cfg.setdefault("uq", {}).setdefault("screening", {})["n_orbits"] = int(n_orbits)
        if n_points is not None:
            cfg.setdefault("uq", {}).setdefault("screening", {})["n_points"] = int(n_points)
        plugin, samples, train, held, dtype, seed = prepare(cfg)
        screen_cfg = cfg.get("uq", {}).get("screening", {})
        traj_info = _build_trajectories(screen_cfg, seed=seed, dtype=dtype, config=cfg)
        trajectories = traj_info["trajectories"]
        time_weighting = _resolve_time_weighting(screen_cfg)
        weights = [_time_weights(t) for t in trajectories] if time_weighting == "kepler_r2" else [None] * len(trajectories)
        scores = plugin.score_ensemble(trajectories, scoring=scoring, weights=weights)
        order = sorted(range(len(scores)), key=lambda i: float(scores[i].risk_score), reverse=True)
        selected = order[: min(int(top_n), len(order))]
        band = band_label(cfg)
        case_meta.append({
            "band": band,
            "n_trajectories_total": len(trajectories),
            "n_attributed": len(selected),
            "scoring": scoring,
            "mask_fraction": float(mask_fraction),
        })
        if cfg.get("_config_path"):
            inputs[f"config_{ci}"] = cfg["_config_path"]
        data_path = cfg.get("data", {}).get("path")
        if data_path:
            inputs[f"data_{ci}"] = data_path

        for rank, idx in enumerate(selected, start=1):
            traj = trajectories[idx]
            w = None if weights is None else weights[idx]
            profile = supervisor_log_profiles(plugin, traj, scoring=scoring, weights=w)
            attr = summarize_log_attribution(profile)
            nn_error = nearest_neighbor_error_magnitude(
                torch.as_tensor(traj, dtype=dtype), held.positions, held.error
            ).detach().cpu()
            mask = masking_validation(
                profile,
                nn_error,
                mask_fraction=mask_fraction,
                aggregator=aggregator,
                seed=seed + idx,
            )
            row = {
                "band": band,
                "rank": rank,
                "trajectory_id": idx,
                "risk_score": float(scores[idx].risk_score),
                "min_radius": float(torch.linalg.norm(torch.as_tensor(traj, dtype=torch.float64), dim=-1).min()),
                "mean_radius": float(torch.linalg.norm(torch.as_tensor(traj, dtype=torch.float64), dim=-1).mean()),
                **attr,
            }
            summary_rows.append(row)
            masking_rows.append({"band": band, "rank": rank, "trajectory_id": idx, **mask})

            top_idx = _mask_top_indices(profile, mask_fraction)
            for local_rank, point_idx in enumerate(top_idx[:10].tolist(), start=1):
                segment_rows.append({
                    "band": band,
                    "trajectory_id": idx,
                    "point_rank": local_rank,
                    "point_index": int(point_idx),
                    "radius": float(profile["radius"][point_idx]),
                    "point_risk": float(profile["point_risk"][point_idx]),
                    "true_error_nn": float(nn_error[point_idx]),
                    **{term: float(profile["terms"][term][point_idx]) for term in _TERMS},
                })

    text_files = {
        "log_attribution.md": _summary_md(summary_rows, masking_rows),
        "attribution_summary.csv": csv_text(
            summary_rows,
            [
                "band", "rank", "trajectory_id", "risk_score", "min_radius", "mean_radius",
                "log_expected_error", "log_altitude_weight", "log_ood_factor",
                "log_point_risk_mean", "log_point_risk_additive_sum", "point_risk_p95",
                "point_risk_mean", "dominant_log_term",
                "share_abs_expected_error", "share_abs_altitude_weight", "share_abs_ood_factor",
            ],
        ),
        "masking_validation.csv": csv_text(
            masking_rows,
            [
                "band", "rank", "trajectory_id", "mask_fraction", "n_points", "n_masked",
                "aggregator", "true_error_base", "true_error_after_top_mask",
                "true_error_after_random_mask", "top_mask_drop_fraction",
                "random_mask_drop_fraction", "top_mask_beats_random",
            ],
        ),
        "top_segments.csv": csv_text(
            segment_rows,
            [
                "band", "trajectory_id", "point_rank", "point_index", "radius", "point_risk",
                "true_error_nn", "log_expected_error", "log_altitude_weight", "log_ood_factor",
            ],
        ),
    }
    config_dict: dict[str, Any] = {
        "n_orbits": n_orbits,
        "n_points": n_points,
        "top_n": int(top_n),
        "mask_fraction": float(mask_fraction),
        "scoring": scoring,
        "aggregator": aggregator,
        "method": "exact_log_factor_decomposition_not_shap",
    }
    payload: dict[str, Any] = {
        "cases": case_meta,
        "settings": config_dict,
    }
    manifest = write_run_artifacts(
        out_dir,
        tool="run_vespuq_log_attribution",
        config=config_dict,
        json_files={"log_attribution.json": payload},
        text_files=text_files,
        inputs=inputs,
    )
    return {
        "out_dir": str(out_dir),
        "cases": case_meta,
        "summary_rows": summary_rows,
        "masking_rows": masking_rows,
        "manifest": manifest,
    }
