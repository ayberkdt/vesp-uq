"""Gate diagnostics for VESP-UQ development decisions.

This module implements the "measure first, change second" gate used for the next VESP-UQ research
steps. It keeps the diagnostics deliberately close to quantities the existing layer already exposes:
trajectory supervisor factors, the local predictive ``3x3`` covariance, held-out calibration
metrics, and a finite-difference curl proxy for the residual force field.

The outputs are meant to guide architecture choices, not to tune a model:

* Measurement A: do the supervisor factors separate, or are they all altitude shadows?
* Measurement B: does the predictive covariance have a changing physical direction / anisotropy?
* Measurement C: is the measured residual force field meaningfully non-conservative?
* Consistency audit: do existing benchmark report numbers still match the current config/code path?

Everything here targets force-model residuals, never position error.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from vesp.uq.baselines import prepare
from vesp.uq.cli import csv_text, fmt_float
from vesp.uq.experiment import _build_trajectories, _resolve_time_weighting, _time_weights
from vesp.uq.io.run_artifacts import write_run_artifacts
from vesp.uq.metrics import local_radial_frame, safe_log
from vesp.uq.ranking import average_ranks
from vesp.uq.suite import band_label

_REGIONS = ("all", "low", "mid", "high")
_CALIB_KEYS = ("z_std", "picp_90", "ellipsoid_picp_90", "radial_z_std", "tangential_z_std")
_FACTOR_NAMES = ("log_s_rms", "log_w_alt", "log_ood_factor")
_AXIS_NAMES = ("radial", "tangential", "normal")


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
    y = torch.as_tensor(y, dtype=torch.float64).reshape(-1)
    mask = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[mask], y[mask]
    if x.numel() < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.norm(x) * torch.linalg.norm(y)
    if float(denom) <= 0.0:
        return float("nan")
    return float((x * y).sum() / denom)


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    """Average ranks for Spearman correlations; ties receive their midpoint rank (vectorized).

    Uses the shared 1-based helper -- the constant offset vs the previous 0-based ranks is
    irrelevant since these ranks only ever feed the shift-invariant Pearson correlation.
    """

    return average_ranks(x, base=1.0)


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    x = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
    y = torch.as_tensor(y, dtype=torch.float64).reshape(-1)
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    return _pearson(_rankdata(x[mask]), _rankdata(y[mask]))


def _corr_matrix(columns: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    return {
        left: {right: _pearson(columns[left], columns[right]) for right in columns}
        for left in columns
    }


def _mean_std(values: torch.Tensor) -> dict[str, float]:
    v = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    v = v[torch.isfinite(v)]
    if v.numel() == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {
        "mean": float(v.mean()),
        "std": float(v.std(unbiased=False)) if v.numel() > 1 else 0.0,
    }


def _total_sources(config: dict) -> int:
    model = config.get("model", {}) or {}
    if model.get("type") == "multishell":
        counts = model.get("n_sources_per_shell", [])
        if isinstance(counts, int):
            return int(counts) * len(model.get("shell_alphas", []))
        return sum(int(c) for c in counts)
    return int(model.get("n_source", 0))


def _n_shells(config: dict) -> int:
    model = config.get("model", {}) or {}
    if model.get("type") == "multishell":
        return len(model.get("shell_alphas", []) or [])
    return 1


def _absolute_altitude_weight(radius: torch.Tensor, *, h_floor: float, reference_h: float) -> torch.Tensor:
    h = (radius - 1.0).clamp_min(float(h_floor))
    return float(reference_h) / h


def _weighted_mean(x: torch.Tensor, weights: torch.Tensor | None = None) -> float:
    values = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
    if weights is None:
        return float(values.mean())
    w = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
    total = float(w.sum())
    if total <= 0.0:
        return float(values.mean())
    return float((values * (w / total)).sum())


def _flat_sample_indices(total: int, max_points: int | None) -> set[int] | None:
    if max_points is None or int(max_points) <= 0 or total <= int(max_points):
        return None
    step = max(1, math.ceil(total / int(max_points)))
    return set(range(0, total, step))


def _trajectory_frame(traj: torch.Tensor) -> torch.Tensor:
    """Approximate RTN frame for a position-only trajectory: radial, along-track, normal."""

    pos = torch.as_tensor(traj, dtype=torch.float64)
    radial = pos / torch.linalg.norm(pos, dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
    if pos.shape[0] == 1:
        return local_radial_frame(pos)
    prev_pos = torch.roll(pos, shifts=1, dims=0)
    next_pos = torch.roll(pos, shifts=-1, dims=0)
    prev_pos[0] = pos[0]
    next_pos[-1] = pos[-1]
    tangent = next_pos - prev_pos
    tangent = tangent - (tangent * radial).sum(dim=-1, keepdim=True) * radial
    fallback = local_radial_frame(pos)[:, 1, :]
    norm = torch.linalg.norm(tangent, dim=-1, keepdim=True)
    tangent = torch.where(norm > 1.0e-14, tangent / norm.clamp_min(1.0e-30), fallback)
    normal = torch.linalg.cross(radial, tangent)
    normal = normal / torch.linalg.norm(normal, dim=-1, keepdim=True).clamp_min(1.0e-30)
    return torch.stack([radial, tangent, normal], dim=1)


def flatten_trajectories_with_frames(
    trajectories: list[torch.Tensor],
    *,
    max_points: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten trajectories and matching RTN frames, optionally taking a global stride sample."""

    total = sum(int(torch.as_tensor(t).shape[0]) for t in trajectories)
    sample = _flat_sample_indices(total, max_points)
    positions: list[torch.Tensor] = []
    frames: list[torch.Tensor] = []
    offset = 0
    for traj in trajectories:
        t = torch.as_tensor(traj, dtype=torch.float64)
        frame = _trajectory_frame(t)
        if sample is None:
            idx = torch.arange(t.shape[0])
        else:
            local = [i for i in range(t.shape[0]) if offset + i in sample]
            idx = torch.tensor(local, dtype=torch.long)
        if idx.numel() > 0:
            positions.append(t[idx])
            frames.append(frame[idx])
        offset += int(t.shape[0])
    if not positions:
        raise ValueError("trajectory sampling produced no points")
    return torch.cat(positions, dim=0), torch.cat(frames, dim=0)


def parse_benchmark_report(path: str | Path) -> dict:
    """Parse the small subset of existing benchmark reports needed for consistency checks."""

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    dataset = None
    sources = None
    m = re.search(r"dataset:\s*`([^`]+)`", text)
    if m:
        dataset = m.group(1).replace("/", "\\")
    m = re.search(r"sources:\s*(\d+)", text)
    if m:
        sources = int(m.group(1))

    calibration: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or "---" in stripped:
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 10 or cells[0] not in _REGIONS:
            continue
        keys = ("mean_radius", "rmse", "mean_pred_std", "mean_epi_std", "z_std",
                "picp_90", "ellipsoid_picp_90", "mean_d2", "nll")
        calibration[cells[0]] = {
            key: float(value)
            for key, value in zip(keys, cells[1:], strict=False)
            if _as_float(value) is not None
        }
    return {"path": str(path), "dataset": dataset, "sources": sources, "calibration": calibration, "text": text}


def consistency_audit_rows(
    config: dict,
    *,
    report_path: str | Path | None,
    calibration: dict,
    tolerance: float = 0.05,
) -> list[dict[str, Any]]:
    """Compare current config/evaluation results with an existing benchmark Markdown report."""

    band = band_label(config)
    rows: list[dict[str, Any]] = []
    if report_path is None:
        rows.append({"band": band, "check": "report_exists", "status": "missing",
                     "detail": "no report path supplied"})
        return rows
    path = Path(report_path)
    if not path.exists():
        rows.append({"band": band, "check": "report_exists", "status": "missing",
                     "detail": str(path)})
        return rows

    parsed = parse_benchmark_report(path)
    expected_dataset = str(config.get("data", {}).get("path", "")).replace("/", "\\")
    report_dataset = parsed.get("dataset")
    rows.append({
        "band": band,
        "check": "dataset_path",
        "status": "ok" if report_dataset == expected_dataset else "warn",
        "config_value": expected_dataset,
        "report_value": report_dataset,
        "detail": "report dataset matches config" if report_dataset == expected_dataset else "report dataset differs",
    })

    expected_sources = _total_sources(config)
    report_sources = parsed.get("sources")
    rows.append({
        "band": band,
        "check": "source_count",
        "status": "ok" if report_sources == expected_sources else "warn",
        "config_value": expected_sources,
        "report_value": report_sources,
        "detail": "report source count matches config" if report_sources == expected_sources else "report source count differs",
    })

    text_lower = str(parsed.get("text", "")).lower()
    n_shells = _n_shells(config)
    stale_shell_phrase = (
        (n_shells != 3 and ("three shells" in text_lower or "3-shell" in text_lower))
        or (n_shells != 2 and ("two shells" in text_lower or "two-shell" in text_lower))
    )
    rows.append({
        "band": band,
        "check": "shell_phrase",
        "status": "warn" if stale_shell_phrase else "ok",
        "config_value": n_shells,
        "report_value": "contains shell-count phrase" if stale_shell_phrase else "",
        "detail": "report prose may describe a stale shell geometry" if stale_shell_phrase else "no stale shell phrase detected",
    })

    report_cal = parsed.get("calibration", {})
    for region in _REGIONS:
        actual = calibration.get(region)
        expected = report_cal.get(region)
        if not isinstance(actual, dict) or not isinstance(expected, dict):
            continue
        for metric in ("z_std", "picp_90", "ellipsoid_picp_90"):
            a = _as_float(actual.get(metric))
            r = _as_float(expected.get(metric))
            if a is None or r is None:
                continue
            diff = abs(a - r)
            rows.append({
                "band": band,
                "check": f"{region}_{metric}",
                "status": "ok" if diff <= float(tolerance) else "warn",
                "actual_value": a,
                "report_value": r,
                "diff": diff,
                "detail": "current evaluation matches report" if diff <= float(tolerance) else "current evaluation differs from report",
            })
    return rows


def measurement_a_factor_separation(plugin, trajectories, weights=None) -> dict:
    """Measurement A: supervisor factor correlation/separation across trajectories."""

    trajs = [plugin._prep_positions(t).detach().cpu() for t in trajectories]
    lengths = [int(t.shape[0]) for t in trajs]
    all_pos = torch.cat(trajs, dim=0)
    pred = plugin.predict_uncertainty(all_pos)
    radius = pred.radius.detach().cpu().to(torch.float64)
    expected = pred.expected_error.detach().cpu().to(torch.float64)
    risk_cfg = {
        "h_floor": 1.0e-3,
        "reference_h": max(float(plugin.low_altitude_radius) - 1.0, 1.0e-3),
        "domain_weight": float(getattr(plugin, "domain_weight", 1.0)),
    }
    w_alt = _absolute_altitude_weight(
        radius, h_floor=risk_cfg["h_floor"], reference_h=risk_cfg["reference_h"]
    )
    if getattr(plugin, "domain_support", False):
        d_ood = plugin.domain_support_score(all_pos).detach().cpu().to(torch.float64)
    else:
        d_ood = torch.zeros_like(radius)
    ood_factor = 1.0 + risk_cfg["domain_weight"] * d_ood.clamp_min(0.0)

    log_s = safe_log(expected)
    log_w = safe_log(w_alt)
    log_o = safe_log(ood_factor)

    traj_rows: list[dict[str, Any]] = []
    offset = 0
    weight_list = [None] * len(trajs) if weights is None else list(weights)
    for i, (n, w) in enumerate(zip(lengths, weight_list, strict=True)):
        sl = slice(offset, offset + n)
        offset += n
        ww = None if w is None else torch.as_tensor(w, dtype=torch.float64)
        r_i = radius[sl]
        traj_rows.append({
            "trajectory": i,
            "log_s_rms": _weighted_mean(log_s[sl], ww),
            "log_w_alt": _weighted_mean(log_w[sl], ww),
            "log_ood_factor": _weighted_mean(log_o[sl], ww),
            "log_min_altitude": float(safe_log((r_i.min() - 1.0).reshape(1).clamp_min(1.0e-6))[0]),
            "min_radius": float(r_i.min()),
            "mean_radius": _weighted_mean(r_i, ww),
        })

    columns = {name: torch.tensor([r[name] for r in traj_rows], dtype=torch.float64) for name in _FACTOR_NAMES}
    columns["log_min_altitude"] = torch.tensor([r["log_min_altitude"] for r in traj_rows], dtype=torch.float64)
    corr = _corr_matrix({name: columns[name] for name in _FACTOR_NAMES})
    altitude_corr = {
        name: {
            "pearson": _pearson(columns[name], columns["log_min_altitude"]),
            "spearman": _spearman(columns[name], columns["log_min_altitude"]),
        }
        for name in _FACTOR_NAMES
    }
    offdiag = [
        abs(corr[a][b])
        for i, a in enumerate(_FACTOR_NAMES)
        for b in _FACTOR_NAMES[i + 1 :]
        if math.isfinite(corr[a][b])
    ]
    altitude_abs = [
        abs(v["pearson"]) for v in altitude_corr.values() if math.isfinite(v["pearson"])
    ]
    return {
        "n_trajectories": len(traj_rows),
        "n_points": int(radius.numel()),
        "correlation": corr,
        "altitude_correlation": altitude_corr,
        "max_abs_factor_corr": max(offdiag) if offdiag else float("nan"),
        "mean_abs_factor_corr": sum(offdiag) / len(offdiag) if offdiag else float("nan"),
        "max_abs_altitude_corr": max(altitude_abs) if altitude_abs else float("nan"),
        "trajectory_rows": traj_rows,
    }


def measurement_b_covariance_modes(
    plugin,
    trajectories,
    *,
    altitude_bands: dict | None = None,
    max_points: int = 2000,
) -> dict:
    """Measurement B: dominant covariance direction and RTN variance shares."""

    positions, frames = flatten_trajectories_with_frames(list(trajectories), max_points=max_points)
    cov = plugin.predict_covariance_3x3(positions).covariance.detach().cpu().to(torch.float64)
    frame = frames.to(torch.float64)
    cov_rtn = torch.einsum("nij,njk,nlk->nil", frame, cov, frame)
    diag = torch.diagonal(cov_rtn, dim1=-2, dim2=-1).clamp_min(0.0)
    trace = diag.sum(dim=-1).clamp_min(torch.finfo(torch.float64).tiny)
    shares = diag / trace.unsqueeze(-1)
    eigvals, eigvecs = torch.linalg.eigh(0.5 * (cov_rtn + cov_rtn.transpose(-1, -2)))
    dominant = eigvecs[:, :, -1].abs().argmax(dim=-1)
    radius = torch.linalg.norm(positions.to(torch.float64), dim=-1)
    log_h = safe_log((radius - 1.0).clamp_min(1.0e-6))
    lam_min = eigvals[:, 0].clamp_min(torch.finfo(torch.float64).tiny)
    anisotropy_ratio = eigvals[:, -1].clamp_min(0.0) / lam_min

    axis_counts = {
        _AXIS_NAMES[i]: int((dominant == i).sum())
        for i in range(3)
    }
    axis_fractions = {
        axis: count / max(1, int(dominant.numel()))
        for axis, count in axis_counts.items()
    }
    share_summary = {
        _AXIS_NAMES[i]: {
            **_mean_std(shares[:, i]),
            "pearson_with_log_altitude": _pearson(shares[:, i], log_h),
            "spearman_with_log_altitude": _spearman(shares[:, i], log_h),
        }
        for i in range(3)
    }

    band_rows: list[dict[str, Any]] = []
    bands = altitude_bands or {"low": [1.03, 1.15], "mid": [1.15, 1.35], "high": [1.35, 1.60]}
    for name, rng in bands.items():
        if rng is None:
            continue
        lo, hi = float(rng[0]), float(rng[1])
        mask = (radius >= lo) & (radius <= hi)
        if int(mask.sum()) == 0:
            continue
        row = {
            "region": name,
            "n": int(mask.sum()),
            "radius_mean": float(radius[mask].mean()),
            "anisotropy_ratio_mean": float(anisotropy_ratio[mask].mean()),
            "dominant_radial_fraction": float((dominant[mask] == 0).to(torch.float64).mean()),
            "dominant_tangential_fraction": float((dominant[mask] == 1).to(torch.float64).mean()),
            "dominant_normal_fraction": float((dominant[mask] == 2).to(torch.float64).mean()),
        }
        for i, axis in enumerate(_AXIS_NAMES):
            row[f"{axis}_variance_share_mean"] = float(shares[mask, i].mean())
        band_rows.append(row)

    return {
        "n_points": int(positions.shape[0]),
        "axis_counts": axis_counts,
        "axis_fractions": axis_fractions,
        "dominant_axis_max_fraction": max(axis_fractions.values()) if axis_fractions else float("nan"),
        "variance_share_summary": share_summary,
        "anisotropy_ratio": _mean_std(anisotropy_ratio),
        "band_rows": band_rows,
    }


def measurement_c_curl_ratio(
    positions: torch.Tensor,
    field: torch.Tensor,
    *,
    max_points: int = 512,
    k_neighbors: int = 24,
) -> dict:
    """Measurement C: scattered local-linear curl proxy for a residual force field.

    The local derivative is estimated by a least-squares affine fit in each sampled neighborhood.
    ``scaled_curl_ratio`` multiplies the RMS curl by the median neighbor spacing before dividing by
    RMS field magnitude, making the reported number dimensionless.
    """

    pos = torch.as_tensor(positions, dtype=torch.float64).detach().cpu()
    vec = torch.as_tensor(field, dtype=torch.float64).detach().cpu()
    if pos.ndim != 2 or pos.shape[-1] != 3 or vec.shape != pos.shape:
        raise ValueError("positions and field must both have shape (N, 3)")
    n = int(pos.shape[0])
    if n < 5:
        raise ValueError("at least five points are required for the curl diagnostic")
    sample = sorted(_flat_sample_indices(n, max_points) or set(range(n)))
    k = min(max(4, int(k_neighbors)), n - 1)
    dist = torch.cdist(pos[sample], pos)
    neighbor_idx = torch.argsort(dist, dim=1)[:, 1 : k + 1]

    curls: list[torch.Tensor] = []
    spacings: list[torch.Tensor] = []
    for row, center_idx in enumerate(sample):
        idx = neighbor_idx[row]
        dx = pos[idx] - pos[center_idx]
        df = vec[idx] - vec[center_idx]
        sol = torch.linalg.lstsq(dx, df).solution  # sol[coord, component]
        curl = torch.stack([
            sol[1, 2] - sol[2, 1],
            sol[2, 0] - sol[0, 2],
            sol[0, 1] - sol[1, 0],
        ])
        curls.append(curl)
        spacings.append(torch.median(torch.linalg.norm(dx, dim=-1)))

    curl_tensor = torch.stack(curls)
    spacing = torch.stack(spacings)
    curl_norm = torch.linalg.norm(curl_tensor, dim=-1)
    field_norm = torch.linalg.norm(vec[sample], dim=-1)
    rms_curl = float(torch.sqrt(torch.mean(curl_norm**2)))
    rms_field = float(torch.sqrt(torch.mean(field_norm**2)))
    median_spacing = float(torch.median(spacing))
    scaled = rms_curl * median_spacing / max(rms_field, torch.finfo(torch.float64).tiny)
    return {
        "n_points_total": n,
        "n_points_sampled": len(sample),
        "k_neighbors": k,
        "rms_curl": rms_curl,
        "rms_field": rms_field,
        "median_neighbor_spacing": median_spacing,
        "scaled_curl_ratio": float(scaled),
        "curl_norm_median": float(torch.median(curl_norm)),
        "curl_norm_p95": float(torch.quantile(curl_norm, 0.95)),
    }


def _resolve_metadata_sidecar(data_path: str | Path) -> Path:
    return Path(str(data_path) + ".metadata.json")


def _resolve_maybe_relative_path(raw_path: str | Path, *, data_path: Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    for candidate in (Path.cwd() / path, data_path.parent / path):
        if candidate.exists():
            return candidate
    return path


def measurement_c_sh_potential_curl_control(
    data_path: str | Path | None,
    positions: torch.Tensor,
    field: torch.Tensor,
    *,
    max_points: int = 128,
    k_neighbors: int = 24,
    finite_difference_step: float | None = None,
    threshold: float = 1.0e-6,
) -> dict[str, Any]:
    """Structured curl sanity check for SH residual CSVs generated from a scalar potential.

    The scattered local-linear curl proxy can report material curl on sparse samples from a known
    conservative field. For SH-derived residual datasets we can instead go back to the scalar
    potential generator recorded in the sidecar metadata, compute the acceleration on matched
    central-difference stencils, and measure the curl of that structured control. This is a sanity
    gate for architecture decisions; the scattered proxy remains useful as a diagnostic artifact.
    """

    if data_path is None or str(data_path) == "":
        return {"structured_control_status": "unavailable", "structured_control_method": "none"}
    csv_path = Path(data_path)
    metadata_path = _resolve_metadata_sidecar(csv_path)
    if not metadata_path.exists():
        return {
            "structured_control_status": "unavailable",
            "structured_control_method": "sh_potential_central_difference",
            "structured_control_note": f"metadata sidecar not found: {metadata_path}",
        }

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "structured_control_status": "error",
            "structured_control_method": "sh_potential_central_difference",
            "structured_control_error": str(exc),
        }

    required = ("gravity_model_path", "degree_min", "degree_max")
    if not all(key in metadata for key in required):
        return {
            "structured_control_status": "unavailable",
            "structured_control_method": "sh_potential_central_difference",
            "structured_control_note": "metadata does not describe an SH scalar-potential residual",
        }

    try:
        import numpy as np

        from vesp.data.real_gravity import read_pds_sha, residual_acceleration_finite_difference
    except Exception as exc:  # pragma: no cover - dependency/import failure path
        return {
            "structured_control_status": "error",
            "structured_control_method": "sh_potential_central_difference",
            "structured_control_error": str(exc),
        }

    pos = torch.as_tensor(positions, dtype=torch.float64).detach().cpu()
    vec = torch.as_tensor(field, dtype=torch.float64).detach().cpu()
    if pos.ndim != 2 or pos.shape[-1] != 3 or vec.shape != pos.shape or int(pos.shape[0]) < 5:
        return {
            "structured_control_status": "unavailable",
            "structured_control_method": "sh_potential_central_difference",
            "structured_control_note": "positions and field must both have shape (N, 3), N >= 5",
        }

    n = int(pos.shape[0])
    sample = sorted(_flat_sample_indices(n, max_points) or set(range(n)))
    query = pos[sample].numpy()
    fd_step = float(finite_difference_step or metadata.get("finite_difference_step", 1.0e-4))
    degree_min = int(metadata["degree_min"])
    degree_max = int(metadata["degree_max"])
    try:
        model_path = _resolve_maybe_relative_path(metadata["gravity_model_path"], data_path=csv_path)
        model = read_pds_sha(
            model_path,
            max_degree=degree_max,
            name=metadata.get("gravity_model"),
            strict=True,
        )
        jac = np.zeros((query.shape[0], 3, 3), dtype=np.float64)  # [point, d/dcoord, component]
        for axis in range(3):
            plus = query.copy()
            minus = query.copy()
            plus[:, axis] += fd_step
            minus[:, axis] -= fd_step
            acc_plus = residual_acceleration_finite_difference(
                model,
                plus,
                degree_min=degree_min,
                degree_max=degree_max,
                step=fd_step,
            )
            acc_minus = residual_acceleration_finite_difference(
                model,
                minus,
                degree_min=degree_min,
                degree_max=degree_max,
                step=fd_step,
            )
            if metadata.get("acceleration_output") == "physical":
                radius = float(metadata.get("R_body", model.reference_radius_km))
                acc_plus = acc_plus / radius
                acc_minus = acc_minus / radius
            jac[:, axis, :] = (acc_plus - acc_minus) / (2.0 * fd_step)
    except Exception as exc:
        return {
            "structured_control_status": "error",
            "structured_control_method": "sh_potential_central_difference",
            "structured_control_error": str(exc),
        }

    curl = np.stack(
        [
            jac[:, 1, 2] - jac[:, 2, 1],
            jac[:, 2, 0] - jac[:, 0, 2],
            jac[:, 0, 1] - jac[:, 1, 0],
        ],
        axis=1,
    )
    curl_norm = torch.as_tensor(np.linalg.norm(curl, axis=1), dtype=torch.float64)
    field_norm = torch.linalg.norm(vec[sample], dim=-1)
    k = min(max(4, int(k_neighbors)), n - 1)
    dist = torch.cdist(pos[sample], pos)
    neighbor_idx = torch.argsort(dist, dim=1)[:, 1 : k + 1]
    spacing = torch.median(torch.linalg.norm(pos[neighbor_idx] - pos[sample].unsqueeze(1), dim=-1))
    rms_curl = float(torch.sqrt(torch.mean(curl_norm**2)))
    rms_field = float(torch.sqrt(torch.mean(field_norm**2)))
    scaled = rms_curl * float(spacing) / max(rms_field, torch.finfo(torch.float64).tiny)
    status = "pass" if float(scaled) <= float(threshold) else "warn"
    return {
        "structured_control_status": status,
        "structured_control_method": "sh_potential_central_difference",
        "structured_control_n_points": len(sample),
        "structured_control_step": fd_step,
        "structured_control_scaled_curl_ratio": float(scaled),
        "structured_control_threshold": float(threshold),
        "structured_control_rms_curl": rms_curl,
        "structured_control_rms_field": rms_field,
        "structured_control_note": (
            "central-difference curl of the metadata SH scalar-potential generator"
        ),
    }


def _component_z_gap(calibration: dict) -> float:
    gaps: list[float] = []
    for region in _REGIONS:
        metrics = calibration.get(region)
        if not isinstance(metrics, dict):
            continue
        r = _as_float(metrics.get("radial_z_std"))
        t = _as_float(metrics.get("tangential_z_std"))
        if r is not None and t is not None:
            gaps.append(abs(r - t))
    return max(gaps) if gaps else float("nan")


def decision_table_rows(case_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rule-derived decisions from the gate measurements."""

    any_factor_separation = any(
        (_as_float(c["measurement_a"].get("mean_abs_factor_corr")) or 1.0) < 0.80
        for c in case_results
    )
    max_z_gap = max((_component_z_gap(c["calibration"]) for c in case_results), default=float("nan"))
    any_cov_axis_varies = any(
        (_as_float(c["measurement_b"].get("dominant_axis_max_fraction")) or 1.0) < 0.90
        for c in case_results
    )
    max_curl_excess = max(
        (_as_float(c["measurement_c"].get("excess_scaled_curl_ratio")) or 0.0)
        for c in case_results
    )
    max_curl_raw = max(
        (_as_float(c["measurement_c"].get("scaled_curl_ratio")) or 0.0)
        for c in case_results
    )
    max_curl_control = max(
        (_as_float(c["measurement_c"].get("conservative_control_scaled_curl_ratio")) or 0.0)
        for c in case_results
    )
    structured_values = [
        _as_float(c["measurement_c"].get("structured_control_scaled_curl_ratio"))
        for c in case_results
    ]
    max_structured_control = max((v for v in structured_values if v is not None), default=float("nan"))
    all_emulated_sh = all(c.get("residual_field_kind") == "emulated_sh_residual" for c in case_results)
    structured_sh_passes = (
        all_emulated_sh
        and all(c["measurement_c"].get("structured_control_status") == "pass" for c in case_results)
    )
    any_consistency_warn = any(
        row.get("status") == "warn"
        for c in case_results
        for row in c.get("consistency_rows", [])
    )
    if structured_sh_passes:
        helmholtz_rationale = "no - structured SH curl sanity check passed"
        helmholtz_decision = "do not add; keep scattered Measurement C as a diagnostic proxy only"
        helmholtz_uncertainty = (
            "The metadata scalar-potential control is conservative; scattered excess curl on SH "
            "residuals is treated as proxy/sampling error, not evidence for Helmholtz."
        )
    elif all_emulated_sh and max_curl_excess >= 0.10:
        helmholtz_rationale = "no - SH structured curl sanity check unavailable/failed"
        helmholtz_decision = "do not add; repair Measurement C before considering Helmholtz"
        helmholtz_uncertainty = (
            "Known conservative SH-residual cases should not open the Helmholtz gate. Without a "
            "passing structured control, the scattered proxy is not decision-grade."
        )
    else:
        helmholtz_rationale = "no for SH residual; conditional for real NN residual"
        helmholtz_decision = (
            "do not add unless real surrogate residual curl is large"
            if max_curl_excess < 0.10 else "investigate Helmholtz extension"
        )
        helmholtz_uncertainty = (
            "Scattered finite-difference curl is proxy-limited; the conservative-control floor "
            "is subtracted before making this decision."
        )

    rows = [
        {
            "proposal": "report_and_number_consistency",
            "rationale_verified": "yes",
            "gate_result": "warnings present" if any_consistency_warn else "no warnings",
            "decision": "fix reports before citing numbers" if any_consistency_warn else "reports are usable",
            "uncertainty": "This checks current benchmark Markdown only; manuscript tables still need G1 number audit.",
        },
        {
            "proposal": "attribution_level_1_log_factor_split",
            "rationale_verified": "yes",
            "gate_result": (
                "factor separation exists in at least one band" if any_factor_separation
                else "factors are highly correlated"
            ),
            "decision": "implement/report, with low-information warning where degenerate",
            "uncertainty": "Requires masking test before any attribution claim beyond score decomposition.",
        },
        {
            "proposal": "anisotropic_heteroscedastic_noise",
            "rationale_verified": "conditional",
            "gate_result": f"max radial-vs-tangential z_std gap = {fmt_float(max_z_gap)}",
            "decision": "prototype diagonal RTN noise" if (_as_float(max_z_gap) or 0.0) >= 0.20 else "keep isotropic for now",
            "uncertainty": "Before/after z_std and PICP must improve without creating over-confidence.",
        },
        {
            "proposal": "covariance_mode_attribution_level_2",
            "rationale_verified": "conditional",
            "gate_result": "dominant covariance axis varies" if any_cov_axis_varies else "dominant axis mostly fixed",
            "decision": "report physical-mode attribution" if any_cov_axis_varies else "hold; likely degenerate",
            "uncertainty": "RTN frame uses generated trajectory tangent; external trajectory CSVs should use their own time order.",
        },
        {
            "proposal": "helmholtz_non_conservative_extension",
            "rationale_verified": helmholtz_rationale,
            "gate_result": (
                f"structured control = {fmt_float(max_structured_control)}; "
                f"max scattered excess curl = {fmt_float(max_curl_excess)} "
                f"(raw {fmt_float(max_curl_raw)}, conservative-control {fmt_float(max_curl_control)})"
            ),
            "decision": helmholtz_decision,
            "uncertainty": helmholtz_uncertainty,
        },
        {
            "proposal": "source_geometry_auto_selection",
            "rationale_verified": "yes",
            "gate_result": "existing geometry sweep/config evidence already points to a placement lever",
            "decision": "automate selection after report consistency cleanup",
            "uncertainty": "Criterion must reproduce or beat the current manual L90 surface-leaning choice.",
        },
        {
            "proposal": "do_not_add_shap_rich_posterior_source_tracing",
            "rationale_verified": "yes",
            "gate_result": "no gate supports these changes",
            "decision": "do not implement",
            "uncertainty": "Revisit only if a separate measured failure points specifically at these mechanisms.",
        },
    ]
    return rows


def _factor_corr_rows(case: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    band = case["band"]
    a = case["measurement_a"]
    for left, values in a["correlation"].items():
        for right, corr in values.items():
            rows.append({"band": band, "kind": "factor_factor", "left": left, "right": right, "pearson": corr})
    for name, values in a["altitude_correlation"].items():
        rows.append({
            "band": band,
            "kind": "factor_altitude",
            "left": name,
            "right": "log_min_altitude",
            "pearson": values.get("pearson"),
            "spearman": values.get("spearman"),
        })
    return rows


def _covariance_rows(case: dict[str, Any]) -> list[dict[str, Any]]:
    band = case["band"]
    out = []
    b = case["measurement_b"]
    for axis, payload in b["variance_share_summary"].items():
        out.append({
            "band": band,
            "region": "all",
            "axis": axis,
            "variance_share_mean": payload.get("mean"),
            "variance_share_std": payload.get("std"),
            "pearson_with_log_altitude": payload.get("pearson_with_log_altitude"),
            "spearman_with_log_altitude": payload.get("spearman_with_log_altitude"),
            "dominant_axis_fraction": b["axis_fractions"].get(axis),
        })
    for row in b["band_rows"]:
        for axis in _AXIS_NAMES:
            out.append({
                "band": band,
                "region": row["region"],
                "axis": axis,
                "n": row["n"],
                "radius_mean": row["radius_mean"],
                "variance_share_mean": row.get(f"{axis}_variance_share_mean"),
                "dominant_axis_fraction": row.get(f"dominant_{axis}_fraction"),
                "anisotropy_ratio_mean": row.get("anisotropy_ratio_mean"),
            })
    return out


def _curl_rows(case: dict[str, Any]) -> list[dict[str, Any]]:
    row = {"band": case["band"]}
    row.update(case["measurement_c"])
    return [row]


def _calibration_rows(case: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for region in _REGIONS:
        metrics = case["calibration"].get(region)
        if not isinstance(metrics, dict):
            continue
        row = {"band": case["band"], "region": region}
        for key in _CALIB_KEYS:
            row[key] = metrics.get(key)
        rows.append(row)
    return rows


def _summary_md(case_results: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> str:
    lines = [
        "# VESP-UQ Gate Diagnostics",
        "",
        "Diagnostics for the measure-first development gates. These outputs guide architecture and "
        "attribution choices; they are not position-error claims.",
        "",
        "## Cases",
        "",
        "| band | trajectories | factor mean | max factor corr | max altitude corr | cov points | curl excess | structured curl | consistency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in case_results:
        warn_count = sum(1 for r in case["consistency_rows"] if r.get("status") == "warn")
        a = case["measurement_a"]
        b = case["measurement_b"]
        c = case["measurement_c"]
        lines.append(
            f"| {case['band']} | {a['n_trajectories']} | {fmt_float(a['mean_abs_factor_corr'])} | "
            f"{fmt_float(a['max_abs_factor_corr'])} | {fmt_float(a['max_abs_altitude_corr'])} | "
            f"{b['n_points']} | {fmt_float(c.get('excess_scaled_curl_ratio'))} | "
            f"{fmt_float(c.get('structured_control_scaled_curl_ratio'))} | "
            f"{'WARN ' + str(warn_count) if warn_count else 'ok'} |"
        )
    lines += [
        "",
        "## Decision Table",
        "",
        "| proposal | gate result | decision | uncertainty |",
        "| --- | --- | --- | --- |",
    ]
    for row in decisions:
        lines.append(
            f"| {row['proposal']} | {row['gate_result']} | {row['decision']} | {row['uncertainty']} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- `S_rms` is represented by the existing per-point expected force error.",
        "- `W_alt` uses the absolute altitude weight so trajectories share one altitude scale.",
        "- `d_OOD` is the domain-support distance factor; it is not a dynamical-instability metric.",
        "- The scattered curl diagnostic is a local-linear finite-difference proxy. It is reported "
        "as a diagnostic, not as proof of non-conservative physics. For SH-derived residual CSVs "
        "with metadata, the decision gate uses a structured central-difference curl control from "
        "the recorded scalar-potential generator. If the structured control passes but the "
        "scattered proxy reports material excess curl, the excess is treated as proxy/sampling "
        "error rather than evidence for Helmholtz.",
        "",
    ]
    return "\n".join(lines)


def run_gate_diagnostics(
    configs: list[dict],
    *,
    report_paths: Sequence[str | Path | None] | None = None,
    out_dir: str | Path = "outputs/gate_diagnostics",
    n_orbits: int | None = None,
    n_points: int | None = None,
    max_cov_points: int = 2000,
    max_curl_points: int = 512,
    curl_k: int = 24,
    report_tolerance: float = 0.05,
) -> dict:
    """Run all gate diagnostics over the supplied configs and write manifest-backed artifacts."""

    report_paths = list(report_paths or [None] * len(configs))
    if len(report_paths) != len(configs):
        raise ValueError("report_paths must be omitted or have the same length as configs")

    case_results: list[dict[str, Any]] = []
    for config, report_path in zip(configs, report_paths, strict=True):
        cfg = copy.deepcopy(config)
        if n_orbits is not None:
            cfg.setdefault("uq", {}).setdefault("screening", {})["n_orbits"] = int(n_orbits)
        if n_points is not None:
            cfg.setdefault("uq", {}).setdefault("screening", {})["n_points"] = int(n_points)

        plugin, samples, train, held, dtype, seed = prepare(cfg)
        bands = cfg.get("evaluation", {}).get("altitude_bands")
        calibration = plugin.evaluate_calibration(held.positions, held.error, altitude_bands=bands)
        screen_cfg = cfg.get("uq", {}).get("screening", {})
        traj_info = _build_trajectories(screen_cfg, seed=seed, dtype=dtype, config=cfg)
        trajectories = traj_info["trajectories"]
        time_weighting = _resolve_time_weighting(screen_cfg)
        weights = [_time_weights(t) for t in trajectories] if time_weighting == "kepler_r2" else None

        data_path = str(cfg.get("data", {}).get("path") or "")
        residual_curl = measurement_c_curl_ratio(
            samples.positions, samples.error, max_points=max_curl_points, k_neighbors=curl_k
        )
        conservative_pred = plugin.predict_uncertainty(samples.positions).mean_error.detach().cpu()
        conservative_curl = measurement_c_curl_ratio(
            samples.positions, conservative_pred, max_points=max_curl_points, k_neighbors=curl_k
        )
        residual_curl["conservative_control_scaled_curl_ratio"] = conservative_curl["scaled_curl_ratio"]
        residual_curl["excess_scaled_curl_ratio"] = max(
            0.0,
            float(residual_curl["scaled_curl_ratio"])
            - float(conservative_curl["scaled_curl_ratio"]),
        )
        residual_curl["control_note"] = (
            "same scattered curl proxy applied to the equivalent-source posterior mean"
        )
        residual_curl.update(
            measurement_c_sh_potential_curl_control(
                data_path,
                samples.positions,
                samples.error,
                max_points=min(int(max_curl_points), 128),
                k_neighbors=curl_k,
            )
        )

        residual_field_kind = (
            "emulated_sh_residual" if "lunar_grail" in data_path.lower() else "synthetic_or_unknown"
        )

        case = {
            "band": band_label(cfg),
            "config_path": cfg.get("_config_path"),
            "dataset": str(cfg.get("data", {}).get("path") or (samples.metadata or {}).get("mode", "synthetic")),
            "residual_field_kind": residual_field_kind,
            "n_shells": _n_shells(cfg),
            "n_sources_total": _total_sources(cfg),
            "calibration": calibration,
            "measurement_a": measurement_a_factor_separation(plugin, trajectories, weights=weights),
            "measurement_b": measurement_b_covariance_modes(
                plugin, trajectories, altitude_bands=bands, max_points=max_cov_points
            ),
            "measurement_c": residual_curl,
        }
        case["consistency_rows"] = consistency_audit_rows(
            cfg, report_path=report_path, calibration=calibration, tolerance=report_tolerance
        )
        # Avoid storing every per-trajectory factor row in the JSON artifact; the summary CSV has them
        # when needed and this keeps the artifact compact.
        case["measurement_a"] = {
            k: v for k, v in case["measurement_a"].items() if k != "trajectory_rows"
        }
        case_results.append(case)

    decisions = decision_table_rows(case_results)
    factor_rows = [row for case in case_results for row in _factor_corr_rows(case)]
    covariance_rows = [row for case in case_results for row in _covariance_rows(case)]
    curl_rows = [row for case in case_results for row in _curl_rows(case)]
    calibration_rows = [row for case in case_results for row in _calibration_rows(case)]
    consistency_rows = [row for case in case_results for row in case["consistency_rows"]]

    text_files = {
        "gate_diagnostics.md": _summary_md(case_results, decisions),
        "measurement_a_factor_correlations.csv": csv_text(
            factor_rows, ["band", "kind", "left", "right", "pearson", "spearman"]
        ),
        "measurement_b_covariance_modes.csv": csv_text(
            covariance_rows,
            [
                "band", "region", "axis", "n", "radius_mean", "variance_share_mean",
                "variance_share_std", "pearson_with_log_altitude", "spearman_with_log_altitude",
                "dominant_axis_fraction", "anisotropy_ratio_mean",
            ],
        ),
        "measurement_c_curl.csv": csv_text(
            curl_rows,
            [
                "band", "n_points_total", "n_points_sampled", "k_neighbors", "rms_curl",
                "rms_field", "median_neighbor_spacing", "scaled_curl_ratio",
                "conservative_control_scaled_curl_ratio", "excess_scaled_curl_ratio",
                "structured_control_status", "structured_control_method",
                "structured_control_scaled_curl_ratio", "structured_control_threshold",
                "structured_control_n_points", "structured_control_step",
                "curl_norm_median", "curl_norm_p95", "control_note", "structured_control_note",
                "structured_control_error",
            ],
        ),
        "calibration_current.csv": csv_text(
            calibration_rows,
            ["band", "region", "z_std", "picp_90", "ellipsoid_picp_90", "radial_z_std", "tangential_z_std"],
        ),
        "consistency_audit.csv": csv_text(
            consistency_rows,
            ["band", "check", "status", "config_value", "actual_value", "report_value", "diff", "detail"],
        ),
        "decision_table.csv": csv_text(
            decisions,
            ["proposal", "rationale_verified", "gate_result", "decision", "uncertainty"],
        ),
    }
    payload = {
        "cases": case_results,
        "decision_table": decisions,
        "settings": {
            "n_orbits": n_orbits,
            "n_points": n_points,
            "max_cov_points": max_cov_points,
            "max_curl_points": max_curl_points,
            "curl_k": curl_k,
            "report_tolerance": report_tolerance,
        },
    }
    inputs: dict[str, str | Path] = {}
    for i, cfg in enumerate(configs):
        if cfg.get("_config_path"):
            inputs[f"config_{i}"] = cfg["_config_path"]
        data_path = cfg.get("data", {}).get("path")
        if data_path:
            inputs[f"data_{i}"] = data_path
    for i, report in enumerate(report_paths):
        if report is not None:
            inputs[f"report_{i}"] = report
    manifest = write_run_artifacts(
        out_dir,
        tool="run_vespuq_gate_diagnostics",
        config={
            "configs": [cfg.get("_config_path") for cfg in configs],
            "reports": [str(r) if r is not None else None for r in report_paths],
            "settings": payload["settings"],
        },
        json_files={"gate_diagnostics.json": payload},
        text_files=text_files,
        inputs=inputs,
    )
    return {"out_dir": str(out_dir), "cases": case_results, "decision_table": decisions, "manifest": manifest}
