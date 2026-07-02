"""Prototype diagonal RTN covariance calibration for VESP-UQ.

The production VESP-UQ covariance is physics-driven by the equivalent-source posterior plus a
scalar altitude-aware aleatoric floor. This module implements the deliberately smaller next step
suggested by the gate diagnostics: fit a held-out, diagonal local-frame variance scaling and report
whether it improves radial/tangential calibration without creating over-confidence.

The pointwise calibration data used here contains positions but no velocity, so the local frame is
radial plus two pooled tangential directions rather than a true trajectory RTN frame. That makes the
prototype appropriate for a before/after calibration gate, not for final trajectory-frame claims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vesp.uq.baselines.assembly import prepare
from vesp.uq.cli import csv_text, fmt_float
from vesp.uq.io.run_artifacts import write_run_artifacts
from vesp.uq.metrics import component_calibration_metrics, local_radial_frame, vector_calibration_metrics

_NORMAL_HALF_WIDTH_90 = 1.6448536269514722
DEFAULT_ALTITUDE_BANDS = {"low": [1.03, 1.15], "mid": [1.15, 1.35], "high": [1.35, 1.60]}


@dataclass(frozen=True)
class RTNVarianceScale:
    """One radial + pooled-tangential variance scale for a radius interval."""

    name: str
    lo: float
    hi: float
    radial_scale: float
    tangential_scale: float
    n_points: int
    radial_z_std_fit: float
    tangential_z_std_fit: float


@dataclass(frozen=True)
class RTNVarianceScaleModel:
    """Piecewise radius-binned diagonal local-frame variance scaling."""

    global_scale: RTNVarianceScale
    bands: tuple[RTNVarianceScale, ...] = ()
    allow_shrink: bool = True
    min_scale: float = 0.25
    max_scale: float = 4.0
    target_z_std: float = 1.0

    def scale_for_radius(self, radius: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(radial_scale, tangential_scale)`` per query radius."""

        r = torch.as_tensor(radius, dtype=torch.float64)
        radial = torch.full_like(r, float(self.global_scale.radial_scale))
        tangential = torch.full_like(r, float(self.global_scale.tangential_scale))
        for band in self.bands:
            mask = (r >= float(band.lo)) & (r <= float(band.hi))
            radial = torch.where(mask, torch.as_tensor(band.radial_scale, dtype=r.dtype, device=r.device), radial)
            tangential = torch.where(
                mask, torch.as_tensor(band.tangential_scale, dtype=r.dtype, device=r.device), tangential
            )
        return radial, tangential

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": "position-only radial + pooled tangential",
            "allow_shrink": self.allow_shrink,
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
            "target_z_std": self.target_z_std,
            "global": self.global_scale.__dict__,
            "bands": [band.__dict__ for band in self.bands],
        }


def local_frame_residual_and_variance(
    positions: torch.Tensor,
    residuals: torch.Tensor,
    covariances: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate residual vectors and covariance diagonals into the local radial/tangential frame."""

    if positions.shape != residuals.shape or positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError("positions and residuals must both have shape (N, 3)")
    if covariances.shape != (positions.shape[0], 3, 3):
        raise ValueError("covariances must have shape (N, 3, 3)")
    pos = positions.to(torch.float64)
    res = residuals.to(torch.float64)
    cov = covariances.to(torch.float64)
    frame = local_radial_frame(pos)
    local_res = torch.einsum("nij,nj->ni", frame, res)
    local_cov = torch.einsum("nij,njk,nlk->nil", frame, cov, frame)
    local_var = torch.diagonal(local_cov, dim1=-2, dim2=-1).clamp_min(0.0)
    return local_res, local_var


def _z_std(errors: torch.Tensor, variances: torch.Tensor) -> float:
    std = torch.sqrt(variances.to(torch.float64).clamp_min(torch.finfo(torch.float64).tiny))
    z = errors.to(torch.float64).reshape(-1) / std.reshape(-1)
    z = z[torch.isfinite(z)]
    if z.numel() < 2:
        return float("nan")
    return float(torch.std(z, unbiased=False).detach().cpu())


def _scale_from_z(
    z_std: float,
    *,
    target_z_std: float,
    allow_shrink: bool,
    min_scale: float,
    max_scale: float,
) -> float:
    if not math.isfinite(z_std) or z_std <= 0.0:
        return 1.0
    scale = (z_std / max(float(target_z_std), 1.0e-12)) ** 2
    if not allow_shrink:
        scale = max(1.0, scale)
    return min(max(float(scale), float(min_scale)), float(max_scale))


def _fit_one_scale(
    name: str,
    lo: float,
    hi: float,
    positions: torch.Tensor,
    residuals: torch.Tensor,
    covariances: torch.Tensor,
    *,
    target_z_std: float,
    allow_shrink: bool,
    min_scale: float,
    max_scale: float,
) -> RTNVarianceScale:
    local_res, local_var = local_frame_residual_and_variance(positions, residuals, covariances)
    radial_z = _z_std(local_res[:, 0], local_var[:, 0])
    tangential_z = _z_std(local_res[:, 1:], local_var[:, 1:])
    return RTNVarianceScale(
        name=str(name),
        lo=float(lo),
        hi=float(hi),
        radial_scale=_scale_from_z(
            radial_z,
            target_z_std=target_z_std,
            allow_shrink=allow_shrink,
            min_scale=min_scale,
            max_scale=max_scale,
        ),
        tangential_scale=_scale_from_z(
            tangential_z,
            target_z_std=target_z_std,
            allow_shrink=allow_shrink,
            min_scale=min_scale,
            max_scale=max_scale,
        ),
        n_points=int(positions.shape[0]),
        radial_z_std_fit=radial_z,
        tangential_z_std_fit=tangential_z,
    )


def fit_rtn_variance_scale_model(
    positions: torch.Tensor,
    residuals: torch.Tensor,
    covariances: torch.Tensor,
    *,
    altitude_bands: dict[str, list[float]] | None = None,
    min_band_points: int = 30,
    target_z_std: float = 1.0,
    allow_shrink: bool = True,
    min_scale: float = 0.25,
    max_scale: float = 4.0,
) -> RTNVarianceScaleModel:
    """Fit global and optional radius-binned radial/tangential variance scales."""

    if positions.shape[0] < 2:
        raise ValueError("at least two calibration points are required")
    radius = torch.linalg.norm(positions.to(torch.float64), dim=-1)
    global_scale = _fit_one_scale(
        "all",
        float(radius.min().detach().cpu()),
        float(radius.max().detach().cpu()),
        positions,
        residuals,
        covariances,
        target_z_std=target_z_std,
        allow_shrink=allow_shrink,
        min_scale=min_scale,
        max_scale=max_scale,
    )

    bands: list[RTNVarianceScale] = []
    for name, rng in (altitude_bands or {}).items():
        if rng is None or len(rng) != 2:
            continue
        lo, hi = float(rng[0]), float(rng[1])
        mask = (radius >= lo) & (radius <= hi)
        if int(mask.sum()) < int(min_band_points):
            continue
        bands.append(
            _fit_one_scale(
                name,
                lo,
                hi,
                positions[mask],
                residuals[mask],
                covariances[mask],
                target_z_std=target_z_std,
                allow_shrink=allow_shrink,
                min_scale=min_scale,
                max_scale=max_scale,
            )
        )

    return RTNVarianceScaleModel(
        global_scale=global_scale,
        bands=tuple(bands),
        allow_shrink=bool(allow_shrink),
        min_scale=float(min_scale),
        max_scale=float(max_scale),
        target_z_std=float(target_z_std),
    )


def apply_rtn_variance_scale(
    positions: torch.Tensor,
    covariances: torch.Tensor,
    model: RTNVarianceScaleModel,
) -> torch.Tensor:
    """Apply the fitted local-frame variance scaling and rotate back to world coordinates."""

    if covariances.shape != (positions.shape[0], 3, 3):
        raise ValueError("covariances must have shape (N, 3, 3)")
    pos = positions.to(torch.float64)
    cov = covariances.to(torch.float64)
    frame = local_radial_frame(pos)
    local_cov = torch.einsum("nij,njk,nlk->nil", frame, cov, frame)
    radial_scale, tangential_scale = model.scale_for_radius(torch.linalg.norm(pos, dim=-1))
    scales = torch.stack([radial_scale, tangential_scale, tangential_scale], dim=-1)
    scaled_local = local_cov * torch.sqrt(scales).unsqueeze(-1) * torch.sqrt(scales).unsqueeze(-2)
    world = torch.einsum("nai,nab,nbj->nij", frame, scaled_local, frame)
    return 0.5 * (world + world.transpose(-1, -2))


def rtn_calibration_summary(
    positions: torch.Tensor,
    residuals: torch.Tensor,
    covariances: torch.Tensor,
) -> dict[str, float]:
    """Compact radial/tangential plus vector calibration summary."""

    comp = component_calibration_metrics(residuals, covariances, positions)
    vec = vector_calibration_metrics(residuals, covariances)
    local_res, local_var = local_frame_residual_and_variance(positions, residuals, covariances)
    radial_picp = float(
        torch.mean(
            (local_res[:, 0].abs() <= _NORMAL_HALF_WIDTH_90 * torch.sqrt(local_var[:, 0].clamp_min(0.0))).to(
                torch.float64
            )
        ).detach().cpu()
    )
    tangential_picp = float(
        torch.mean(
            (
                local_res[:, 1:].abs()
                <= _NORMAL_HALF_WIDTH_90 * torch.sqrt(local_var[:, 1:].clamp_min(0.0))
            ).to(torch.float64)
        ).detach().cpu()
    )
    return {
        "n": int(positions.shape[0]),
        "radial_z_std": float(comp.get("radial_z_std", float("nan"))),
        "tangential_z_std": float(comp.get("tangential_z_std", float("nan"))),
        "radial_picp_90": radial_picp,
        "tangential_picp_90": tangential_picp,
        "ellipsoid_picp_90": float(vec.get("ellipsoid_picp_90", float("nan"))),
        "mean_mahalanobis_d2": float(vec.get("mean_mahalanobis_d2", float("nan"))),
    }


def calibration_error_score(summary: dict[str, float]) -> float:
    """Single scalar used only to compare before/after prototype calibration."""

    terms = [
        abs(float(summary["radial_z_std"]) - 1.0),
        abs(float(summary["tangential_z_std"]) - 1.0),
        abs(float(summary["ellipsoid_picp_90"]) - 0.90),
    ]
    return float(sum(terms) / len(terms))


def creates_overconfidence(
    summary: dict[str, float],
    *,
    max_z_std: float = 1.10,
    min_picp_90: float = 0.88,
) -> bool:
    """Conservative guardrail for deciding whether the prototype is admissible."""

    return (
        float(summary["radial_z_std"]) > float(max_z_std)
        or float(summary["tangential_z_std"]) > float(max_z_std)
        or float(summary["ellipsoid_picp_90"]) < float(min_picp_90)
        or float(summary["radial_picp_90"]) < float(min_picp_90)
        or float(summary["tangential_picp_90"]) < float(min_picp_90)
    )


def _select_points(n: int, *, max_points: int | None, seed: int) -> torch.Tensor:
    idx = torch.arange(n)
    if max_points is None or int(max_points) <= 0 or n <= int(max_points):
        return idx
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(int(seed)))
    return perm[: int(max_points)].sort().values


def _band_label(config: dict) -> str:
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


def _fit_eval_indices(n: int, *, fit_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if n < 4:
        raise ValueError("at least four held-out points are required for RTN prototype split")
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(int(seed)))
    n_fit = min(max(2, int(round(float(fit_fraction) * n))), n - 2)
    return perm[:n_fit], perm[n_fit:]


def _region_masks(radius: torch.Tensor, bands: dict[str, list[float]]) -> dict[str, torch.Tensor]:
    masks = {"all": torch.ones_like(radius, dtype=torch.bool)}
    for name, rng in bands.items():
        if rng is None or len(rng) != 2:
            continue
        lo, hi = float(rng[0]), float(rng[1])
        masks[str(name)] = (radius >= lo) & (radius <= hi)
    return masks


def _scale_rows(case_band: str, model: RTNVarianceScaleModel) -> list[dict[str, Any]]:
    rows = []
    for scale in (model.global_scale, *model.bands):
        rows.append(
            {
                "band": case_band,
                "region": scale.name,
                "lo": scale.lo,
                "hi": scale.hi,
                "n_fit_points": scale.n_points,
                "radial_z_std_fit": scale.radial_z_std_fit,
                "tangential_z_std_fit": scale.tangential_z_std_fit,
                "radial_variance_scale": scale.radial_scale,
                "tangential_variance_scale": scale.tangential_scale,
            }
        )
    return rows


def _summary_row(case_band: str, region: str, before: dict, after: dict) -> dict[str, Any]:
    before_error = calibration_error_score(before)
    after_error = calibration_error_score(after)
    overconf = creates_overconfidence(after)
    improved = after_error < before_error
    return {
        "band": case_band,
        "region": region,
        "n_eval_points": before["n"],
        "before_error_score": before_error,
        "after_error_score": after_error,
        "error_score_delta": after_error - before_error,
        "overconfidence": overconf,
        "decision": "candidate" if improved and not overconf else "hold",
        "before_radial_z_std": before["radial_z_std"],
        "after_radial_z_std": after["radial_z_std"],
        "before_tangential_z_std": before["tangential_z_std"],
        "after_tangential_z_std": after["tangential_z_std"],
        "before_radial_picp_90": before["radial_picp_90"],
        "after_radial_picp_90": after["radial_picp_90"],
        "before_tangential_picp_90": before["tangential_picp_90"],
        "after_tangential_picp_90": after["tangential_picp_90"],
        "before_ellipsoid_picp_90": before["ellipsoid_picp_90"],
        "after_ellipsoid_picp_90": after["ellipsoid_picp_90"],
        "before_mean_mahalanobis_d2": before["mean_mahalanobis_d2"],
        "after_mean_mahalanobis_d2": after["mean_mahalanobis_d2"],
    }


def _summary_md(cases: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# RTN Noise Prototype",
        "",
        "Held-out before/after check for a diagonal local-frame variance scaling. The frame is "
        "position-only radial plus pooled tangential directions, because the calibration samples do "
        "not carry velocity/along-track metadata.",
        "",
        "| band | fit/eval | before error | after error | radial z | tangential z | "
        "ellipsoid PICP90 | case decision |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    by_case = {(row["band"], row["region"]): row for row in rows}
    for case in cases:
        row = by_case.get((case["band"], "all"))
        if row is None:
            continue
        lines.append(
            "| {band} | {fit}/{eval} | {before} | {after} | {rz0}->{rz1} | "
            "{tz0}->{tz1} | {p0}->{p1} | {decision} |".format(
                band=case["band"],
                fit=case["n_fit"],
                eval=case["n_eval"],
                before=fmt_float(row["before_error_score"]),
                after=fmt_float(row["after_error_score"]),
                rz0=fmt_float(row["before_radial_z_std"]),
                rz1=fmt_float(row["after_radial_z_std"]),
                tz0=fmt_float(row["before_tangential_z_std"]),
                tz1=fmt_float(row["after_tangential_z_std"]),
                p0=fmt_float(row["before_ellipsoid_picp_90"], ".3f"),
                p1=fmt_float(row["after_ellipsoid_picp_90"], ".3f"),
                decision=case.get("case_decision", row["decision"]),
            )
        )
        if case.get("regional_holds"):
            lines.append(
                f"| {case['band']} regional holds |  |  |  |  |  |  | "
                f"{', '.join(case['regional_holds'])} |"
            )
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "A candidate requires a lower aggregate calibration error and no over-confidence guardrail "
            "violation (`z_std <= 1.10`, component/ellipsoid PICP90 >= 0.88).",
        ]
    )
    return "\n".join(lines) + "\n"


def run_rtn_noise_prototype(
    configs: list[dict],
    *,
    out_dir: str | Path = "outputs/rtn_noise_prototype",
    fit_fraction: float = 0.5,
    max_points: int | None = None,
    min_band_points: int = 30,
    allow_shrink: bool = True,
    min_scale: float = 0.25,
    max_scale: float = 4.0,
    max_z_std: float = 1.10,
    min_picp_90: float = 0.88,
) -> dict:
    """Run the RTN-style prototype over configs and write manifest-backed artifacts."""

    rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []

    for config in configs:
        plugin, _samples, _train, held, _dtype, seed = prepare(config)
        case_band = _band_label(config)
        idx = _select_points(held.n, max_points=max_points, seed=seed + 1009)
        pos = held.positions[idx]
        err = held.error[idx]
        pred = plugin.predict_covariance_3x3(pos)
        residual = err - pred.mean_error.detach().cpu()
        cov = pred.covariance.detach().cpu()

        fit_idx, eval_idx = _fit_eval_indices(pos.shape[0], fit_fraction=fit_fraction, seed=seed + 2003)
        bands = config.get("evaluation", {}).get("altitude_bands") or DEFAULT_ALTITUDE_BANDS
        model = fit_rtn_variance_scale_model(
            pos[fit_idx],
            residual[fit_idx],
            cov[fit_idx],
            altitude_bands=bands,
            min_band_points=min_band_points,
            allow_shrink=allow_shrink,
            min_scale=min_scale,
            max_scale=max_scale,
        )
        adjusted = apply_rtn_variance_scale(pos[eval_idx], cov[eval_idx], model)
        radius_eval = torch.linalg.norm(pos[eval_idx].to(torch.float64), dim=-1)
        masks = _region_masks(radius_eval, bands)
        case_rows = []
        for region, mask in masks.items():
            if int(mask.sum()) < 10:
                continue
            before = rtn_calibration_summary(pos[eval_idx][mask], residual[eval_idx][mask], cov[eval_idx][mask])
            after = rtn_calibration_summary(pos[eval_idx][mask], residual[eval_idx][mask], adjusted[mask])
            row = _summary_row(case_band, region, before, after)
            row["overconfidence"] = creates_overconfidence(after, max_z_std=max_z_std, min_picp_90=min_picp_90)
            row["decision"] = (
                "candidate"
                if row["after_error_score"] < row["before_error_score"] and not row["overconfidence"]
                else "hold"
            )
            rows.append(row)
            case_rows.append(row)
        scale_rows.extend(_scale_rows(case_band, model))
        all_row = next((row for row in case_rows if row["region"] == "all"), None)
        regional_holds = [
            row["region"] for row in case_rows if row["region"] != "all" and row["decision"] != "candidate"
        ]
        if all_row is None or all_row["decision"] != "candidate":
            case_decision = "hold"
        elif regional_holds:
            case_decision = "partial"
        else:
            case_decision = "candidate"
        cases.append(
            {
                "band": case_band,
                "config_path": config.get("_config_path"),
                "n_heldout_used": int(pos.shape[0]),
                "n_fit": int(fit_idx.numel()),
                "n_eval": int(eval_idx.numel()),
                "model": model.to_dict(),
                "overall": all_row,
                "case_decision": case_decision,
                "regional_holds": regional_holds,
            }
        )

    settings = {
        "fit_fraction": fit_fraction,
        "max_points": max_points,
        "min_band_points": min_band_points,
        "allow_shrink": allow_shrink,
        "min_scale": min_scale,
        "max_scale": max_scale,
        "max_z_std": max_z_std,
        "min_picp_90": min_picp_90,
    }
    text_files = {
        "rtn_noise_prototype.md": _summary_md(cases, rows),
        "rtn_noise_summary.csv": csv_text(
            rows,
            [
                "band",
                "region",
                "n_eval_points",
                "before_error_score",
                "after_error_score",
                "error_score_delta",
                "overconfidence",
                "decision",
                "before_radial_z_std",
                "after_radial_z_std",
                "before_tangential_z_std",
                "after_tangential_z_std",
                "before_radial_picp_90",
                "after_radial_picp_90",
                "before_tangential_picp_90",
                "after_tangential_picp_90",
                "before_ellipsoid_picp_90",
                "after_ellipsoid_picp_90",
                "before_mean_mahalanobis_d2",
                "after_mean_mahalanobis_d2",
            ],
        ),
        "rtn_noise_scales.csv": csv_text(
            scale_rows,
            [
                "band",
                "region",
                "lo",
                "hi",
                "n_fit_points",
                "radial_z_std_fit",
                "tangential_z_std_fit",
                "radial_variance_scale",
                "tangential_variance_scale",
            ],
        ),
    }
    payload = {"cases": cases, "summary_rows": rows, "scale_rows": scale_rows, "settings": settings}
    manifest = write_run_artifacts(
        out_dir,
        tool="run_vespuq_rtn_noise_prototype",
        config={"configs": configs, "settings": settings},
        json_files={"rtn_noise_prototype.json": payload},
        text_files=text_files,
        seed=[cfg.get("seed") for cfg in configs],
        config_path=",".join(str(cfg.get("_config_path", "")) for cfg in configs),
    )
    return {"out_dir": str(out_dir), "cases": cases, "rows": rows, "scale_rows": scale_rows, "manifest": manifest}
