"""Run the diagonal RTN-style covariance calibration prototype.

This is a gate artifact, not a production model toggle. It fits a held-out radial + pooled
tangential variance scaling on one half of the existing held-out split, evaluates before/after
calibration on the other half, and writes manifest-backed artifacts.
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
from typing import Any

import torch

from vesp.common.config import load_config
from vesp.uq.io.run_artifacts import write_run_artifacts
from vesp.uq.risk_baselines import prepare
from vesp.uq.rtn_noise import (
    apply_rtn_variance_scale,
    calibration_error_score,
    creates_overconfidence,
    fit_rtn_variance_scale_model,
    rtn_calibration_summary,
)
from vesp.uq.suite import band_label

_DEFAULT_BANDS = {"low": [1.03, 1.15], "mid": [1.15, 1.35], "high": [1.35, 1.60]}


def _csv_text(rows: list[dict[str, Any]], columns: list[str]) -> str:
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in columns})
    return handle.getvalue()


def _fmt(value: Any, spec: str = ".3g") -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return format(v, spec) if torch.isfinite(torch.tensor(v)) else "n/a"


def _load_configs(paths: list[str]) -> list[dict]:
    configs = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"config not found: {path}")
        cfg = load_config(str(p))
        cfg.setdefault("_config_path", str(p))
        configs.append(cfg)
    return configs


def _select_points(n: int, *, max_points: int | None, seed: int) -> torch.Tensor:
    idx = torch.arange(n)
    if max_points is None or int(max_points) <= 0 or n <= int(max_points):
        return idx
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(int(seed)))
    return perm[: int(max_points)].sort().values


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


def _scale_rows(case_band: str, model) -> list[dict[str, Any]]:
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
    decision = "candidate" if improved and not overconf else "hold"
    return {
        "band": case_band,
        "region": region,
        "n_eval_points": before["n"],
        "before_error_score": before_error,
        "after_error_score": after_error,
        "error_score_delta": after_error - before_error,
        "overconfidence": overconf,
        "decision": decision,
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
    """Run the prototype over configs and write artifacts."""

    rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []

    for config in configs:
        plugin, _samples, _train, held, _dtype, seed = prepare(config)
        case_band = band_label(config)
        idx = _select_points(held.n, max_points=max_points, seed=seed + 1009)
        pos = held.positions[idx]
        err = held.error[idx]
        pred = plugin.predict_covariance_3x3(pos)
        residual = err - pred.mean_error.detach().cpu()
        cov = pred.covariance.detach().cpu()

        fit_idx, eval_idx = _fit_eval_indices(pos.shape[0], fit_fraction=fit_fraction, seed=seed + 2003)
        bands = config.get("evaluation", {}).get("altitude_bands") or _DEFAULT_BANDS
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
            row["overconfidence"] = creates_overconfidence(
                after, max_z_std=max_z_std, min_picp_90=min_picp_90
            )
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
            row["region"]
            for row in case_rows
            if row["region"] != "all" and row["decision"] != "candidate"
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

    md = _summary_md(cases, rows)
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
        "rtn_noise_prototype.md": md,
        "rtn_noise_summary.csv": _csv_text(
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
        "rtn_noise_scales.csv": _csv_text(
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


def _summary_md(cases: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# RTN Noise Prototype",
        "",
        "Held-out before/after check for a diagonal local-frame variance scaling. The frame is "
        "position-only radial plus pooled tangential directions, because the calibration samples do "
        "not carry velocity/along-track metadata.",
        "",
        "| band | fit/eval | before error | after error | radial z | tangential z | ellipsoid PICP90 | case decision |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    by_case = {(row["band"], row["region"]): row for row in rows}
    for case in cases:
        row = by_case.get((case["band"], "all"))
        if row is None:
            continue
        lines.append(
            "| {band} | {fit}/{eval} | {before} | {after} | {rz0}->{rz1} | {tz0}->{tz1} | "
            "{p0}->{p1} | {decision} |".format(
                band=case["band"],
                fit=case["n_fit"],
                eval=case["n_eval"],
                before=_fmt(row["before_error_score"]),
                after=_fmt(row["after_error_score"]),
                rz0=_fmt(row["before_radial_z_std"]),
                rz1=_fmt(row["after_radial_z_std"]),
                tz0=_fmt(row["before_tangential_z_std"]),
                tz1=_fmt(row["after_tangential_z_std"]),
                p0=_fmt(row["before_ellipsoid_picp_90"], ".3f"),
                p1=_fmt(row["after_ellipsoid_picp_90"], ".3f"),
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="VESP-UQ diagonal RTN-style noise prototype.")
    parser.add_argument("--configs", nargs="+", required=True, help="one or more VESP-UQ YAML configs")
    parser.add_argument("--out", default="outputs/rtn_noise_prototype")
    parser.add_argument("--fit-fraction", type=float, default=0.5)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--min-band-points", type=int, default=30)
    parser.add_argument("--no-shrink", action="store_true", help="only widen local-frame variance")
    parser.add_argument("--min-scale", type=float, default=0.25)
    parser.add_argument("--max-scale", type=float, default=4.0)
    parser.add_argument("--max-z-std", type=float, default=1.10)
    parser.add_argument("--min-picp-90", type=float, default=0.88)
    parser.add_argument("--quick", action="store_true", help="cap held-out points for a fast wiring pass")
    args = parser.parse_args(argv)

    configs = _load_configs(args.configs)
    max_points = args.max_points
    if args.quick:
        max_points = 600 if max_points is None else min(max_points, 600)
    result = run_rtn_noise_prototype(
        configs,
        out_dir=args.out,
        fit_fraction=args.fit_fraction,
        max_points=max_points,
        min_band_points=args.min_band_points,
        allow_shrink=not args.no_shrink,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        max_z_std=args.max_z_std,
        min_picp_90=args.min_picp_90,
    )
    print(f"saved_rtn_noise_prototype: {Path(result['out_dir']) / 'rtn_noise_prototype.md'}")
    for case in result["cases"]:
        overall = case.get("overall") or {}
        print(
            f"{case['band']}: before_error={_fmt(overall.get('before_error_score'))} "
            f"after_error={_fmt(overall.get('after_error_score'))} "
            f"decision={case.get('case_decision', overall.get('decision', 'n/a'))}"
        )


if __name__ == "__main__":
    main()
