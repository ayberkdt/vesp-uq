"""Raw-vs-calibrated force-error reliability for VESP-UQ (WP7).

The manuscript states the calibration layer is *intended* to improve coverage; this module makes
that auditable by reporting empirical reliability before and after a split-conformal scale. The
held-out residual set is split into a calibration half (which fits one multiplicative conformal
scale via :func:`vesp.uq.conformal.fit_conformal_scale`) and a disjoint evaluation half on which
the raw and calibrated per-component coverages (PICP) are measured at several nominal levels, per
altitude band, with sharpness (mean predictive std / interval width).

Everything concerns force-model error (``a_reference - a_surrogate``); none of it is a
position-error or orbit-covariance diagnostic, and the conformal scale is a measured post-hoc
correction, not a guarantee.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch

from vesp.uq.conformal import fit_conformal_scale
from vesp.uq.io.run_artifacts import write_run_artifacts
from vesp.uq.risk_baselines import prepare
from vesp.uq.suite import _csv, _fmt, _pm, band_label, git_commit_hash, mean_std

NOMINAL_LEVELS = (0.50, 0.68, 0.80, 0.90, 0.95)
_DEFAULT_BANDS = {"low": [1.03, 1.15], "mid": [1.15, 1.35], "high": [1.35, 1.60]}


def _half_width(level: float) -> float:
    """Standard-normal half-interval width for a central coverage ``level`` (``z`` such that
    ``P(|Z| <= z) = level``)."""

    return float(math.sqrt(2.0) * torch.erfinv(torch.tensor(float(level), dtype=torch.float64)))


def _cal_eval_split(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(int(seed) + 2027)
    perm = torch.randperm(n, generator=g)
    half = n // 2
    return perm[:half], perm[half:]


def _band_reliability(z: torch.Tensor, std_c: torch.Tensor, scale: float) -> dict:
    """Raw and conformally-calibrated PICP + z-std + sharpness for one component subset."""

    out = {"n": int(z.numel()), "z_std_raw": float(z.std()), "scale": float(scale)}
    z_cal = z / max(scale, 1e-30)
    out["z_std_calibrated"] = float(z_cal.std())
    out["mean_pred_std_raw"] = float(std_c.mean())
    out["mean_pred_std_calibrated"] = float(scale * std_c.mean())
    for level in NOMINAL_LEVELS:
        hw = _half_width(level)
        out[f"picp_{int(round(level * 100))}_raw"] = float((z.abs() <= hw).to(torch.float64).mean())
        out[f"picp_{int(round(level * 100))}_calibrated"] = float(
            (z_cal.abs() <= hw).to(torch.float64).mean()
        )
    # interval sharpness at the 90% level (2 * half-width * mean std)
    out["interval_width_90_raw"] = 2.0 * _half_width(0.90) * out["mean_pred_std_raw"]
    out["interval_width_90_calibrated"] = 2.0 * _half_width(0.90) * out["mean_pred_std_calibrated"]
    return out


def reliability_run(config: dict, *, seed: int) -> dict:
    """Per-band raw vs calibrated reliability at one seed (conformal scale fit on a disjoint half)."""

    cfg = copy.deepcopy(config)
    cfg["seed"] = int(seed)
    plugin, _samples, _train, held, _dtype, _ = prepare(cfg)
    bands = cfg.get("evaluation", {}).get("altitude_bands") or _DEFAULT_BANDS

    cov = plugin.predict_covariance_3x3(held.positions)
    mean_err = cov.mean_error.to(torch.float64)
    std3 = cov.std_components.to(torch.float64)
    target = held.error.to(torch.float64)
    residual = target - mean_err
    radius = torch.linalg.norm(held.positions.to(torch.float64), dim=-1)

    n = int(residual.shape[0])
    cal_idx, eval_idx = _cal_eval_split(n, seed)
    # one global component-max conformal scale fit on the calibration half
    calibrator = fit_conformal_scale(std3[cal_idx], residual[cal_idx], alpha=0.10, mode="component_max")
    scale = float(calibrator.scale)

    # evaluation half, flattened to per-component z with aligned radii
    z_all = (residual[eval_idx] / std3[eval_idx].clamp_min(1e-30)).reshape(-1)
    std_all = std3[eval_idx].reshape(-1)
    radius_c = radius[eval_idx].repeat_interleave(3)

    regions = {"all": _band_reliability(z_all, std_all, scale)}
    for name, rng in bands.items():
        if rng is None:
            continue
        lo, hi = float(rng[0]), float(rng[1])
        mask = (radius_c >= lo) & (radius_c <= hi)
        if int(mask.sum()) >= 30:
            regions[name] = _band_reliability(z_all[mask], std_all[mask], scale)
    return {"band": band_label(cfg), "seed": int(seed), "scale": scale, "regions": regions}


def _aggregate(runs) -> dict:
    keys = sorted({(r["band"], region) for r in runs for region in r["regions"]})
    metric_names = None
    out = {}
    for band, region in keys:
        entries = [r["regions"][region] for r in runs if region in r["regions"] and r["band"] == band]
        if not entries:
            continue
        if metric_names is None:
            metric_names = [k for k in entries[0] if k != "n"]
        agg = {m: mean_std([e.get(m) for e in entries]) for m in metric_names}
        agg["n_seeds"] = len(entries)
        out[(band, region)] = agg
    return out


def _reliability_csv(agg: dict) -> str:
    metric_cols = ["scale", "z_std_raw", "z_std_calibrated",
                   "mean_pred_std_raw", "mean_pred_std_calibrated"]
    for level in NOMINAL_LEVELS:
        p = int(round(level * 100))
        metric_cols += [f"picp_{p}_raw", f"picp_{p}_calibrated"]
    cols = ["band", "region", "n_seeds"] + [f"{m}_mean" for m in metric_cols]
    rows = [cols]
    for (band, region), a in sorted(agg.items()):
        rows.append([band, region, a["n_seeds"]] + [a[m]["mean"] for m in metric_cols])
    return _csv(rows)


def _reliability_md(agg: dict) -> str:
    lines = [
        "# VESP-UQ Reliability: Raw vs Calibrated (WP7)",
        "",
        "Per-component held-out force-error coverage before and after a split-conformal scale "
        "(fit on a disjoint calibration half, `component_max` mode). `z_std` ~ 1 and `PICP90` ~ 0.90 "
        "are well-calibrated. Mean +/- std across seeds.",
        "",
        "| band | region | conformal scale | z_std raw->cal | PICP90 raw->cal | PICP68 raw->cal |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for (band, region), a in sorted(agg.items()):
        lines.append(
            f"| {band} | {region} | {_pm(a['scale'], '.3f')} | "
            f"{_fmt(a['z_std_raw']['mean'], '.3f')} -> {_fmt(a['z_std_calibrated']['mean'], '.3f')} | "
            f"{_fmt(a['picp_90_raw']['mean'], '.3f')} -> {_fmt(a['picp_90_calibrated']['mean'], '.3f')} | "
            f"{_fmt(a['picp_68_raw']['mean'], '.3f')} -> {_fmt(a['picp_68_calibrated']['mean'], '.3f')} |"
        )
    lines += [
        "",
        "A conformal scale > 1 means the raw band under-covered and was inflated; bins whose "
        "calibrated PICP90 still misses 0.90 are flagged for the report. Conformal calibration is a "
        "measured post-hoc correction on exchangeable held-out samples, not a universal guarantee.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _plot_reliability(runs, out_dir: Path) -> dict:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}
    bands = sorted({r["band"] for r in runs})
    artifacts = {}
    levels = list(NOMINAL_LEVELS)
    for band in bands:
        band_runs = [r for r in runs if r["band"] == band]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="ideal")
        for kind, color in (("raw", "#c0504d"), ("calibrated", "#4f81bd")):
            ys = []
            for level in levels:
                p = int(round(level * 100))
                vals = [r["regions"]["all"][f"picp_{p}_{kind}"] for r in band_runs if "all" in r["regions"]]
                ys.append(sum(vals) / len(vals) if vals else float("nan"))
            ax.plot(levels, ys, marker="o", color=color, label=kind, linewidth=1.4)
        ax.set_title(f"{band}: reliability (all altitudes)")
        ax.set_xlabel("nominal coverage")
        ax.set_ylabel("empirical PICP")
        ax.set_xlim(0.4, 1.0)
        ax.set_ylim(0.4, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend()
        name = f"fig_reliability_raw_vs_calibrated_{band}.png"
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=130)
        plt.close(fig)
        artifacts[name] = out_dir / name
    return artifacts


def run_reliability(
    configs, *, seeds=(0, 1, 2, 3, 4), out_dir="outputs/calibration_reliability/", make_plots: bool = True,
) -> dict:
    """Run the raw-vs-calibrated reliability study over configs x seeds; write tables + diagrams."""

    out_dir = Path(out_dir)
    runs = [reliability_run(cfg, seed=s) for cfg in configs for s in seeds]
    agg = _aggregate(runs)

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_files = _plot_reliability(runs, out_dir) if make_plots else {}
    write_run_artifacts(
        out_dir,
        tool="run_calibration_reliability",
        config=configs[0],
        json_files={"calibration_reliability_meta.json": {
            "git_commit": git_commit_hash(), "seeds": list(seeds),
            "nominal_levels": list(NOMINAL_LEVELS),
            "conformal_mode": "component_max", "alpha": 0.10,
        }},
        text_files={
            "calibration_reliability.csv": _reliability_csv(agg),
            "calibration_reliability.md": _reliability_md(agg),
        },
        artifact_files=artifact_files,
        manifest_name="manifest.json",
    )
    return {"out_dir": str(out_dir), "agg": agg}
