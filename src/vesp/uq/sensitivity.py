"""Source-geometry and regularization sensitivity for VESP-UQ (WP8).

Sweeps the equivalent-source count and the L2 regularization strength and records, for each
variant, how well the fitted layer reconstructs the held-out force error and how well-conditioned
the source posterior is. This locates whether VESP-UQ performance is limited by source placement,
under/over-regularization, or low-altitude conditioning -- without claiming any physical density
recovery (the equivalent sources are a mathematical basis).

Metrics per variant (mean +/- std across seeds):
* relative acceleration RMSE on held-out residuals (RMSE / RMS of the true force error),
* mean predictive std, z-std, PICP90, ellipsoid PICP90 (calibration quality),
* effective source count and top-5% source contribution (source concentration),
* max shell energy fraction (shell-cancellation / collapse proxy),
* fit runtime.

Targets force-model error, never position error.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import torch

from vesp.uq.io.run_artifacts import write_run_artifacts
from vesp.uq.risk_baselines import prepare
from vesp.uq.suite import _csv, _pm, band_label, git_commit_hash, mean_std

_METRICS = (
    "rel_accel_rmse",
    "mean_pred_std",
    "z_std",
    "picp_90",
    "ellipsoid_picp_90",
    "effective_source_count",
    "top_5pct_source_contribution",
    "max_shell_energy_fraction",
    "fit_seconds",
)


def _variant_metrics(cfg: dict, seed: int) -> dict:
    """Fit one config variant and extract calibration + source-health metrics on held-out data."""

    cfg = copy.deepcopy(cfg)
    cfg["seed"] = int(seed)
    t0 = time.perf_counter()
    plugin, _samples, _train, held, _dtype, _ = prepare(cfg)
    fit_seconds = time.perf_counter() - t0

    bands = cfg.get("evaluation", {}).get("altitude_bands")
    calib = plugin.evaluate_calibration(held.positions, held.error, altitude_bands=bands)["all"]
    rms_target = float(torch.sqrt(torch.mean(held.error.to(torch.float64) ** 2)))
    rel_rmse = float(calib["rmse"]) / rms_target if rms_target > 0 else float("nan")

    health = plugin.source_health()
    shell_dist = health.get("shell_energy_distribution") or []
    max_frac = max((float(s.get("energy_fraction", 0.0)) for s in shell_dist), default=float("nan"))

    return {
        "rel_accel_rmse": rel_rmse,
        "mean_pred_std": float(calib.get("mean_pred_std", float("nan"))),
        "z_std": float(calib.get("z_std", float("nan"))),
        "picp_90": float(calib.get("picp_90", float("nan"))),
        "ellipsoid_picp_90": float(calib.get("ellipsoid_picp_90", float("nan")))
        if calib.get("ellipsoid_picp_90") is not None else float("nan"),
        "effective_source_count": float(health.get("effective_source_count", float("nan"))),
        "top_5pct_source_contribution": float(health.get("top_5pct_source_contribution", float("nan"))),
        "max_shell_energy_fraction": max_frac,
        "fit_seconds": fit_seconds,
    }


def _scaled_source_counts(base_counts, target_total: int) -> list[int]:
    base = [int(c) for c in base_counts]
    total = sum(base) or 1
    scaled = [max(1, int(round(c * target_total / total))) for c in base]
    return scaled


def source_geometry_sweep(config: dict, *, seed: int, n_sources_targets) -> list[dict]:
    """Vary total source count (scaling per-shell counts) and record metrics."""

    base_counts = config.get("model", {}).get("n_sources_per_shell")
    if not base_counts:
        raise SystemExit("source geometry sweep requires model.n_sources_per_shell in the config")
    rows = []
    for target in n_sources_targets:
        cfg = copy.deepcopy(config)
        counts = _scaled_source_counts(base_counts, int(target))
        cfg.setdefault("model", {})["n_sources_per_shell"] = counts
        m = _variant_metrics(cfg, seed)
        rows.append({"band": band_label(config), "seed": int(seed),
                     "n_sources_total": int(sum(counts)), **m})
    return rows


def regularization_sweep(config: dict, *, seed: int, lambdas) -> list[dict]:
    """Vary the fixed L2 regularization strength and record metrics."""

    rows = []
    for lam in lambdas:
        cfg = copy.deepcopy(config)
        reg = cfg.setdefault("uq", {}).setdefault("regularization", {})
        reg["method"] = "fixed"
        reg["lambda_l2"] = float(lam)
        m = _variant_metrics(cfg, seed)
        rows.append({"band": band_label(config), "seed": int(seed), "lambda_l2": float(lam), **m})
    return rows


def _aggregate(rows, key: str) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["band"], r[key]), []).append(r)
    out = {}
    for gk, rs in groups.items():
        agg = {m: mean_std([r[m] for r in rs]) for m in _METRICS}
        agg["n_seeds"] = len(rs)
        out[gk] = agg
    return out


def _sweep_csv(agg: dict, key: str) -> str:
    cols = ["band", key, "n_seeds"]
    for m in _METRICS:
        cols += [f"{m}_mean", f"{m}_std"]
    rows = [cols]
    for (band, val), a in sorted(agg.items()):
        row = [band, val, a["n_seeds"]]
        for m in _METRICS:
            row += [a[m]["mean"], a[m]["std"]]
        rows.append(row)
    return _csv(rows)


def _sweep_md(geom_agg: dict, reg_agg: dict) -> str:
    lines = [
        "# VESP-UQ Source-Geometry & Regularization Sensitivity (WP8)",
        "",
        "Held-out force-error reconstruction and source-posterior conditioning as a function of the "
        "equivalent-source count and the L2 regularization strength. Mean +/- std across seeds. The "
        "equivalent sources are a mathematical basis -- these diagnostics say nothing about physical "
        "lunar density.",
        "",
        "## Source-geometry sweep",
        "",
        "| band | n_sources | rel accel RMSE | z_std | PICP90 | eff. sources | max shell energy frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (band, val), a in sorted(geom_agg.items()):
        lines.append(
            f"| {band} | {int(val)} | {_pm(a['rel_accel_rmse'], '.3f')} | {_pm(a['z_std'], '.3f')} | "
            f"{_pm(a['picp_90'], '.3f')} | {_pm(a['effective_source_count'], '.0f')} | "
            f"{_pm(a['max_shell_energy_fraction'], '.3f')} |"
        )
    lines += [
        "",
        "## Regularization sweep",
        "",
        "| band | lambda_l2 | rel accel RMSE | mean pred std | z_std | PICP90 | eff. sources |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (band, val), a in sorted(reg_agg.items()):
        lines.append(
            f"| {band} | {val:g} | {_pm(a['rel_accel_rmse'], '.3f')} | {_pm(a['mean_pred_std'], '.3e')} | "
            f"{_pm(a['z_std'], '.3f')} | {_pm(a['picp_90'], '.3f')} | "
            f"{_pm(a['effective_source_count'], '.0f')} |"
        )
    lines += [
        "",
        "Interpretation: a rel-RMSE that flattens with more sources indicates source placement is not "
        "the limiting factor; a z_std that drifts from 1 across lambda indicates under/over-"
        "regularization of the predictive spread. Cautious, geometry-level reading only.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _plot_regularization(reg_agg: dict, out_path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    bands = sorted({b for (b, _) in reg_agg})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for band in bands:
        items = sorted([(val, a) for (b, val), a in reg_agg.items() if b == band], key=lambda t: t[0])
        xs = [it[0] for it in items]
        rmse = [it[1]["rel_accel_rmse"]["mean"] for it in items]
        zstd = [it[1]["z_std"]["mean"] for it in items]
        axes[0].plot(xs, rmse, marker="o", label=band)
        axes[1].plot(xs, zstd, marker="o", label=band)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("lambda_l2")
    axes[0].set_ylabel("relative accel RMSE")
    axes[0].set_title("Reconstruction vs regularization")
    axes[1].set_xscale("log")
    axes[1].axhline(1.0, color="k", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("lambda_l2")
    axes[1].set_ylabel("z_std (1 = calibrated)")
    axes[1].set_title("Calibration vs regularization")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def run_sensitivity(
    configs, *, seeds=(0, 1, 2), n_sources_targets=None, lambdas=(1.0, 10.0, 30.0, 100.0, 300.0),
    out_dir="outputs/sensitivity/", make_plots: bool = True,
) -> dict:
    """Run the source-geometry + regularization sensitivity over configs x seeds; write artifacts."""

    out_dir = Path(out_dir)
    geom_rows, reg_rows = [], []
    for cfg in configs:
        base_counts = cfg.get("model", {}).get("n_sources_per_shell") or [1]
        base_total = sum(int(c) for c in base_counts)
        targets = n_sources_targets or [max(1, base_total // 2), base_total, base_total * 2]
        for seed in seeds:
            geom_rows += source_geometry_sweep(cfg, seed=seed, n_sources_targets=targets)
            reg_rows += regularization_sweep(cfg, seed=seed, lambdas=lambdas)

    geom_agg = _aggregate(geom_rows, "n_sources_total")
    reg_agg = _aggregate(reg_rows, "lambda_l2")

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_files = {}
    if make_plots and _plot_regularization(reg_agg, out_dir / "regularization_sensitivity.png"):
        artifact_files["regularization_sensitivity.png"] = out_dir / "regularization_sensitivity.png"

    geom_runs_cols = ["band", "seed", "n_sources_total", *_METRICS]
    reg_runs_cols = ["band", "seed", "lambda_l2", *_METRICS]
    write_run_artifacts(
        out_dir,
        tool="run_source_sensitivity",
        config=configs[0],
        json_files={"sensitivity_meta.json": {
            "git_commit": git_commit_hash(), "seeds": list(seeds),
            "lambdas": list(lambdas), "n_sources_targets": n_sources_targets,
        }},
        text_files={
            "source_geometry_sensitivity.csv": _sweep_csv(geom_agg, "n_sources_total"),
            "source_geometry_sensitivity_runs.csv": _csv(
                [geom_runs_cols] + [[r.get(c) for c in geom_runs_cols] for r in geom_rows]),
            "regularization_sensitivity.csv": _sweep_csv(reg_agg, "lambda_l2"),
            "regularization_sensitivity_runs.csv": _csv(
                [reg_runs_cols] + [[r.get(c) for c in reg_runs_cols] for r in reg_rows]),
            "source_geometry_sensitivity.md": _sweep_md(geom_agg, reg_agg),
        },
        artifact_files=artifact_files,
        manifest_name="manifest.json",
    )
    return {"out_dir": str(out_dir), "geom_agg": geom_agg, "reg_agg": reg_agg}
