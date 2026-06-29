"""Source-geometry vs low-altitude calibration sweep for VESP-UQ (E8 / calibration link).

The documented weakest point of the layer is the under-confident low-altitude band (`z_std < 1`,
`PICP90 < 0.90`). Two mechanisms could cause it: (a) the equivalent-source *geometry* under-resolves
the low-altitude force-error field (a placement problem), or (b) the 2-parameter heteroscedastic
noise law cannot match the misfit shape (a noise-model problem). This sweep distinguishes them by
re-placing the sources (shell radii, shell count, per-shell density, weight mode) while holding the
total source count and the noise law fixed, and reading the **per-band** held-out calibration.

If a geometry brings the low band's `z_std` toward 1 and `PICP90` toward 0.90 without wrecking the
reconstruction RMSE, the limitation was placement. If *no* geometry does, the misfit is irreducible
by placement and the noise law is the binding limitation -- a mechanistic conclusion either way.

Geometries are specified by shell radii (alphas) + per-shell source fractions, so the same set
scales to any config's base source count. Targets force-model error, never position error; the
equivalent sources remain a mathematical basis (no density claim).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from vesp.uq.io.run_artifacts import write_run_artifacts
from vesp.uq.risk_baselines import prepare
from vesp.uq.suite import _csv, _fmt, _pm, band_label, git_commit_hash, mean_std


@dataclass(frozen=True)
class GeometrySpec:
    alphas: tuple[float, ...]       # shell radii (fraction of body radius, interior < 1)
    fractions: tuple[float, ...]    # per-shell share of the total source count (sums ~1)
    weight_mode: str = "surface_area"
    note: str = ""


# The total source count is held at the config's base; only placement / density / weights vary.
GEOMETRIES: dict[str, GeometrySpec] = {
    "baseline": GeometrySpec((0.70, 0.86, 0.95), (0.30, 0.40, 0.30), "surface_area",
                             "current 3-shell geometry"),
    "surface_dense": GeometrySpec((0.85, 0.92, 0.97), (0.30, 0.40, 0.30), "surface_area",
                                  "shells pushed toward the surface (closer to low-alt queries)"),
    "deep": GeometrySpec((0.55, 0.72, 0.88), (0.30, 0.40, 0.30), "surface_area",
                         "deeper shells"),
    "five_shell": GeometrySpec((0.70, 0.80, 0.88, 0.94, 0.98), (0.2, 0.2, 0.2, 0.2, 0.2),
                               "surface_area", "finer radial resolution (5 shells)"),
    "two_shell_surface": GeometrySpec((0.85, 0.95), (0.5, 0.5), "surface_area",
                                      "two near-surface shells"),
    "single_surface": GeometrySpec((0.95,), (1.0,), "surface_area", "one near-surface shell"),
    "baseline_uniform_w": GeometrySpec((0.70, 0.86, 0.95), (0.30, 0.40, 0.30), "uniform",
                                       "baseline placement, uniform weights"),
    "wide_span": GeometrySpec((0.50, 0.72, 0.94), (0.30, 0.40, 0.30), "surface_area",
                              "wide radial span"),
}

_BANDS = ("all", "low", "mid", "high")
_CALIB_KEYS = ("z_std", "picp_90", "picp_68", "rmse", "mean_pred_std", "ellipsoid_picp_90")


def _counts_for(spec: GeometrySpec, base_total: int) -> list[int]:
    counts = [max(1, int(round(base_total * f))) for f in spec.fractions]
    return counts


def geometry_calib_run(config: dict, *, seed: int, geometry: str) -> dict:
    """Fit one geometry variant and read per-band held-out calibration + reconstruction RMSE."""

    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry {geometry!r}; choices: {sorted(GEOMETRIES)}")
    spec = GEOMETRIES[geometry]
    cfg = copy.deepcopy(config)
    cfg["seed"] = int(seed)
    base_total = sum(int(c) for c in cfg.get("model", {}).get("n_sources_per_shell", [128]))
    model = cfg.setdefault("model", {})
    model["type"] = "multishell"
    model["shell_alphas"] = list(spec.alphas)
    model["n_sources_per_shell"] = _counts_for(spec, base_total)
    model["weight_mode"] = spec.weight_mode

    t0 = time.perf_counter()
    plugin, _samples, _train, held, _dtype, _ = prepare(cfg)
    fit_seconds = time.perf_counter() - t0

    bands = cfg.get("evaluation", {}).get("altitude_bands")
    calib = plugin.evaluate_calibration(held.positions, held.error, altitude_bands=bands)
    rms_target = float(torch.sqrt(torch.mean(held.error.to(torch.float64) ** 2)))

    health = plugin.source_health()
    out = {
        "band": band_label(cfg), "seed": int(seed), "geometry": geometry,
        "n_sources_total": int(sum(model["n_sources_per_shell"])),
        "n_shells": len(spec.alphas), "weight_mode": spec.weight_mode,
        "fit_seconds": fit_seconds,
        "rel_accel_rmse": float(calib["all"]["rmse"]) / rms_target if rms_target > 0 else float("nan"),
        "low_high_epistemic_std_ratio": calib.get("low_high_epistemic_std_ratio"),
        "effective_source_count": float(health.get("effective_source_count", float("nan"))),
    }
    for region in _BANDS:
        bm = calib.get(region)
        if not isinstance(bm, dict):
            continue
        for k in _CALIB_KEYS:
            v = bm.get(k)
            out[f"{region}_{k}"] = float(v) if v is not None else float("nan")
    return out


def _aggregate(rows) -> dict:
    metric_keys = None
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["band"], r["geometry"]), []).append(r)
    out = {}
    for key, rs in groups.items():
        if metric_keys is None:
            metric_keys = [k for k in rs[0] if k not in ("band", "geometry", "seed", "weight_mode")]
        agg = {m: mean_std([r.get(m) for r in rs]) for m in metric_keys}
        agg["n_seeds"] = len(rs)  # type: ignore[assignment]
        agg["weight_mode"] = rs[0]["weight_mode"]
        out[key] = agg
    return out


def _verdict(agg: dict) -> dict:
    """Per band: which geometry gets the low-altitude z_std closest to 1, vs baseline."""

    verdict = {}
    bands = sorted({b for (b, _) in agg})
    for band in bands:
        cands = []
        for (b, geom), a in agg.items():
            if b != band:
                continue
            z = a.get("low_z_std", {}).get("mean")
            if z is not None:
                cands.append((geom, z, a.get("low_picp_90", {}).get("mean"),
                              a.get("rel_accel_rmse", {}).get("mean")))
        if not cands:
            continue
        base = next((c for c in cands if c[0] == "baseline"), None)
        best = min(cands, key=lambda c: abs(c[1] - 1.0))
        verdict[band] = {
            "baseline_low_z_std": base[1] if base else None,
            "baseline_low_picp_90": base[2] if base else None,
            "best_geometry": best[0],
            "best_low_z_std": best[1],
            "best_low_picp_90": best[2],
            "best_rel_rmse": best[3],
            # geometry "fixes" calibration if it moves low z_std materially closer to 1 than baseline
            "geometry_improves_low_calibration": bool(
                base is not None and abs(best[1] - 1.0) < abs(base[1] - 1.0) - 0.05
            ),
        }
    return verdict


def _summary_csv(agg: dict) -> str:
    cols = ["band", "geometry", "weight_mode", "n_seeds", "n_sources_total_mean", "n_shells_mean",
            "rel_accel_rmse_mean", "all_z_std_mean", "low_z_std_mean", "low_z_std_std",
            "low_picp_90_mean", "mid_z_std_mean", "high_z_std_mean", "effective_source_count_mean"]
    def m(a, k):
        return a.get(k, {}).get("mean")

    rows = [cols]
    for (band, geom), a in sorted(agg.items()):
        rows.append([band, geom, a["weight_mode"], a["n_seeds"], m(a, "n_sources_total"),
                     m(a, "n_shells"), m(a, "rel_accel_rmse"), m(a, "all_z_std"), m(a, "low_z_std"),
                     a.get("low_z_std", {}).get("std"), m(a, "low_picp_90"),
                     m(a, "mid_z_std"), m(a, "high_z_std"), m(a, "effective_source_count")])
    return _csv(rows)


def _summary_md(agg: dict, verdict: dict) -> str:
    lines = [
        "# VESP-UQ Source-Geometry vs Low-Altitude Calibration (E8)",
        "",
        "Does the equivalent-source *placement* drive the under-confident low-altitude band "
        "(`z_std < 1`, `PICP90 < 0.90`), or is the 2-parameter noise law the binding limitation? "
        "Total source count and the noise law are held fixed; only shell radii / count / density / "
        "weights vary. Mean +/- std across seeds. `z_std` ~ 1 and `PICP90` ~ 0.90 are well-calibrated.",
        "",
        "| band | geometry | rel RMSE | low z_std | low PICP90 | mid z_std | high z_std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (band, geom), a in sorted(agg.items()):
        lines.append(
            f"| {band} | {geom} | {_pm(a['rel_accel_rmse'], '.3f')} | {_pm(a['low_z_std'], '.3f')} | "
            f"{_pm(a['low_picp_90'], '.3f')} | {_pm(a['mid_z_std'], '.3f')} | {_pm(a['high_z_std'], '.3f')} |"
        )
    lines += ["", "## Verdict (low-altitude calibration)", ""]
    for band, v in sorted(verdict.items()):
        improves = v["geometry_improves_low_calibration"]
        lines.append(
            f"- **{band}**: baseline low z_std = {_fmt(v['baseline_low_z_std'], '.3f')} "
            f"(PICP90 {_fmt(v['baseline_low_picp_90'], '.3f')}); best geometry "
            f"`{v['best_geometry']}` -> z_std {_fmt(v['best_low_z_std'], '.3f')} "
            f"(PICP90 {_fmt(v['best_low_picp_90'], '.3f')}, rel-RMSE {_fmt(v['best_rel_rmse'], '.3f')}). "
            + ("**Geometry materially improves low-altitude calibration.**" if improves else
               "Geometry does NOT materially fix low-altitude calibration -> the noise law is the "
               "binding limitation (mechanistic conclusion).")
        )
    lines += [
        "",
        "Interpretation: a geometry that pulls low z_std toward 1 implies placement under-resolution "
        "was the cause (fixable). If every geometry leaves z_std far from 1, the 2-parameter "
        "heteroscedastic law cannot match the low-altitude misfit shape -- which is the honest, "
        "publishable explanation for the residual under-confidence (and motivates conformal/angular "
        "refinement as future work). No physical density is inferred from source placement.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _plot(agg: dict, out_path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    bands = sorted({b for (b, _) in agg})
    fig, axes = plt.subplots(1, len(bands), figsize=(8 * max(1, len(bands)), 5.5), squeeze=False)
    for bi, band in enumerate(bands):
        ax = axes[0][bi]
        items = sorted([(g, a) for (b, g), a in agg.items() if b == band], key=lambda t: t[0])
        labels = [it[0] for it in items]
        for region, color in (("low", "#c0504d"), ("mid", "#4f81bd"), ("high", "#9bbb59")):
            vals = [it[1].get(f"{region}_z_std", {}).get("mean") for it in items]
            vals = [v if v is not None else 0.0 for v in vals]
            ax.plot(labels, vals, marker="o", label=f"{region} z_std", color=color)
        ax.axhline(1.0, color="k", linewidth=0.9, linestyle="--", alpha=0.6, label="calibrated (z_std=1)")
        ax.set_title(f"{band}: z_std by geometry")
        ax.set_ylabel("z_std (1 = calibrated)")
        ax.tick_params(axis="x", rotation=40)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def run_geometry_calibration(
    configs, *, seeds=(0, 1, 2, 3, 4), geometries=None, out_dir="outputs/geometry_calibration/",
    make_plots=True,
) -> dict:
    """Run the geometry x calibration sweep over configs x seeds; write tables + figure + manifest."""

    out_dir = Path(out_dir)
    geometries = list(geometries) if geometries else list(GEOMETRIES)
    rows = [geometry_calib_run(cfg, seed=s, geometry=g)
            for cfg in configs for s in seeds for g in geometries]
    agg = _aggregate(rows)
    verdict = _verdict(agg)

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_files = {}
    if make_plots and _plot(agg, out_dir / "geometry_calibration.png"):
        artifact_files["geometry_calibration.png"] = out_dir / "geometry_calibration.png"

    runs_cols = ["band", "seed", "geometry", "weight_mode", "n_sources_total", "n_shells",
                 "rel_accel_rmse", "all_z_std", "low_z_std", "low_picp_90", "mid_z_std", "high_z_std"]
    write_run_artifacts(
        out_dir,
        tool="run_geometry_calibration",
        config=configs[0],
        json_files={"geometry_calibration_meta.json": {
            "git_commit": git_commit_hash(), "seeds": list(seeds), "geometries": geometries,
            "verdict": verdict,
        }},
        text_files={
            "geometry_calibration.csv": _summary_csv(agg),
            "geometry_calibration_runs.csv": _csv([runs_cols] + [[r.get(c) for c in runs_cols] for r in rows]),
            "geometry_calibration.md": _summary_md(agg, verdict),
        },
        artifact_files=artifact_files,
        manifest_name="manifest.json",
    )
    return {"out_dir": str(out_dir), "agg": agg, "verdict": verdict}
