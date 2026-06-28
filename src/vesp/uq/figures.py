"""Publication-figure rendering for the VESP-UQ evidence pack.

The renderer consumes already-produced audit / benchmark artifacts and writes static PNG+PDF
figures. It intentionally does not refit models or rescore trajectories; the IAC pack should be a
claim-mapped presentation layer over reproducible run outputs.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from vesp.common.artifacts import atomic_write_json

FIGURE_STEMS = (
    "reliability_diagram",
    "sigma_vs_altitude",
    "risk_vs_true_error",
    "mc_vs_stm_agreement",
    "l60_l90_band_comparison",
)

_COLORS = {
    "all": "#4C78A8",
    "low": "#F58518",
    "mid": "#54A24B",
    "high": "#B279A2",
    "flagged": "#D62728",
    "accepted": "#4C78A8",
    "stm": "#4C78A8",
    "mc": "#F58518",
    "l60": "#4C78A8",
    "l90": "#F58518",
}


def render_iac_figures(
    *,
    train_run: str | Path = "outputs/vespuq_smoke",
    iac_dir: str | Path = "outputs/iac",
    linear_dir: str | Path = "outputs/linear_propagation",
    benchmarks_dir: str | Path = "benchmarks",
    out_dir: str | Path = "outputs/iac_pack/figures",
) -> dict[str, Any]:
    """Render all evidence-pack figures and return a manifest-like summary."""

    train_run = Path(train_run)
    iac_dir = Path(iac_dir)
    linear_dir = Path(linear_dir)
    benchmarks_dir = Path(benchmarks_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        "calibration_by_band": train_run / "calibration_by_band.csv",
        "trajectory_scores": train_run / "trajectory_scores.csv",
        "force_error_scores": iac_dir / "force_error_scores.csv",
        "covariance_propagation": benchmarks_dir / "covariance_propagation.md",
        "linear_propagation_states": linear_dir / "linear_propagation_states.csv",
        "l60_report": benchmarks_dir / "vespuq_real_lunar_report.md",
        "l90_report": benchmarks_dir / "vespuq_real_lunar_L90_report.md",
    }

    figure_entries = [
        _render_reliability(inputs["calibration_by_band"], out_dir),
        _render_sigma_vs_altitude(inputs["calibration_by_band"], out_dir),
        _render_risk_vs_error(
            preferred_path=inputs["trajectory_scores"],
            fallback_path=inputs["force_error_scores"],
            out_dir=out_dir,
        ),
        _render_mc_vs_stm(
            md_path=inputs["covariance_propagation"],
            fallback_states_path=inputs["linear_propagation_states"],
            out_dir=out_dir,
        ),
        _render_l60_l90_comparison(
            l60_path=inputs["l60_report"],
            l90_path=inputs["l90_report"],
            out_dir=out_dir,
        ),
    ]
    manifest = {
        "figure_schema_version": 1,
        "out_dir": str(out_dir),
        "inputs": {name: str(path) for name, path in inputs.items()},
        "figures": figure_entries,
    }
    atomic_write_json(out_dir / "figures_manifest.json", manifest)
    return manifest


def _render_reliability(path: Path, out_dir: Path) -> dict[str, Any]:
    stem = "reliability_diagram"
    rows = _calibration_rows(path)
    if not rows:
        return _placeholder_figure(stem, out_dir, f"Missing calibration data: {path}")

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    nominal = [0.50, 0.68, 0.90, 0.95]
    for row in rows:
        observed = [_float(row.get(f"picp_{int(level * 100)}")) for level in nominal]
        if any(value is None for value in observed):
            continue
        band = row.get("band", "band")
        ax.plot(
            nominal,
            observed,
            marker="o",
            linewidth=2,
            label=str(band),
            color=_COLORS.get(str(band), None),
        )
    ax.plot([0.45, 1.0], [0.45, 1.0], color="#555555", linestyle="--", linewidth=1, label="ideal")
    ax.set_xlim(0.48, 0.97)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("Nominal predictive coverage")
    ax.set_ylabel("Empirical component coverage")
    ax.set_title("Reliability by altitude band")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    return _save_figure(fig, out_dir, stem, status="ok", source=path)


def _render_sigma_vs_altitude(path: Path, out_dir: Path) -> dict[str, Any]:
    stem = "sigma_vs_altitude"
    rows = [row for row in _calibration_rows(path) if row.get("band") != "all"]
    if not rows:
        return _placeholder_figure(stem, out_dir, f"Missing altitude-band sigma data: {path}")

    rows = sorted(rows, key=lambda row: _float(row.get("mean_radius")) or 0.0)
    radius = [_float(row.get("mean_radius")) for row in rows]
    predictive = [_float(row.get("mean_pred_std")) for row in rows]
    epistemic = [_float(row.get("mean_epistemic_std")) for row in rows]
    labels = [str(row.get("band", "")) for row in rows]
    if any(value is None for value in radius + predictive + epistemic):
        return _placeholder_figure(stem, out_dir, f"Calibration table lacks sigma columns: {path}")

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(radius, predictive, marker="o", linewidth=2, label="predictive sigma", color="#4C78A8")
    ax.plot(radius, epistemic, marker="s", linewidth=2, label="epistemic sigma", color="#F58518")
    for x, y, label in zip(radius, predictive, labels, strict=True):
        ax.annotate(label, (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xlabel("Mean radius (normalized)")
    ax.set_ylabel("Mean acceleration std")
    ax.set_title("Uncertainty grows toward low altitude")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return _save_figure(fig, out_dir, stem, status="ok", source=path)


def _render_risk_vs_error(*, preferred_path: Path, fallback_path: Path, out_dir: Path) -> dict[str, Any]:
    stem = "risk_vs_true_error"
    rows, source = _trajectory_rows(preferred_path)
    if not rows:
        rows, source = _trajectory_rows(fallback_path)
    if not rows:
        return _placeholder_figure(stem, out_dir, f"Missing trajectory risk data: {preferred_path}")

    xs: list[float] = []
    ys: list[float] = []
    flags: list[int] = []
    for row in rows:
        risk = _first_float(row, ("risk_score", "force_risk", "mean_point_risk", "p95_point_risk"))
        err = _first_float(row, ("true_error", "true_force_error", "oracle_error"))
        flagged = _first_float(row, ("flagged_for_rerun", "flagged"))
        if risk is None or err is None:
            continue
        xs.append(risk)
        ys.append(err)
        flags.append(int(flagged or 0))
    if not xs:
        return _placeholder_figure(stem, out_dir, f"Risk table lacks risk/error columns: {source}")

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    accepted_x = [x for x, flag in zip(xs, flags, strict=True) if not flag]
    accepted_y = [y for y, flag in zip(ys, flags, strict=True) if not flag]
    flagged_x = [x for x, flag in zip(xs, flags, strict=True) if flag]
    flagged_y = [y for y, flag in zip(ys, flags, strict=True) if flag]
    ax.scatter(accepted_x, accepted_y, s=30, alpha=0.72, label="accepted", color=_COLORS["accepted"])
    ax.scatter(flagged_x, flagged_y, s=38, alpha=0.82, label="flagged", color=_COLORS["flagged"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("VESP-UQ force-risk score")
    ax.set_ylabel("Supplied true force-error metric")
    ax.set_title("Risk ranking against held-out force error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    return _save_figure(fig, out_dir, stem, status="ok", source=source, points=len(xs))


def _render_mc_vs_stm(*, md_path: Path, fallback_states_path: Path, out_dir: Path) -> dict[str, Any]:
    stem = "mc_vs_stm_agreement"
    records = _parse_mc_stm_table(md_path)
    if records:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        n_values = [rec["n"] for rec in records]
        rel_errors = [rec["rel_error_pct"] for rec in records]
        ax.plot(n_values, rel_errors, marker="o", linewidth=2, color=_COLORS["mc"], label="MC sampling error")
        ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1, label="STM reference")
        ax.set_xscale("log")
        ax.set_xlabel("Monte Carlo samples")
        ax.set_ylabel("Relative error vs STM (%)")
        ax.set_title("Monte Carlo convergence to STM covariance")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
        return _save_figure(fig, out_dir, stem, status="ok", source=md_path, points=len(records))

    rows = _read_csv_rows(fallback_states_path) if fallback_states_path.exists() else []
    if not rows:
        return _placeholder_figure(stem, out_dir, f"Missing MC-vs-STM benchmark data: {md_path}")

    times = [_first_float(row, ("time", "t")) for row in rows]
    sigmas = [_first_float(row, ("position_sigma", "pos_sigma", "trace_position_cov")) for row in rows]
    pairs = [(t, s) for t, s in zip(times, sigmas, strict=True) if t is not None and s is not None]
    if not pairs:
        return _placeholder_figure(stem, out_dir, f"Linear propagation table lacks sigma columns: {fallback_states_path}")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot([p[0] for p in pairs], [p[1] for p in pairs], linewidth=2, color=_COLORS["stm"])
    ax.set_xlabel("Propagation time")
    ax.set_ylabel("Position sigma")
    ax.set_title("STM covariance growth (MC comparison unavailable)")
    ax.grid(True, alpha=0.25)
    return _save_figure(fig, out_dir, stem, status="fallback", source=fallback_states_path, points=len(pairs))


def _render_l60_l90_comparison(*, l60_path: Path, l90_path: Path, out_dir: Path) -> dict[str, Any]:
    stem = "l60_l90_band_comparison"
    l60 = _parse_calibration_table_from_report(l60_path)
    l90 = _parse_calibration_table_from_report(l90_path)
    bands = [band for band in ("low", "mid", "high") if band in l60 and band in l90]
    if not bands:
        return _placeholder_figure(stem, out_dir, f"Missing L60/L90 calibration tables: {l60_path}, {l90_path}")

    l60_z = [l60[band]["z_std"] for band in bands]
    l90_z = [l90[band]["z_std"] for band in bands]
    l60_picp = [l60[band]["picp_90"] for band in bands]
    l90_picp = [l90[band]["picp_90"] for band in bands]
    if any(value is None for value in l60_z + l90_z + l60_picp + l90_picp):
        return _placeholder_figure(stem, out_dir, "L60/L90 report table lacks z_std or picp_90")

    plt = _plt()
    fig, (ax_z, ax_picp) = plt.subplots(1, 2, figsize=(9.0, 4.2))
    x = list(range(len(bands)))
    width = 0.36
    ax_z.bar([v - width / 2 for v in x], l60_z, width=width, label="L60", color=_COLORS["l60"])
    ax_z.bar([v + width / 2 for v in x], l90_z, width=width, label="L90", color=_COLORS["l90"])
    ax_z.axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    ax_z.set_xticks(x, bands)
    ax_z.set_ylim(0.0, 1.2)
    ax_z.set_ylabel("z_std")
    ax_z.set_title("Sharpness")
    ax_z.grid(True, axis="y", alpha=0.25)

    ax_picp.bar([v - width / 2 for v in x], l60_picp, width=width, label="L60", color=_COLORS["l60"])
    ax_picp.bar([v + width / 2 for v in x], l90_picp, width=width, label="L90", color=_COLORS["l90"])
    ax_picp.axhline(0.90, color="#555555", linestyle="--", linewidth=1)
    ax_picp.set_xticks(x, bands)
    ax_picp.set_ylim(0.0, 1.05)
    ax_picp.set_ylabel("PICP90")
    ax_picp.set_title("Coverage")
    ax_picp.grid(True, axis="y", alpha=0.25)
    ax_picp.legend(frameon=False)
    fig.suptitle("Surrogate-band comparison")
    return _save_figure(fig, out_dir, stem, status="ok", source=[l60_path, l90_path])


def _placeholder_figure(stem: str, out_dir: Path, message: str) -> dict[str, Any]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.axis("off")
    ax.text(
        0.5,
        0.55,
        stem.replace("_", " ").title(),
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=9, wrap=True)
    return _save_figure(fig, out_dir, stem, status="missing_data", message=message)


def _save_figure(fig: Any, out_dir: Path, stem: str, **metadata: Any) -> dict[str, Any]:
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    _plt().close(fig)
    return {
        "name": stem,
        "png": str(png),
        "pdf": str(pdf),
        **{key: _jsonable(value) for key, value in metadata.items()},
    }


def _calibration_rows(path: Path) -> list[dict[str, str]]:
    return _read_csv_rows(path) if path.exists() else []


def _trajectory_rows(path: Path) -> tuple[list[dict[str, str]], Path]:
    return (_read_csv_rows(path), path) if path.exists() else ([], path)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _parse_mc_stm_table(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    records: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("| MC, N ="):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        n_match = re.search(r"N\s*=\s*([0-9]+)", cells[0])
        if not n_match:
            continue
        rel_error = _float(cells[2].replace("%", ""))
        if rel_error is None:
            continue
        records.append({"n": float(n_match.group(1)), "rel_error_pct": rel_error})
    return records


def _parse_calibration_table_from_report(path: Path) -> dict[str, dict[str, float | None]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, float | None]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not re.match(r"\|\s*(all|low|mid|high)\s*\|", line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 8:
            continue
        rows[cells[0]] = {
            "mean_radius": _float(cells[1]),
            "rmse": _float(cells[2]),
            "mean_pred_std": _float(cells[3]),
            "mean_epi_std": _float(cells[4]),
            "z_std": _float(cells[5]),
            "picp_90": _float(cells[6]),
        }
    return rows


def _first_float(row: Mapping[str, Any], names: Iterable[str]) -> float | None:
    for name in names:
        value = _float(row.get(name))
        if value is not None:
            return value
    return None


def _float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "figure.dpi": 300,
    })
    return plt
