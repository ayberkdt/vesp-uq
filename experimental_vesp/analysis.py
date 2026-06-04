"""Experiment analysis and report generation for discrete VESP runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


@dataclass(frozen=True)
class AnalysisThresholds:
    """Heuristic thresholds for normalized synthetic MVP runs."""

    acceleration_good: float = 1.0e-3
    acceleration_watch: float = 1.0e-2
    potential_good: float = 1.0e-4
    potential_watch: float = 1.0e-3
    low_high_ratio_watch: float = 10.0
    shell_dominance_watch: float = 0.85
    sigma_abs_max_watch: float = 50.0
    monopole_watch: float = 1.0e-3


def _torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint_summary(path: str | Path) -> dict:
    """Load the metadata and stored metrics from a VESP checkpoint."""

    ckpt_path = Path(path)
    ckpt = _torch_load(ckpt_path)
    metrics = ckpt.get("metrics", {})
    config = ckpt.get("config", {})
    return {
        "path": ckpt_path,
        "name": ckpt_path.stem,
        "metrics": metrics,
        "config": config,
        "shell_radii": tuple(float(v) for v in ckpt.get("shell_radii", ())),
        "n_sources": int(ckpt["sigma"].numel()) if "sigma" in ckpt else None,
    }


def _metric(metrics: dict, key: str, default: float | None = None) -> float | None:
    value = metrics.get(key, default)
    if value is None:
        return None
    return float(value)


def _status(value: float | None, good: float, watch: float) -> str:
    if value is None:
        return "missing"
    if value <= good:
        return "good"
    if value <= watch:
        return "watch"
    return "risk"


def low_high_altitude_ratio(metrics: dict) -> float | None:
    bins = metrics.get("altitude_binned_error") or []
    if len(bins) < 2:
        return None
    low = float(bins[0]["acceleration_rmse"])
    high = float(bins[-1]["acceleration_rmse"])
    if high <= 0.0:
        return None
    return low / high


def shell_energy_fractions(metrics: dict) -> list[float]:
    diagnostics = metrics.get("diagnostics") or {}
    energies = [float(v) for v in diagnostics.get("shell_energy", [])]
    total = sum(max(v, 0.0) for v in energies)
    if total <= 0.0:
        return []
    return [max(v, 0.0) / total for v in energies]


def interpret_experiment(summary: dict, thresholds: AnalysisThresholds | None = None) -> dict:
    """Return scalar summary and human-readable findings for one experiment."""

    thresholds = thresholds or AnalysisThresholds()
    metrics = summary["metrics"]
    diagnostics = metrics.get("diagnostics") or {}

    acc_rmse = _metric(metrics, "acceleration_rmse")
    pot_rmse = _metric(metrics, "potential_rmse")
    radial_rmse = _metric(metrics, "radial_acceleration_rmse")
    cross_rmse = _metric(metrics, "cross_radial_acceleration_rmse")
    sigma_abs_max = float(diagnostics.get("sigma_abs_max", 0.0))
    monopole = float(diagnostics.get("monopole_leakage", 0.0))
    dipole = float(diagnostics.get("dipole_leakage", 0.0))
    inference_seconds = _metric(metrics, "inference_seconds_per_batch")
    altitude_ratio = low_high_altitude_ratio(metrics)
    energy_fractions = shell_energy_fractions(metrics)

    findings: list[str] = []
    risks: list[str] = []
    next_steps: list[str] = []

    acc_status = _status(acc_rmse, thresholds.acceleration_good, thresholds.acceleration_watch)
    pot_status = _status(pot_rmse, thresholds.potential_good, thresholds.potential_watch)

    if acc_status == "good":
        findings.append("Acceleration fit is in the good range for a normalized MVP smoke run.")
    elif acc_status == "watch":
        findings.append("Acceleration fit is usable for ablation, but not yet a strong baseline.")
        next_steps.append("Tune shell radii, source count, and regularization before adding MaxEnt.")
    else:
        risks.append("Acceleration RMSE is high; this run should be treated as a geometry or regularization diagnostic.")
        next_steps.append("Try fewer/more sources, stronger L2, and a shell radius closer to the hidden residual scale.")

    if pot_status == "good":
        findings.append("Potential fit is also tight, so the fitted field is not only matching acceleration locally.")
    elif pot_status == "watch":
        findings.append("Potential fit is moderate; combined potential+acceleration weighting may need tuning.")
    else:
        risks.append("Potential RMSE is high; acceleration-only behavior may be overfitting local gradients.")

    if altitude_ratio is not None:
        if altitude_ratio > thresholds.low_high_ratio_watch:
            risks.append(
                f"Low-altitude error is {altitude_ratio:.1f}x the high-altitude error; near-surface stability needs attention."
            )
            next_steps.append("Run low-altitude ablations with stronger source penalties or less aggressive near-surface shells.")
        else:
            findings.append(f"Altitude error ratio is controlled at {altitude_ratio:.1f}x low/high.")

    if energy_fractions:
        dominant = max(energy_fractions)
        dominant_idx = energy_fractions.index(dominant)
        findings.append(
            "Shell energy fractions: "
            + ", ".join(f"shell {i}={frac:.2%}" for i, frac in enumerate(energy_fractions))
            + "."
        )
        if dominant > thresholds.shell_dominance_watch and len(energy_fractions) > 1:
            risks.append(f"Shell {dominant_idx} holds {dominant:.1%} of source energy; multi-shell may be collapsing.")
            next_steps.append("Increase shell-wise penalty on the dominant shell or test an alternate shell set.")

    if sigma_abs_max > thresholds.sigma_abs_max_watch:
        risks.append(f"Maximum source amplitude is large ({sigma_abs_max:.3g}); watch for ill-conditioned source maps.")

    if monopole > thresholds.monopole_watch:
        risks.append(f"Monopole leakage is above the heuristic watch level ({monopole:.3e}).")
        next_steps.append("Increase lambda_moment or explicitly inspect residual low-degree content.")
    else:
        findings.append(f"Monopole leakage is controlled ({monopole:.3e}).")

    if dipole:
        findings.append(f"Dipole leakage diagnostic is {dipole:.3e}; compare this across ablations rather than as an absolute truth.")

    if radial_rmse is not None and cross_rmse is not None:
        findings.append(f"Radial/cross-radial RMSE: {radial_rmse:.3e} / {cross_rmse:.3e}.")

    if inference_seconds is not None:
        findings.append(f"Inference time per evaluation batch is {inference_seconds:.4f} s.")

    if not next_steps:
        next_steps.append("Use this run as a baseline and compare against shell/source-count ablations.")

    return {
        "name": summary["name"],
        "path": str(summary["path"]),
        "n_sources": summary["n_sources"],
        "shell_radii": summary["shell_radii"],
        "acceleration_rmse": acc_rmse,
        "potential_rmse": pot_rmse,
        "acceleration_status": acc_status,
        "potential_status": pot_status,
        "low_high_altitude_ratio": altitude_ratio,
        "shell_energy_fractions": energy_fractions,
        "findings": findings,
        "risks": risks,
        "next_steps": next_steps,
    }


def compare_experiments(reports: list[dict]) -> dict:
    """Compare interpreted experiment reports."""

    usable = [r for r in reports if r.get("acceleration_rmse") is not None]
    if not usable:
        return {"findings": ["No comparable acceleration RMSE values were found."], "ranking": []}

    ranking = sorted(usable, key=lambda r: float(r["acceleration_rmse"]))
    best = ranking[0]
    findings = [f"Best acceleration RMSE: {best['name']} ({best['acceleration_rmse']:.6e})."]

    if len(ranking) >= 2:
        second = ranking[1]
        ratio = float(second["acceleration_rmse"]) / max(float(best["acceleration_rmse"]), 1.0e-30)
        findings.append(f"Second-best run is {second['name']}; best is {ratio:.2f}x lower in acceleration RMSE.")

    single = [r for r in usable if len(r.get("shell_radii", ())) == 1]
    multi = [r for r in usable if len(r.get("shell_radii", ())) > 1]
    if single and multi:
        best_single = min(single, key=lambda r: float(r["acceleration_rmse"]))
        best_multi = min(multi, key=lambda r: float(r["acceleration_rmse"]))
        delta = float(best_multi["acceleration_rmse"]) / max(float(best_single["acceleration_rmse"]), 1.0e-30)
        if delta < 0.9:
            findings.append("Multi-shell improves over the best single-shell baseline; Stage 2 is promising.")
        elif delta <= 1.1:
            findings.append("Multi-shell is roughly tied with single-shell; inspect shell energy before moving forward.")
        else:
            findings.append(
                f"Multi-shell is currently {delta:.2f}x worse than single-shell; tune shell penalties/radii before Stage 3."
            )

    return {"findings": findings, "ranking": ranking}


def make_markdown_report(checkpoint_paths: Iterable[str | Path]) -> str:
    summaries = [load_checkpoint_summary(path) for path in checkpoint_paths]
    reports = [interpret_experiment(summary) for summary in summaries]
    comparison = compare_experiments(reports)

    lines = [
        "# VESP Experiment Analysis",
        "",
        "This report uses heuristic MVP thresholds for normalized synthetic or normalized-radius runs.",
        "Treat the conclusions as ablation guidance, not as final physical validation.",
        "",
        "## Summary Table",
        "",
        "| Run | Shells | Sources | Acc RMSE | Pot RMSE | Low/High Alt Ratio | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]

    for report in reports:
        shells = ", ".join(f"{v:.2f}" for v in report["shell_radii"]) or "-"
        ratio = report["low_high_altitude_ratio"]
        ratio_text = "-" if ratio is None else f"{ratio:.2f}"
        lines.append(
            f"| {report['name']} | {shells} | {report['n_sources']} | "
            f"{report['acceleration_rmse']:.6e} | {report['potential_rmse']:.6e} | "
            f"{ratio_text} | acc={report['acceleration_status']}, pot={report['potential_status']} |"
        )

    lines.extend(["", "## Comparison", ""])
    for finding in comparison["findings"]:
        lines.append(f"- {finding}")

    for report in reports:
        lines.extend(["", f"## {report['name']}", "", "Findings:"])
        lines.extend(f"- {item}" for item in report["findings"])
        lines.append("")
        lines.append("Risks:")
        lines.extend(f"- {item}" for item in (report["risks"] or ["No major heuristic risk flagged."]))
        lines.append("")
        lines.append("Next steps:")
        lines.extend(f"- {item}" for item in report["next_steps"])

    lines.extend(
        [
            "",
            "## Continue / Redesign Signal",
            "",
            "- Continue if acceleration error is controlled, source amplitudes stay bounded, and multi-shell improves stability or OOD behavior.",
            "- Redesign if multi-shell consistently underperforms single-shell, shell energy collapses, or near-surface error dominates.",
            "- Add MaxEnt only after the discrete equivalent-source baseline has a reliable ablation story.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown_report(checkpoint_paths: Iterable[str | Path], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(make_markdown_report(checkpoint_paths), encoding="utf-8")
    return output


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="One or more VESP checkpoint .pt files")
    parser.add_argument("--output", default="outputs/analysis_report.md")
    args = parser.parse_args(argv)
    output = write_markdown_report(args.checkpoints, args.output)
    print(f"analysis_report: {output}")
    print(make_markdown_report(args.checkpoints))


if __name__ == "__main__":
    main()

