"""Physical acceleration-error budget screening for VESP-UQ.

Flags trajectories whose estimated *force-model* error exceeds a user-defined physical
acceleration-error tolerance (e.g. ``1e-8 m/s^2``). The budget is converted into the model's score
units via an explicit acceleration scale (``body.acceleration_scale_m_s2`` or a physical
``body.acceleration_units``) and compared against an absolute-scale trajectory risk score; it never
infers a physical scale and never silently falls back to normalized units.

    python scripts/run_physical_budget_screening.py --config configs/vespuq/vespuq_smoke.yaml \
        --budget 1e-8 --units m/s^2 --scoring expected_abs_p95

Set ``uq.physical_budget.enabled: true`` in the config (with a ``value``) to run without CLI flags.

Outputs (under --out-dir, default outputs/physical_budget):
    physical_budget_screening.json, physical_budget_screening.md, physical_budget_scores.csv

This flags force-risk exceedances against a tolerance; it is not a position-accuracy or
orbit-covariance diagnostic and does not guarantee safety.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vesp.common.config import load_config
from vesp.uq.experiment import run_vespuq
from vesp.uq.physical_units import acceleration_to_physical, resolve_acceleration_scale
from vesp.uq.scoring import is_absolute_scoring, is_relative_scoring
from vesp.uq.thresholds import PHYSICAL_BUDGET_SCORINGS


def _configure_physical_budget(config: dict, args) -> dict:
    """Merge CLI flags into the config's physical-budget block and wire up the screening keys."""

    physical_cfg = dict(config.get("uq", {}).get("physical_budget", {}) or {})
    value = args.budget if args.budget is not None else physical_cfg.get("value")
    units = args.units or physical_cfg.get("units", "m/s^2")
    scoring = args.scoring or physical_cfg.get("scoring", "expected_abs_p95")
    max_rerun_fraction = (
        args.max_rerun_fraction
        if args.max_rerun_fraction is not None
        else physical_cfg.get("max_rerun_fraction")
    )
    enabled = bool(physical_cfg.get("enabled", False)) or args.budget is not None

    if not enabled:
        raise SystemExit(
            "physical-budget screening is not enabled: set uq.physical_budget.enabled=true "
            "(with a value) in the config, or pass --budget on the command line."
        )
    if value is None:
        raise SystemExit("a physical budget value is required (config uq.physical_budget.value or --budget)")
    if is_relative_scoring(scoring) or not is_absolute_scoring(scoring):
        raise SystemExit(
            f"physical-budget screening requires an absolute scoring mode {PHYSICAL_BUDGET_SCORINGS}; "
            f"got --scoring/{scoring!r} (relative supervisor scores cannot be compared to a budget)."
        )

    # Wire the resolved budget into the config so run_vespuq screens with the physical threshold.
    config.setdefault("uq", {}).setdefault("risk", {})["scoring"] = scoring
    config["uq"].setdefault("screening", {})["threshold_source"] = "physical_budget"
    config["uq"]["physical_budget"] = {
        "enabled": True,
        "value": float(value),
        "units": str(units),
        "scoring": scoring,
        "max_rerun_fraction": (float(max_rerun_fraction) if max_rerun_fraction is not None else None),
    }
    return config["uq"]["physical_budget"]


def run_physical_budget_screening(config: dict) -> dict:
    """Run the VESP-UQ pipeline under a physical budget and assemble the screening report dict."""

    report = run_vespuq(config)
    tables = report.pop("_tables")
    screen = report["experiment_3_screening"]
    sc = screen["screen"]
    scale = resolve_acceleration_scale(config)
    units = screen.get("threshold_physical_units") or config["uq"]["physical_budget"]["units"]

    header = tables["trajectory_header"]
    i_id = header.index("trajectory_id")
    i_risk = header.index("risk_score")
    i_flag = header.index("flagged_for_rerun")
    i_true = header.index("true_error")
    model_threshold = float(sc["threshold"])

    rows = []
    for r in tables["trajectory_rows"]:
        risk_model = float(r[i_risk])
        true_model = float(r[i_true])
        rows.append({
            "trajectory_id": int(r[i_id]),
            "risk_score_model": risk_model,
            "risk_score_physical": float(acceleration_to_physical(risk_model, scale, units)),
            "true_force_error_model": true_model,
            "true_force_error_physical": float(acceleration_to_physical(true_model, scale, units)),
            "above_budget": int(risk_model >= model_threshold),
            "flagged": int(r[i_flag]),
        })

    n_above = sc.get("n_above_threshold")
    capped = bool(
        sc.get("max_rerun_fraction") is not None
        and n_above is not None
        and sc["n_flagged"] < n_above
    )
    result = {
        "config_path": config.get("_config_path"),
        "error_basis": "true_force_model_error",
        "scope_note": (
            "Physical budget screening flags trajectories whose estimated force-risk exceeds a "
            "user-defined acceleration-error tolerance. It is not a position-accuracy or "
            "orbit-covariance diagnostic and does not guarantee safety."
        ),
        "scoring": screen["scoring"],
        "scoring_scale": screen.get("scoring_scale"),
        "physical_budget": {
            "value": screen.get("threshold_physical_value"),
            "units": units,
            "scoring": config["uq"]["physical_budget"]["scoring"],
            "max_rerun_fraction": config["uq"]["physical_budget"]["max_rerun_fraction"],
        },
        "threshold": {
            "model_units": model_threshold,
            "physical_value": screen.get("threshold_physical_value"),
            "physical_units": units,
            "acceleration_scale_m_s2": screen.get("acceleration_scale_m_s2"),
        },
        "physical_conversion_available": scale.physical,
        "n_trajectories": sc["n_trajectories"],
        "n_above_budget": n_above,
        "n_flagged": sc["n_flagged"],
        "flagged_fraction": sc["rerun_fraction"],
        "max_rerun_fraction_capped": capped,
        "_rows": rows,
    }
    return result


def _screening_md(result: dict) -> str:
    def f(x, s=".4e"):
        return "n/a" if x is None else format(float(x), s)

    pb = result["physical_budget"]
    thr = result["threshold"]
    flagged = sorted(
        (r for r in result["_rows"] if r["flagged"]),
        key=lambda r: r["risk_score_model"],
        reverse=True,
    )[:10]
    top_lines = [
        f"| {r['trajectory_id']} | {f(r['risk_score_model'])} | {f(r['risk_score_physical'])} | "
        f"{f(r['true_force_error_physical'])} |"
        for r in flagged
    ] or ["| (none) | | | |"]
    return "\n".join([
        "# VESP-UQ Physical Acceleration-Budget Screening",
        "",
        "**Flags trajectories whose estimated FORCE-MODEL error exceeds a user-defined "
        "acceleration-error tolerance.** The physical budget is converted into the model's score "
        "units via an explicit acceleration scale; relative (ranking) scores are rejected. This is "
        "not a position-accuracy or orbit-covariance diagnostic and does not guarantee safety.",
        "",
        f"- config: `{result.get('config_path')}`",
        f"- physical budget: **{f(pb['value'])} {pb['units']}**  |  scoring: `{result['scoring']}` "
        f"(`{result['scoring_scale']}`)",
        f"- acceleration scale: 1 model unit = {f(thr['acceleration_scale_m_s2'])} m/s^2",
        f"- converted model-unit threshold: {f(thr['model_units'])}",
        f"- trajectories: {result['n_trajectories']}  |  above budget: {result['n_above_budget']}  |  "
        f"flagged: {result['n_flagged']} ({result['flagged_fraction']:.1%})",
        f"- max_rerun_fraction capped the result: {result['max_rerun_fraction_capped']}"
        + (f" (cap {pb['max_rerun_fraction']})" if pb["max_rerun_fraction"] is not None else ""),
        "",
        ("**Zero alarms:** no trajectory's estimated force-risk exceeded the physical budget on this "
         "set." if result["n_flagged"] == 0 else
         "**Nonzero alarms:** the trajectories below carry estimated force-risk at or above the "
         "physical budget and are flagged for high-fidelity rerun."),
        "",
        "## Top flagged trajectories",
        "",
        "| trajectory_id | risk (model units) | risk (physical) | true force error (physical) |",
        "| ---: | ---: | ---: | ---: |",
        *top_lines,
        "",
        "Interpretation: a flagged trajectory is one whose VESP-UQ force-risk estimate meets or "
        "exceeds the acceleration-error tolerance. The true force error column is a held-out "
        "diagnostic, not a guarantee. Both model-normalized and physical values are reported.",
        "",
    ]) + "\n"


def run_and_write(config: dict, *, out_dir: Path) -> dict:
    result = run_physical_budget_screening(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = result["_rows"]
    markdown = _screening_md(result)
    serializable = {k: v for k, v in result.items() if k != "_rows"}

    (out_dir / "physical_budget_screening.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    (out_dir / "physical_budget_screening.md").write_text(markdown, encoding="utf-8")
    csv_header = [
        "trajectory_id", "risk_score_model", "risk_score_physical",
        "true_force_error_model", "true_force_error_physical", "above_budget", "flagged",
    ]
    lines = [",".join(csv_header)] + [",".join(str(r[h]) for h in csv_header) for r in rows]
    (out_dir / "physical_budget_scores.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result["_markdown"] = markdown
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="VESP-UQ physical acceleration-budget screening.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--budget", type=float, default=None, help="physical acceleration-error budget value")
    parser.add_argument("--units", default=None, help="m/s^2 | km/s^2 | mm/s^2 | um/s^2")
    parser.add_argument("--scoring", default=None, help="absolute scoring mode (default expected_abs_p95)")
    parser.add_argument("--max-rerun-fraction", type=float, default=None)
    parser.add_argument("--out-dir", default="outputs/physical_budget")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config.setdefault("_config_path", args.config)
    _configure_physical_budget(config, args)
    result = run_and_write(config, out_dir=Path(args.out_dir))
    print(result["_markdown"].encode("ascii", "replace").decode("ascii"))
    print(f"saved_physical_budget_screening: {Path(args.out_dir) / 'physical_budget_screening.md'}")


if __name__ == "__main__":
    main()
