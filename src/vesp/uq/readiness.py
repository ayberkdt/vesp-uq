"""Pre-results readiness orchestration for VESP-UQ.

This module owns the repeatable "measure before result production" gate. It runs diagnostics,
attribution, the RTN covariance prototype, and optionally geometry auto-selection over the same
config set, then writes one top-level manifest-backed report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vesp.uq.attribution import run_log_attribution
from vesp.uq.cli import csv_text, fmt_float
from vesp.uq.gate_diagnostics import run_gate_diagnostics
from vesp.uq.geometry_calibration import run_geometry_calibration
from vesp.uq.io.run_artifacts import verify_manifest, write_run_artifacts
from vesp.uq.rtn_noise_prototype import run_rtn_noise_prototype

DEFAULT_CONFIGS = [
    "configs/vespuq/vespuq_real_lunar.yaml",
    "configs/vespuq/vespuq_real_lunar_L90.yaml",
]
DEFAULT_REPORTS = [
    "benchmarks/vespuq_real_lunar_report.md",
    "benchmarks/vespuq_real_lunar_L90_report.md",
]

QUICK_N_ORBITS_GATE = 120
QUICK_N_ORBITS_ATTRIBUTION = 160
QUICK_N_POINTS = 48
QUICK_MAX_COV_POINTS = 600
QUICK_MAX_CURL_POINTS = 256
QUICK_TOP_N = 20
QUICK_RTN_MAX_POINTS = 600


def _subrun_artifact_files(*dirs: Path) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for directory in dirs:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file():
                artifacts[str(path.relative_to(directory.parent)).replace("\\", "/")] = path
    return artifacts


def summarize_readiness(
    *,
    gate_result: dict,
    attribution_result: dict,
    rtn_result: dict,
    geometry_result: dict | None = None,
    verifications: list[dict],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build top-level readiness status and one row per major subsystem."""

    gate_warnings = sum(
        1
        for case in gate_result.get("cases", [])
        for row in case.get("consistency_rows", [])
        if row.get("status") == "warn"
    )
    gate_cases = len(gate_result.get("cases", []))
    gate_ok = gate_cases > 0 and gate_warnings == 0

    attr_cases = len(attribution_result.get("cases", []))
    masking_rows = attribution_result.get("masking_rows", [])
    mask_win_rate = (
        sum(1 for row in masking_rows if bool(row.get("top_mask_beats_random"))) / len(masking_rows)
        if masking_rows
        else float("nan")
    )
    attribution_ok = attr_cases > 0 and len(masking_rows) > 0

    rtn_cases = rtn_result.get("cases", [])
    rtn_decisions = [str(case.get("case_decision", "missing")) for case in rtn_cases]
    rtn_ok = len(rtn_cases) > 0 and all(decision in {"candidate", "partial", "hold"} for decision in rtn_decisions)
    rtn_promote = bool(rtn_decisions) and all(decision == "candidate" for decision in rtn_decisions)

    geometry_verdict = (geometry_result or {}).get("verdict") or {}
    geometry_ran = geometry_result is not None
    geometry_ok = (not geometry_ran) or bool(geometry_verdict)
    geometry_detail = "skipped"
    if geometry_ran:
        bits = [
            f"{band}: best={verdict.get('best_geometry')}"
            for band, verdict in sorted(geometry_verdict.items())
        ]
        geometry_detail = "; ".join(bits) if bits else "no verdict"

    provenance_ok = all(bool(v.get("ok")) and not v.get("unlisted") for v in verifications)

    rows = [
        {
            "check": "gate_diagnostics_abc",
            "status": "ok" if gate_ok else "block",
            "detail": f"{gate_cases} cases, {gate_warnings} consistency warnings",
        },
        {
            "check": "exact_log_attribution",
            "status": "ok" if attribution_ok else "block",
            "detail": f"{attr_cases} cases, masking win-rate {fmt_float(mask_win_rate, '.2f')}",
        },
        {
            "check": "rtn_noise_prototype",
            "status": "ok" if rtn_ok else "block",
            "detail": "case decisions: " + ", ".join(rtn_decisions or ["missing"]),
        },
        {
            "check": "geometry_auto_selection",
            "status": "ok" if geometry_ran and geometry_ok else ("block" if geometry_ran else "skipped"),
            "detail": geometry_detail,
        },
        {
            "check": "subrun_provenance",
            "status": "ok" if provenance_ok else "block",
            "detail": f"{sum(len(v.get('verified', [])) for v in verifications)} files verified",
        },
        {
            "check": "prototype_promotion",
            "status": "hold" if not rtn_promote else "candidate",
            "detail": "RTN prototype is not integrated unless every case is candidate",
        },
    ]
    blockers = [row for row in rows if row["status"] == "block"]
    summary = {
        "status": "ready_for_controlled_result_runs" if not blockers else "not_ready",
        "blockers": [row["check"] for row in blockers],
        "gate_consistency_warnings": gate_warnings,
        "attribution_mask_win_rate": mask_win_rate,
        "rtn_case_decisions": rtn_decisions,
        "rtn_production_promotion": "candidate" if rtn_promote else "hold",
        "geometry_auto_selection": "ran" if geometry_ran else "skipped",
        "geometry_verdict": geometry_verdict,
        "provenance_ok": provenance_ok,
    }
    return summary, rows


def _summary_md(summary: dict[str, Any], rows: list[dict[str, Any]], out_dir: Path) -> str:
    lines = [
        "# VESP-UQ System Readiness",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| check | status | detail |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['check']} | {row['status']} | {row['detail']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This report only gates the system before result production; it does not change production covariance.",
            "- SHAP/LIME remains absent. Attribution is exact log-factor decomposition plus masking validation.",
            "- RTN noise remains a prototype unless every case clears the case-level candidate guardrail.",
            "- Helmholtz/non-conservative extension remains closed unless Measurement C passes a stronger sanity check.",
            "",
            "## Subreports",
            "",
            f"- `{out_dir / 'gate_diagnostics' / 'gate_diagnostics.md'}`",
            f"- `{out_dir / 'log_attribution' / 'log_attribution.md'}`",
            f"- `{out_dir / 'rtn_noise_prototype' / 'rtn_noise_prototype.md'}`",
            f"- `{out_dir / 'geometry_calibration' / 'geometry_calibration.md'}` "
            "(full readiness, or quick with `--include-geometry`)",
        ]
    )
    return "\n".join(lines) + "\n"


def run_system_readiness(
    configs: list[dict],
    *,
    report_paths: list[str | None] | None = None,
    out_dir: str | Path = "outputs/system_readiness",
    quick: bool = False,
    include_geometry: bool = True,
) -> dict:
    """Run all pre-result gates and write a top-level readiness artifact."""

    out = Path(out_dir)
    gate_dir = out / "gate_diagnostics"
    attribution_dir = out / "log_attribution"
    rtn_dir = out / "rtn_noise_prototype"
    geometry_dir = out / "geometry_calibration"

    gate_n_orbits = QUICK_N_ORBITS_GATE if quick else None
    attribution_n_orbits = QUICK_N_ORBITS_ATTRIBUTION if quick else None
    n_points = QUICK_N_POINTS if quick else None
    max_cov_points = QUICK_MAX_COV_POINTS if quick else 2000
    max_curl_points = QUICK_MAX_CURL_POINTS if quick else 512
    top_n = QUICK_TOP_N if quick else 25
    rtn_max_points = QUICK_RTN_MAX_POINTS if quick else None

    gate_result = run_gate_diagnostics(
        configs,
        report_paths=report_paths,
        out_dir=gate_dir,
        n_orbits=gate_n_orbits,
        n_points=n_points,
        max_cov_points=max_cov_points,
        max_curl_points=max_curl_points,
    )
    attribution_result = run_log_attribution(
        configs,
        out_dir=attribution_dir,
        n_orbits=attribution_n_orbits,
        n_points=n_points,
        top_n=top_n,
    )
    rtn_result = run_rtn_noise_prototype(
        configs,
        out_dir=rtn_dir,
        max_points=rtn_max_points,
    )
    geometry_result = None
    if include_geometry:
        geometry_result = run_geometry_calibration(
            configs,
            seeds=[0] if quick else (0, 1, 2, 3, 4),
            geometries=["baseline", "surface_dense", "deep"] if quick else None,
            out_dir=geometry_dir,
            make_plots=not quick,
        )

    verify_dirs = [gate_dir, attribution_dir, rtn_dir]
    if include_geometry:
        verify_dirs.append(geometry_dir)
    verifications = [verify_manifest(d) for d in verify_dirs]
    summary, rows = summarize_readiness(
        gate_result=gate_result,
        attribution_result=attribution_result,
        rtn_result=rtn_result,
        geometry_result=geometry_result,
        verifications=verifications,
    )
    settings = {
        "quick": bool(quick),
        "configs": [cfg.get("_config_path") for cfg in configs],
        "reports": report_paths,
        "gate_n_orbits": gate_n_orbits,
        "attribution_n_orbits": attribution_n_orbits,
        "n_points": n_points,
        "max_cov_points": max_cov_points,
        "max_curl_points": max_curl_points,
        "top_n": top_n,
        "rtn_max_points": rtn_max_points,
        "include_geometry": include_geometry,
    }
    payload = {
        "summary": summary,
        "rows": rows,
        "subrun_verifications": verifications,
        "settings": settings,
    }
    artifact_dirs = [gate_dir, attribution_dir, rtn_dir]
    if include_geometry:
        artifact_dirs.append(geometry_dir)
    artifact_files = _subrun_artifact_files(*artifact_dirs)
    manifest = write_run_artifacts(
        out,
        tool="run_vespuq_system_readiness",
        config=settings,
        json_files={"system_readiness.json": payload},
        text_files={
            "system_readiness.md": _summary_md(summary, rows, out),
            "system_readiness_checks.csv": csv_text(rows, ["check", "status", "detail"]),
        },
        artifact_files=artifact_files,
        seed=[cfg.get("seed") for cfg in configs],
        config_path=",".join(str(cfg.get("_config_path", "")) for cfg in configs),
    )
    return {
        "out_dir": str(out),
        "summary": summary,
        "rows": rows,
        "manifest": manifest,
    }
