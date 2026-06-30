"""Run VESP-UQ gate diagnostics before architecture / attribution decisions.

Examples
--------
    python scripts/run_vespuq_gate_diagnostics.py \
        --configs configs/vespuq/vespuq_real_lunar.yaml configs/vespuq/vespuq_real_lunar_L90.yaml \
        --reports benchmarks/vespuq_real_lunar_report.md benchmarks/vespuq_real_lunar_L90_report.md \
        --quick --out outputs/gate_diagnostics_quick

    python scripts/run_vespuq_gate_diagnostics.py \
        --configs configs/vespuq/vespuq_real_lunar.yaml configs/vespuq/vespuq_real_lunar_L90.yaml \
        --reports benchmarks/vespuq_real_lunar_report.md benchmarks/vespuq_real_lunar_L90_report.md \
        --n-orbits 2000 --max-cov-points 4000 --out outputs/gate_diagnostics
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vesp.uq.cli import load_configs
from vesp.uq.gate_diagnostics import run_gate_diagnostics

_QUICK_N_ORBITS = 120
_QUICK_N_POINTS = 48
_QUICK_MAX_COV_POINTS = 600
_QUICK_MAX_CURL_POINTS = 256


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="VESP-UQ measurement-gate diagnostics.")
    parser.add_argument("--configs", nargs="+", required=True, help="one or more VESP-UQ YAML configs")
    parser.add_argument("--reports", nargs="*", default=None,
                        help="optional benchmark Markdown reports aligned with --configs")
    parser.add_argument("--out", default="outputs/gate_diagnostics/")
    parser.add_argument("--n-orbits", type=int, default=None,
                        help="override uq.screening.n_orbits for diagnostics")
    parser.add_argument("--n-points", type=int, default=None,
                        help="override uq.screening.n_points for generated trajectories")
    parser.add_argument("--max-cov-points", type=int, default=2000,
                        help="max generated trajectory points used for covariance eigen diagnostics")
    parser.add_argument("--max-curl-points", type=int, default=512,
                        help="max residual-field points used for the local curl diagnostic")
    parser.add_argument("--curl-k", type=int, default=24,
                        help="neighbors per local least-squares curl fit")
    parser.add_argument("--report-tolerance", type=float, default=0.05,
                        help="allowed absolute difference when comparing current calibration to reports")
    parser.add_argument("--quick", action="store_true",
                        help="small real-data diagnostic pass for wiring / first look")
    args = parser.parse_args(argv)

    configs = load_configs(args.configs)
    reports = args.reports
    if reports is not None and len(reports) == 0:
        reports = None
    if reports is not None and len(reports) != len(configs):
        raise SystemExit("--reports must be omitted or have the same number of paths as --configs")

    n_orbits = args.n_orbits
    n_points = args.n_points
    max_cov_points = args.max_cov_points
    max_curl_points = args.max_curl_points
    if args.quick:
        n_orbits = _QUICK_N_ORBITS if n_orbits is None else min(n_orbits, _QUICK_N_ORBITS)
        n_points = _QUICK_N_POINTS if n_points is None else min(n_points, _QUICK_N_POINTS)
        max_cov_points = min(max_cov_points, _QUICK_MAX_COV_POINTS)
        max_curl_points = min(max_curl_points, _QUICK_MAX_CURL_POINTS)

    result = run_gate_diagnostics(
        configs,
        report_paths=reports,
        out_dir=args.out,
        n_orbits=n_orbits,
        n_points=n_points,
        max_cov_points=max_cov_points,
        max_curl_points=max_curl_points,
        curl_k=args.curl_k,
        report_tolerance=args.report_tolerance,
    )
    print(f"saved_gate_diagnostics: {Path(result['out_dir']) / 'gate_diagnostics.md'}")
    for case in result["cases"]:
        warnings = sum(1 for row in case["consistency_rows"] if row.get("status") == "warn")
        print(
            f"{case['band']}: factor_corr={case['measurement_a']['mean_abs_factor_corr']:.3f} "
            f"curl_excess={case['measurement_c']['excess_scaled_curl_ratio']:.3g} "
            f"consistency_warnings={warnings}"
        )


if __name__ == "__main__":
    main()
