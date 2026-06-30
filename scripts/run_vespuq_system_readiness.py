"""Run the VESP-UQ pre-results readiness gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from vesp.uq.cli import aligned_existing_reports, load_configs
from vesp.uq.readiness import DEFAULT_CONFIGS, DEFAULT_REPORTS, run_system_readiness


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the VESP-UQ pre-results readiness gate.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=DEFAULT_CONFIGS,
        help="VESP-UQ configs; defaults to real lunar L60 and L90",
    )
    parser.add_argument(
        "--reports",
        nargs="*",
        default=None,
        help="benchmark reports aligned with --configs; omit to use L60/L90 defaults",
    )
    parser.add_argument("--out", default="outputs/system_readiness")
    parser.add_argument("--quick", action="store_true", help="fast readiness pass for iteration")
    parser.add_argument(
        "--include-geometry",
        action="store_true",
        help="also run geometry auto-selection in quick mode",
    )
    parser.add_argument(
        "--skip-geometry",
        action="store_true",
        help="skip geometry auto-selection even in full mode",
    )
    args = parser.parse_args(argv)

    configs = load_configs(args.configs)
    reports = aligned_existing_reports(args.reports, n_configs=len(configs), defaults=DEFAULT_REPORTS)
    include_geometry = (not args.quick or args.include_geometry) and not args.skip_geometry
    result = run_system_readiness(
        configs,
        report_paths=reports,
        out_dir=args.out,
        quick=args.quick,
        include_geometry=include_geometry,
    )
    print(f"saved_system_readiness: {Path(result['out_dir']) / 'system_readiness.md'}")
    print(f"status={result['summary']['status']}")
    if result["summary"]["blockers"]:
        print("blockers=" + ",".join(result["summary"]["blockers"]))
    print("rtn_production_promotion=" + result["summary"]["rtn_production_promotion"])


if __name__ == "__main__":
    main()
