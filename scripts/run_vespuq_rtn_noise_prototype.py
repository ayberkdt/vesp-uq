"""Run the diagonal RTN-style covariance calibration prototype."""

from __future__ import annotations

import argparse
from pathlib import Path

from vesp.uq.cli import fmt_float, load_configs
from vesp.uq.rtn_noise_prototype import run_rtn_noise_prototype


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

    max_points = args.max_points
    if args.quick:
        max_points = 600 if max_points is None else min(max_points, 600)
    result = run_rtn_noise_prototype(
        load_configs(args.configs),
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
            f"{case['band']}: before_error={fmt_float(overall.get('before_error_score'))} "
            f"after_error={fmt_float(overall.get('after_error_score'))} "
            f"decision={case.get('case_decision', overall.get('decision', 'n/a'))}"
        )


if __name__ == "__main__":
    main()
