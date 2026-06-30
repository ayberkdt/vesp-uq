"""Run exact VESP-UQ log-factor attribution + masking validation.

This is not SHAP/LIME. It uses the closed-form supervisor factors:

    log(point_risk) = log(expected_error) + log(altitude_weight) + log(1 + k*d_OOD)

Example
-------
    python scripts/run_vespuq_log_attribution.py \
        --configs configs/vespuq/vespuq_real_lunar.yaml configs/vespuq/vespuq_real_lunar_L90.yaml \
        --quick --out outputs/log_attribution_quick
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vesp.uq.attribution import run_log_attribution
from vesp.uq.cli import load_configs

_QUICK_N_ORBITS = 160
_QUICK_N_POINTS = 48
_QUICK_TOP_N = 20


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="VESP-UQ exact log-factor attribution.")
    parser.add_argument("--configs", nargs="+", required=True, help="one or more VESP-UQ configs")
    parser.add_argument("--out", default="outputs/log_attribution/")
    parser.add_argument("--n-orbits", type=int, default=None)
    parser.add_argument("--n-points", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=25,
                        help="number of highest-risk trajectories attributed per config")
    parser.add_argument("--mask-fraction", type=float, default=0.10,
                        help="fraction of points masked in the validation test")
    parser.add_argument("--scoring", default="supervisor_rel_p95")
    parser.add_argument("--aggregator", default="p95", choices=("mean", "p95", "max"),
                        help="true-error profile aggregator used in masking validation")
    parser.add_argument("--quick", action="store_true",
                        help="small L60/L90 pass for fast first-look artifacts")
    args = parser.parse_args(argv)

    n_orbits = args.n_orbits
    n_points = args.n_points
    top_n = args.top_n
    if args.quick:
        n_orbits = _QUICK_N_ORBITS if n_orbits is None else min(n_orbits, _QUICK_N_ORBITS)
        n_points = _QUICK_N_POINTS if n_points is None else min(n_points, _QUICK_N_POINTS)
        top_n = min(top_n, _QUICK_TOP_N)

    result = run_log_attribution(
        load_configs(args.configs),
        out_dir=args.out,
        n_orbits=n_orbits,
        n_points=n_points,
        top_n=top_n,
        mask_fraction=args.mask_fraction,
        scoring=args.scoring,
        aggregator=args.aggregator,
    )
    print(f"saved_log_attribution: {Path(result['out_dir']) / 'log_attribution.md'}")
    for case in result["cases"]:
        rows = [r for r in result["masking_rows"] if r["band"] == case["band"]]
        wins = sum(1 for r in rows if bool(r["top_mask_beats_random"]))
        print(
            f"{case['band']}: attributed={case['n_attributed']} "
            f"mask_win_rate={wins / max(1, len(rows)):.2f}"
        )


if __name__ == "__main__":
    main()
