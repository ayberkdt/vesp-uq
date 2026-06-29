"""VESP-UQ drift-boundary characterization (B): where does force-risk predict position drift?

Combines the controlled trajectory families (WP9) with the multi-horizon covariance propagation
(WP10) into a (family x horizon) grid of Spearman(force-risk, position dispersion), and reports the
horizon up to which force-risk predicts drift in each family. Characterization only -- the
dispersion is the force-error-posterior sigma, not a validated position-error propagation.

    python scripts/run_drift_boundary.py \
        --configs configs/vespuq/vespuq_real_lunar.yaml configs/vespuq/vespuq_real_lunar_L90.yaml \
        --seeds 0 1 --n-orbits 20 --out outputs/drift_boundary/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vesp.common.config import load_config
from vesp.uq.drift_boundary import DEFAULT_FAMILIES, run_drift_boundary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="VESP-UQ drift-boundary characterization (B).")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--families", nargs="+", default=None, help=f"default: {list(DEFAULT_FAMILIES)}")
    parser.add_argument("--n-orbits", type=int, default=20)
    parser.add_argument("--n-points", type=int, default=120)
    parser.add_argument("--out", default="outputs/drift_boundary/")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    configs = []
    for path in args.configs:
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"config not found: {path}")
        cfg = load_config(str(p))
        cfg.setdefault("_config_path", str(p))
        configs.append(cfg)

    seeds = [0] if args.quick else args.seeds
    n_orbits = min(args.n_orbits, 8) if args.quick else args.n_orbits
    families = args.families or (["low_alt_near_circular", "high_alt_transfer"] if args.quick else None)

    result = run_drift_boundary(
        configs, seeds=seeds, families=families, n_orbits=n_orbits, n_points=args.n_points,
        out_dir=args.out, make_plots=not args.no_plots,
    )
    for (band, fam), v in result["verdict"].items():
        print(f"{band}/{fam}: predicts_drift_up_to={v['predicts_drift_up_to_periods']} periods")
    print(f"saved_drift_boundary: {Path(args.out) / 'drift_boundary.md'}")


if __name__ == "__main__":
    main()
