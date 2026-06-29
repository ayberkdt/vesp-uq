"""VESP-UQ trajectory-family diversity study (WP9).

Scores VESP-UQ and baselines against trajectory true FORCE-model error across controlled orbit
families (near-circular, eccentric perilune, polar, equatorial, inclined, descent arc, high-alt
transfer, OOD low-alt), reporting per-family ranking + whether VESP-UQ adds value beyond altitude.

    python scripts/run_trajectory_families.py \
        --configs configs/vespuq/vespuq_real_lunar.yaml configs/vespuq/vespuq_real_lunar_L90.yaml \
        --seeds 0 1 2 --n-orbits 2000 --out outputs/trajectory_families/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vesp.common.config import load_config
from vesp.uq.family_study import run_trajectory_families
from vesp.uq.trajectory_families import FAMILIES


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="VESP-UQ trajectory-family diversity study (WP9).")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--families", nargs="+", default=None, help=f"subset of {sorted(FAMILIES)}")
    parser.add_argument("--n-orbits", type=int, default=2000)
    parser.add_argument("--n-points", type=int, default=120)
    parser.add_argument("--out", default="outputs/trajectory_families/")
    parser.add_argument("--primary-fraction", type=float, default=0.20)
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
    n_orbits = min(args.n_orbits, 200) if args.quick else args.n_orbits

    run_trajectory_families(
        configs, seeds=seeds, families=args.families, n_orbits=n_orbits, n_points=args.n_points,
        primary_fraction=args.primary_fraction, out_dir=args.out, make_plots=not args.no_plots,
    )
    print(f"saved_trajectory_families: {Path(args.out) / 'trajectory_family_summary.md'}")


if __name__ == "__main__":
    main()
