"""VESP-UQ force-risk vs trajectory-drift multi-horizon diagnostic (WP10).

Reports (10.1) force-error ranking and (10.2) drift ranking at 1/6/12/60 orbital periods, using the
linearized force-error covariance propagator for the position dispersion. Horizons are unit-free
(orbital periods); the dispersion is the force-error-posterior sigma -- a diagnostic, not a
validated position-error propagation.

    python scripts/run_drift_horizon.py \
        --configs configs/vespuq/vespuq_real_lunar.yaml configs/vespuq/vespuq_real_lunar_L90.yaml \
        --seeds 0 1 2 --n-orbits 80 --out outputs/drift_horizon/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vesp.common.config import load_config
from vesp.uq.drift_horizon import run_drift_horizon


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="VESP-UQ force-risk vs drift multi-horizon (WP10).")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-orbits", type=int, default=80)
    parser.add_argument("--n-points", type=int, default=120)
    parser.add_argument("--out", default="outputs/drift_horizon/")
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
    n_orbits = min(args.n_orbits, 24) if args.quick else args.n_orbits

    run_drift_horizon(
        configs, seeds=seeds, n_orbits=n_orbits, n_points=args.n_points,
        out_dir=args.out, make_plots=not args.no_plots,
    )
    print(f"saved_drift_horizon: {Path(args.out) / 'drift_horizon.md'}")


if __name__ == "__main__":
    main()
