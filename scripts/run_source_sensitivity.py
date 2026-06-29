"""VESP-UQ source-geometry & regularization sensitivity (WP8).

Sweeps the equivalent-source count and the L2 regularization strength and reports held-out
force-error reconstruction + source-posterior conditioning, mean +/- std across seeds. Fast (no
trajectory scoring). The equivalent sources are a mathematical basis -- no density-recovery claim.

    python scripts/run_source_sensitivity.py \
        --configs configs/vespuq/vespuq_real_lunar.yaml configs/vespuq/vespuq_real_lunar_L90.yaml \
        --seeds 0 1 2 --out outputs/sensitivity/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vesp.common.config import load_config
from vesp.uq.sensitivity import run_sensitivity


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="VESP-UQ source/regularization sensitivity (WP8).")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-sources-targets", nargs="+", type=int, default=None,
                        help="explicit total source counts (default: base/2, base, base*2)")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[1.0, 10.0, 30.0, 100.0, 300.0])
    parser.add_argument("--out", default="outputs/sensitivity/")
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
    lambdas = [10.0, 100.0] if args.quick else args.lambdas

    run_sensitivity(
        configs, seeds=seeds, n_sources_targets=args.n_sources_targets, lambdas=lambdas,
        out_dir=args.out, make_plots=not args.no_plots,
    )
    print(f"saved_sensitivity: {Path(args.out) / 'source_geometry_sensitivity.md'}")


if __name__ == "__main__":
    main()
