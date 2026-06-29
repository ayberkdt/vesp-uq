"""VESP-UQ raw-vs-calibrated reliability study (WP7).

Reports empirical held-out force-error coverage (PICP at several nominal levels, per altitude band)
before and after a split-conformal scale, plus reliability diagrams. The conformal scale is fit on
a disjoint calibration half so the reported coverage is not the fit objective.

    python scripts/run_calibration_reliability.py \
        --configs configs/vespuq/vespuq_real_lunar.yaml configs/vespuq/vespuq_real_lunar_L90.yaml \
        --seeds 0 1 2 3 4 --out outputs/calibration_reliability/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vesp.common.config import load_config
from vesp.uq.calibration_reliability import run_reliability

_QUICK_SEEDS = (0, 1)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="VESP-UQ raw-vs-calibrated reliability (WP7).")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", default="outputs/calibration_reliability/")
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
    seeds = list(_QUICK_SEEDS) if args.quick else args.seeds

    run_reliability(configs, seeds=seeds, out_dir=args.out, make_plots=not args.no_plots)
    print(f"saved_calibration_reliability: {Path(args.out) / 'calibration_reliability.md'}")


if __name__ == "__main__":
    main()
