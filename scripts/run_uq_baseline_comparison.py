"""VESP-UQ vs a Gaussian-process UQ baseline (WP-D).

Fits the VESP-UQ plugin and an exact-GP residual UQ model on the same train/held split and compares
them on identical per-band calibration, trajectory-screening decision quality, and runtime.

    python scripts/run_uq_baseline_comparison.py \
        --configs configs/vespuq/vespuq_real_lunar.yaml configs/vespuq/vespuq_real_lunar_L90.yaml \
        --seeds 0 1 2 --out outputs/uq_baseline_comparison/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vesp.common.config import load_config
from vesp.uq.uq_baseline_comparison import run_uq_baseline_comparison


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="VESP-UQ vs GP UQ baseline comparison (WP-D).")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-orbits", type=int, default=None,
                        help="override each config's screening orbit count (caps screening cost)")
    parser.add_argument("--out", default="outputs/uq_baseline_comparison/")
    parser.add_argument("--quick", action="store_true")
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

    result = run_uq_baseline_comparison(configs, seeds=seeds, n_orbits=args.n_orbits, out_dir=args.out)
    for (band, model), a in sorted(result["decision_agg"].items()):
        auroc = a["auroc"]["mean"]
        print(f"{band} {model}: AUROC={auroc:.3f}" if auroc is not None else f"{band} {model}: AUROC=n/a")
    print(f"saved_uq_baseline_comparison: {Path(args.out) / 'uq_baseline_comparison.md'}")


if __name__ == "__main__":
    main()
