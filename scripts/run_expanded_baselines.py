"""VESP-UQ expanded baseline comparison (WP5).

Compares trivial single-feature baselines against altitude+uncertainty / altitude+OOD hybrids, a
learned ridge supervisor, and an altitude-bin empirical residual-RMSE lookup. Hybrid weights and
the ridge lambda are selected on a validation split; results are reported on a disjoint test split,
mean +/- std across seeds. Targets trajectory true FORCE-model error.

    python scripts/run_expanded_baselines.py \
        --configs configs/vespuq/vespuq_real_lunar.yaml configs/vespuq/vespuq_real_lunar_L90.yaml \
        --seeds 0 1 2 3 4 --out outputs/expanded_baselines/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vesp.common.config import load_config
from vesp.uq.ablation import run_expanded_baselines

_QUICK_SEEDS = (0, 1)
_QUICK_N_ORBITS = 200


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="VESP-UQ expanded baseline comparison (WP5).")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", default="outputs/expanded_baselines/")
    parser.add_argument("--primary-fraction", type=float, default=0.20)
    parser.add_argument("--n-orbits", type=int, default=None,
                        help="override uq.screening.n_orbits for all configs (speed vs power)")
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
        if args.quick:
            screen = cfg.setdefault("uq", {}).setdefault("screening", {})
            screen["n_orbits"] = min(int(screen.get("n_orbits", _QUICK_N_ORBITS)), _QUICK_N_ORBITS)
        elif args.n_orbits is not None:
            cfg.setdefault("uq", {}).setdefault("screening", {})["n_orbits"] = int(args.n_orbits)
        configs.append(cfg)
    seeds = list(_QUICK_SEEDS) if args.quick else args.seeds

    run_expanded_baselines(
        configs, seeds=seeds, out_dir=args.out,
        primary_fraction=args.primary_fraction, make_plots=not args.no_plots,
    )
    print(f"saved_expanded_baselines: {Path(args.out) / 'expanded_baselines.md'}")


if __name__ == "__main__":
    main()
