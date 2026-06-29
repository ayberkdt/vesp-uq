"""VESP-UQ learned supervisor (Design A): validation-tuned supervisor exponents.

Keeps the supervisor's physical multiplicative form but learns three non-negative exponents on a
validation split (`point_risk = expected_error^b1 * rel_alt^b2 * (1 + b3 * domain_risk)`), then
compares the hand-set supervisor (beta=(1,1,1)) against the learned one on a disjoint test split.
Default behavior is unchanged; this is an offline study of the supervisor's weighting headroom.

    python scripts/run_learned_supervisor.py \
        --configs configs/vespuq/vespuq_real_lunar.yaml configs/vespuq/vespuq_real_lunar_L90.yaml \
        --seeds 0 1 2 3 4 --n-orbits 3000 --out outputs/learned_supervisor/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vesp.common.config import load_config
from vesp.uq.ablation import run_learned_supervisor

_QUICK_SEEDS = (0, 1)
_QUICK_N_ORBITS = 200


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="VESP-UQ learned supervisor (Design A).")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", default="outputs/learned_supervisor/")
    parser.add_argument("--primary-fraction", type=float, default=0.20)
    parser.add_argument("--n-orbits", type=int, default=None,
                        help="override uq.screening.n_orbits for all configs (speed vs power)")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)

    configs = []
    for path in args.configs:
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"config not found: {path}")
        cfg = load_config(str(p))
        cfg.setdefault("_config_path", str(p))
        if args.quick:
            cfg.setdefault("uq", {}).setdefault("screening", {})["n_orbits"] = _QUICK_N_ORBITS
        elif args.n_orbits is not None:
            cfg.setdefault("uq", {}).setdefault("screening", {})["n_orbits"] = int(args.n_orbits)
        configs.append(cfg)
    seeds = list(_QUICK_SEEDS) if args.quick else args.seeds

    result = run_learned_supervisor(
        configs, seeds=seeds, out_dir=args.out, primary_fraction=args.primary_fraction,
    )
    for band, betas in result["betas_by_band"].items():
        print(f"{band}: learned betas (per seed) = {betas}")
    print(f"saved_learned_supervisor: {Path(args.out) / 'learned_supervisor.md'}")


if __name__ == "__main__":
    main()
