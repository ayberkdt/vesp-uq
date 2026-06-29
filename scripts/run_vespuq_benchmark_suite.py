"""Unified VESP-UQ journal benchmark suite (multi-seed / multi-fraction / multi-selector).

Fans a VESP-UQ config out over seeds x rerun-fractions x selectors and writes journal-ready
calibration robustness (Table A), ranking robustness (Table B), rerun-budget curves, a selector
ablation, and altitude-controlled incremental-value diagnostics -- all with a provenance manifest
(git commit + SHA-256 checksums). Everything targets trajectory true FORCE-model error.

Examples
--------
    python scripts/run_vespuq_benchmark_suite.py \
        --configs configs/vespuq/vespuq_real_lunar.yaml configs/vespuq/vespuq_real_lunar_L90.yaml \
        --seeds 0 1 2 3 4 \
        --rerun-fractions 0.05 0.10 0.15 0.20 0.30 \
        --selectors random min_altitude low_altitude_exposure uncertainty_only domain_support \
                    knn_p95 vespuq_supervisor \
        --out outputs/benchmark_suite/

    # fast wiring check on the smoke config
    python scripts/run_vespuq_benchmark_suite.py --configs configs/vespuq/vespuq_smoke.yaml --quick

    # list the run matrix without computing anything
    python scripts/run_vespuq_benchmark_suite.py --configs configs/vespuq/vespuq_smoke.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vesp.common.config import load_config
from vesp.uq.suite import (
    DEFAULT_FRACTIONS,
    DEFAULT_SELECTORS,
    plan_matrix,
    run_suite,
)

_QUICK_SEEDS = (0, 1)
_QUICK_N_ORBITS = 200


def _load_configs(paths, *, quick: bool) -> list[dict]:
    configs = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"config not found: {path}")
        cfg = load_config(str(p))
        cfg.setdefault("_config_path", str(p))
        if quick:
            screen = cfg.setdefault("uq", {}).setdefault("screening", {})
            current = int(screen.get("n_orbits", _QUICK_N_ORBITS))
            screen["n_orbits"] = min(current, _QUICK_N_ORBITS)
        configs.append(cfg)
    return configs


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="VESP-UQ journal benchmark suite.")
    parser.add_argument("--configs", nargs="+", required=True, help="one or more VESP-UQ YAML configs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--rerun-fractions", nargs="+", type=float, default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--selectors", nargs="+", default=list(DEFAULT_SELECTORS))
    parser.add_argument("--out", default="outputs/benchmark_suite/")
    parser.add_argument("--primary-fraction", type=float, default=0.20)
    parser.add_argument("--n-orbits", type=int, default=None,
                        help="override uq.screening.n_orbits for all configs (speed vs power)")
    parser.add_argument("--quick", action="store_true", help="2 seeds + capped n_orbits (wiring check)")
    parser.add_argument("--dry-run", action="store_true", help="print the run matrix and exit")
    parser.add_argument("--no-plots", action="store_true", help="skip PNG plots (CSV/MD only)")
    args = parser.parse_args(argv)

    seeds = list(_QUICK_SEEDS) if args.quick else args.seeds
    configs = _load_configs(args.configs, quick=args.quick)

    if args.dry_run:
        plan = plan_matrix(configs, seeds, args.rerun_fractions, args.selectors)
        print(json.dumps({"n_runs": len(plan), "matrix": plan}, indent=2))
        return

    result = run_suite(
        configs,
        seeds=seeds,
        rerun_fractions=args.rerun_fractions,
        selectors=args.selectors,
        out_dir=args.out,
        primary_fraction=args.primary_fraction,
        make_plots=not args.no_plots,
        n_orbits=args.n_orbits,
    )
    meta = result["meta"]
    if meta["missing_selectors"]:
        print(f"warning: selectors not available on this plugin/config: {meta['missing_selectors']}",
              file=sys.stderr)
    print(f"bands={meta['bands']} seeds={meta['seeds']} primary={meta['primary_fraction']:.0%} "
          f"git={meta['git_commit']}")
    print(f"saved_benchmark_suite: {Path(args.out) / 'benchmark_summary.md'}")


if __name__ == "__main__":
    main()
