"""VESP-UQ source-geometry vs low-altitude calibration sweep (E8 / calibration link).

Distinguishes whether the under-confident low-altitude band (z_std < 1) is caused by equivalent-
source placement or by the heteroscedastic noise law, by re-placing the sources (shell radii /
count / density / weights) at fixed total count and reading per-band held-out calibration.

    python scripts/run_geometry_calibration.py \
        --configs configs/vespuq/vespuq_real_lunar.yaml configs/vespuq/vespuq_real_lunar_L90.yaml \
        --seeds 0 1 2 3 4 --out outputs/geometry_calibration/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vesp.common.config import load_config
from vesp.uq.geometry_calibration import GEOMETRIES, run_geometry_calibration


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="VESP-UQ geometry x calibration sweep (E8).")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--geometries", nargs="+", default=None, help=f"subset of {sorted(GEOMETRIES)}")
    parser.add_argument("--out", default="outputs/geometry_calibration/")
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
    geometries = args.geometries or (["baseline", "surface_dense", "deep"] if args.quick else None)

    result = run_geometry_calibration(
        configs, seeds=seeds, geometries=geometries, out_dir=args.out, make_plots=not args.no_plots,
    )
    for band, v in result["verdict"].items():
        print(f"{band}: best={v['best_geometry']} low_z_std={v['best_low_z_std']:.3f} "
              f"improves={v['geometry_improves_low_calibration']}")
    print(f"saved_geometry_calibration: {Path(args.out) / 'geometry_calibration.md'}")


if __name__ == "__main__":
    main()
