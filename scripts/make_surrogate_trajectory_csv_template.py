"""Emit a schema-correct surrogate trajectory CSV template (Format B) for VESP-UQ.

When the ST-LRPS (or any) surrogate is ready, its trajectories are scored by VESP-UQ through the
external-CSV path (``uq.screening.trajectory_source: csv``). This script writes a small, valid
**Format B** example so the surrogate export can be matched to the exact column schema the loader
expects -- and (by default) loads it straight back through
:func:`vesp.uq.io.load_trajectory_csv` to prove the round-trip works before any real data exists.

The Format B columns (positions + surrogate/reference acceleration pairs) are::

    trajectory_id, t, x, y, z, ax_sur, ay_sur, az_sur, ax_ref, ay_ref, az_ref

with ``residual = reference - surrogate`` becoming the true force-model error the loader fits and
scores. To consume a real export, point a config at it::

    uq:
      screening:
        trajectory_source: csv
        trajectory_path: data/stlrps_trajectories.csv
        true_error_source: residual_csv

Columns ``x,y,z`` default to model/normalized units and ``a*_sur/a*_ref`` to model-normalized
acceleration unless the config supplies an explicit position/acceleration scale. This template is
illustrative geometry only -- it is not a physics propagation.

    python scripts/make_surrogate_trajectory_csv_template.py --out data/surrogate_template.csv \
        --n-trajectories 3 --n-points 8
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

HEADER = [
    "trajectory_id", "t", "x", "y", "z",
    "ax_sur", "ay_sur", "az_sur", "ax_ref", "ay_ref", "az_ref",
]


def _template_rows(n_trajectories: int, n_points: int, residual_scale: float) -> list[list]:
    """Illustrative circular trajectories with a small, altitude-growing surrogate residual.

    The reference acceleration is a toy central -1/r^2 pull along the inward radial; the surrogate is
    the reference minus a small residual that grows at low radius -- mirroring the real pattern where
    force-model error is largest near periapsis. This is geometry for schema validation, not physics.
    """

    rows: list[list] = []
    for tid in range(n_trajectories):
        radius = 1.2 + 0.25 * tid  # distinct altitude per trajectory
        for k in range(n_points):
            theta = 2.0 * math.pi * k / max(1, n_points)
            x, y, z = radius * math.cos(theta), radius * math.sin(theta), 0.0
            r = math.sqrt(x * x + y * y + z * z)
            # reference: central inward acceleration ~ -1/r^2 * r_hat
            mag = 1.0 / (r * r)
            ax_ref, ay_ref, az_ref = -mag * x / r, -mag * y / r, -mag * z / r
            # residual grows as radius shrinks (low-altitude force-model error)
            res = residual_scale / (r * r)
            ax_sur, ay_sur, az_sur = ax_ref - res * x / r, ay_ref - res * y / r, az_ref - res * z / r
            rows.append([tid, k, x, y, z, ax_sur, ay_sur, az_sur, ax_ref, ay_ref, az_ref])
    return rows


def _validate(path: Path) -> str:
    """Load the written CSV back through the real loader and return a one-line summary."""

    from vesp.uq.io import load_trajectory_csv

    ds = load_trajectory_csv(str(path))
    has_acc = ds.has_accelerations
    resid = "yes" if ds.residual_accelerations is not None else "no"
    return (f"loaded {ds.n_trajectories} trajectories / {ds.total_points} points "
            f"| acceleration pairs: {has_acc} | residual force error: {resid}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Write a VESP-UQ surrogate trajectory CSV template.")
    parser.add_argument("--out", default="data/surrogate_template.csv")
    parser.add_argument("--n-trajectories", type=int, default=3)
    parser.add_argument("--n-points", type=int, default=8)
    parser.add_argument("--residual-scale", type=float, default=2e-3,
                        help="magnitude of the synthetic surrogate-vs-reference force residual")
    parser.add_argument("--no-validate", action="store_true", help="skip loading the CSV back")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = _template_rows(args.n_trajectories, args.n_points, args.residual_scale)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        writer.writerows(rows)

    print(f"wrote_surrogate_template: {out} ({len(rows)} rows, Format B)")
    print(f"columns: {','.join(HEADER)}")
    if not args.no_validate:
        print(f"validation: {_validate(out)}")
    print("wire a real export with uq.screening.trajectory_source=csv, "
          "trajectory_path=<file>, true_error_source=residual_csv")


if __name__ == "__main__":
    main()
