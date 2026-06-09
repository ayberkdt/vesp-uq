"""Load external surrogate trajectory ensembles from CSV (Format A or B).

``load_trajectory_csv`` reads a flat CSV of trajectory points, groups them by ``trajectory_id``,
sorts each trajectory by ``t`` (when present), and returns a :class:`TrajectoryDataset`. When the
CSV carries surrogate/reference acceleration pairs (Format B) it also exposes the residual force
error ``reference - surrogate`` per trajectory and via :func:`flatten_acceleration_pairs` for
``VESPUQPlugin.fit``.

Units default to verbatim (the historical behavior). When explicit metadata is supplied, the loader
can convert into the model's working units before scoring/fitting -- positions via a
:class:`~vesp.common.units.PositionScaler`, and accelerations from a physical unit
(``m/s^2`` ...) into model-normalized units via an
:class:`~vesp.uq.physical_units.AccelerationScale`. Physical acceleration units without an available
scale raise a clear error (never a silent normalized fallback); see :mod:`vesp.uq.physical_units`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from vesp.common.units import PositionScaler
from vesp.uq.io.trajectory_schema import (
    _ID_ALIASES,
    _POS_ALIASES,
    _REF_ALIASES,
    _SUR_ALIASES,
    _TIME_ALIASES,
    POSITION_COLUMNS,
    REFERENCE_COLUMNS,
    SURROGATE_COLUMNS,
    TrajectoryDataset,
)
from vesp.uq.physical_units import (
    MODEL_UNITS,
    AccelerationScale,
    acceleration_to_model_units,
    is_physical_units,
    normalize_units,
)


def _first_match(fields: set[str], options) -> str | None:
    return next((o for o in options if o in fields), None)


def _resolve_group(fields: set[str], aliases: dict[str, tuple[str, ...]]) -> dict[str, str] | None:
    """Return logical->actual column map if *all* logical names resolve, else ``None``."""

    selected: dict[str, str] = {}
    for logical, opts in aliases.items():
        match = _first_match(fields, opts)
        if match is None:
            return None
        selected[logical] = match
    return selected


def _id_sort_key(ids):
    """Sort trajectory ids numerically when all parse as numbers, else lexicographically."""

    try:
        numeric = {i: float(i) for i in ids}
        return lambda i: (0, numeric[i])
    except (TypeError, ValueError):
        return lambda i: (1, str(i))


def load_trajectory_csv(
    path: str | Path,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    position_scaler: PositionScaler | None = None,
    acceleration_scale: AccelerationScale | None = None,
    acceleration_units: str = MODEL_UNITS,
) -> TrajectoryDataset:
    """Load a trajectory ensemble CSV into a :class:`TrajectoryDataset`.

    Required columns: ``trajectory_id, x, y, z`` (``t`` is optional but recommended -- it is used
    to sort each trajectory and populate ``times``). Acceleration-pair columns
    ``ax_sur, ay_sur, az_sur, ax_ref, ay_ref, az_ref`` enable residual-force-error fitting.

    Behavior:
      - missing required columns -> ``ValueError``;
      - non-contiguous / string trajectory ids are supported;
      - variable point counts per trajectory are supported;
      - rows are grouped by ``trajectory_id`` and sorted by ``t`` (stable when ``t`` is absent).

    Units (default: verbatim, the historical behavior):
      - ``position_scaler`` (optional) maps CSV positions into model-normalized coordinates;
      - ``acceleration_units`` declares the CSV acceleration units. If physical (``m/s^2`` ...) it is
        converted into model-normalized units via ``acceleration_scale`` (required and physical in
        that case, else ``ValueError``). ``model_normalized_accel`` (default) leaves accelerations
        verbatim.
    """

    accel_units = normalize_units(acceleration_units)
    convert_accel = accel_units != MODEL_UNITS

    path = Path(path)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"trajectory CSV has no header: {path}")
        fields = set(reader.fieldnames)

        id_col = _first_match(fields, _ID_ALIASES)
        if id_col is None:
            raise ValueError(
                f"trajectory CSV {path} is missing the trajectory id column "
                f"(one of {_ID_ALIASES}); found {sorted(fields)}"
            )
        pos_cols = _resolve_group(fields, _POS_ALIASES)
        if pos_cols is None:
            raise ValueError(
                f"trajectory CSV {path} is missing position columns x, y, z; found {sorted(fields)}"
            )
        time_col = _first_match(fields, _TIME_ALIASES)
        sur_cols = _resolve_group(fields, _SUR_ALIASES)
        ref_cols = _resolve_group(fields, _REF_ALIASES)
        has_accel = sur_cols is not None and ref_cols is not None
        if (sur_cols is None) != (ref_cols is None):
            raise ValueError(
                f"trajectory CSV {path} has only one of the surrogate / reference acceleration "
                f"blocks; both {SURROGATE_COLUMNS} and {REFERENCE_COLUMNS} are required for "
                f"acceleration-pair (Format B) mode"
            )
        if convert_accel and has_accel and (acceleration_scale is None or not acceleration_scale.physical):
            raise ValueError(
                f"trajectory CSV {path} declares physical acceleration units "
                f"{acceleration_units!r} but no physical acceleration_scale was supplied; pass an "
                "AccelerationScale (e.g. from body.acceleration_scale_m_s2) or use "
                "model_normalized_accel"
            )

        # Group rows by id, preserving first-appearance order; keep an enumeration index so a
        # missing time column still yields a stable within-trajectory order.
        groups: dict[str, list[dict]] = {}
        order: list[str] = []
        for idx, row in enumerate(reader):
            tid = row[id_col]
            if tid not in groups:
                groups[tid] = []
                order.append(tid)
            rec = {
                "row": idx,
                "pos": [float(row[pos_cols[c]]) for c in ("x", "y", "z")],
            }
            if time_col is not None:
                rec["t"] = float(row[time_col])
            if has_accel:
                rec["sur"] = [float(row[sur_cols[c]]) for c in SURROGATE_COLUMNS]
                rec["ref"] = [float(row[ref_cols[c]]) for c in REFERENCE_COLUMNS]
            groups[tid].append(rec)

    if not groups:
        raise ValueError(f"trajectory CSV has no data rows: {path}")

    sorted_ids = sorted(order, key=_id_sort_key(order))
    trajectories: list[torch.Tensor] = []
    times: list[torch.Tensor] = []
    sur_list: list[torch.Tensor] = []
    ref_list: list[torch.Tensor] = []
    res_list: list[torch.Tensor] = []

    def _accel_to_model(values: torch.Tensor) -> torch.Tensor:
        if not convert_accel:
            return values
        return acceleration_to_model_units(values, acceleration_scale, source_units=accel_units).to(
            dtype=dtype, device=device
        )

    for tid in sorted_ids:
        recs = groups[tid]
        recs.sort(key=lambda r: (r.get("t", 0.0), r["row"]))  # by time, stable on row index
        pos = torch.tensor([r["pos"] for r in recs], dtype=dtype, device=device)
        if position_scaler is not None:
            pos = position_scaler.to_model_positions(pos)
        trajectories.append(pos)
        if time_col is not None:
            times.append(torch.tensor([r["t"] for r in recs], dtype=dtype, device=device))
        if has_accel:
            sur = _accel_to_model(torch.tensor([r["sur"] for r in recs], dtype=dtype, device=device))
            ref = _accel_to_model(torch.tensor([r["ref"] for r in recs], dtype=dtype, device=device))
            sur_list.append(sur)
            ref_list.append(ref)
            res_list.append(ref - sur)

    positions_converted = (
        position_scaler is not None
        and position_scaler.units.normalize_positions
        and position_scaler.units.position_units != "normalized"
    )
    metadata = {
        "path": str(path),
        "format": "B_acceleration_pairs" if has_accel else "A_positions_only",
        "has_time": time_col is not None,
        "n_trajectories": len(trajectories),
        # Explicit per-file unit handling. Defaults reproduce the historical verbatim behavior; a
        # supplied PositionScaler / physical AccelerationScale converts into the model's units.
        "units": {
            "csv_acceleration_units": accel_units,
            "acceleration_converted_to_model": bool(convert_accel and has_accel),
            "acceleration_scale_m_s2": (
                acceleration_scale.scale_m_s2 if (convert_accel and has_accel) else None
            ),
            "positions_converted_to_model": bool(positions_converted),
            "position_units": (
                position_scaler.units.position_units if position_scaler is not None else "as_supplied"
            ),
        },
    }
    return TrajectoryDataset(
        trajectories=trajectories,
        trajectory_ids=sorted_ids,
        times=times if time_col is not None else None,
        surrogate_accelerations=sur_list if has_accel else None,
        reference_accelerations=ref_list if has_accel else None,
        residual_accelerations=res_list if has_accel else None,
        metadata=metadata,
    )


def flatten_acceleration_pairs(dataset: TrajectoryDataset):
    """Flatten a Format-B dataset to ``(positions, surrogate_acc, reference_acc)`` for fitting.

    Concatenates all trajectory points into ``(M, 3)`` tensors suitable for ``VESPUQPlugin.fit``.
    Raises ``ValueError`` if the dataset has no acceleration pairs.
    """

    if not dataset.has_accelerations:
        raise ValueError(
            "dataset has no surrogate/reference acceleration pairs (positions-only Format A); "
            "load a Format-B CSV to fit residual-force error"
        )
    positions = torch.cat(dataset.trajectories, dim=0)
    surrogate = torch.cat(dataset.surrogate_accelerations, dim=0)
    reference = torch.cat(dataset.reference_accelerations, dim=0)
    return positions, surrogate, reference
