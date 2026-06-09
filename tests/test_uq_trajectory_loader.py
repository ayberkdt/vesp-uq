"""Tests for the external trajectory / surrogate-output CSV loader."""

from __future__ import annotations

import pytest
import torch

from vesp.common.units import PositionScaler, UnitConfig
from vesp.uq.io import TrajectoryDataset, flatten_acceleration_pairs, load_trajectory_csv
from vesp.uq.physical_units import resolve_acceleration_scale


def _write(path, header, rows):
    lines = [",".join(header)]
    lines += [",".join(str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_positions_only_format(tmp_path):
    # two trajectories, variable length (3 and 2 points)
    rows = [
        [0, 0.0, 1.1, 0.0, 0.0],
        [0, 1.0, 1.2, 0.0, 0.0],
        [0, 2.0, 1.3, 0.0, 0.0],
        [1, 0.0, 1.5, 0.0, 0.0],
        [1, 1.0, 1.6, 0.0, 0.0],
    ]
    csv = _write(tmp_path / "a.csv", ["trajectory_id", "t", "x", "y", "z"], rows)
    ds = load_trajectory_csv(csv)
    assert isinstance(ds, TrajectoryDataset)
    assert ds.n_trajectories == 2
    assert ds.total_points == 5
    assert [t.shape[0] for t in ds.trajectories] == [3, 2]  # variable length supported
    assert ds.has_accelerations is False
    assert ds.residual_accelerations is None
    assert ds.times is not None and ds.times[0].tolist() == [0.0, 1.0, 2.0]
    assert ds.metadata["format"] == "A_positions_only"


def test_acceleration_pair_format_and_residual(tmp_path):
    header = [
        "trajectory_id", "t", "x", "y", "z",
        "ax_sur", "ay_sur", "az_sur", "ax_ref", "ay_ref", "az_ref",
    ]
    rows = [
        [7, 0.0, 1.1, 0.0, 0.0, 0.10, 0.20, 0.30, 0.11, 0.22, 0.33],
        [7, 1.0, 1.2, 0.0, 0.0, 1.00, 1.00, 1.00, 1.50, 2.00, 2.50],
    ]
    csv = _write(tmp_path / "b.csv", header, rows)
    ds = load_trajectory_csv(csv)
    assert ds.has_accelerations is True
    res = ds.residual_accelerations[0]  # reference - surrogate
    assert torch.allclose(res[0], torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64))
    assert torch.allclose(res[1], torch.tensor([0.50, 1.00, 1.50], dtype=torch.float64))
    # flatten for fitting
    pos, sur, ref = flatten_acceleration_pairs(ds)
    assert pos.shape == (2, 3) and sur.shape == (2, 3) and ref.shape == (2, 3)
    assert torch.allclose(ref - sur, torch.cat(ds.residual_accelerations, dim=0))


def test_missing_required_column_raises(tmp_path):
    # no position columns
    csv = _write(tmp_path / "bad.csv", ["trajectory_id", "t"], [[0, 0.0], [0, 1.0]])
    with pytest.raises(ValueError):
        load_trajectory_csv(csv)
    # no id column
    csv2 = _write(tmp_path / "bad2.csv", ["t", "x", "y", "z"], [[0.0, 1.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        load_trajectory_csv(csv2)


def test_half_acceleration_block_raises(tmp_path):
    header = ["trajectory_id", "t", "x", "y", "z", "ax_sur", "ay_sur", "az_sur"]  # surrogate only
    csv = _write(tmp_path / "half.csv", header, [[0, 0.0, 1.1, 0.0, 0.0, 0.1, 0.1, 0.1]])
    with pytest.raises(ValueError):
        load_trajectory_csv(csv)


def test_non_contiguous_ids_and_time_sorting(tmp_path):
    # ids 10 and 2 (non-contiguous, out of order); rows shuffled in time
    rows = [
        [10, 2.0, 1.30, 0.0, 0.0],
        [2, 1.0, 1.51, 0.0, 0.0],
        [10, 0.0, 1.10, 0.0, 0.0],
        [10, 1.0, 1.20, 0.0, 0.0],
        [2, 0.0, 1.50, 0.0, 0.0],
    ]
    csv = _write(tmp_path / "nc.csv", ["trajectory_id", "t", "x", "y", "z"], rows)
    ds = load_trajectory_csv(csv)
    # numeric sort -> id 2 before id 10
    assert ds.trajectory_ids == ["2", "10"]
    # trajectory 10 sorted by t -> x ascending 1.10, 1.20, 1.30
    traj10 = ds.trajectories[1]
    assert traj10[:, 0].tolist() == [1.10, 1.20, 1.30]


def test_positions_only_without_time_column(tmp_path):
    rows = [[0, 1.1, 0.0, 0.0], [0, 1.2, 0.0, 0.0]]
    csv = _write(tmp_path / "not.csv", ["trajectory_id", "x", "y", "z"], rows)
    ds = load_trajectory_csv(csv)
    assert ds.times is None
    assert ds.trajectories[0].shape == (2, 3)


def test_flatten_rejects_positions_only(tmp_path):
    csv = _write(tmp_path / "a.csv", ["trajectory_id", "t", "x", "y", "z"], [[0, 0.0, 1.1, 0.0, 0.0]])
    ds = load_trajectory_csv(csv)
    with pytest.raises(ValueError):
        flatten_acceleration_pairs(ds)


# --------------------------------------------------------------- unit handling

_ACCEL_HEADER = [
    "trajectory_id", "t", "x", "y", "z",
    "ax_sur", "ay_sur", "az_sur", "ax_ref", "ay_ref", "az_ref",
]


def _accel_csv(path):
    rows = [
        [0, 0.0, 1.1, 0.0, 0.0, 0.10, 0.20, 0.30, 0.11, 0.22, 0.33],
        [0, 1.0, 1.2, 0.0, 0.0, 1.00, 1.00, 1.00, 1.50, 2.00, 2.50],
    ]
    return _write(path, _ACCEL_HEADER, rows)


def test_default_units_verbatim_metadata_dict(tmp_path):
    ds = load_trajectory_csv(_accel_csv(tmp_path / "v.csv"))
    u = ds.metadata["units"]
    assert u["acceleration_converted_to_model"] is False
    assert u["acceleration_scale_m_s2"] is None
    assert u["positions_converted_to_model"] is False
    # residual unchanged from the verbatim reference - surrogate
    assert torch.allclose(ds.residual_accelerations[0][0], torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64))


def test_physical_acceleration_units_converted_to_model(tmp_path):
    # CSV accelerations in m/s^2; scale says 1 model unit == 1e-6 m/s^2 -> model = value / 1e-6
    scale = resolve_acceleration_scale(
        {"body": {"acceleration_units": "model_normalized_accel", "acceleration_scale_m_s2": 1.0e-6}}
    )
    ds = load_trajectory_csv(
        _accel_csv(tmp_path / "p.csv"), acceleration_scale=scale, acceleration_units="m/s^2"
    )
    u = ds.metadata["units"]
    assert u["acceleration_converted_to_model"] is True
    assert u["acceleration_scale_m_s2"] == pytest.approx(1.0e-6)
    # residual (ref - sur) in m/s^2 was [0.01,0.02,0.03] -> model units /1e-6
    assert torch.allclose(
        ds.residual_accelerations[0][0],
        torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64) / 1.0e-6,
    )


def test_physical_units_without_scale_raises(tmp_path):
    with pytest.raises(ValueError):
        load_trajectory_csv(_accel_csv(tmp_path / "ns.csv"), acceleration_units="m/s^2")


def test_unsupported_acceleration_units_raises(tmp_path):
    with pytest.raises(ValueError):
        load_trajectory_csv(_accel_csv(tmp_path / "bad.csv"), acceleration_units="furlongs/s^2")


def test_position_scaler_normalizes_positions(tmp_path):
    rows = [[0, 0.0, 1738.0, 0.0, 0.0], [0, 1.0, 3476.0, 0.0, 0.0]]  # km positions
    csv = _write(tmp_path / "km.csv", ["trajectory_id", "t", "x", "y", "z"], rows)
    scaler = PositionScaler(UnitConfig(R_body=1738.0, normalize_positions=True, position_units="km"))
    ds = load_trajectory_csv(csv, position_scaler=scaler)
    # 1738 km / 1738 -> 1.0 body radii, 3476 -> 2.0
    assert ds.trajectories[0][:, 0].tolist() == [1.0, 2.0]
    assert ds.metadata["units"]["positions_converted_to_model"] is True
