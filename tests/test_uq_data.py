"""Tests for the surrogate-agnostic VESP-UQ data interface."""

from __future__ import annotations

import pytest
import torch

from vesp.uq.data import (
    UQSamples,
    load_uq_samples_from_csv,
    make_synthetic_uq_samples,
    split_uq_samples,
    split_uq_samples_by_config,
    validate_uq_samples,
)


def _write(path, header, rows):
    lines = [",".join(header)] + [",".join(str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_direct_error_csv(tmp_path):
    p = _write(
        tmp_path / "err.csv",
        ["x", "y", "z", "ax_err", "ay_err", "az_err"],
        [[1.1, 0.0, 0.0, 0.3, -0.2, 0.1], [0.0, 1.2, 0.0, 0.0, 0.5, -0.4]],
    )
    s = load_uq_samples_from_csv(p)
    assert s.n == 2
    assert s.metadata["mode"] == "error"
    assert torch.allclose(s.error[0], torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64))
    # direct-error mode: surrogate is zero, reference equals error
    assert torch.allclose(s.surrogate, torch.zeros_like(s.surrogate))
    assert torch.allclose(s.reference, s.error)


def test_load_reference_surrogate_csv_computes_difference(tmp_path):
    p = _write(
        tmp_path / "rs.csv",
        ["x", "y", "z", "ax_ref", "ay_ref", "az_ref", "ax_sur", "ay_sur", "az_sur"],
        [[1.1, 0.0, 0.0, 1.0, 2.0, 3.0, 0.4, 0.5, 0.6]],
    )
    s = load_uq_samples_from_csv(p)
    assert s.metadata["mode"] == "reference_surrogate"
    # error = reference - surrogate
    assert torch.allclose(s.error[0], torch.tensor([0.6, 1.5, 2.4], dtype=torch.float64))


def test_legacy_residual_columns_load_as_error(tmp_path):
    p = _write(
        tmp_path / "resid.csv",
        ["x", "y", "z", "Delta a_x", "Delta a_y", "Delta a_z"],
        [[1.1, 0.0, 0.0, 0.1, 0.2, 0.3]],
    )
    s = load_uq_samples_from_csv(p)
    assert s.metadata["mode"] == "error"
    assert torch.allclose(s.error[0], torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64))


def test_missing_columns_raise_clear_error(tmp_path):
    p = _write(tmp_path / "bad.csv", ["x", "y", "z", "foo"], [[1.0, 0.0, 0.0, 9.0]])
    with pytest.raises(ValueError, match="error columns"):
        load_uq_samples_from_csv(p, mode="error")
    with pytest.raises(ValueError, match="reference columns"):
        load_uq_samples_from_csv(p, mode="reference_surrogate")


def test_missing_position_columns_raise(tmp_path):
    p = _write(tmp_path / "nopos.csv", ["a", "ax_err", "ay_err", "az_err"], [[1, 2, 3, 4]])
    with pytest.raises(ValueError, match="position columns"):
        load_uq_samples_from_csv(p)


def test_split_is_deterministic_with_seed():
    s = make_synthetic_uq_samples(n=100, seed=3)
    a1, b1 = split_uq_samples(s, train_fraction=0.7, seed=11)
    a2, b2 = split_uq_samples(s, train_fraction=0.7, seed=11)
    a3, _ = split_uq_samples(s, train_fraction=0.7, seed=12)
    assert a1.n == 70 and b1.n == 30
    assert torch.allclose(a1.positions, a2.positions)  # same seed -> same split
    assert not torch.allclose(a1.positions, a3.positions)  # different seed -> different split


def test_validate_rejects_bad_shapes():
    with pytest.raises(ValueError):
        validate_uq_samples(UQSamples(positions=torch.zeros(4, 2), error=torch.zeros(4, 3)))
    with pytest.raises(ValueError):
        validate_uq_samples(UQSamples(positions=torch.zeros(4, 3), error=torch.zeros(3, 3)))


# ------------------------------------------------------------------ spatial splits (R2WP-7)
def test_altitude_disjoint_split_separates_radius_bands():
    s = make_synthetic_uq_samples(n=600, seed=5)
    train, held, info = split_uq_samples_by_config(
        s, {"method": "altitude_disjoint", "held_quantile": [0.0, 0.3], "buffer": 0.02}, seed=0
    )
    assert info["method"] == "altitude_disjoint"
    # every held radius sits below every train radius, with at least the buffer between them
    assert float(held.radius.max()) + 0.02 <= float(train.radius.min()) + 1e-12
    assert info["n_train"] == train.n and info["n_held"] == held.n
    assert train.n + held.n + info["n_dropped_buffer"] == s.n


def test_angular_block_split_holds_out_whole_cells():
    s = make_synthetic_uq_samples(n=800, seed=6)
    train, held, info = split_uq_samples_by_config(
        s, {"method": "angular_block", "n_blocks": 10, "buffer_deg": 5.0}, train_fraction=0.7, seed=1
    )
    assert info["method"] == "angular_block"
    assert train.n > 0 and held.n > 0
    # the angular buffer must hold: no train direction within 5 deg of a held direction
    tr = train.positions / torch.linalg.norm(train.positions, dim=-1, keepdim=True)
    hd = held.positions / torch.linalg.norm(held.positions, dim=-1, keepdim=True)
    max_cos = (tr @ hd.transpose(0, 1)).max()
    assert float(max_cos) < torch.cos(torch.deg2rad(torch.tensor(5.0))) + 1e-9


def test_trajectory_group_split_never_splits_a_group():
    s = make_synthetic_uq_samples(n=90, seed=7)
    groups = [f"traj{i % 9}" for i in range(90)]
    train, held, info = split_uq_samples_by_config(
        s, {"method": "trajectory_group"}, train_fraction=0.7, seed=2, groups=groups
    )
    assert info["n_train_groups"] + info["n_held_groups"] == 9
    # reconstruct group membership by matching positions back to the parent set
    def _labels(part):
        out = []
        for p in part.positions:
            j = int(torch.argmin(torch.linalg.norm(s.positions - p, dim=-1)))
            out.append(groups[j])
        return set(out)

    assert _labels(train).isdisjoint(_labels(held))


def test_split_by_config_random_default_and_failclosed():
    s = make_synthetic_uq_samples(n=100, seed=8)
    train, held, info = split_uq_samples_by_config(s, None, train_fraction=0.7, seed=3)
    assert info["method"] == "random" and train.n == 70 and held.n == 30
    with pytest.raises(ValueError, match="unknown split method"):
        split_uq_samples_by_config(s, {"method": "altitute_disjoint"})
    with pytest.raises(ValueError, match="key"):
        split_uq_samples_by_config(s, {"method": "angular_block", "n_block": 8})
    with pytest.raises(ValueError, match="group labels"):
        split_uq_samples_by_config(s, {"method": "trajectory_group"})


def test_group_column_loads_into_metadata(tmp_path):
    p = _write(
        tmp_path / "grp.csv",
        ["x", "y", "z", "ax_err", "ay_err", "az_err", "traj_id"],
        [[1.1, 0.0, 0.0, 0.1, 0.2, 0.3, "a"], [0.0, 1.2, 0.0, 0.2, 0.1, 0.0, "b"]],
    )
    s = load_uq_samples_from_csv(p)
    assert s.metadata["groups"] == ["a", "b"]
