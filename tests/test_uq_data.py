"""Tests for the surrogate-agnostic VESP-UQ data interface."""

from __future__ import annotations

import pytest
import torch

from vesp.uq.data import (
    UQSamples,
    load_uq_samples_from_csv,
    make_synthetic_uq_samples,
    split_uq_samples,
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
