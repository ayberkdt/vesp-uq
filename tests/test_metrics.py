"""Tests for altitude band metric behavior, including the None-vs-empty contract."""

import torch

from vesp.core.metrics import altitude_band_errors


def _zeros(n):
    pos = torch.zeros((n, 3), dtype=torch.float64)
    return pos, torch.zeros((n, 3), dtype=torch.float64), torch.zeros((n, 3), dtype=torch.float64)


def test_none_bands_use_defaults():
    pos, pred, tgt = _zeros(2)
    pos[0, 0] = 1.05
    pos[1, 0] = 1.40
    result = altitude_band_errors(pos, pred, tgt, bands=None, warn_empty=False)
    # default low/mid/high bands are applied
    assert "low_altitude_acceleration_rmse" in result
    assert "high_altitude_acceleration_rmse" in result


def test_empty_dict_skips_band_computation():
    """An explicit empty dict means 'no bands' (used for single-band OOD subsets)."""

    pos, pred, tgt = _zeros(1)
    pos[0, 0] = 1.70
    result = altitude_band_errors(pos, pred, tgt, bands={}, warn_empty=True)
    assert not any(k.endswith("_altitude_acceleration_rmse") for k in result)
    assert result["low_to_high_error_ratio"] is None
