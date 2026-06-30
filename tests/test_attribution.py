from __future__ import annotations

import math

import torch

from vesp.uq.attribution import masking_validation, summarize_log_attribution


def _profile():
    terms = {
        "log_expected_error": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        "log_altitude_weight": torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64),
        "log_ood_factor": torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64),
    }
    log_point_risk = terms["log_expected_error"] + terms["log_altitude_weight"] + terms["log_ood_factor"]
    return {
        "terms": terms,
        "log_point_risk": log_point_risk,
        "point_risk": torch.exp(log_point_risk),
        "weights": None,
    }


def test_log_attribution_is_exact_additive():
    summary = summarize_log_attribution(_profile())

    expected = 2.0
    altitude = 0.5
    ood = 0.1
    assert math.isclose(summary["log_expected_error"], expected)
    assert math.isclose(summary["log_altitude_weight"], altitude)
    assert math.isclose(summary["log_ood_factor"], ood)
    assert math.isclose(summary["log_point_risk_additive_sum"], expected + altitude + ood)
    assert summary["dominant_log_term"] == "log_expected_error"


def test_masking_validation_top_mask_can_beat_random():
    profile = _profile()
    true_error = torch.tensor([0.1, 0.2, 10.0], dtype=torch.float64)

    out = masking_validation(profile, true_error, mask_fraction=1 / 3, aggregator="max", seed=3)

    assert out["n_masked"] == 1
    assert out["true_error_after_top_mask"] == 0.2
    assert out["top_mask_drop_fraction"] > 0.9
    assert out["top_mask_beats_random"] is True
