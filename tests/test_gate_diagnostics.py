from __future__ import annotations

import math

import torch

from vesp.uq.gate_diagnostics import (
    decision_table_rows,
    flatten_trajectories_with_frames,
    measurement_c_curl_ratio,
    parse_benchmark_report,
)


def test_flatten_trajectories_with_frames_shapes():
    trajs = [
        torch.tensor([[1.1, 0.0, 0.0], [0.0, 1.2, 0.0], [-1.1, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, -1.2, 0.0], [1.3, 0.0, 0.0]], dtype=torch.float64),
    ]

    positions, frames = flatten_trajectories_with_frames(trajs, max_points=None)

    assert positions.shape == (5, 3)
    assert frames.shape == (5, 3, 3)
    eye = torch.eye(3, dtype=torch.float64).expand(5, 3, 3)
    assert torch.allclose(torch.einsum("nij,nkj->nik", frames, frames), eye, atol=1.0e-12)


def test_curl_ratio_zero_for_constant_field():
    grid = torch.tensor(
        [
            [1.1, 0.0, 0.0],
            [0.0, 1.1, 0.0],
            [0.0, 0.0, 1.1],
            [-1.1, 0.0, 0.0],
            [0.0, -1.1, 0.0],
            [0.0, 0.0, -1.1],
        ],
        dtype=torch.float64,
    )
    field = torch.ones_like(grid) * torch.tensor([0.2, -0.1, 0.05], dtype=torch.float64)

    result = measurement_c_curl_ratio(grid, field, max_points=6, k_neighbors=4)

    assert math.isclose(result["scaled_curl_ratio"], 0.0, abs_tol=1.0e-12)
    assert result["n_points_sampled"] == 6


def test_decision_table_uses_structured_sh_control_over_scattered_proxy():
    case = {
        "residual_field_kind": "emulated_sh_residual",
        "measurement_a": {"mean_abs_factor_corr": 0.95},
        "measurement_b": {"dominant_axis_max_fraction": 1.0},
        "measurement_c": {
            "scaled_curl_ratio": 1.2,
            "conservative_control_scaled_curl_ratio": 0.9,
            "excess_scaled_curl_ratio": 0.3,
            "structured_control_status": "pass",
            "structured_control_scaled_curl_ratio": 1.0e-12,
        },
        "calibration": {},
        "consistency_rows": [],
    }

    rows = decision_table_rows([case])
    helmholtz = next(row for row in rows if row["proposal"] == "helmholtz_non_conservative_extension")

    assert helmholtz["rationale_verified"] == "no - structured SH curl sanity check passed"
    assert "scattered Measurement C" in helmholtz["decision"]


def test_parse_benchmark_report_extracts_calibration_table(tmp_path):
    report = tmp_path / "report.md"
    report.write_text(
        "\n".join(
            [
                "dataset: `data/example.csv`",
                "sources: 128",
                "| band | mean_radius | rmse | mean_pred_std | mean_epi_std | z_std | picp_90 | ell_picp_90 | mean_d2 | nll |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                "| low | 1.1 | 2e-4 | 3e-4 | 4e-5 | 0.9 | 0.88 | 0.91 | 2.7 | -1.0 |",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_benchmark_report(report)

    assert parsed["dataset"] == "data\\example.csv"
    assert parsed["sources"] == 128
    assert parsed["calibration"]["low"]["z_std"] == 0.9
    assert parsed["calibration"]["low"]["ellipsoid_picp_90"] == 0.91
