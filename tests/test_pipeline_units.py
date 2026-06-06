"""End-to-end regression test for acceleration-unit consistency in the solve.

This guards the exact failure that was previously silent: a real CSV stores a
physical ``km/s^2`` acceleration while the model predicts a normalized gradient,
so without the loader conversion the joint potential+acceleration ridge solve is
internally inconsistent by a factor of ``R_body`` and abandons the potential fit.
"""

import csv
import json

import torch

from vesp.common.units import UnitConfig
from vesp.core.kernels import evaluate_kernel
from vesp.core.operators import build_joint_operator
from vesp.core.solvers import RidgeSolveConfig, solve_discrete_ridge
from vesp.core.sources import make_shell_sources
from vesp.data.dataset import load_csv_dataset

R_BODY_KM = 1738.0


def _write_physical_csv(tmp_path):
    """Build a self-consistent normalized field, export acceleration as physical."""

    sources = make_shell_sources([0.8], 96, body_radius=1.0, dtype=torch.float64)
    gen = torch.Generator().manual_seed(0)
    sigma_true = torch.randn(sources.n_sources, generator=gen, dtype=torch.float64)
    sigma_true = sigma_true - sigma_true.mean()

    directions = torch.randn((256, 3), generator=gen, dtype=torch.float64)
    directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
    radii = 1.1 + 0.4 * torch.rand((256, 1), generator=gen, dtype=torch.float64)
    query = directions * radii

    out = evaluate_kernel(query, sources.positions, sources.weights * sigma_true)
    potential = out.potential  # [N, 1], normalized coordinates
    accel_normalized_gradient = out.acceleration  # [N, 3], dU/d(x / R_body)
    accel_physical = accel_normalized_gradient / R_BODY_KM  # km/s^2

    path = tmp_path / "phys.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z", "Delta U", "Delta a_x", "Delta a_y", "Delta a_z"])
        for x, u, a in zip(query, potential, accel_physical):
            w.writerow([float(x[0]), float(x[1]), float(x[2]), float(u[0]), float(a[0]), float(a[1]), float(a[2])])
    metadata = {
        "position_units": "normalized",
        "potential_units": "km^2/s^2",
        "acceleration_units": "km/s^2",
        "acceleration_output": "physical",
        "R_body": R_BODY_KM,
        "R_body_units": "km",
    }
    path.with_suffix(path.suffix + ".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return path, sources, sigma_true, potential, accel_normalized_gradient


def test_physical_csv_loads_back_to_normalized_gradient(tmp_path):
    path, _, _, _, accel_norm = _write_physical_csv(tmp_path)
    data = load_csv_dataset(
        path,
        dtype=torch.float64,
        unit_config=UnitConfig(R_body=1.0, normalize_positions=True, position_units="normalized"),
    )
    assert torch.allclose(data.acceleration, accel_norm, rtol=1e-9, atol=1e-12)


def test_joint_solve_recovers_potential_from_physical_csv(tmp_path):
    """With the loader conversion, the joint solve fits BOTH potential and acceleration."""

    path, sources, _, _, _ = _write_physical_csv(tmp_path)
    data = load_csv_dataset(
        path,
        dtype=torch.float64,
        unit_config=UnitConfig(R_body=1.0, normalize_positions=True, position_units="normalized"),
    )
    bundle = build_joint_operator(
        data.positions,
        sources,
        potential=data.potential,
        acceleration=data.acceleration,
        use_potential=True,
        use_acceleration=True,
        sign=1.0,
    )
    sigma = solve_discrete_ridge(
        bundle.operator,
        bundle.target,
        sources.positions,
        sources.weights,
        sources.shell_ids,
        RidgeSolveConfig(method="augmented_lstsq", lambda_l2=1e-10, column_normalize=True, lambda_moment=0.0),
    )
    out = evaluate_kernel(data.positions, sources.positions, sources.weights * sigma)
    pot_rel = float(
        torch.sqrt(torch.mean((out.potential - data.potential) ** 2))
        / torch.sqrt(torch.mean(data.potential ** 2))
    )
    acc_rel = float(
        torch.sqrt(torch.mean(torch.sum((out.acceleration - data.acceleration) ** 2, dim=-1)))
        / torch.sqrt(torch.mean(torch.sum(data.acceleration ** 2, dim=-1)))
    )
    # Both must be tiny; pre-fix the potential error was ~100%.
    assert pot_rel < 1e-2
    assert acc_rel < 1e-2
