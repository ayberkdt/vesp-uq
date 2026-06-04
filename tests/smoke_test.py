from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from experimental_vesp.data import ResidualGravityData, split_data
from experimental_vesp.evaluate import evaluate_model
from experimental_vesp.kernels import evaluate_kernel
from experimental_vesp.sources import make_shell_sources
from experimental_vesp.models import DiscreteVESP
from experimental_vesp.train_discrete import solve_ridge


def main() -> None:
    dtype = torch.float64
    generator = torch.Generator().manual_seed(11)
    source_set = make_shell_sources([0.72], 64, dtype=dtype)
    sigma_truth = torch.randn(source_set.n_sources, generator=generator, dtype=dtype)
    sigma_truth = sigma_truth - sigma_truth.mean()

    directions = torch.randn((256, 3), generator=generator, dtype=dtype)
    directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
    radii = 1.05 + 0.45 * torch.rand((256, 1), generator=generator, dtype=dtype)
    positions = directions * radii

    out = evaluate_kernel(positions, source_set.positions, source_set.weights * sigma_truth)
    data = ResidualGravityData(positions=positions, potential=out.potential, acceleration=out.acceleration)
    train, val = split_data(data, train_fraction=0.8, seed=11)
    model = DiscreteVESP(source_set, dtype=dtype)
    config = {
        "kernel": {"source_chunk_size": 64, "softening": 0.0},
        "loss": {
            "lambda_potential": 0.1,
            "lambda_acceleration": 1.0,
            "lambda_l2": 1e-10,
            "lambda_moment": 0.0,
            "lambda_dipole": 1.0,
        },
    }
    solve_ridge(model, train, config, device=torch.device("cpu"))
    metrics = evaluate_model(model, val, batch_size=128, source_chunk_size=64, device="cpu")
    print("potential_rmse", metrics["potential_rmse"])
    print("acceleration_rmse", metrics["acceleration_rmse"])
    assert metrics["acceleration_rmse"] < 1e-3


if __name__ == "__main__":
    main()
