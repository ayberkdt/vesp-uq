import torch

from experimental_vesp.data import ResidualGravityData, split_data
from experimental_vesp.kernels import evaluate_kernel
from experimental_vesp.models import DiscreteVESP
from experimental_vesp.sources import single_shell_sources
from experimental_vesp.train_discrete import solve_ridge
from experimental_vesp.evaluate import evaluate_model


def test_same_family_synthetic_recovery():
    dtype = torch.float64
    generator = torch.Generator().manual_seed(5)
    sources = single_shell_sources(0.72, 48, dtype=dtype)
    sigma_truth = torch.randn(sources.n_sources, generator=generator, dtype=dtype)
    directions = torch.randn(192, 3, generator=generator, dtype=dtype)
    directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
    x = directions * (1.05 + 0.4 * torch.rand(192, 1, generator=generator, dtype=dtype))
    out = evaluate_kernel(x, sources.positions, sources.weights * sigma_truth)
    data = ResidualGravityData(x, out.potential, out.acceleration)
    train, val = split_data(data, train_fraction=0.8, seed=5)
    model = DiscreteVESP(sources, dtype=dtype)
    config = {
        "kernel": {"source_chunk_size": 48},
        "loss": {"lambda_potential": 0.2, "lambda_acceleration": 1.0, "lambda_l2": 1e-10},
        "training": {"ridge_method": "augmented_lstsq", "column_normalize": True},
    }
    solve_ridge(model, train, config, device=torch.device("cpu"))
    metrics = evaluate_model(model, val, batch_size=128, source_chunk_size=48, device="cpu")
    assert metrics["acceleration_rmse"] < 1.0e-6

