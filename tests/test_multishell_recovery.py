import torch

from experimental_vesp.synthetic import make_multishell_truth_case
from experimental_vesp.train import run_from_config


def _cfg(model, output_dir="outputs/test_multishell"):
    return {
        "seed": 9,
        "device": "cpu",
        "dtype": "float64",
        "output": {"output_dir": str(output_dir), "run_name": f"test_{model['type']}"},
        "data": {
            "type": "synthetic",
            "seed": 9,
            "synthetic_n_query": 192,
            "synthetic_truth_shell_radii": [0.5, 0.78, 0.86],
            "synthetic_n_truth_sources": [48, 64, 48],
        },
        "model": model,
        "kernel": {"source_chunk_size": 256},
        "solver": {"type": "ridge", "ridge_method": "augmented_lstsq", "column_normalize": True},
        "loss": {
            "use_potential": True,
            "use_acceleration": True,
            "lambda_potential": 0.2,
            "lambda_acceleration": 1.0,
            "lambda_l2": 1e-8,
            "lambda_moment": 1e-6,
            "lambda_dipole": 1.0,
            "shell_energy_weights": [],
        },
        "split": {"type": "random", "train_fraction": 0.8},
    }


def test_multishell_model_beats_single_on_multishell_truth(tmp_path):
    single = run_from_config(_cfg({"type": "discrete", "shell_alpha": 0.86, "n_source": 160}, tmp_path))
    multi = run_from_config(
        _cfg({"type": "multishell", "shell_alphas": [0.5, 0.78, 0.86], "n_sources_per_shell": [48, 64, 48]}, tmp_path)
    )
    assert multi["acceleration_rmse"] < single["acceleration_rmse"]
