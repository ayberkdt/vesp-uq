import numpy as np
import pytest
import torch

from vesp.core.sources import SourceSet
from vesp.uq.linear_propagation import score_stm_dispersion
from vesp.uq.plugin import VESPUQPlugin


@pytest.fixture
def dummy_plugin():
    sources = SourceSet(
        positions=torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.6]]),
        weights=torch.tensor([1.0, 1.0]),
        shell_ids=torch.tensor([0, 0]),
        shell_radii=torch.tensor([0.5, 0.6])
    )
    plugin = VESPUQPlugin(sources=sources, eps=0.0, acceleration_sign=1.0)
    # mock posterior
    plugin.posterior = type("MockPosterior", (), {
        "mean": torch.tensor([0.1, -0.1], dtype=torch.float64),
        "cov": torch.eye(2, dtype=torch.float64) * 0.01,
    })
    return plugin


def test_score_stm_dispersion(dummy_plugin):
    initial_states = np.array([
        [1.2, 0.0, 0.0, 0.0, 0.9, 0.0],  # state 1
        [1.3, 0.0, 0.0, 0.0, 0.85, 0.0], # state 2
    ])

    scores = score_stm_dispersion(
        dummy_plugin,
        initial_states,
        duration_s=100.0,
        output_dt_s=10.0,
        device="cpu",
    )

    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (2,)
    assert scores.dtype == torch.float64
    assert bool(torch.all(torch.isfinite(scores)))
    assert bool(torch.all(scores >= 0.0))
