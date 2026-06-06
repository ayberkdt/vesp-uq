import math

import pytest
import torch

from vesp.extensions.entropy import (
    effective_source_entropy,
    entropy_regularization_loss,
    positive_negative_entropy,
    relative_entropy_to_uniform,
    shannon_entropy,
    shell_energy_balance_entropy,
    shell_energy_fractions,
)


def test_shannon_entropy_uniform_is_log_n():
    p = torch.full((8,), 1.0 / 8.0, dtype=torch.float64)
    assert abs(float(shannon_entropy(p)) - math.log(8)) < 1e-9


def test_effective_source_entropy_uniform_vs_concentrated():
    uniform = torch.ones(16, dtype=torch.float64)
    concentrated = torch.zeros(16, dtype=torch.float64)
    concentrated[0] = 1.0
    assert abs(float(effective_source_entropy(uniform)) - math.log(16)) < 1e-9
    assert float(effective_source_entropy(concentrated)) < 1e-6


def test_relative_entropy_to_uniform_is_zero_for_uniform():
    uniform = torch.ones(10, dtype=torch.float64)
    assert abs(float(relative_entropy_to_uniform(uniform))) < 1e-9
    concentrated = torch.zeros(10, dtype=torch.float64)
    concentrated[0] = 1.0
    assert float(relative_entropy_to_uniform(concentrated)) > 0.0


def test_positive_negative_entropy_counts_both_signs():
    sigma = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.float64)
    # two uniform groups of size 2 -> each log(2)
    assert abs(float(positive_negative_entropy(sigma)) - 2.0 * math.log(2)) < 1e-9


def test_shell_energy_balance_entropy_balanced_vs_collapsed():
    weights = torch.ones(4, dtype=torch.float64)
    shell_ids = torch.tensor([0, 0, 1, 1])
    balanced = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    collapsed = torch.tensor([1.0, 1.0, 1.0e-6, 1.0e-6], dtype=torch.float64)
    assert abs(float(shell_energy_balance_entropy(balanced, weights, shell_ids)) - math.log(2)) < 1e-6
    assert float(shell_energy_balance_entropy(collapsed, weights, shell_ids)) < 1e-3


def test_shell_energy_fractions_sum_to_one():
    weights = torch.ones(4, dtype=torch.float64)
    shell_ids = torch.tensor([0, 0, 1, 1])
    sigma = torch.tensor([2.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    fractions = shell_energy_fractions(sigma, weights, shell_ids)
    assert abs(float(fractions.sum()) - 1.0) < 1e-9


def test_entropy_regularization_loss_signs_and_modes():
    sigma = torch.tensor([1.0, 0.5, -0.5, -1.0], dtype=torch.float64)
    weights = torch.ones(4, dtype=torch.float64)
    shell_ids = torch.tensor([0, 0, 1, 1])
    # entropy modes return negative entropy (a loss to minimize)
    assert float(entropy_regularization_loss(sigma, weights, mode="abs")) < 0.0
    assert float(entropy_regularization_loss(sigma, weights, mode="positive_negative")) < 0.0
    # KL mode returns a non-negative divergence
    assert float(entropy_regularization_loss(sigma, weights, mode="relative_uniform")) >= 0.0
    # shell_balance requires shell_ids
    with pytest.raises(ValueError):
        entropy_regularization_loss(sigma, weights, mode="shell_balance")
    assert float(entropy_regularization_loss(sigma, weights, mode="shell_balance", shell_ids=shell_ids)) <= 0.0
    with pytest.raises(ValueError):
        entropy_regularization_loss(sigma, weights, mode="nonexistent")
