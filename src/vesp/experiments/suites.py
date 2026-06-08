"""Named groups of experiment configs.

A *suite* is just an ordered list of experiment YAML paths that should be run and
summarized together. ``ci`` is the small, fast subset used by the pre-results check
and unit tests (no real-lunar data, quick sweeps).
"""

from __future__ import annotations


# Paths are relative to the repository root.
SUITES: dict[str, list[str]] = {
    "synthetic": [
        "configs/experiments/synthetic_exact_recovery.yaml",
        "configs/experiments/synthetic_shell_radius_mismatch.yaml",
        "configs/experiments/synthetic_multishell_truth.yaml",
        "configs/experiments/synthetic_l2_sweep.yaml",
        "configs/experiments/synthetic_entropy_pareto.yaml",
    ],
    "real_lunar": [
        "configs/experiments/real_lunar_ridge_baseline.yaml",
        "configs/experiments/real_lunar_l2_sweep.yaml",
        "configs/experiments/real_lunar_entropy_pareto.yaml",
    ],
    # Fast, data-free subset for CI and the pre-results check. Run with --quick.
    "ci": [
        "configs/experiments/synthetic_exact_recovery.yaml",
        "configs/experiments/synthetic_l2_sweep.yaml",
        "configs/experiments/synthetic_entropy_pareto.yaml",
    ],
}

SUITES["all"] = SUITES["synthetic"] + SUITES["real_lunar"]


def resolve_suite(name: str) -> list[str]:
    key = name.strip().lower()
    if key not in SUITES:
        raise KeyError(f"unknown suite {name!r}; known: {sorted(SUITES)}")
    return list(SUITES[key])
