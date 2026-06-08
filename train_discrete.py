"""Convenience wrapper for the core single-run entry point (``vesp.training.train``).

This is the supported way to run ONE config. For multi-run experiments / sweeps and the
cross-experiment summary, use the experiment framework:
    python scripts/run_experiment_suite.py --experiment E0
"""

from vesp.training.train import main


if __name__ == "__main__":
    main()
