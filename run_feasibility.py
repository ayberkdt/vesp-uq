"""DEPRECATED convenience wrapper for ``vesp.training.feasibility``.

Superseded by the experiment framework. Prefer:
    python scripts/run_experiment_suite.py --suite synthetic
    python scripts/run_experiment_suite.py --experiment E0

Still works (deprecate-and-delegate); the underlying runner prints a deprecation banner.
"""

from vesp.training.feasibility import main


if __name__ == "__main__":
    main()
