"""DEPRECATED convenience wrapper for ``vesp.training.run_ablation``.

Superseded by the experiment framework. Prefer:
    python scripts/run_experiment_suite.py --experiment E3
    python scripts/run_experiment_suite.py --config configs/experiments/<name>.yaml

Still works (deprecate-and-delegate); the underlying runner prints a deprecation banner.
"""

from vesp.training.run_ablation import main


if __name__ == "__main__":
    main()
