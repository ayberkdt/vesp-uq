"""Shared deprecation helper for superseded entry points.

The experiment-first framework (``vesp.experiments`` /
``scripts/run_experiment_suite.py``) supersedes the older one-off orchestration
runners (``run_ablation``, ``maxent_pareto``, ``feasibility``, the ``stage*`` shims).
Those runners still work — they are *deprecated, not removed* — so existing commands
keep functioning while users migrate. This helper emits a single consistent notice.
"""

from __future__ import annotations

import warnings


def warn_superseded(old: str, replacement: str) -> None:
    """Emit a ``DeprecationWarning`` and a visible banner for a superseded entry point.

    ``old`` is the deprecated command/module; ``replacement`` is the recommended
    experiment-framework command to use instead.
    """

    message = (
        f"{old} is deprecated and superseded by the experiment framework. "
        f"Use: {replacement}. It still runs for now (deprecate-and-delegate), "
        f"but new experiments should go through scripts/run_experiment_suite.py."
    )
    warnings.warn(message, DeprecationWarning, stacklevel=2)
    print(f"[DEPRECATED] {message}")
