"""Experiment-first framework for falsifying the MaxEnt-VESP idea.

This package turns the deterministic VESP solvers into a set of *named, repeatable
experiments* so the core scientific question can actually be tested:

    Does the MaxEnt (entropy-regularized) component add value beyond a classical
    ridge / Tikhonov equivalent-source fit?

It does not implement any new physics or solver. It is pure orchestration on top of
``vesp.training.train_discrete.run``:

- ``runner``    — load an experiment YAML, expand it into trials, run each trial and
                  collect its metrics + diagnostics.
- ``summarize`` — flatten a list of run metrics into the standard summary row and
                  write the combined CSV / Markdown / Pareto artifacts (+ optional plots).
- ``registry``  — the catalogue of the core experiments (E0–E5) and the questions
                  (Q1–Q6) each one answers.
- ``suites``    — named groups of experiment configs (``synthetic``, ``real_lunar``,
                  ``ci``) used by the runner scripts and the pre-results check.

See ``docs/SCIENTIFIC_CLAIMS.md`` for what these experiments may and may not be used
to claim.
"""

from __future__ import annotations

from vesp.experiments.registry import CORE_EXPERIMENTS, ExperimentInfo, experiment_info
from vesp.experiments.runner import (
    Trial,
    expand_trials,
    git_commit_hash,
    load_experiment_config,
    run_experiment,
)
from vesp.experiments.summarize import (
    SUMMARY_COLUMNS,
    summary_row,
    write_suite_artifacts,
)
from vesp.experiments.suites import SUITES, resolve_suite

__all__ = [
    "CORE_EXPERIMENTS",
    "ExperimentInfo",
    "experiment_info",
    "Trial",
    "expand_trials",
    "git_commit_hash",
    "load_experiment_config",
    "run_experiment",
    "SUMMARY_COLUMNS",
    "summary_row",
    "write_suite_artifacts",
    "SUITES",
    "resolve_suite",
]
