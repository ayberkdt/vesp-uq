"""Load, expand and run experiment YAML configs.

Experiment config schema
------------------------

An experiment file is a YAML mapping with:

``experiment``
    ``name`` (required), ``kind`` (``single`` | ``sweep``, optional/informational)
    and a free-text ``description`` / ``question`` for the report.
``base_config``
    A full VESP config (the same schema consumed by ``vesp.training.train``). It is
    merged onto the package defaults before every trial.
``sweep`` (optional)
    Either:

    * ``{"coupled_grid": [{"paths": [...], "values": [...]}, ...]}`` — a cartesian
      product over axes, where every path in an axis receives the *same* value for a
      given trial. The coupling is what lets the L2 sweep keep ``loss.lambda_l2`` and
      ``solver.lambda_l2`` in lock-step, instead of taking their cartesian product.
    * an explicit list ``[{"name": ..., "set": {"<dotted.path>": value, ...}}, ...]``.

If ``sweep`` is absent the experiment is a single run of ``base_config``.

This module is deliberately thin: each trial is handed to
``vesp.training.train_discrete.run`` which already performs the solve, evaluation,
diagnostics, acceptability screening and per-run artifact writing.
"""

from __future__ import annotations

import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from vesp.common.config import merge_defaults
from vesp.core.models import MultiShellDiscreteVESP
from vesp.training.train_discrete import run


_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Trial:
    """A single concrete run: a unique name plus a fully merged config."""

    name: str
    config: dict


@dataclass
class ExperimentRunResult:
    """Outcome of running one experiment (which may contain many trials)."""

    name: str
    config_path: str | None
    rows: list[dict] = field(default_factory=list)
    metrics: list[dict] = field(default_factory=list)


def git_commit_hash() -> str | None:
    """Return the current git commit hash, or ``None`` if unavailable."""

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    commit = out.stdout.strip()
    return commit or None


def load_experiment_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"experiment config {path} must be a YAML mapping")
    if "base_config" not in cfg or not isinstance(cfg["base_config"], dict):
        raise ValueError(f"experiment config {path} must contain a 'base_config' mapping")
    experiment = cfg.get("experiment", {})
    if not isinstance(experiment, dict) or not experiment.get("name"):
        raise ValueError(f"experiment config {path} must contain experiment.name")
    return cfg


def _set_nested(config: dict, dotted_path: str, value: Any) -> None:
    node = config
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def _fmt_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "-".join(_fmt_value(v) for v in value)
    if isinstance(value, float):
        return f"{value:g}".replace(".", "p").replace("-", "m").replace("+", "")
    return str(value).replace(".", "p")


def _coupled_axes(sweep: dict) -> list[dict]:
    axes = sweep["coupled_grid"]
    if not isinstance(axes, list) or not axes:
        raise ValueError("sweep.coupled_grid must be a non-empty list of axes")
    normalized: list[dict] = []
    for axis in axes:
        paths = axis.get("paths")
        if isinstance(paths, str):
            paths = [paths]
        values = axis.get("values")
        if not paths or values is None:
            raise ValueError("each coupled_grid axis needs 'paths' and 'values'")
        if not isinstance(values, list):
            values = [values]
        normalized.append({"paths": list(paths), "values": list(values)})
    return normalized


def expand_trials(experiment_cfg: dict, *, quick: bool = False, quick_per_axis: int = 3) -> list[Trial]:
    """Expand an experiment config into concrete trials.

    ``quick`` subsamples each sweep axis to at most ``quick_per_axis`` values
    (first / middle / last) so CI and the pre-results check can exercise the full
    code path without running the entire grid.
    """

    base = merge_defaults(deepcopy(experiment_cfg["base_config"]))
    name = str(experiment_cfg["experiment"]["name"])
    sweep = experiment_cfg.get("sweep")

    if not sweep:
        return [Trial(name=name, config=base)]

    # Explicit list of trials.
    if isinstance(sweep, list):
        trials: list[Trial] = []
        chosen = _subsample(sweep, quick_per_axis) if quick else sweep
        for idx, spec in enumerate(chosen):
            cfg = deepcopy(base)
            overrides = spec.get("set", {})
            for dotted, value in overrides.items():
                _set_nested(cfg, dotted, value)
            trial_name = str(spec.get("name", f"{name}_trial{idx}"))
            trials.append(Trial(name=trial_name, config=cfg))
        return trials

    if not isinstance(sweep, dict) or "coupled_grid" not in sweep:
        raise ValueError("sweep must be a list of trials or a {coupled_grid: [...]} mapping")

    axes = _coupled_axes(sweep)
    if quick:
        for axis in axes:
            axis["values"] = _subsample(axis["values"], quick_per_axis)

    trials = []
    value_lists = [axis["values"] for axis in axes]
    for combo in product(*value_lists):
        cfg = deepcopy(base)
        label_parts: list[str] = []
        for axis, value in zip(axes, combo):
            for dotted in axis["paths"]:
                _set_nested(cfg, dotted, value)
            leaf = axis["paths"][0].split(".")[-1]
            label_parts.append(f"{leaf}_{_fmt_value(value)}")
        trial_name = f"{name}__" + "__".join(label_parts)
        trials.append(Trial(name=trial_name, config=cfg))
    return trials


def _subsample(values: list, max_keep: int) -> list:
    if max_keep <= 0 or len(values) <= max_keep:
        return list(values)
    if max_keep == 1:
        return [values[0]]
    if max_keep == 2:
        return [values[0], values[-1]]
    mid = values[len(values) // 2]
    return [values[0], mid, values[-1]]


def _run_single_trial(trial: Trial, output_root: Path) -> dict:
    cfg = deepcopy(trial.config)
    cfg.setdefault("output", {})
    cfg["output"]["output_dir"] = str(output_root)
    cfg["output"]["run_name"] = trial.name
    cfg["checkpoint_name"] = f"{trial.name}.pt"
    model_type = cfg.get("model", {}).get("type", "discrete")
    if model_type == "multishell":
        return run(cfg, model_cls=MultiShellDiscreteVESP)
    return run(cfg)


def run_experiment(
    experiment_cfg: dict,
    *,
    output_root: str | Path,
    config_path: str | Path | None = None,
    quick: bool = False,
    git_commit: str | None = None,
    continue_on_error: bool = True,
) -> ExperimentRunResult:
    """Run every trial of one experiment, writing per-run artifacts under ``output_root``.

    Returns the collected summary rows (one per trial) plus the raw metrics dicts.
    Per-run artifacts (``config.yaml``, ``metrics.json``, ``diagnostics.json``,
    ``summary.txt``, ``shell_energy.csv``, ``altitude_binned_error.csv``,
    ``target_scales.json``, ``run_manifest.json``) are written by ``run`` into
    ``output_root/<run_name>/``.
    """

    # Imported here to avoid a circular import (summarize imports nothing heavy, but
    # keeping the dependency one-directional makes the package easier to reason about).
    from vesp.experiments.summarize import summary_row

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if git_commit is None:
        git_commit = git_commit_hash()
    experiment_name = str(experiment_cfg["experiment"]["name"])
    config_path_str = str(config_path) if config_path is not None else None

    result = ExperimentRunResult(name=experiment_name, config_path=config_path_str)
    for trial in expand_trials(experiment_cfg, quick=quick):
        print(f"===== experiment={experiment_name} trial={trial.name} =====")
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        start = time.perf_counter()
        try:
            metrics = _run_single_trial(trial, output_root)
        except Exception as exc:  # noqa: BLE001 - record and continue the sweep
            if not continue_on_error:
                raise
            print(f"FAILED experiment={experiment_name} trial={trial.name}: {exc}")
            row = summary_row(
                trial.name,
                trial.config,
                {},
                experiment=experiment_name,
                config_path=config_path_str,
                git_commit=git_commit,
                timestamp=timestamp,
                runtime_sec=time.perf_counter() - start,
                error=str(exc)[:300],
            )
            result.rows.append(row)
            continue
        result.metrics.append(metrics)
        result.rows.append(
            summary_row(
                trial.name,
                trial.config,
                metrics,
                experiment=experiment_name,
                config_path=config_path_str,
                git_commit=git_commit,
                timestamp=timestamp,
                runtime_sec=time.perf_counter() - start,
            )
        )
    return result
