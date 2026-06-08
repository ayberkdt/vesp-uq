"""Run one or more experiment configs and write a combined suite summary.

Examples
--------
Run a single experiment::

    python scripts/run_experiment_suite.py --config configs/experiments/synthetic_l2_sweep.yaml

Run a named suite (fast subset)::

    python scripts/run_experiment_suite.py --suite ci --quick

Run a core experiment by id::

    python scripts/run_experiment_suite.py --experiment E3

Outputs land under ``outputs/suites/<suite_name>/``:

    suite_summary.csv   combined standardized metrics (one row per trial)
    suite_summary.md    human-readable table
    pareto_data.csv     columns for L2 / entropy trade-off plots
    README.md           description + acceptability tally
    runs/<run_name>/    per-run artifacts (metrics.json, diagnostics.json, ...)
    *.png               optional plots (if matplotlib is available)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vesp.experiments.registry import experiment_info
from vesp.experiments.runner import git_commit_hash, load_experiment_config, run_experiment
from vesp.experiments.suites import resolve_suite
from vesp.experiments.summarize import write_suite_artifacts


def _resolve_config_paths(args: argparse.Namespace) -> tuple[str, list[Path]]:
    if args.suite:
        return args.suite, [ROOT / p for p in resolve_suite(args.suite)]
    if args.experiment:
        info = experiment_info(args.experiment)
        return info.name, [ROOT / info.config]
    if args.config:
        paths = [Path(p) for p in args.config]
        suite_name = args.name or (paths[0].stem if len(paths) == 1 else "custom_suite")
        return suite_name, paths
    raise SystemExit("provide one of --config, --suite or --experiment")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", nargs="+", help="one or more experiment YAML configs")
    group.add_argument("--suite", help="named suite (synthetic, real_lunar, ci, all)")
    group.add_argument("--experiment", help="core experiment id (E0..E5)")
    parser.add_argument("--name", help="suite name override (used for the output directory)")
    parser.add_argument("--output-root", default=None, help="base output dir (default outputs/suites/<name>)")
    parser.add_argument("--quick", action="store_true", help="subsample each sweep axis for a fast pass")
    parser.add_argument("--no-plots", action="store_true", help="skip optional plot generation")
    args = parser.parse_args(argv)

    suite_name, config_paths = _resolve_config_paths(args)
    suite_dir = Path(args.output_root) if args.output_root else (ROOT / "outputs" / "suites" / suite_name)
    runs_dir = suite_dir / "runs"
    suite_dir.mkdir(parents=True, exist_ok=True)

    git_commit = git_commit_hash()
    all_rows: list[dict] = []
    experiments: list[str] = []
    for config_path in config_paths:
        experiment_cfg = load_experiment_config(config_path)
        experiments.append(str(experiment_cfg["experiment"]["name"]))
        result = run_experiment(
            experiment_cfg,
            output_root=runs_dir,
            config_path=config_path,
            quick=args.quick,
            git_commit=git_commit,
        )
        all_rows.extend(result.rows)

    artifacts = write_suite_artifacts(
        suite_dir,
        all_rows,
        suite_name=suite_name,
        experiments=experiments,
        git_commit=git_commit,
        make_plots=not args.no_plots,
    )
    print(f"suite_dir: {suite_dir}")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
