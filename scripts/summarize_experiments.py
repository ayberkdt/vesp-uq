"""Rebuild a suite summary from existing per-run output directories.

Useful for regenerating the combined CSV / Markdown / Pareto table (and plots)
without re-running the experiments, e.g. after editing the summary columns or
plotting code.

    python scripts/summarize_experiments.py --runs-dir outputs/suites/synthetic/runs

It scans ``<runs-dir>/<run_name>/{metrics.json, diagnostics.json, config.yaml}`` and
writes the suite artifacts to the parent of ``runs-dir`` (or ``--output-root``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vesp.experiments.runner import git_commit_hash
from vesp.experiments.summarize import summary_row, write_suite_artifacts


def _load_run(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.yaml"
    if not metrics_path.exists() or not config_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    diagnostics_path = run_dir / "diagnostics.json"
    if diagnostics_path.exists():
        with diagnostics_path.open("r", encoding="utf-8") as handle:
            metrics["diagnostics"] = json.load(handle)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return {"run_name": run_dir.name, "config": config, "metrics": metrics}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", required=True, help="directory containing per-run output subdirectories")
    parser.add_argument("--output-root", default=None, help="where to write suite artifacts (default: parent of runs-dir)")
    parser.add_argument("--suite-name", default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        raise SystemExit(f"runs-dir not found: {runs_dir}")
    suite_dir = Path(args.output_root) if args.output_root else runs_dir.parent
    suite_name = args.suite_name or suite_dir.name

    rows: list[dict] = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        loaded = _load_run(run_dir)
        if loaded is None:
            continue
        rows.append(summary_row(loaded["run_name"], loaded["config"], loaded["metrics"]))

    if not rows:
        raise SystemExit(f"no runs with metrics.json found under {runs_dir}")

    artifacts = write_suite_artifacts(
        suite_dir,
        rows,
        suite_name=suite_name,
        git_commit=git_commit_hash(),
        make_plots=not args.no_plots,
    )
    print(f"suite_dir: {suite_dir} ({len(rows)} runs)")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
