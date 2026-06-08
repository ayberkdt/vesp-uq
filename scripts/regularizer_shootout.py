"""E7 driver: run the regularizer shootout (L2 vs entropy) and write the verdict.

    python scripts/regularizer_shootout.py --config configs/experiments/synthetic_regularizer_shootout.yaml
    python scripts/regularizer_shootout.py --config configs/experiments/synthetic_regularizer_shootout_concentrated.yaml --quick

Runs every trial through the standard experiment runner, writes the usual suite artifacts
(`suite_summary.csv`, etc.), then aligns the ridge and MaxEnt families by data error and
writes a data-driven verdict (`shootout_verdict.md`, `shootout_matched.csv`).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vesp.experiments.runner import git_commit_hash, load_experiment_config, run_experiment
from vesp.experiments.shootout import shootout_report
from vesp.experiments.summarize import write_suite_artifacts


def _write_matched_csv(path: Path, matched: list[dict]) -> None:
    if not matched:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in matched:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(matched)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="an E7 shootout experiment YAML")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--quick", action="store_true", help="subsample sweep axes for a fast pass")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    experiment_cfg = load_experiment_config(config_path)
    name = str(experiment_cfg["experiment"]["name"])
    suite_dir = Path(args.output_root) if args.output_root else (ROOT / "outputs" / "suites" / name)
    runs_dir = suite_dir / "runs"

    git_commit = git_commit_hash()
    result = run_experiment(
        experiment_cfg,
        output_root=runs_dir,
        config_path=config_path,
        quick=args.quick,
        git_commit=git_commit,
    )
    write_suite_artifacts(
        suite_dir,
        result.rows,
        suite_name=name,
        experiments=[name],
        git_commit=git_commit,
        make_plots=not args.no_plots,
    )

    markdown, matched, _tally = shootout_report(result.rows)
    verdict_path = suite_dir / "shootout_verdict.md"
    verdict_path.write_text(markdown, encoding="utf-8")
    _write_matched_csv(suite_dir / "shootout_matched.csv", matched)

    # encoding-safe console output (Windows consoles may not be UTF-8); the .md file keeps the marks
    print(markdown.encode("ascii", "replace").decode("ascii"))
    print(f"shootout_verdict: {verdict_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
