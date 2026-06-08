"""Run the minimum safety checklist before reporting numerical results.

The base check is experiment-first and fast (no real-lunar data load):

    1. pytest tests/
    2. smoke test
    3. E0  synthetic exact recovery        (sanity: kernel / solver / metrics)
    4. E3  synthetic L2 sweep   (--quick)  (Tikhonov stability region)
    5. E4  synthetic entropy Pareto (--quick) (does entropy beat ridge?)

Optional flags:

    --include-real-lunar   also run the real lunar suite (loads the GRAIL-derived
                           band-limited residual CSV; heavier).
    --include-legacy       also run the legacy deterministic single/multi/OOD/feasibility
                           configs (kept for continuity).
    --dry-run              print the commands without executing them.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _base_commands() -> list[list[str]]:
    py = sys.executable
    suite = ["scripts/run_experiment_suite.py"]
    return [
        [py, "-m", "pytest", "tests/"],
        [py, "scripts/smoke_test.py"],
        [py, *suite, "--experiment", "E0", "--no-plots"],
        [py, *suite, "--experiment", "E3", "--quick", "--no-plots"],
        [py, *suite, "--experiment", "E4", "--quick", "--no-plots"],
        # E7: regularizer shootout (L2 vs entropy at matched error), data-driven verdict.
        [py, "scripts/regularizer_shootout.py", "--config", "configs/experiments/synthetic_regularizer_shootout.yaml", "--quick", "--no-plots"],
        # E8: source-geometry shootout (is the low-altitude bottleneck geometry?), ranked verdict.
        [py, "scripts/geometry_shootout.py", "--config", "configs/experiments/synthetic_geometry_shootout.yaml", "--quick", "--no-plots"],
        # Stage 3C: calibrated posterior uncertainty (MaxEnt-as-uncertainty) calibration check.
        [py, "-m", "vesp.training.uncertainty", "--config", "configs/uncertainty_synthetic_ood.yaml"],
    ]


def _legacy_commands() -> list[list[str]]:
    py = sys.executable
    return [
        [py, "-m", "vesp.training.train", "--config", "configs/discrete_single_shell.yaml"],
        [py, "-m", "vesp.training.train", "--config", "configs/discrete_multishell.yaml"],
        [py, "-m", "vesp.training.train", "--config", "configs/altitude_ood.yaml"],
        [py, "-m", "vesp.training.feasibility", "--config", "configs/feasibility_suite.yaml"],
    ]


def _real_lunar_commands() -> list[list[str]]:
    py = sys.executable
    return [
        [py, "scripts/run_experiment_suite.py", "--suite", "real_lunar", "--quick"],
        # Stage 3C+ heteroscedastic per-band calibration on real lunar (in-distribution).
        [py, "-m", "vesp.training.uncertainty", "--config", "configs/uncertainty_real_lunar.yaml"],
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--include-real-lunar", action="store_true", help="Also run the real lunar suite.")
    parser.add_argument("--include-legacy", action="store_true", help="Also run legacy deterministic configs.")
    args = parser.parse_args(argv)

    commands = _base_commands()
    if args.include_legacy:
        commands += _legacy_commands()
    if args.include_real_lunar:
        commands += _real_lunar_commands()

    for index, command in enumerate(commands, start=1):
        display = " ".join(command)
        print(f"[{index}] {display}")
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode != 0:
            print(f"PRE-RESULTS CHECK FAILED at step {index}: {display}")
            return completed.returncode

    print("PRE-RESULTS CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
