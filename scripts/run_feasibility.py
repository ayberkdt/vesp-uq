"""DEPRECATED shim for the feasibility suite. Prefer the experiment framework:

    python scripts/run_experiment_suite.py --suite synthetic

Kept working for continuity (deprecate-and-delegate); the underlying runner prints a
deprecation banner.
"""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    print("[DEPRECATED] scripts/run_feasibility.py -> prefer scripts/run_experiment_suite.py --suite synthetic")
    subprocess.check_call([sys.executable, "-m", "vesp.training.feasibility", "--config", "configs/feasibility_suite.yaml"], cwd=ROOT)


if __name__ == "__main__":
    main()
