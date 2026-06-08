"""DEPRECATED stage shim. Prefer the experiment framework:

    python scripts/run_experiment_suite.py --experiment E0

(Equivalent single run: ``python -m vesp.training.train --config configs/discrete_single_shell.yaml``.)
Kept working for continuity (deprecate-and-delegate).
"""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    print("[DEPRECATED] scripts/run_stage1.py -> prefer scripts/run_experiment_suite.py --experiment E0")
    subprocess.check_call([sys.executable, "-m", "vesp.training.train", "--config", "configs/discrete_single_shell.yaml"], cwd=ROOT)


if __name__ == "__main__":
    main()
