"""DEPRECATED stage shim. Prefer the experiment framework:

    python scripts/run_experiment_suite.py --experiment E2

(Equivalent single run: ``python -m vesp.training.train --config configs/discrete_multishell.yaml``.)
Kept working for continuity (deprecate-and-delegate).
"""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    print("[DEPRECATED] scripts/run_stage2.py -> prefer scripts/run_experiment_suite.py --experiment E2")
    subprocess.check_call([sys.executable, "-m", "vesp.training.train", "--config", "configs/discrete_multishell.yaml"], cwd=ROOT)


if __name__ == "__main__":
    main()
