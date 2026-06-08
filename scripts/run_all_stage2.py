"""DEPRECATED orchestration shim. Prefer the experiment framework / pre-results check:

    python scripts/run_experiment_suite.py --suite synthetic
    python scripts/pre_results_check.py

Kept working for continuity (deprecate-and-delegate).
"""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    print("[DEPRECATED] scripts/run_all_stage2.py -> prefer scripts/run_experiment_suite.py --suite synthetic")
    subprocess.check_call([sys.executable, str(ROOT / "train_multishell.py"), "--config", str(ROOT / "configs" / "discrete_multishell.yaml")])
    subprocess.check_call([sys.executable, str(ROOT / "run_ablation.py"), "--config", str(ROOT / "configs" / "synthetic_stress_multishell.yaml")])


if __name__ == "__main__":
    main()
