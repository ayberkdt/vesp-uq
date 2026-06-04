from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.check_call([sys.executable, str(ROOT / "train_multishell.py"), "--config", str(ROOT / "configs" / "discrete_multishell.yaml")])
    subprocess.check_call([sys.executable, str(ROOT / "run_ablation.py"), "--config", str(ROOT / "configs" / "synthetic_stress_multishell.yaml")])


if __name__ == "__main__":
    main()

