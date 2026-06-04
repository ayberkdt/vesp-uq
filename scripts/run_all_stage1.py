from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.check_call([sys.executable, str(ROOT / "train_discrete.py"), "--config", str(ROOT / "configs" / "discrete_single_shell.yaml")])
    subprocess.check_call([sys.executable, str(ROOT / "train_discrete.py"), "--config", str(ROOT / "configs" / "altitude_ood.yaml")])


if __name__ == "__main__":
    main()

