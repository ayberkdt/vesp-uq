from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.check_call([sys.executable, "-m", "experimental_vesp.train", "--config", "configs/discrete_multishell.yaml"], cwd=ROOT)


if __name__ == "__main__":
    main()

