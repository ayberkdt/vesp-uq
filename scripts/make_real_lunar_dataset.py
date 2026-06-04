from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "experimental_vesp.real_gravity",
            "--model",
            "gl0420a",
            "--n-query",
            "1024",
            "--degree-min",
            "2",
            "--degree-max",
            "60",
            "--output",
            "data/lunar_grail_gl0420a_L60_residual.csv",
            "--acceleration-output",
            "physical",
        ],
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()

