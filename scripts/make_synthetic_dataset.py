from pathlib import Path
import argparse
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experimental_vesp.data import make_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic_residual.csv")
    parser.add_argument("--n-query", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    data = make_synthetic_dataset(n_query=args.n_query, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "DeltaU", "Deltaax", "Deltaay", "Deltaaz"])
        for x, u, a in zip(data.positions, data.potential, data.acceleration):
            writer.writerow([float(x[0]), float(x[1]), float(x[2]), float(u[0]), float(a[0]), float(a[1]), float(a[2])])
    print(f"synthetic_dataset: {output}")


if __name__ == "__main__":
    main()

