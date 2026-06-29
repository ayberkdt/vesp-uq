"""Render the WP-B/C/D paper-rigor figures from the benchmark-suite + GP-baseline CSVs.

    python scripts/render_paper_figures.py \
        --benchmark-dir outputs/benchmark_suite --baseline-dir outputs/uq_baseline_comparison \
        --out outputs/paper_figures/

Refits nothing; missing study CSVs render an explicit placeholder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vesp.uq.figures import render_paper_figures


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Render VESP-UQ paper-rigor figures (WP-B/C/D).")
    parser.add_argument("--benchmark-dir", default="outputs/benchmark_suite")
    parser.add_argument("--baseline-dir", default="outputs/uq_baseline_comparison")
    parser.add_argument("--out", default="outputs/paper_figures/")
    args = parser.parse_args(argv)

    manifest = render_paper_figures(
        benchmark_dir=args.benchmark_dir, baseline_dir=args.baseline_dir, out_dir=args.out,
    )
    for fig in manifest["figures"]:
        print(f"{fig['name']}: {fig.get('status', '?')} -> {fig['png']}")
    print(f"saved_paper_figures_manifest: {Path(args.out) / 'paper_figures_manifest.json'}")


if __name__ == "__main__":
    main()
