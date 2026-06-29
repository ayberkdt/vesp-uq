"""Generate the VESP-UQ journal validation report + LaTeX tables (WP12).

Reads the study CSVs produced by the benchmark suite and the M2/M3 scripts and emits
``journal_validation_report.md`` plus ``latex_tables/table_*.tex``. Sections without a CSV are
marked pending (naming the script to run); the Phase-14 ranking verdict and the claims
supported/unsupported table are derived from the measured numbers only.

    python scripts/run_journal_report.py --outputs outputs/ --out outputs/journal/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vesp.uq.journal_report import write_report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Generate the VESP-UQ journal validation report (WP12).")
    parser.add_argument("--outputs", default="outputs/", help="root containing the study output dirs")
    parser.add_argument("--out", default=None, help="where to write the report (default: --outputs root)")
    parser.add_argument("--figures", action="store_true",
                        help="also render the WP-B/C/D paper figures into <out>/figures")
    args = parser.parse_args(argv)

    root = Path(args.outputs)
    if not root.exists():
        raise SystemExit(f"outputs root not found: {root}")
    result = write_report(root, out_dir=args.out)
    print(f"verdict: {result['verdict']['overall']}")
    print(f"available studies: {result['available']}")
    if result["missing"]:
        print(f"pending studies: {result['missing']}")
    out_dir = Path(args.out) if args.out else root
    print(f"saved_journal_report: {out_dir / 'journal_validation_report.md'}")
    if args.figures:
        from vesp.uq.figures import render_paper_figures

        manifest = render_paper_figures(
            benchmark_dir=root / "benchmark_suite",
            baseline_dir=root / "uq_baseline_comparison",
            out_dir=out_dir / "figures",
        )
        ok = sum(1 for f in manifest["figures"] if f.get("status") == "ok")
        print(f"saved_paper_figures: {ok}/{len(manifest['figures'])} rendered -> {out_dir / 'figures'}")


if __name__ == "__main__":
    main()
