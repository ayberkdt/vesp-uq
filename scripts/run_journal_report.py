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


if __name__ == "__main__":
    main()
