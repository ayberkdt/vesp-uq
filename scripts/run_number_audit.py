"""G1 -- no-orphan-number auditor: fail CI if any reported number is not CSV-traceable.

Audits a generated Markdown report (and, optionally, the LaTeX tables) so that every data number
traces to a source CSV cell recorded in a checksummed run manifest. A hand-entered or fabricated
number matches no CSV cell and is reported as an orphan; the process then exits non-zero.

    python scripts/run_number_audit.py \
        --report outputs/journal/journal_validation_report.md \
        --csv-dir outputs/journal outputs/benchmark_suite \
        --tables-dir outputs/journal/latex
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vesp.uq.integrity.number_audit import audit_latex_tables, audit_report_numbers


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="VESP-UQ no-orphan-number auditor (G1).")
    parser.add_argument("--report", required=True, help="Markdown report to audit")
    parser.add_argument("--csv-dir", nargs="+", required=True,
                        help="one or more dirs holding the source CSVs (+ their manifests)")
    parser.add_argument("--tables-dir", default=None,
                        help="optional dir of LaTeX *.tex tables to audit as well")
    parser.add_argument("--tol", type=float, default=5e-3,
                        help="match tolerance: |v-c| <= tol*max(|v|,|c|,1) (default 5e-3)")
    parser.add_argument("--min-digits", type=int, default=2,
                        help="ignore tokens with fewer significant digits (default 2)")
    args = parser.parse_args(argv)

    report = Path(args.report)
    if not report.exists():
        raise SystemExit(f"report not found: {report}")

    result = audit_report_numbers(report, args.csv_dir, tol=args.tol, min_digits=args.min_digits)
    man = result["manifests"]
    print(f"[number-audit] {report}: checked {result['checked']} numbers, "
          f"{result['n_orphans']} orphan(s)")
    print(f"[number-audit] manifests: {man['checked']} CSV(s), "
          f"{len(man['unmanifested'])} unmanifested, {len(man['changed'])} changed")
    for o in result["orphans"]:
        print(f"  ORPHAN line {o['line']}: {o['value']}  (no CSV match within tol)")
    for p in man["unmanifested"]:
        print(f"  UNMANIFESTED CSV: {p}")
    for p in man["changed"]:
        print(f"  CHANGED CSV (checksum mismatch): {p}")

    ok = result["ok"]
    if args.tables_dir:
        tex = audit_latex_tables(args.tables_dir, args.csv_dir, tol=args.tol,
                                 min_digits=args.min_digits)
        print(f"[number-audit] latex: {tex['n_orphans']} orphan(s) across {len(tex['tables'])} table(s)")
        for name, orphans in tex["tables"].items():
            for o in orphans:
                print(f"  ORPHAN {name} line {o['line']}: {o['value']}")
        ok = ok and tex["ok"]

    print(f"[number-audit] {'OK -- every number is sourced' if ok else 'FAILED -- orphan / manifest issue'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
