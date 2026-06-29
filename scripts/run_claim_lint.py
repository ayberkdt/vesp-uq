"""G6 -- forbidden-claim linter: fail CI if a report / manuscript over-claims.

Scans a generated Markdown report (and, optionally, a manuscript ``.tex``) for phrasing the project
has ruled out (validated operational covariance, density recovery, "outperforms all baselines", a
guaranteed risk bound, end-to-end trajectory correction / ST-LRPS validation). A hit excused by an
``<!-- evidence: ... -->`` tag or a disclaimer ("future work", "not validated", ...) is allowed;
anything else exits non-zero.

    python scripts/run_claim_lint.py \
        --report outputs/journal/journal_validation_report.md \
        --manuscript paper/vesp_uq.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vesp.uq.integrity.claim_lint import lint_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="VESP-UQ forbidden-claim linter (G6).")
    parser.add_argument("--report", required=True, help="Markdown report to lint")
    parser.add_argument("--manuscript", default=None, help="optional manuscript .tex to also lint")
    args = parser.parse_args(argv)

    report = Path(args.report)
    if not report.exists():
        raise SystemExit(f"report not found: {report}")

    result = lint_report(report, manuscript=args.manuscript)
    for v in result["violations"]:
        print(f"  FORBIDDEN [{v['claim']}] {v['source']} line {v['line']}: {v['match']!r}")
    print(f"[claim-lint] {result['n_violations']} violation(s)")
    print(f"[claim-lint] {'OK -- no forbidden claim' if result['ok'] else 'FAILED'}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
