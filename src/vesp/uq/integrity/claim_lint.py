"""G6 -- forbidden-claim linter.

Integrity includes not over-claiming in prose. This linter scans a generated report (and an optional
manuscript ``.tex``) for phrasing the project has explicitly ruled out -- a *validated* operational
orbit/state covariance, density recovery, "outperforms all baselines", a guaranteed risk bound,
end-to-end trajectory correction, end-to-end ST-LRPS validation -- and fails CI on a hit.

The forbidden list mirrors ``docs/SCIENTIFIC_CLAIMS.md`` ("Not implemented yet -- do not claim") and
the journal plan's Phase-14 claim discipline. A hit is **excused** only when the same line carries an
explicit disclaimer: an ``<!-- evidence: ... -->`` / ``% evidence: ...`` tag, or a negation marker
(``future work``, ``not attempted``, ``not validated``, ``do not claim``, ``diagnostic only``,
``limitation``). That lets the auto-generated claims table legitimately *list* a forbidden claim as
future work while still catching an affirmative over-claim in prose.
"""

from __future__ import annotations

import re
from pathlib import Path

# (label, compiled pattern). Patterns are case-insensitive and intentionally phrase-specific so an
# ordinary mention ("the local force-error covariance") does not trip; only the over-claim does.
FORBIDDEN: tuple[tuple[str, re.Pattern], ...] = (
    ("validated operational orbit/state covariance",
     re.compile(r"validated\b[^.\n]{0,40}\b(?:orbit|state|operational)\b[^.\n]{0,40}\bcovariance\b"
                r"|validated\b[^.\n]{0,40}\bcovariance\b[^.\n]{0,40}\b(?:orbit|state|propagat)",
                re.IGNORECASE)),
    ("density recovery",
     re.compile(r"\bdensity\s+recovery\b|\brecover[a-z]*\b[^.\n]{0,25}\bdensity\b", re.IGNORECASE)),
    ("outperforms all baselines",
     re.compile(r"\boutperform[a-z]*\s+all\b|\bbeats?\s+(?:all|every)\b[^.\n]{0,20}\bbaseline",
                re.IGNORECASE)),
    ("guaranteed risk bound",
     re.compile(r"\bguarantee[a-z]*\b[^.\n]{0,30}\b(?:risk|error|coverage)?\s*bound\b", re.IGNORECASE)),
    ("end-to-end trajectory correction",
     re.compile(r"\btrajectory\s+correction\b|\bcorrect[a-z]*\b[^.\n]{0,15}\btrajectory\b",
                re.IGNORECASE)),
    ("end-to-end ST-LRPS validation",
     re.compile(r"\bend-to-end\b[^.\n]{0,30}\bst-?lrps\b|\bst-?lrps\b[^.\n]{0,25}\bvalidat",
                re.IGNORECASE)),
    ("learned/generative noise model",
     re.compile(r"\b(?:learned|generative|neural)\b[^.\n]{0,25}\bnoise\s+model\b", re.IGNORECASE)),
)

# A line carrying any of these is excused (it is disclaiming, not asserting).
_DISCLAIMER = re.compile(
    r"<!--\s*evidence:|%\s*evidence:|\bfuture work\b|\bnot attempted\b|\bnot validated\b|"
    r"\bdo not claim\b|\bdiagnostic only\b|\blimitation\b|\bnot a\b[^.\n]{0,20}\bclaim\b",
    re.IGNORECASE,
)
_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)


def _strip_fenced_code(text: str) -> str:
    return _FENCED_CODE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def scan_text(text: str, *, source: str = "") -> list[dict]:
    """Return the forbidden-claim violations in ``text`` (each: label, line, the matched snippet).

    Lines inside fenced code blocks are ignored; a line carrying an evidence tag or a disclaimer
    marker is excused.
    """

    violations: list[dict] = []
    for line_no, line in enumerate(_strip_fenced_code(text).splitlines(), start=1):
        if _DISCLAIMER.search(line):
            continue
        for label, pattern in FORBIDDEN:
            m = pattern.search(line)
            if m:
                violations.append({"source": source, "line": line_no, "claim": label,
                                   "match": m.group(0).strip()})
    return violations


def lint_report(report_path: str | Path, *, manuscript: str | Path | None = None) -> dict:
    """Lint a Markdown report (and an optional manuscript ``.tex``) for forbidden claims.

    Returns ``{"violations": [...], "n_violations": int, "ok": bool}``. ``ok`` is true only when no
    un-excused forbidden phrase appears in either file.
    """

    violations = scan_text(Path(report_path).read_text(encoding="utf-8"), source=str(report_path))
    if manuscript is not None:
        violations += scan_text(Path(manuscript).read_text(encoding="utf-8"), source=str(manuscript))
    return {"violations": violations, "n_violations": len(violations), "ok": not violations}
