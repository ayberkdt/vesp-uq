"""G1 -- no-orphan-number auditor (the strongest anti-fabrication feature).

Every numeric value in a generated report (or LaTeX table) must be traceable to a cell in a source
CSV that is itself recorded, with a matching SHA-256, in a run manifest. A hand-entered or
fabricated number then cannot survive this audit: it matches no CSV cell and is flagged as an
**orphan**.

Public API
----------
``audit_report_numbers(report_path, csv_dirs, *, tol, min_digits)`` -- audit a Markdown report.
``audit_latex_tables(tables_dir, csv_dirs, *, tol)`` -- audit every ``*.tex`` table in a directory.
``collect_csv_values(csv_dirs)`` / ``verify_csv_manifests(csv_dirs)`` -- the reusable building blocks.

What counts as an auditable number
----------------------------------
Only *data-like* tokens are audited; the following are deliberately **skipped** (documented allowlist):

* pure integers with no decimal point / exponent / percent sign (seed counts, ``n_seeds``, years);
* tokens with fewer than ``min_digits`` significant digits (``0.5`` at the default ``min_digits=2``);
* numbers that are part of an identifier -- a digit touching a letter or ``_`` (``L90``, ``p95``,
  ``picp_90``, ``gl0420a``);
* numbers in a structural reference context (``Table 3``, ``Section 3b``, ``Figure 2``, ``WP-A``,
  ``Phase-14``, ``v0.2``, a Markdown ``#`` heading line, a ``§`` reference);
* anything inside fenced ``` code blocks or inline ``code`` spans (commands, file names);
* the literal markers ``n/a`` and ``pending``.

Matching tolerance
------------------
A token value is *sourced* if some CSV cell matches it within ``abs(v - c) <= tol * max(|v|, |c|, 1)``
-- relative ``tol`` for magnitudes above 1, an absolute ``tol`` floor below 1 (which absorbs the
fixed-decimal rounding of rendered tables, e.g. ``0.856`` for a true ``0.8557``). A percentage token
(``20%``) is sourced if *either* its face value (``20``) or its fraction (``0.20``) matches.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Iterable
from pathlib import Path

from vesp.common.artifacts import compute_file_sha256

# Context keywords that mark a following number as a structural reference, not data.
_REFERENCE_CONTEXT = re.compile(
    r"(?:table|section|sect\.|figure|fig\.|phase|wp-|appendix|app\.|eq\.|equation|"
    r"version|chapter|§)\s*[-:]?\s*$",
    re.IGNORECASE,
)
# A numeric token: optional sign, a mantissa (with a decimal point OR plain digits), optional
# exponent, optional trailing percent. Lookarounds (not consumed chars) reject a token that is part
# of a longer number / identifier: a digit touching a letter or ``_`` is a name (``L90``, ``p95``,
# ``picp_90``), not a value -- so it never becomes a false orphan.
_NUMBER = re.compile(
    r"(?<![A-Za-z_0-9.])"
    r"(?P<num>[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)"
    r"(?P<pct>%?)"
    r"(?![A-Za-z_])"
)
_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`]*`")
_SKIP_TOKENS = ("n/a", "pending")
_MANIFEST_NAMES = ("manifest.json", "run_manifest.json")


def _strip_code(text: str) -> str:
    """Blank out fenced + inline code spans (preserving line count) so commands are not audited."""

    def _blank(match: re.Match) -> str:
        return re.sub(r"[^\n]", " ", match.group(0))  # keep newlines, drop everything else

    return _INLINE_CODE.sub(_blank, _FENCED_CODE.sub(_blank, text))


def _significant_digits(mantissa: str) -> int:
    """Count significant digits in a numeric mantissa string (sign / point stripped, no leading 0s)."""

    digits = mantissa.lstrip("+-").replace(".", "").lstrip("0")
    return len(digits)


def _candidate_values(num: str, is_percent: bool) -> list[float]:
    """The value(s) a token may legitimately denote: the literal, plus its /100 form if a percent."""

    try:
        v = float(num)
    except ValueError:
        return []
    if is_percent:
        return [v, v / 100.0]
    return [v]


def _auditable(num: str, is_percent: bool, *, min_digits: int) -> bool:
    """Whether a matched numeric token is a data value we should trace (vs an int / year / count)."""

    is_float = ("." in num) or ("e" in num) or ("E" in num)
    if not (is_float or is_percent):  # pure integers (seeds, years, counts) are not audited
        return False
    return _significant_digits(num) >= int(min_digits)


def _iter_number_tokens(text: str, *, min_digits: int):
    """Yield ``(value_candidates, raw, line_no)`` for each auditable numeric token in ``text``."""

    cleaned = _strip_code(text)
    for line_no, line in enumerate(cleaned.splitlines(), start=1):
        if line.lstrip().startswith("#"):  # Markdown heading: titles, not data
            continue
        for m in _NUMBER.finditer(line):
            num, pct = m.group("num"), m.group("pct")
            is_percent = pct == "%"
            if not _auditable(num, is_percent, min_digits=min_digits):
                continue
            before = line[: m.start("num")]
            if _REFERENCE_CONTEXT.search(before.rstrip()[-16:]):  # "Table 3", "Phase-14", ...
                continue
            candidates = _candidate_values(num, is_percent)
            if candidates:
                yield candidates, num + pct, line_no


def collect_csv_values(csv_dirs: Iterable[str | Path]) -> list[float]:
    """Parse every cell of every ``*.csv`` under ``csv_dirs`` into a flat list of finite floats."""

    values: list[float] = []
    seen: set[Path] = set()
    for d in csv_dirs:
        for path in sorted(Path(d).rglob("*.csv")):
            rp = path.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.reader(handle):
                    for cell in row:
                        cell = cell.strip()
                        if not cell:
                            continue
                        try:
                            v = float(cell)
                        except ValueError:
                            continue
                        if math.isfinite(v):
                            values.append(v)
    return values


def _is_sourced(candidates: list[float], csv_values: Iterable[float], tol: float) -> bool:
    """True if any candidate value matches a CSV value within ``tol * max(|v|, |c|, 1)``."""

    for v in candidates:
        for c in csv_values:
            if abs(v - c) <= tol * max(abs(v), abs(c), 1.0):
                return True
    return False


def verify_csv_manifests(csv_dirs: Iterable[str | Path]) -> dict:
    """Check every source CSV is recorded in a run manifest with a matching SHA-256.

    Scans ``csv_dirs`` for ``manifest.json`` / ``run_manifest.json``, indexes their ``artifacts``
    entries by file basename + SHA-256, then re-hashes every ``*.csv`` on disk. A CSV that no
    manifest lists is ``unmanifested``; one whose on-disk hash differs from the manifest is
    ``changed``. ``ok`` is true only when no CSV is unmanifested or changed.
    """

    manifest_sha: dict[str, set[str]] = {}
    for d in csv_dirs:
        for name in _MANIFEST_NAMES:
            for mpath in sorted(Path(d).rglob(name)):
                try:
                    manifest = json.loads(mpath.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                for entry in (manifest.get("artifacts") or {}).values():
                    sha = entry.get("sha256")
                    epath = entry.get("path")
                    if sha and epath:
                        manifest_sha.setdefault(Path(epath).name, set()).add(str(sha))

    unmanifested: list[str] = []
    changed: list[str] = []
    checked = 0
    seen: set[Path] = set()
    for d in csv_dirs:
        for path in sorted(Path(d).rglob("*.csv")):
            rp = path.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            checked += 1
            recorded = manifest_sha.get(path.name)
            if not recorded:
                unmanifested.append(str(path))
            elif compute_file_sha256(path) not in recorded:
                changed.append(str(path))
    return {
        "checked": checked,
        "unmanifested": unmanifested,
        "changed": changed,
        "ok": not unmanifested and not changed,
    }


def _audit_text(text: str, csv_values: list[float], *, tol: float, min_digits: int) -> list[dict]:
    """Return the list of orphan tokens (no CSV match) in ``text``."""

    orphans: list[dict] = []
    for candidates, raw, line_no in _iter_number_tokens(text, min_digits=min_digits):
        if raw.lower() in _SKIP_TOKENS:
            continue
        if not _is_sourced(candidates, csv_values, tol):
            orphans.append({"value": raw, "line": line_no, "candidates": candidates})
    return orphans


def audit_report_numbers(
    report_path: str | Path,
    csv_dirs: Iterable[str | Path],
    *,
    tol: float = 5e-3,
    min_digits: int = 2,
) -> dict:
    """Audit every data number in a Markdown report against the source CSVs + their manifests.

    Returns ``{"orphans", "checked", "n_orphans", "manifests", "ok"}``. ``ok`` is true only when no
    orphan number is found **and** every source CSV is manifested with a matching checksum.
    """

    csv_dirs = [Path(d) for d in csv_dirs]
    text = Path(report_path).read_text(encoding="utf-8")
    csv_values = collect_csv_values(csv_dirs)
    orphans = _audit_text(text, csv_values, tol=tol, min_digits=min_digits)
    manifests = verify_csv_manifests(csv_dirs)
    checked = sum(1 for _ in _iter_number_tokens(_strip_code(text), min_digits=min_digits))
    return {
        "report": str(report_path),
        "orphans": orphans,
        "n_orphans": len(orphans),
        "checked": checked,
        "manifests": manifests,
        "ok": not orphans and manifests["ok"],
    }


def audit_latex_tables(
    tables_dir: str | Path,
    csv_dirs: Iterable[str | Path],
    *,
    tol: float = 5e-3,
    min_digits: int = 2,
) -> dict:
    """Audit every number in every ``*.tex`` table under ``tables_dir`` against the source CSVs.

    Returns ``{"tables": {name: orphans}, "n_orphans", "ok"}``. LaTeX tables hold only rendered data,
    so an orphan here is the same red flag as in the report.
    """

    csv_values = collect_csv_values([Path(d) for d in csv_dirs])
    per_table: dict[str, list[dict]] = {}
    for tex in sorted(Path(tables_dir).glob("*.tex")):
        text = tex.read_text(encoding="utf-8")
        orphans = _audit_text(text, csv_values, tol=tol, min_digits=min_digits)
        if orphans:
            per_table[tex.name] = orphans
    n_orphans = sum(len(v) for v in per_table.values())
    return {"tables_dir": str(tables_dir), "tables": per_table, "n_orphans": n_orphans,
            "ok": n_orphans == 0}
