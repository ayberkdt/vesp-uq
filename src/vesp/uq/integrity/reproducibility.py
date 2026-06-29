"""G5 -- determinism / reproducibility gate.

A byte-reproducible result cannot be quietly hand-edited: rerun the same ``(config, seed)`` and the
data tables must come out identical. Timing columns (``runtime_*``) legitimately vary run to run, so
they are dropped by a normalizer before the byte comparison; everything else -- every metric, every
aggregate -- must match exactly.

``normalize_csv_text`` re-emits a CSV with the ignored columns removed; ``assert_reproducible``
compares the normalized text of a set of files across two output directories and raises
:class:`ReproducibilityError` on the first divergence, with a short unified diff for the offending
file. The suite orchestration that runs a config twice and calls this lives in
:func:`vesp.uq.suite.run_reproducibility_check`.
"""

from __future__ import annotations

import csv
import difflib
import io
from collections.abc import Iterable, Sequence
from pathlib import Path

# Canonical data tables whose bytes must be reproducible across reruns (timing excluded).
REPRO_FILES = ("benchmark_runs.csv", "decision_quality.csv", "calibration_summary.csv")
# Column-name prefixes dropped before comparison (wall-clock timing is not part of the result).
DEFAULT_IGNORE_PREFIXES = ("runtime_",)


class ReproducibilityError(AssertionError):
    """Raised when a normalized output file differs between two runs of the same (config, seed)."""


def normalize_csv_text(text: str, *, ignore_prefixes: Sequence[str] = DEFAULT_IGNORE_PREFIXES) -> str:
    """Return ``text`` as canonical CSV with columns whose header starts with an ignored prefix removed.

    A header-less or empty CSV is returned unchanged (nothing to drop). Cell strings are preserved
    verbatim, so the comparison stays exact for everything that is not timing.
    """

    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return text
    header = rows[0]
    keep = [i for i, name in enumerate(header)
            if not any(name.startswith(p) for p in ignore_prefixes)]
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    for row in rows:
        writer.writerow([row[i] for i in keep if i < len(row)])
    return out.getvalue()


def _diff(name: str, a: str, b: str) -> str:
    diff = difflib.unified_diff(a.splitlines(), b.splitlines(), fromfile=f"a/{name}",
                                tofile=f"b/{name}", lineterm="")
    return "\n".join(list(diff)[:20])  # first few differing lines are enough to localize it


def compare_outputs(
    dir_a: str | Path,
    dir_b: str | Path,
    *,
    filenames: Iterable[str] = REPRO_FILES,
    ignore_prefixes: Sequence[str] = DEFAULT_IGNORE_PREFIXES,
) -> dict:
    """Compare the normalized text of ``filenames`` across two output dirs (no raising).

    Returns ``{"files": {name: {"status": ..., "diff": ...}}, "ok": bool, "checked": int}`` where
    each status is ``identical`` / ``differs`` / ``missing`` (absent from one or both dirs).
    """

    dir_a, dir_b = Path(dir_a), Path(dir_b)
    files: dict[str, dict] = {}
    for name in filenames:
        pa, pb = dir_a / name, dir_b / name
        if not pa.exists() or not pb.exists():
            files[name] = {"status": "missing", "diff": ""}
            continue
        na = normalize_csv_text(pa.read_text(encoding="utf-8"), ignore_prefixes=ignore_prefixes)
        nb = normalize_csv_text(pb.read_text(encoding="utf-8"), ignore_prefixes=ignore_prefixes)
        if na == nb:
            files[name] = {"status": "identical", "diff": ""}
        else:
            files[name] = {"status": "differs", "diff": _diff(name, na, nb)}
    ok = bool(files) and all(f["status"] == "identical" for f in files.values())
    return {"files": files, "ok": ok, "checked": len(files)}


def assert_reproducible(
    dir_a: str | Path,
    dir_b: str | Path,
    *,
    filenames: Iterable[str] = REPRO_FILES,
    ignore_prefixes: Sequence[str] = DEFAULT_IGNORE_PREFIXES,
) -> dict:
    """Like :func:`compare_outputs`, but raise :class:`ReproducibilityError` on any non-identical file."""

    report = compare_outputs(dir_a, dir_b, filenames=filenames, ignore_prefixes=ignore_prefixes)
    bad = {n: f for n, f in report["files"].items() if f["status"] != "identical"}
    if bad:
        lines = [f"{n}: {f['status']}" + (f"\n{f['diff']}" if f["diff"] else "")
                 for n, f in bad.items()]
        raise ReproducibilityError(
            "non-reproducible outputs (after dropping timing columns):\n" + "\n".join(lines))
    return report
