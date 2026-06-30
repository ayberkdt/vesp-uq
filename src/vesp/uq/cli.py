"""Small shared helpers for the VESP-UQ surface.

Config loading, deterministic CSV rendering, and compact float formatting used by both the thin
command-line wrappers under ``scripts/`` and the library runners (e.g. ``gate_diagnostics``,
``attribution``, ``readiness``). Kept dependency-light (std lib + ``vesp.common.config`` only) so it
is cheap to import from anywhere.
"""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Any

from vesp.common.config import load_config


def load_configs(paths: list[str]) -> list[dict]:
    """Load YAML configs and attach their source path for downstream provenance."""

    configs = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"config not found: {path}")
        cfg = load_config(str(p))
        cfg.setdefault("_config_path", str(p))
        configs.append(cfg)
    return configs


def aligned_existing_reports(
    paths: list[str] | None,
    *,
    n_configs: int,
    defaults: list[str] | None = None,
) -> list[str | None]:
    """Resolve optional report paths aligned with config paths.

    ``None`` means use provided defaults when they fit; an empty list means no reports.
    Missing report files are represented as ``None`` so callers can emit an explicit audit row.
    """

    if paths is None:
        paths = list(defaults or [])[:n_configs] if defaults and n_configs <= len(defaults) else []
    if len(paths) == 0:
        return [None] * n_configs
    if len(paths) != n_configs:
        raise SystemExit("--reports must be omitted, empty, or have the same length as --configs")
    return [str(Path(p)) if Path(p).exists() else None for p in paths]


def csv_text(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Render dictionaries as a deterministic CSV string."""

    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in columns})
    return handle.getvalue()


def fmt_float(value: Any, spec: str = ".3g") -> str:
    """Compact numeric formatting for CLI and Markdown summaries."""

    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return format(v, spec) if math.isfinite(v) else "n/a"
