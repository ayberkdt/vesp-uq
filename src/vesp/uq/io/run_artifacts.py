"""Shared run-artifact writer for VESP-UQ scripts.

The benchmark / audit / screening scripts historically wrote bare JSON/MD/CSV files with no
provenance. This helper routes every output through the atomic writers in
:mod:`vesp.common.artifacts`, injects a small ``_provenance`` block into each JSON, and writes a
``run_manifest.json`` recording the config snapshot, seed, environment, and a SHA-256 checksum +
byte size for every emitted file -- so a result can be traced back to the exact config and verified.

It deliberately does **not** use :func:`vesp.common.artifacts.write_run_manifest` /
``ensure_run_layout`` (those create a training-oriented ``checkpoints/`` subdirectory); it writes the
manifest directly while mirroring the same schema fields.
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vesp.common.artifacts import (
    RUN_MANIFEST_SCHEMA_VERSION,
    atomic_write_json,
    atomic_write_text,
    compute_file_sha256,
    json_safe,
    utc_now_iso,
)

MANIFEST_NAME = "run_manifest.json"


def write_run_artifacts(
    out_dir: str | Path,
    *,
    tool: str,
    config: Mapping[str, Any] | None = None,
    json_files: Mapping[str, Mapping[str, Any]] | None = None,
    text_files: Mapping[str, str] | None = None,
    inputs: Mapping[str, str | Path] | None = None,
    seed: Any = None,
    config_path: str | None = None,
    manifest_name: str = MANIFEST_NAME,
) -> dict:
    """Write a VESP-UQ script's outputs atomically with a provenance manifest + checksums.

    ``json_files`` maps filename -> JSON-able payload (a ``_provenance`` block is injected unless one
    is already present); ``text_files`` maps filename -> text (Markdown / CSV). ``inputs`` maps a
    logical name -> path of a file the run CONSUMED (saved models, datasets, trajectory CSVs); each
    existing input is checksummed into the manifest's ``inputs`` block so results trace to exact
    input bytes (mirrors :func:`vesp.common.artifacts.write_run_manifest`). Returns the manifest
    dict. Output filenames are preserved exactly; ``run_manifest.json`` is added alongside them.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = utc_now_iso()
    cfg_map: Mapping[str, Any] = config if isinstance(config, Mapping) else {}
    if seed is None:
        seed = cfg_map.get("seed")
    if config_path is None:
        config_path = cfg_map.get("_config_path")
    provenance = {
        "tool": tool,
        "generated_at": generated_at,
        "seed": seed,
        "config_path": config_path,
    }

    written: list[str] = []
    for name, payload in (json_files or {}).items():
        body = dict(payload)
        body.setdefault("_provenance", provenance)
        atomic_write_json(out_dir / name, body)
        written.append(name)
    for name, text in (text_files or {}).items():
        atomic_write_text(out_dir / name, text)
        written.append(name)

    artifacts = {
        name: {
            "sha256": compute_file_sha256(out_dir / name),
            "bytes": (out_dir / name).stat().st_size,
        }
        for name in written
    }
    input_payload: dict[str, dict[str, Any]] = {}
    for name, input_path in (inputs or {}).items():
        p = Path(input_path)
        if p.exists() and p.is_file():
            input_payload[str(name)] = {
                "path": str(p),
                "sha256": compute_file_sha256(p),
                "bytes": p.stat().st_size,
            }
        else:
            input_payload[str(name)] = {"path": str(p), "missing": True}
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "tool": tool,
        "created_at_utc": generated_at,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": seed,
        "config_path": config_path,
        "config": json_safe(dict(cfg_map)),
        "artifacts": artifacts,
        "inputs": input_payload,
    }
    atomic_write_json(out_dir / manifest_name, manifest)
    return manifest
