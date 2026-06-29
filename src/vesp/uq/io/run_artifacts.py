"""Shared run-artifact writer for VESP-UQ scripts.

The benchmark / audit / screening scripts historically wrote bare JSON/MD/CSV files with no
provenance. This helper routes every output through the atomic writers in
:mod:`vesp.common.artifacts`, injects a small ``_provenance`` block into each JSON, and writes a
``run_manifest.json`` recording the config snapshot, seed, environment, and a SHA-256 checksum +
byte size for every emitted file -- so a result can be traced back to the exact config and verified.

It uses the shared manifest builder from :mod:`vesp.common.artifacts` while keeping this
script-oriented API free of the training layout's ``checkpoints/`` side effect.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vesp.common.artifacts import (
    atomic_write_json,
    atomic_write_text,
    compute_file_sha256,
    utc_now_iso,
    write_manifest,
)

MANIFEST_NAME = "run_manifest.json"
# Manifest filenames a study dir may use (the suite writes the short form).
_MANIFEST_CANDIDATES = ("manifest.json", MANIFEST_NAME)


def write_run_artifacts(
    out_dir: str | Path,
    *,
    tool: str,
    config: Mapping[str, Any] | None = None,
    json_files: Mapping[str, Mapping[str, Any]] | None = None,
    text_files: Mapping[str, str] | None = None,
    artifact_files: Mapping[str, str | Path] | None = None,
    artifact_statuses: Mapping[str, Any] | None = None,
    inputs: Mapping[str, str | Path] | None = None,
    seed: Any = None,
    config_path: str | None = None,
    manifest_name: str = MANIFEST_NAME,
) -> dict:
    """Write a VESP-UQ script's outputs atomically with a provenance manifest + checksums.

    ``json_files`` maps filename -> JSON-able payload (a ``_provenance`` block is injected unless one
    is already present); ``text_files`` maps filename -> text (Markdown / CSV). ``artifact_files``
    maps logical artifact names to files that were already written (for example PNG/PDF figures);
    they are checksummed into the manifest's ``artifacts`` block without being rewritten and marked
    with ``origin: prewritten``. ``artifact_statuses`` adds optional machine-readable status fields.
    ``inputs`` maps a logical name -> path of a file the run CONSUMED (saved models, datasets,
    trajectory CSVs); each existing input is checksummed into the manifest's ``inputs`` block.
    Returns the manifest dict. Output filenames are preserved exactly; ``run_manifest.json`` is
    added alongside them.
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

    artifacts: dict[str, str | Path] = {name: out_dir / name for name in written}
    artifacts.update({str(name): path for name, path in (artifact_files or {}).items()})
    artifact_metadata = {
        name: {"origin": "generated"} for name in written
    }
    artifact_metadata.update(
        {str(name): {"origin": "prewritten"} for name in (artifact_files or {})}
    )
    return write_manifest(
        out_dir / manifest_name,
        created_at_utc=generated_at,
        config=cfg_map,
        artifacts=artifacts,
        inputs=inputs,
        artifact_statuses=artifact_statuses,
        artifact_metadata=artifact_metadata,
        manifest_metadata={
            "tool": tool,
            "seed": seed,
            "config_path": config_path,
        },
    )


def _resolve_artifact_path(run_dir: Path, name: str, entry: Mapping[str, Any]) -> Path | None:
    """Best-effort locate an artifact on disk from its manifest entry (absolute paths may have moved)."""

    raw = entry.get("path")
    candidates: list[Path] = []
    if raw:
        p = Path(str(raw))
        candidates += [p, run_dir / p.name]
    candidates.append(run_dir / name)
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand
    if raw:  # fall back to a recursive search by basename (artifacts in figures/ etc.)
        matches = list(run_dir.rglob(Path(str(raw)).name))
        if matches:
            return matches[0]
    return None


def verify_manifest(run_dir: str | Path, *, manifest_name: str | None = None) -> dict:
    """G7 -- check every artifact a run manifest lists is present on disk with a matching SHA-256.

    Recomputes the SHA-256 of each file in ``run_manifest.json["artifacts"]`` and classifies it as
    ``verified`` / ``changed`` (checksum mismatch) / ``missing`` (absent or marked missing). Files on
    disk that no manifest entry lists are reported as ``unlisted`` (informational -- a stray figure or
    log is not a provenance failure). ``ok`` is true only when nothing is missing or changed.

    ``manifest_name`` forces a specific manifest filename; otherwise ``manifest.json`` then
    ``run_manifest.json`` is tried. A directory with no manifest returns ``ok=False`` with
    ``manifest=None``.
    """

    run_dir = Path(run_dir)
    names = [manifest_name] if manifest_name else list(_MANIFEST_CANDIDATES)
    manifest_path = next((run_dir / n for n in names if (run_dir / n).exists()), None)
    if manifest_path is None:
        return {"dir": str(run_dir), "manifest": None, "ok": False, "verified": [],
                "changed": [], "missing": [], "unlisted": [], "checked": 0}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts: Mapping[str, Any] = manifest.get("artifacts") or {}
    verified: list[str] = []
    changed: list[str] = []
    missing: list[str] = []
    listed: set[Path] = set()
    for name, entry in artifacts.items():
        resolved = _resolve_artifact_path(run_dir, name, entry)
        if entry.get("missing") or resolved is None:
            missing.append(name)
            continue
        listed.add(resolved.resolve())
        recorded = entry.get("sha256")
        if recorded and compute_file_sha256(resolved) != recorded:
            changed.append(name)
        else:
            verified.append(name)

    manifest_files = {(run_dir / n).resolve() for n in _MANIFEST_CANDIDATES}
    unlisted = sorted(
        str(p) for p in run_dir.rglob("*")
        if p.is_file() and p.resolve() not in listed and p.resolve() not in manifest_files
    )
    return {
        "dir": str(run_dir),
        "manifest": str(manifest_path),
        "verified": verified,
        "changed": changed,
        "missing": missing,
        "unlisted": unlisted,
        "checked": len(artifacts),
        "ok": not changed and not missing,
    }
