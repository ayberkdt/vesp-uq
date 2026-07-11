"""Run artifact helpers inspired by the LUNAR_SIMULATION artifact layout."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch

RUN_MANIFEST_SCHEMA_VERSION = "vesp_run_manifest_v1"


@dataclass(frozen=True)
class RunLayout:
    run_dir: Path
    config_yaml: Path
    metrics_json: Path
    diagnostics_json: Path
    altitude_binned_error_csv: Path
    shell_energy_csv: Path
    summary_txt: Path
    run_manifest_json: Path
    checkpoints_dir: Path
    checkpoint_last: Path


def make_run_layout(run_dir: str | Path) -> RunLayout:
    run_dir = Path(run_dir).expanduser().resolve()
    checkpoints_dir = run_dir / "checkpoints"
    return RunLayout(
        run_dir=run_dir,
        config_yaml=run_dir / "config.yaml",
        metrics_json=run_dir / "metrics.json",
        diagnostics_json=run_dir / "diagnostics.json",
        altitude_binned_error_csv=run_dir / "altitude_binned_error.csv",
        shell_energy_csv=run_dir / "shell_energy.csv",
        summary_txt=run_dir / "summary.txt",
        run_manifest_json=run_dir / "run_manifest.json",
        checkpoints_dir=checkpoints_dir,
        checkpoint_last=run_dir / "sigma.pt",
    )


def ensure_run_layout(run_dir: str | Path) -> RunLayout:
    layout = make_run_layout(run_dir)
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    layout.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return layout


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return {str(k): json_safe(v) for k, v in asdict(cast(Any, value)).items()}
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (set, frozenset)):
        items = [json_safe(v) for v in value]
        return sorted(
            items,
            key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=True, default=str),
        )
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "detach") and callable(value.detach):
        try:
            return json_safe(value.detach().cpu().tolist())
        except Exception:
            return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return json_safe(value.item())
        except Exception:
            return str(value)
    return value


def canonical_json_text(payload: Mapping[str, Any], *, indent: int = 2) -> str:
    return (
        json.dumps(
            json_safe(dict(payload)),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
            default=str,
        )
        + "\n"
    )


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def atomic_write_json(path: str | Path, payload: Mapping[str, Any], *, indent: int = 2) -> None:
    atomic_write_text(path, canonical_json_text(payload, indent=indent))


def atomic_torch_save(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def compute_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest_entries(
    files: Mapping[str, str | Path] | None,
    *,
    origin: str,
    statuses: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build consistent manifest entries for produced, prewritten, or consumed files.

    Every entry records ``path`` and ``origin``. Existing files also record SHA-256 and byte size;
    missing files record ``missing: true``. Callers may add a machine-readable ``status`` and
    per-file metadata without changing the common checksum contract.
    """

    payload: dict[str, dict[str, Any]] = {}
    for name, file_path in (files or {}).items():
        p = Path(file_path)
        file_metadata = dict(metadata.get(name, {})) if metadata is not None else {}
        entry_origin = str(file_metadata.pop("origin", origin))
        entry: dict[str, Any] = {"path": str(p), "origin": entry_origin}
        if p.exists() and p.is_file():
            entry.update({"sha256": compute_file_sha256(p), "bytes": p.stat().st_size})
        else:
            entry["missing"] = True
        if statuses is not None and name in statuses:
            entry["status"] = json_safe(statuses[name])
        for key, value in file_metadata.items():
            if key not in entry:
                entry[str(key)] = json_safe(value)
        payload[str(name)] = entry
    return payload


def build_run_manifest(
    *,
    config: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, str | Path] | None = None,
    inputs: Mapping[str, str | Path] | None = None,
    artifact_statuses: Mapping[str, Any] | None = None,
    input_statuses: Mapping[str, Any] | None = None,
    artifact_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    input_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    manifest_metadata: Mapping[str, Any] | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build the shared VESP run-manifest schema without writing it."""

    manifest: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": created_at_utc or utc_now_iso(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": json_safe(dict(config or {})),
        "metrics": json_safe(dict(metrics or {})),
        "artifacts": file_manifest_entries(
            artifacts,
            origin="generated",
            statuses=artifact_statuses,
            metadata=artifact_metadata,
        ),
        "inputs": file_manifest_entries(
            inputs,
            origin="consumed",
            statuses=input_statuses,
            metadata=input_metadata,
        ),
    }
    reserved = set(manifest)
    for key, value in (manifest_metadata or {}).items():
        if key in reserved:
            raise ValueError(f"manifest_metadata cannot replace reserved field {key!r}")
        manifest[str(key)] = json_safe(value)
    return manifest


def write_manifest(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Build and atomically write a manifest to an explicit path."""

    manifest = build_run_manifest(**kwargs)
    atomic_write_json(path, manifest)
    return manifest


def write_run_manifest(
    run_dir: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, str | Path] | None = None,
    inputs: Mapping[str, str | Path] | None = None,
    artifact_statuses: Mapping[str, Any] | None = None,
    input_statuses: Mapping[str, Any] | None = None,
    artifact_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    input_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    manifest_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a compact provenance manifest for a completed run.

    ``artifacts`` are the files the run PRODUCED; ``inputs`` are the files it CONSUMED
    (datasets, trajectory CSVs, saved models). Both get the same path + SHA-256 + byte-size
    treatment, so a result can be traced to the exact inputs as well as verified outputs.
    The ``inputs`` key is additive (older manifests simply lack it).
    """

    layout = ensure_run_layout(run_dir)
    write_manifest(
        layout.run_manifest_json,
        config=config,
        metrics=metrics,
        artifacts=artifacts,
        inputs=inputs,
        artifact_statuses=artifact_statuses,
        input_statuses=input_statuses,
        artifact_metadata=artifact_metadata,
        input_metadata=input_metadata,
        manifest_metadata=manifest_metadata,
    )
    return layout.run_manifest_json


__all__ = [
    "RUN_MANIFEST_SCHEMA_VERSION",
    "RunLayout",
    "atomic_torch_save",
    "atomic_write_json",
    "atomic_write_text",
    "build_run_manifest",
    "compute_file_sha256",
    "ensure_run_layout",
    "file_manifest_entries",
    "json_safe",
    "make_run_layout",
    "utc_now_iso",
    "write_manifest",
    "write_run_manifest",
]
