"""Run artifact helpers inspired by the LUNAR_SIMULATION artifact layout."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return {str(k): json_safe(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "detach") and callable(getattr(value, "detach")):
        try:
            return value.detach().cpu().tolist()
        except Exception:
            return str(value)
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def canonical_json_text(payload: Mapping[str, Any], *, indent: int = 2) -> str:
    return json.dumps(json_safe(dict(payload)), indent=indent, sort_keys=True, ensure_ascii=True, default=str) + "\n"


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


def write_run_manifest(
    run_dir: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, str | Path] | None = None,
) -> Path:
    """Write a compact provenance manifest for a completed run."""

    layout = ensure_run_layout(run_dir)
    artifact_payload: dict[str, dict[str, Any]] = {}
    for name, artifact_path in (artifacts or {}).items():
        p = Path(artifact_path)
        if p.exists() and p.is_file():
            artifact_payload[str(name)] = {
                "path": str(p),
                "sha256": compute_file_sha256(p),
                "bytes": p.stat().st_size,
            }
        else:
            artifact_payload[str(name)] = {"path": str(p), "missing": True}

    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": dict(config or {}),
        "metrics": dict(metrics or {}),
        "artifacts": artifact_payload,
    }
    atomic_write_json(layout.run_manifest_json, manifest)
    return layout.run_manifest_json


__all__ = [
    "RUN_MANIFEST_SCHEMA_VERSION",
    "RunLayout",
    "atomic_torch_save",
    "atomic_write_json",
    "atomic_write_text",
    "compute_file_sha256",
    "ensure_run_layout",
    "json_safe",
    "make_run_layout",
    "utc_now_iso",
    "write_run_manifest",
]
