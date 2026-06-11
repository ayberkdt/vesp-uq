"""Contract tests for the committed band-limited residual datasets (L60 + N11's L90).

Each dataset must carry: the residual CSV header the loaders expect, a metadata sidecar with
the lunar unit contract and the exact degree band, and consistent loader behavior. Tests skip
per-dataset when the CSV is absent (e.g. a fresh clone before generation).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from vesp.common.lunar import validate_lunar_metadata_contract

ROOT = Path(__file__).resolve().parents[1]

DATASETS = (
    pytest.param("data/lunar_grail_gl0420a_L60_residual.csv", 2, 60, id="L60"),
    pytest.param("data/lunar_grail_gl0420a_L90_residual.csv", 31, 90, id="L90"),
)

EXPECTED_COLUMNS = ["x", "y", "z", "Delta U", "Delta a_x", "Delta a_y", "Delta a_z"]


def _dataset(path_str: str) -> Path:
    path = ROOT / path_str
    if not path.is_file():
        pytest.skip(f"dataset not generated locally: {path_str}")
    return path


@pytest.mark.parametrize("path_str, degree_min, degree_max", DATASETS)
def test_residual_csv_header_and_rows(path_str, degree_min, degree_max):
    path = _dataset(path_str)
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        n_rows = sum(1 for _ in reader)
    assert header == EXPECTED_COLUMNS
    assert n_rows >= 256, "residual dataset is suspiciously small"


@pytest.mark.parametrize("path_str, degree_min, degree_max", DATASETS)
def test_metadata_sidecar_contract(path_str, degree_min, degree_max):
    path = _dataset(path_str)
    sidecar = Path(str(path) + ".metadata.json")
    assert sidecar.is_file(), "residual dataset must ship its metadata sidecar"
    meta = json.loads(sidecar.read_text(encoding="utf-8"))

    assert meta["metadata_schema"] == "vesp_lunar_residual_csv_v1"
    # the lunar unit contract (mu / radius) must validate, not just be present
    validate_lunar_metadata_contract(meta)
    assert int(meta["degree_min"]) == degree_min
    assert int(meta["degree_max"]) == degree_max
    assert meta["gravity_model"] == "gl0420a"
    assert meta["position_units"] == "normalized"
    assert meta["acceleration_units"] in {"km/s^2", "normalized"}


@pytest.mark.parametrize("path_str, degree_min, degree_max", DATASETS)
def test_loader_reads_residual_band(path_str, degree_min, degree_max):
    import torch

    from vesp.uq.data import load_uq_samples_from_csv

    path = _dataset(path_str)
    samples = load_uq_samples_from_csv(path)
    n = samples.positions.shape[0]
    assert samples.error.shape == (n, 3)
    radii = torch.linalg.norm(samples.positions, dim=-1)
    assert float(radii.min()) >= 1.0, "query shell must sit outside the body"
    assert float(radii.max()) <= 2.0
    assert bool(torch.isfinite(samples.error).all())
    assert float(samples.error.abs().max()) > 0.0
