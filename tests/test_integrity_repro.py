"""Tests for G5 -- the determinism / reproducibility gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from vesp.common.config import load_config
from vesp.uq.integrity.reproducibility import (
    ReproducibilityError,
    assert_reproducible,
    compare_outputs,
    normalize_csv_text,
)
from vesp.uq.suite import run_reproducibility_check

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "vespuq" / "vespuq_smoke.yaml"

# Two "runs" of benchmark_runs.csv: identical metrics, different timing columns.
RUN_A = "band,seed,spearman,runtime_ms_per_traj,runtime_us_per_point\nL60,0,0.42,1.10,2.0\n"
RUN_B = "band,seed,spearman,runtime_ms_per_traj,runtime_us_per_point\nL60,0,0.42,9.90,8.7\n"
RUN_C = "band,seed,spearman,runtime_ms_per_traj,runtime_us_per_point\nL60,0,0.99,1.10,2.0\n"


def test_normalize_drops_timing_columns():
    out = normalize_csv_text(RUN_A)
    assert "runtime_ms_per_traj" not in out
    assert "runtime_us_per_point" not in out
    assert "spearman" in out and "0.42" in out


def test_timing_only_difference_is_reproducible():
    assert normalize_csv_text(RUN_A) == normalize_csv_text(RUN_B)  # only timing differs


def test_metric_difference_is_caught():
    assert normalize_csv_text(RUN_A) != normalize_csv_text(RUN_C)  # spearman differs


def _write_pair(tmp_path, a_text, b_text, name="benchmark_runs.csv"):
    da, db = tmp_path / "a", tmp_path / "b"
    da.mkdir()
    db.mkdir()
    (da / name).write_text(a_text, encoding="utf-8")
    (db / name).write_text(b_text, encoding="utf-8")
    return da, db


def test_compare_outputs_ok_on_timing_only_diff(tmp_path):
    da, db = _write_pair(tmp_path, RUN_A, RUN_B)
    report = compare_outputs(da, db, filenames=["benchmark_runs.csv"])
    assert report["ok"] is True
    assert report["files"]["benchmark_runs.csv"]["status"] == "identical"


def test_assert_reproducible_raises_on_metric_diff(tmp_path):
    da, db = _write_pair(tmp_path, RUN_A, RUN_C)
    with pytest.raises(ReproducibilityError):
        assert_reproducible(da, db, filenames=["benchmark_runs.csv"])


def test_missing_file_is_not_ok(tmp_path):
    da, db = tmp_path / "a", tmp_path / "b"
    da.mkdir()
    db.mkdir()
    (da / "benchmark_runs.csv").write_text(RUN_A, encoding="utf-8")  # absent in b
    report = compare_outputs(da, db, filenames=["benchmark_runs.csv"])
    assert report["ok"] is False
    assert report["files"]["benchmark_runs.csv"]["status"] == "missing"


@pytest.mark.skipif(not SMOKE_CONFIG.exists(), reason="smoke config missing")
def test_smoke_suite_is_byte_reproducible(tmp_path):
    cfg = load_config(str(SMOKE_CONFIG))
    cfg["_config_path"] = str(SMOKE_CONFIG)
    report = run_reproducibility_check([cfg], seeds=(0,), rerun_fractions=(0.1, 0.2),
                                       out_root=tmp_path / "repro")
    assert report["ok"] is True, report["files"]
    assert report["checked"] == 3  # benchmark_runs, decision_quality, calibration_summary
