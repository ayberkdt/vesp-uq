"""Tests for G7 -- the provenance-completeness checker (verify_manifest)."""

from __future__ import annotations

from vesp.uq.io.run_artifacts import verify_manifest, write_run_artifacts


def _build_run(out_dir):
    """Write two CSVs + a manifest via the real artifact writer."""

    return write_run_artifacts(
        out_dir,
        tool="test_provenance",
        text_files={"a.csv": "x,y\n1,2\n", "b.csv": "p,q\n3,4\n"},
        manifest_name="manifest.json",
    )


def test_clean_dir_verifies(tmp_path):
    _build_run(tmp_path)
    report = verify_manifest(tmp_path)
    assert report["ok"] is True
    assert set(report["verified"]) == {"a.csv", "b.csv"}
    assert report["changed"] == [] and report["missing"] == []


def test_tampered_byte_is_detected(tmp_path):
    _build_run(tmp_path)
    (tmp_path / "a.csv").write_text("x,y\n1,999\n", encoding="utf-8")  # changed after manifest
    report = verify_manifest(tmp_path)
    assert report["ok"] is False
    assert "a.csv" in report["changed"]


def test_missing_artifact_is_detected(tmp_path):
    _build_run(tmp_path)
    (tmp_path / "b.csv").unlink()
    report = verify_manifest(tmp_path)
    assert report["ok"] is False
    assert "b.csv" in report["missing"]


def test_unlisted_file_is_reported_but_not_fatal(tmp_path):
    _build_run(tmp_path)
    (tmp_path / "stray.log").write_text("noise\n", encoding="utf-8")
    report = verify_manifest(tmp_path)
    assert report["ok"] is True  # a stray file is not a provenance failure
    assert any(p.endswith("stray.log") for p in report["unlisted"])


def test_no_manifest_is_not_ok(tmp_path):
    (tmp_path / "lonely.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    report = verify_manifest(tmp_path)
    assert report["ok"] is False
    assert report["manifest"] is None
