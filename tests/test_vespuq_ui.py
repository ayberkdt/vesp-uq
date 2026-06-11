"""Import-safety + pure-logic tests for the VESP-UQ Mission Console.

These tests deliberately NEVER instantiate ``QApplication`` (window construction needs an
interactive/offscreen display, which CI and sandboxed shells may not provide). They pin:

- every UI module imports cleanly when PyQt6 is available (skipped otherwise);
- module import stays cheap -- no torch / vesp.uq / matplotlib at import time (heavy imports
  must remain lazy inside worker callables so the app opens instantly);
- the display-free helpers (run scanning, repo paths) behave.
"""

from __future__ import annotations

import json
import sys

import pytest

pytest.importorskip("PyQt6.QtWidgets")

UI_MODULES = (
    "vesp.ui",
    "vesp.ui.theme",
    "vesp.ui.paths",
    "vesp.ui.jobs",
    "vesp.ui.widgets",
    "vesp.ui.app",
    "vesp.ui.pages.dashboard",
    "vesp.ui.pages.train",
    "vesp.ui.pages.screen",
    "vesp.ui.pages.propagate",
    "vesp.ui.pages.model",
    "vesp.ui.pages.update",
    "vesp.ui.pages.runs",
)


def test_all_ui_modules_import():
    import importlib

    for name in UI_MODULES:
        module = importlib.import_module(name)
        assert module is not None, name


def test_app_exposes_entry_point_and_pages():
    from vesp.ui import app

    assert callable(app.main)
    assert {key for key, _label in app.NAV_ITEMS} == {
        "dashboard", "train", "screen", "propagate", "model", "update", "runs",
    }


def test_ui_imports_stay_light():
    # torch / matplotlib / vesp.uq must NOT be import-time dependencies of the UI shell; they
    # load lazily inside worker callables so the window opens instantly. Checked in a clean
    # subprocess because the surrounding pytest session has already imported torch itself.
    import subprocess

    code = (
        "import importlib, sys\n"
        + "\n".join(f"importlib.import_module({name!r})" for name in UI_MODULES)
        + "\nfor heavy in ('torch', 'matplotlib', 'vesp.uq.plugin'):\n"
        "    assert heavy not in sys.modules, heavy + ' must stay a lazy import'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr


def test_theme_builds_consistent_qss():
    from vesp.ui import theme

    qss = theme.build_qss()
    assert "QMainWindow" in qss and "NavRail" in qss
    for token in ("bg", "accent", "card", "danger"):
        assert theme.TOKENS[token].startswith("#")
    assert theme.TOKENS["accent"] in qss


def test_scan_runs_classifies_kind_and_sorts(tmp_path):
    from vesp.ui.paths import scan_runs

    def _write(run, created, artifacts):
        d = tmp_path / run
        d.mkdir()
        (d / "run_manifest.json").write_text(
            json.dumps(
                {
                    "created_at_utc": created,
                    "metrics": {"n_flagged": 1},
                    "artifacts": dict.fromkeys(artifacts, {"path": "x"}),
                    "inputs": {},
                }
            ),
            encoding="utf-8",
        )

    _write("train_run", "2026-06-10T10:00:00Z", ["vespuq_report_json"])
    _write("serve_run", "2026-06-10T12:00:00Z", ["screening_report_json"])
    _write("bench_run", "2026-06-10T11:00:00Z", ["foo_json"])
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "run_manifest.json").write_text("{not json", encoding="utf-8")

    records = scan_runs(tmp_path)
    assert [r.name for r in records] == ["serve_run", "bench_run", "train_run"]  # newest first
    assert [r.kind for r in records] == ["serve", "other", "train"]
    assert records[0].metrics == {"n_flagged": 1}


def test_repo_root_detects_source_tree():
    from vesp.ui.paths import CONFIG_DIR, ROOT

    assert (ROOT / "pyproject.toml").is_file() or (ROOT / "configs").is_dir()
    assert CONFIG_DIR.name == "vespuq"


def test_launcher_is_a_thin_caller():
    from vesp.ui.paths import ROOT

    launcher = ROOT / "ui" / "app_vespuq.py"
    assert launcher.is_file(), "ui/app_vespuq.py must exist at the repo root"
    text = launcher.read_text(encoding="utf-8")
    assert "from vesp.ui.app import main" in text
    assert len(text.splitlines()) < 40, "launcher must stay a thin caller"
