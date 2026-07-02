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
import os
import sys

import pytest

pytest.importorskip("PyQt6.QtWidgets")

UI_MODULES = (
    "vesp.ui",
    "vesp.ui.theme",
    "vesp.ui.paths",
    "vesp.ui.helpers",
    "vesp.ui.jobs",
    "vesp.ui.widgets",
    "vesp.ui.app",
    "vesp.ui.pages.dashboard",
    "vesp.ui.pages.train",
    "vesp.ui.pages.screen",
    "vesp.ui.pages.compare",
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
        "dashboard", "train", "screen", "compare", "propagate", "model", "update", "runs",
    }
    assert all(label.isascii() for _key, label in app.NAV_ITEMS)


def test_ui_imports_stay_light():
    # torch / matplotlib / vesp.uq must NOT be import-time dependencies of the UI shell; they
    # load lazily inside worker callables so the window opens instantly. Checked in a clean
    # subprocess because the surrounding pytest session has already imported torch itself.
    import subprocess

    from vesp.ui.paths import ROOT

    code = (
        "import importlib, sys\n"
        + "\n".join(f"importlib.import_module({name!r})" for name in UI_MODULES)
        + "\nfor heavy in ('torch', 'matplotlib', 'vesp.uq.plugin'):\n"
        "    assert heavy not in sys.modules, heavy + ' must stay a lazy import'\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env
    )
    assert result.returncode == 0, result.stderr


def test_scoring_modes_cover_backend_registry():
    # The UI scoring list must stay in lockstep with the backend registry: a new mode (e.g. an M1/M2
    # directional score) that is not surfaced here is a UI-drift bug. Compared against the canonical
    # names so the backward-compatible aliases are not required in the dropdown.
    from vesp.ui.helpers import SCORING_MODES, scoring_hint
    from vesp.uq.scoring import SCORING_FUNCTIONS, canonical_scoring_name

    canonical = {canonical_scoring_name(m) for m in SCORING_FUNCTIONS}
    assert set(SCORING_MODES) == canonical
    assert len(SCORING_MODES) == len(set(SCORING_MODES))  # no duplicates / stray aliases
    # every M1/M2 mode carries a tooltip so the new capability is discoverable in the UI
    for mode in ("radial_expected", "anisotropy_gated", "largest_eigenvalue", "expected_epistemic"):
        assert mode in SCORING_MODES
        assert scoring_hint(mode)
    assert scoring_hint("supervisor_rel_p95").endswith("(p95-aggregated per trajectory)")


def test_train_and_screen_pages_expose_full_scoring_list():
    from vesp.ui.helpers import SCORING_MODES as CANON
    from vesp.ui.pages import screen, train

    assert set(CANON) <= set(screen.SCORING_MODES)
    assert set(CANON) <= set(train.SCORING_MODES)
    assert screen.SCORING_MODES[0] == screen.MODEL_DEFAULT   # sentinel stays first
    assert train.SCORING_MODES[0] == train.CONFIG_DEFAULT


def test_theme_builds_consistent_qss():
    from vesp.ui import theme

    qss = theme.build_qss()
    assert "QMainWindow" in qss and "NavRail" in qss
    for token in ("bg", "accent", "card", "danger"):
        assert theme.TOKENS[token].startswith("#")
    assert theme.TOKENS["accent"] in qss


def test_shared_ui_helpers_handle_invalid_values_and_json(tmp_path):
    from vesp.ui.helpers import fmt, safe_read_json

    assert fmt(1.23456, digits=3) == "1.23"
    assert fmt(None) == "--"
    assert fmt(float("nan")) == "--"
    assert fmt(float("inf")) == "--"

    valid = tmp_path / "valid.json"
    valid.write_text('{"value": 3}', encoding="utf-8")
    payload, error = safe_read_json(valid)
    assert payload == {"value": 3}
    assert error is None

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{broken", encoding="utf-8")
    payload, error = safe_read_json(invalid)
    assert payload is None
    assert error

    invalid_utf8 = tmp_path / "invalid_utf8.json"
    invalid_utf8.write_bytes(b"\xff")
    payload, error = safe_read_json(invalid_utf8)
    assert payload is None
    assert error


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
    (tmp_path / "wrong_shape").mkdir()
    (tmp_path / "wrong_shape" / "run_manifest.json").write_text("[]", encoding="utf-8")
    (tmp_path / "wrong_fields").mkdir()
    (tmp_path / "wrong_fields" / "run_manifest.json").write_text(
        '{"created_at_utc": "2026-06-09T00:00:00Z", "artifacts": [], "metrics": 3, "inputs": null}',
        encoding="utf-8",
    )

    records = scan_runs(tmp_path)
    assert [r.name for r in records] == ["serve_run", "bench_run", "train_run", "wrong_fields"]
    assert [r.kind for r in records] == ["serve", "other", "train", "other"]
    assert records[0].metrics == {"n_flagged": 1}
    assert records[-1].metrics == {}


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


def test_train_override_helper_sets_conformal_without_touching_defaults():
    from vesp.ui.pages.train import CONFIG_DEFAULT, apply_training_overrides

    base = {
        "seed": 0,
        "uq": {"risk": {"scoring": "supervisor_rel"}, "conformal": {"enabled": True}},
        "output": {"run_name": "base", "save_model": False},
    }
    unchanged = apply_training_overrides(base, conformal=CONFIG_DEFAULT)
    assert unchanged["uq"]["conformal"] == {"enabled": True}
    assert base["output"]["save_model"] is False

    updated = apply_training_overrides(
        base,
        run_name="ui_run",
        seed=42,
        scoring="expected_abs_p95",
        domain="on",
        conformal="on",
        save_model=True,
    )
    assert updated["output"]["run_name"] == "ui_run"
    assert updated["seed"] == 42
    assert updated["uq"]["risk"]["scoring"] == "expected_abs_p95"
    assert updated["uq"]["risk"]["domain_support"] is True
    assert updated["uq"]["conformal"]["apply"] is True
    assert updated["output"]["save_model"] is True
    assert base["uq"]["conformal"] == {"enabled": True}


def test_model_conformal_summary():
    from vesp.ui.pages.model import _conformal_summary

    assert _conformal_summary(None) == "off"
    assert _conformal_summary({"enabled": False}) == "off"
    assert _conformal_summary(
        {
            "enabled": True,
            "mode": "norm",
            "scope": "global",
            "global": {"scale": 0.81234},
        }
    ) == "0.812 (norm, global)"


def test_compare_helpers_format_missing_and_present_agreement():
    from vesp.ui.pages.compare import _agreement_summary, _drift_mapping

    missing = _agreement_summary({})
    assert missing["spearman"] == "--"
    assert "trajectory CSV" in missing["spearman_hint"]

    present = _agreement_summary(
        {
            "risk_spearman": 0.98765,
            "flag_overlap": 0.5,
            "n_flagged_A": 3,
            "n_flagged_B": 4,
        }
    )
    assert present["spearman"] == "0.9877"
    assert present["iou"] == "0.5"
    assert present["counts"] == "3 / 4"

    drift = _drift_mapping(
        {
            "posterior_distance": {"mean_l2_diff": 1.2, "cov_frob_diff": 3.4, "noise_var_delta": -0.1},
            "domain_shift": {"mean_score_on_A": 0.2, "max_score_on_A": 0.9},
        }
    )
    assert drift["posterior mean L2"] == "1.2"
    assert drift["domain score on A (max)"] == "0.9"
