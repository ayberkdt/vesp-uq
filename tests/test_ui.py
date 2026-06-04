import json
import os
from pathlib import Path

import pytest
import torch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication

from experimental_vesp import ui


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(["vesp-ui-test"])
    return app


def _window(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(ui, "DEFAULT_OUTPUTS", tmp_path)
    window = ui.VespWorkbench()
    yield window
    window.close()


def _write_checkpoint(path: Path, *, run_name: str = "ui_run", normalize_targets: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "dtype": "float64",
        "data": {"type": "csv", "path": "data/lunar_grail_gl0420a_L60_residual.csv"},
        "model": {"type": "discrete", "shell_alpha": 0.86, "n_source": 4},
        "loss": {
            "normalize_targets": normalize_targets,
            "resolved_potential_scale": 2.0,
            "resolved_acceleration_scale": 5.0,
        },
        "output": {"output_dir": str(path.parent.parent), "run_name": run_name},
    }
    torch.save(
        {
            "source_positions": torch.zeros(4, 3, dtype=torch.float64),
            "source_weights": torch.ones(4, dtype=torch.float64),
            "shell_ids": torch.zeros(4, dtype=torch.long),
            "shell_radii": (0.86,),
            "sigma": torch.ones(4, dtype=torch.float64),
            "config": config,
            "metrics": {
                "potential_rmse": 1.0e-4,
                "acceleration_rmse": 2.0e-4,
                "relative_acceleration_rmse": 3.0e-2,
                "angle_deg_p95": 4.0,
                "diagnostics": {
                    "top_5pct_source_contribution": 0.25,
                    "sigma_abs_max": 1.0,
                    "monopole_leakage": 0.0,
                    "dipole_leakage": 0.0,
                    "shell_energy": [1.0],
                },
            },
        },
        path,
    )
    return path


def test_ui_builds_with_root_config_presets(qapp, monkeypatch, tmp_path):
    for window in _window(qapp, monkeypatch, tmp_path):
        assert window.windowTitle() == "VESP Workbench"
        assert Path(window.config_path.text()).name == "discrete_single_shell.yaml"
        assert window.preset_combo.count() == len(ui.CONFIG_PRESETS)
        assert "discrete" in window.summary_labels["model"].text()
        assert window.summary_labels["data"].text() == "synthetic"
        assert window.summary_labels["scaling"].text() == "off"


def test_ui_feasibility_config_routes_to_feasibility_module(qapp, monkeypatch, tmp_path):
    for window in _window(qapp, monkeypatch, tmp_path):
        window._load_config(ui.DEFAULT_FEASIBILITY_CONFIG)
        assert window._is_feasibility_config()
        assert window.summary_labels["model"].text() == "suite"
        assert "feasibility" in window.summary_labels["output"].text()

        called = []
        monkeypatch.setattr(window, "_run_training", lambda module: called.append(module))
        window._run_selected_config()
        assert called == ["experimental_vesp.feasibility"]


def test_ui_train_config_routes_to_unified_train_module(qapp, monkeypatch, tmp_path):
    for window in _window(qapp, monkeypatch, tmp_path):
        window._load_config(ui.DEFAULT_REAL_CONFIG)
        assert not window._is_feasibility_config()
        assert "lunar_grail" in window.summary_labels["data"].text()
        assert window.summary_labels["scaling"].text().startswith("auto:")

        called = []
        monkeypatch.setattr(window, "_run_training", lambda module: called.append(module))
        window._run_selected_config()
        assert called == ["experimental_vesp.train"]


def test_ui_discovers_nested_sigma_without_auto_selecting(qapp, monkeypatch, tmp_path):
    run_dir = tmp_path / "ui_run"
    _write_checkpoint(run_dir / "sigma.pt")
    _write_checkpoint(tmp_path / "top_level.pt", run_name="top_level", normalize_targets=False)

    for window in _window(qapp, monkeypatch, tmp_path):
        labels = [window.checkpoint_list.item(i).text() for i in range(window.checkpoint_list.count())]
        assert "ui_run/sigma.pt" in labels
        assert "top_level.pt" in labels
        assert window.results_table.rowCount() == 0


def test_ui_results_table_reads_manifest_and_target_scales(qapp, monkeypatch, tmp_path):
    run_dir = tmp_path / "ui_run"
    ckpt = _write_checkpoint(run_dir / "sigma.pt")
    (run_dir / "run_manifest.json").write_text(json.dumps({"schema_version": "vesp_run_manifest_v1"}), encoding="utf-8")
    (run_dir / "target_scales.json").write_text(
        json.dumps({"potential_scale": 7.0, "acceleration_scale": 11.0}),
        encoding="utf-8",
    )

    for window in _window(qapp, monkeypatch, tmp_path):
        for i in range(window.checkpoint_list.count()):
            item = window.checkpoint_list.item(i)
            if Path(item.data(Qt.ItemDataRole.UserRole)) == ckpt.resolve():
                item.setSelected(True)
                break
        window._update_results_table()

        assert window.results_table.rowCount() == 1
        values = [window.results_table.item(0, col).text() for col in range(window.results_table.columnCount())]
        assert values[2] == "lunar_grail_gl0420a_L60_residual.csv"
        assert values[3] == "on"
        assert values[4] == "7.000e+00"
        assert values[5] == "1.100e+01"
        assert values[10] == "ok"


def test_ui_analysis_generates_synced_table_and_pdf(qapp, monkeypatch, tmp_path):
    run_dir = tmp_path / "ui_run"
    ckpt = _write_checkpoint(run_dir / "sigma.pt")

    for window in _window(qapp, monkeypatch, tmp_path):
        for i in range(window.checkpoint_list.count()):
            item = window.checkpoint_list.item(i)
            if Path(item.data(Qt.ItemDataRole.UserRole)) == ckpt.resolve():
                item.setSelected(True)
                break

        window._generate_analysis()

        assert window.analysis_table.rowCount() == 1
        assert window.results_table.rowCount() == 1
        assert window.current_pdf_path is not None
        assert window.current_pdf_path.exists()
        assert window.current_pdf_path.suffix == ".pdf"
        assert "PDF:" in window.pdf_status.text()
        assert "VESP Experiment Analysis" in window.current_report_markdown


def test_ui_plot_gallery_lists_and_previews_png(qapp, monkeypatch, tmp_path):
    plot_path = tmp_path / "altitude_error.png"
    pixmap = QPixmap(120, 80)
    pixmap.fill(Qt.GlobalColor.red)
    assert pixmap.save(str(plot_path))

    for window in _window(qapp, monkeypatch, tmp_path):
        window._refresh_plot_gallery(tmp_path)

        assert window.plot_list.count() == 1
        assert window.analysis_tabs.tabText(2) == "Plots (1)"
        assert window.plot_preview.pixmap() is not None
