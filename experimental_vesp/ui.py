"""PyQt6 workbench for running and analyzing discrete VESP experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from PyQt6.QtCore import QSize, QProcess, Qt
from PyQt6.QtGui import QAction, QFont, QPixmap, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
DEFAULT_SINGLE_CONFIG = CONFIG_ROOT / "discrete_single_shell.yaml"
DEFAULT_MULTI_CONFIG = CONFIG_ROOT / "discrete_multishell.yaml"
DEFAULT_ALTITUDE_CONFIG = CONFIG_ROOT / "altitude_ood.yaml"
DEFAULT_REAL_CONFIG = CONFIG_ROOT / "real_lunar_gl0420a.yaml"
DEFAULT_REAL_MULTI_CONFIG = CONFIG_ROOT / "real_lunar_gl0420a_multishell.yaml"
DEFAULT_FEASIBILITY_CONFIG = CONFIG_ROOT / "feasibility_suite.yaml"
DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"

CONFIG_PRESETS = {
    "Single Shell": DEFAULT_SINGLE_CONFIG,
    "Multi Shell": DEFAULT_MULTI_CONFIG,
    "Altitude OOD": DEFAULT_ALTITUDE_CONFIG,
    "Real Lunar": DEFAULT_REAL_CONFIG,
    "Real Lunar Multi": DEFAULT_REAL_MULTI_CONFIG,
    "Feasibility Suite": DEFAULT_FEASIBILITY_CONFIG,
}

try:
    from .analysis import interpret_experiment, load_checkpoint_summary, make_markdown_report, write_markdown_report
    from .advanced_analysis import make_advanced_report, write_analysis_pdf
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT))
    from experimental_vesp.analysis import interpret_experiment, load_checkpoint_summary, make_markdown_report, write_markdown_report
    from experimental_vesp.advanced_analysis import make_advanced_report, write_analysis_pdf


class VespWorkbench(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VESP Workbench")
        self.resize(1320, 820)
        self.process: QProcess | None = None
        self.checkpoint_paths: list[Path] = []
        self.current_report_markdown = ""
        self.current_pdf_path: Path | None = None
        self.current_assets_dir: Path | None = None
        self.summary_labels: dict[str, QLabel] = {}
        self.active_config: dict = {}

        self._build_menu()
        self._build_ui()
        self._apply_style()
        self._load_config(DEFAULT_SINGLE_CONFIG)
        self._refresh_output_checkpoints()

    def _build_menu(self) -> None:
        open_outputs = QAction("Open Outputs Folder", self)
        open_outputs.triggered.connect(lambda: self._add_checkpoints_from_dir(DEFAULT_OUTPUTS))

        refresh = QAction("Refresh Checkpoints", self)
        refresh.triggered.connect(self._refresh_output_checkpoints)

        load_root_configs = QAction("Root Config Presets", self)
        load_root_configs.triggered.connect(lambda: self._load_config(DEFAULT_SINGLE_CONFIG))

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(load_root_configs)
        file_menu.addAction(open_outputs)
        file_menu.addAction(refresh)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_header())

        tabs = QTabWidget()
        tabs.addTab(self._build_run_tab(), "Run")
        tabs.addTab(self._build_analysis_tab(), "Analysis")
        tabs.addTab(self._build_results_tab(), "Results")
        root_layout.addWidget(tabs, stretch=1)
        self.setCentralWidget(root)

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("TopHeader")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 18, 24, 16)
        layout.setSpacing(18)

        title_block = QVBoxLayout()
        title_block.setSpacing(4)
        title = QLabel("MaxEnt-VESP Workbench")
        title.setObjectName("HeaderTitle")
        subtitle = QLabel("Stage 1-2 deterministic feasibility")
        subtitle.setObjectName("HeaderSubtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)

        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        for text in ("Unit-safe CSV", "Target scales", "Run manifest"):
            chip = QLabel(text)
            chip.setObjectName("HeaderChip")
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            status_row.addWidget(chip)
        status_row.addStretch(1)

        layout.addLayout(title_block, stretch=1)
        layout.addLayout(status_row, stretch=0)
        return header

    def _button(self, text: str, icon: QStyle.StandardPixmap) -> QPushButton:
        button = QPushButton(text)
        button.setIcon(self.style().standardIcon(icon))
        button.setIconSize(QSize(16, 16))
        return button

    def _summary_card(self, key: str, title: str) -> QFrame:
        card = QFrame()
        card.setObjectName("SummaryCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(5)
        label = QLabel(title)
        label.setObjectName("SummaryTitle")
        value = QLabel("-")
        value.setObjectName("SummaryValue")
        value.setWordWrap(True)
        value.setMinimumWidth(140)
        self.summary_labels[key] = value
        layout.addWidget(label)
        layout.addWidget(value)
        return card

    def _load_selected_preset(self, name: str) -> None:
        path = CONFIG_PRESETS.get(name)
        if path is not None and path != Path(self.config_path.text()):
            self._load_config(path)

    def _read_editor_config(self) -> dict:
        text = self.config_editor.toPlainText() if hasattr(self, "config_editor") else ""
        try:
            loaded = yaml.safe_load(text) if text.strip() else {}
        except yaml.YAMLError:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _refresh_config_summary_from_editor(self) -> None:
        cfg = self._read_editor_config()
        self.active_config = cfg
        if not self.summary_labels:
            return

        if self._is_feasibility_config(cfg):
            base = cfg.get("base", {})
            feasibility = cfg.get("feasibility", {})
            self.summary_labels["model"].setText("suite")
            self.summary_labels["data"].setText("synthetic scenarios")
            self.summary_labels["units"].setText(str(base.get("dtype", "float64")))
            self.summary_labels["scaling"].setText("per scenario")
            self.summary_labels["output"].setText(str(base.get("output", {}).get("output_dir", "outputs/feasibility")))
            self.summary_labels["output"].setToolTip(", ".join(sorted(feasibility.keys())))
            return

        model = cfg.get("model", {})
        data = cfg.get("data", {})
        body = cfg.get("body", {})
        loss = cfg.get("loss", {})
        output = cfg.get("output", {})

        model_type = str(model.get("type", "-"))
        if model_type == "multishell":
            shells = model.get("shell_alphas", [])
            counts = model.get("n_sources_per_shell", [])
            model_text = f"multishell / {len(shells)} shells / {sum(counts) if isinstance(counts, list) else counts} src"
        else:
            model_text = f"{model_type} / alpha {model.get('shell_alpha', '-')} / {model.get('n_source', '-')} src"

        data_type = str(data.get("type", "synthetic"))
        data_path = data.get("path")
        data_text = data_type if not data_path else Path(str(data_path)).name

        position_units = body.get("position_units", "normalized")
        normalize_positions = body.get("normalize_positions", True)
        r_body = body.get("R_body", 1.0)
        units_text = f"R={r_body}, {'normalized' if normalize_positions else position_units}"

        if bool(loss.get("normalize_targets", False)):
            scale_text = f"auto: U={loss.get('potential_scale', 'auto')}, a={loss.get('acceleration_scale', 'auto')}"
        else:
            scale_text = "off"

        run_name = output.get("run_name", "vesp_run")
        output_dir = output.get("output_dir", "outputs")
        self.summary_labels["model"].setText(model_text)
        self.summary_labels["data"].setText(data_text)
        self.summary_labels["units"].setText(units_text)
        self.summary_labels["scaling"].setText(scale_text)
        self.summary_labels["output"].setText(f"{output_dir}/{run_name}")

    def _is_feasibility_config(self, cfg: dict | None = None) -> bool:
        cfg = cfg if cfg is not None else self.active_config
        return isinstance(cfg, dict) and "feasibility" in cfg and "model" not in cfg

    def _build_run_tab(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(14)

        controls = QGroupBox("Experiment")
        grid = QGridLayout(controls)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        self.preset_combo = QComboBox()
        for name in CONFIG_PRESETS:
            self.preset_combo.addItem(name)
        self.preset_combo.currentTextChanged.connect(self._load_selected_preset)

        self.config_path = QLineEdit(str(DEFAULT_SINGLE_CONFIG))
        browse_config = self._button("Browse", QStyle.StandardPixmap.SP_DialogOpenButton)
        browse_config.clicked.connect(self._browse_config)

        save_config = self._button("Save", QStyle.StandardPixmap.SP_DialogSaveButton)
        save_config.clicked.connect(self._save_config)

        run_selected = self._button("Run", QStyle.StandardPixmap.SP_MediaPlay)
        run_selected.setObjectName("PrimaryButton")
        run_selected.clicked.connect(self._run_selected_config)
        stop = self._button("Stop", QStyle.StandardPixmap.SP_MediaStop)
        stop.setObjectName("DangerButton")
        stop.clicked.connect(self._stop_process)

        preset_label = QLabel("Preset")
        preset_label.setObjectName("FieldLabel")
        config_label = QLabel("Config")
        config_label.setObjectName("FieldLabel")
        grid.addWidget(preset_label, 0, 0)
        grid.addWidget(self.preset_combo, 0, 1, 1, 2)
        grid.addWidget(save_config, 0, 3)
        grid.addWidget(run_selected, 0, 4)
        grid.addWidget(stop, 0, 5)
        grid.addWidget(config_label, 1, 0)
        grid.addWidget(self.config_path, 1, 1, 1, 4)
        grid.addWidget(browse_config, 1, 5)

        summary = QFrame()
        summary.setObjectName("SummaryStrip")
        summary_grid = QGridLayout(summary)
        summary_grid.setContentsMargins(0, 0, 0, 0)
        summary_grid.setHorizontalSpacing(12)
        summary_grid.setVerticalSpacing(12)
        for idx, (key, title) in enumerate(
            [
                ("model", "Model"),
                ("data", "Data"),
                ("units", "Units"),
                ("scaling", "Target Scaling"),
                ("output", "Output"),
            ]
        ):
            card = self._summary_card(key, title)
            summary_grid.addWidget(card, 0, idx)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.config_editor = QPlainTextEdit()
        self.config_editor.setFont(QFont("Consolas", 10))
        self.config_editor.setPlaceholderText("YAML config")
        self.config_editor.textChanged.connect(self._refresh_config_summary_from_editor)

        self.run_log = QPlainTextEdit()
        self.run_log.setReadOnly(True)
        self.run_log.setFont(QFont("Consolas", 10))
        self.run_log.setPlaceholderText("Run log")

        splitter.addWidget(self.config_editor)
        splitter.addWidget(self.run_log)
        splitter.setSizes([560, 700])

        layout.addWidget(controls)
        layout.addWidget(summary)
        layout.addWidget(splitter, stretch=1)
        return root

    def _build_analysis_tab(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(14)

        controls = QGroupBox("Checkpoints")
        row = QHBoxLayout(controls)
        add_files = self._button("Add Files", QStyle.StandardPixmap.SP_FileDialogNewFolder)
        add_files.clicked.connect(self._browse_checkpoints)
        add_dir = self._button("Add Folder", QStyle.StandardPixmap.SP_DirOpenIcon)
        add_dir.clicked.connect(lambda: self._choose_checkpoint_dir())
        select_all = self._button("Select All", QStyle.StandardPixmap.SP_DialogApplyButton)
        select_all.clicked.connect(self._select_all_checkpoints)
        remove = self._button("Remove", QStyle.StandardPixmap.SP_DialogDiscardButton)
        remove.clicked.connect(self._remove_selected_checkpoints)
        refresh = self._button("Refresh", QStyle.StandardPixmap.SP_BrowserReload)
        refresh.clicked.connect(self._refresh_output_checkpoints)
        analyze = self._button("Analyze", QStyle.StandardPixmap.SP_FileDialogDetailedView)
        analyze.setObjectName("PrimaryButton")
        analyze.clicked.connect(self._generate_analysis)
        deep_analyze = self._button("Deep Analysis", QStyle.StandardPixmap.SP_ComputerIcon)
        deep_analyze.clicked.connect(self._generate_deep_analysis)
        pdf = self._button("PDF", QStyle.StandardPixmap.SP_FileIcon)
        pdf.clicked.connect(self._export_pdf)
        save = self._button("Save Report", QStyle.StandardPixmap.SP_DialogSaveButton)
        save.clicked.connect(self._save_report)

        for widget in (add_files, add_dir, select_all, remove, refresh, analyze, deep_analyze, pdf, save):
            row.addWidget(widget)
        row.addStretch(1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.checkpoint_list = QListWidget()
        self.checkpoint_list.setObjectName("CheckpointList")
        self.checkpoint_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.checkpoint_list.itemSelectionChanged.connect(self._update_results_table)

        self.analysis_tabs = QTabWidget()

        self.report_view = QTextBrowser()
        self.report_view.setOpenExternalLinks(True)
        self.report_view.setPlaceholderText("Analysis")

        self.analysis_table = self._make_analysis_table()

        plot_root = QWidget()
        plot_layout = QHBoxLayout(plot_root)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(10)
        self.plot_list = QListWidget()
        self.plot_list.setObjectName("PlotList")
        self.plot_list.itemSelectionChanged.connect(self._preview_selected_plot)
        self.plot_preview = QLabel("Generate Deep Analysis to see plots.")
        self.plot_preview.setObjectName("PlotPreview")
        self.plot_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_preview.setMinimumSize(520, 360)
        self.plot_preview.setWordWrap(True)
        plot_scroll = QScrollArea()
        plot_scroll.setWidgetResizable(True)
        plot_scroll.setWidget(self.plot_preview)
        plot_layout.addWidget(self.plot_list, stretch=1)
        plot_layout.addWidget(plot_scroll, stretch=3)

        self.analysis_tabs.addTab(self.report_view, "Report")
        self.analysis_tabs.addTab(self.analysis_table, "Table")
        self.analysis_tabs.addTab(plot_root, "Plots")

        self.pdf_status = QLabel("PDF: -")
        self.pdf_status.setObjectName("PdfStatus")

        splitter.addWidget(self.checkpoint_list)
        splitter.addWidget(self.analysis_tabs)
        splitter.setSizes([420, 880])

        layout.addWidget(controls)
        layout.addWidget(splitter, stretch=1)
        layout.addWidget(self.pdf_status)
        return root

    def _build_results_tab(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(12)

        toolbar = QFrame()
        toolbar.setObjectName("InlineToolbar")
        row = QHBoxLayout(toolbar)
        row.setContentsMargins(12, 10, 12, 10)
        row.setSpacing(8)
        refresh = self._button("Refresh", QStyle.StandardPixmap.SP_BrowserReload)
        refresh.clicked.connect(self._refresh_output_checkpoints)
        analyze = self._button("Analyze Selected", QStyle.StandardPixmap.SP_FileDialogDetailedView)
        analyze.setObjectName("PrimaryButton")
        analyze.clicked.connect(self._generate_analysis)
        row.addWidget(refresh)
        row.addWidget(analyze)
        row.addStretch(1)

        self.results_table = QTableWidget(0, 12)
        self.results_table.setHorizontalHeaderLabels(
            [
                "Run",
                "Model",
                "Data",
                "Target Norm",
                "U Scale",
                "A Scale",
                "Acc RMSE",
                "Rel Acc",
                "Angle p95",
                "Top 5%",
                "Manifest",
                "Path",
            ]
        )
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(11, QHeaderView.ResizeMode.Stretch)
        self.results_table.verticalHeader().setVisible(False)
        layout.addWidget(toolbar)
        layout.addWidget(self.results_table)
        return root

    def _make_analysis_table(self) -> QTableWidget:
        table = QTableWidget(0, 12)
        table.setHorizontalHeaderLabels(
            [
                "Run",
                "Model",
                "Data",
                "Target Norm",
                "U Scale",
                "A Scale",
                "Acc RMSE",
                "Rel Acc",
                "Angle p95",
                "Top 5%",
                "Manifest",
                "Path",
            ]
        )
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(11, QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        return table

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f3f6f8;
            }
            QMenuBar {
                background: #121820;
                color: #edf3f6;
                padding: 5px 10px;
            }
            QMenuBar::item {
                background: transparent;
                padding: 6px 11px;
                border-radius: 4px;
            }
            QMenuBar::item:selected {
                background: #24313d;
            }
            QMenu {
                background: #ffffff;
                color: #141c24;
                border: 1px solid #d6dee5;
            }
            QMenu::item:selected {
                background: #e5f0ed;
            }
            #TopHeader {
                background: #121820;
                border-bottom: 1px solid #263240;
            }
            #HeaderTitle {
                color: #f7fbfc;
                font-size: 22px;
                font-weight: 700;
            }
            #HeaderSubtitle {
                color: #aab8c2;
                font-size: 12px;
            }
            #HeaderChip {
                color: #dce8e4;
                background: #1d2a32;
                border: 1px solid #354752;
                border-radius: 8px;
                padding: 6px 10px;
                font-size: 12px;
            }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #d8e0e6;
                border-radius: 8px;
                margin-top: 12px;
                padding: 12px;
                background: #ffffff;
                color: #17212b;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #2b3742;
            }
            QPushButton {
                background: #1c6674;
                color: white;
                border: 0;
                border-radius: 7px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #247b8b;
            }
            QPushButton:pressed {
                background: #15505c;
            }
            #PrimaryButton {
                background: #0f766e;
            }
            #PrimaryButton:hover {
                background: #12867d;
            }
            #DangerButton {
                background: #9f3a45;
            }
            #DangerButton:hover {
                background: #b64955;
            }
            QToolButton {
                background: #ffffff;
                border: 1px solid #d8e0e6;
                border-radius: 7px;
                padding: 7px;
            }
            QLineEdit, QComboBox, QPlainTextEdit, QTextBrowser, QListWidget, QTableWidget {
                background: #ffffff;
                color: #17212b;
                border: 1px solid #d1dbe3;
                border-radius: 7px;
                selection-background-color: #cfe8e2;
                padding: 6px;
            }
            QPlainTextEdit, QTextBrowser, QListWidget, QTableWidget {
                padding: 0px;
            }
            QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QTextBrowser:focus, QListWidget:focus, QTableWidget:focus {
                border: 1px solid #1c6674;
            }
            QComboBox::drop-down {
                border: 0;
                width: 24px;
            }
            QTabWidget::pane {
                border: 0;
                background: #f3f6f8;
            }
            QTabBar::tab {
                background: #1c2731;
                color: #b9c7cf;
                padding: 10px 18px;
                margin-right: 2px;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #141c24;
                font-weight: 600;
            }
            QTabBar::tab:hover:!selected {
                background: #2a3845;
                color: #ffffff;
            }
            #SummaryStrip {
                background: transparent;
            }
            #SummaryCard {
                background: #ffffff;
                border: 1px solid #d8e0e6;
                border-radius: 8px;
            }
            #SummaryTitle {
                color: #6d7a86;
                font-size: 11px;
                font-weight: 600;
            }
            #SummaryValue {
                color: #17212b;
                font-size: 13px;
                font-weight: 700;
            }
            #FieldLabel {
                color: #4e5d68;
                font-weight: 600;
            }
            #InlineToolbar {
                background: #ffffff;
                border: 1px solid #d8e0e6;
                border-radius: 8px;
            }
            #CheckpointList::item {
                padding: 8px;
            }
            #CheckpointList::item:selected {
                background: #dcefeb;
                color: #14242a;
            }
            #PlotList::item {
                padding: 8px;
            }
            #PlotList::item:selected {
                background: #dcefeb;
                color: #14242a;
            }
            #PlotPreview {
                background: #ffffff;
                color: #66737c;
                border: 1px solid #d8e0e6;
                border-radius: 8px;
                padding: 14px;
            }
            #PdfStatus {
                color: #3c4a55;
                background: #ffffff;
                border: 1px solid #d8e0e6;
                border-radius: 8px;
                padding: 9px 12px;
                font-weight: 600;
            }
            QHeaderView::section {
                background: #e9eef2;
                color: #32424f;
                padding: 7px;
                border: 0;
                border-right: 1px solid #d4dde5;
                font-weight: 700;
            }
            QTableWidget {
                gridline-color: #e1e7ec;
                alternate-background-color: #f8fafb;
            }
            """
        )

    def _browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Config",
            str(CONFIG_ROOT),
            "YAML files (*.yaml *.yml);;All files (*)",
        )
        if path:
            self._load_config(Path(path))

    def _load_config(self, path: Path) -> None:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Config Error", str(exc))
            return
        self.config_path.setText(str(path))
        for idx, preset_path in enumerate(CONFIG_PRESETS.values()):
            if preset_path.resolve() == path.resolve():
                self.preset_combo.blockSignals(True)
                self.preset_combo.setCurrentIndex(idx)
                self.preset_combo.blockSignals(False)
                break
        self.config_editor.setPlainText(text)
        self._refresh_config_summary_from_editor()

    def _save_config(self) -> None:
        path = Path(self.config_path.text())
        try:
            path.write_text(self.config_editor.toPlainText(), encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
            return
        self._append_log(f"Saved config: {path}\n")

    def _run_selected_config(self) -> None:
        cfg = self._read_editor_config()
        module = "experimental_vesp.feasibility" if self._is_feasibility_config(cfg) else "experimental_vesp.train"
        self._run_training(module)

    def _run_training(self, module: str) -> None:
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Process Running", "An experiment is already running.")
            return

        self._save_config()
        self.run_log.clear()
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        self.process.setProgram(sys.executable)
        self.process.setArguments(["-m", module, "--config", self.config_path.text()])
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._process_finished)
        self._append_log(f"Running: {sys.executable} -m {module} --config {self.config_path.text()}\n\n")
        self.process.start()

    def _read_stdout(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._append_log(text)

    def _read_stderr(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        self._append_log(text)

    def _process_finished(self, exit_code: int, _exit_status=None) -> None:
        self._append_log(f"\nProcess finished with exit code {exit_code}.\n")
        self._refresh_output_checkpoints()

    def _stop_process(self) -> None:
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
            self._append_log("Stopped process.\n")

    def _append_log(self, text: str) -> None:
        self.run_log.moveCursor(QTextCursor.MoveOperation.End)
        self.run_log.insertPlainText(text)
        self.run_log.moveCursor(QTextCursor.MoveOperation.End)

    def _browse_checkpoints(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Checkpoints",
            str(DEFAULT_OUTPUTS),
            "PyTorch checkpoints (*.pt);;All files (*)",
        )
        self._add_checkpoint_paths(Path(path) for path in paths)

    def _choose_checkpoint_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Checkpoint Folder", str(DEFAULT_OUTPUTS))
        if path:
            self._add_checkpoints_from_dir(Path(path))

    def _refresh_output_checkpoints(self) -> None:
        self._add_checkpoints_from_dir(DEFAULT_OUTPUTS)

    def _add_checkpoints_from_dir(self, path: Path) -> None:
        if not path.exists():
            return
        candidates = sorted(path.glob("*.pt"))
        candidates.extend(sorted(path.glob("*/sigma.pt")))
        self._add_checkpoint_paths(candidates)

    def _add_checkpoint_paths(self, paths) -> None:
        existing = {str(path.resolve()) for path in self.checkpoint_paths}
        changed = False
        for raw in paths:
            path = Path(raw)
            if not path.exists() or path.suffix.lower() != ".pt":
                continue
            resolved = path.resolve()
            if str(resolved) in existing:
                continue
            self.checkpoint_paths.append(resolved)
            existing.add(str(resolved))
            display = resolved.name if resolved.parent == DEFAULT_OUTPUTS.resolve() else f"{resolved.parent.name}/{resolved.name}"
            item = QListWidgetItem(display)
            item.setToolTip(str(resolved))
            item.setData(Qt.ItemDataRole.UserRole, str(resolved))
            self.checkpoint_list.addItem(item)
            changed = True
        if changed:
            self._update_results_table()

    def _select_all_checkpoints(self) -> None:
        for i in range(self.checkpoint_list.count()):
            self.checkpoint_list.item(i).setSelected(True)

    def _selected_checkpoint_paths(self) -> list[Path]:
        selected = self.checkpoint_list.selectedItems()
        return [Path(item.data(Qt.ItemDataRole.UserRole)) for item in selected]

    def _remove_selected_checkpoints(self) -> None:
        selected = self.checkpoint_list.selectedItems()
        for item in selected:
            row = self.checkpoint_list.row(item)
            path = Path(item.data(Qt.ItemDataRole.UserRole)).resolve()
            self.checkpoint_paths = [p for p in self.checkpoint_paths if p.resolve() != path]
            self.checkpoint_list.takeItem(row)
        self._update_results_table()

    def _generate_analysis(self) -> None:
        paths = self._selected_checkpoint_paths()
        if not paths:
            QMessageBox.information(self, "No Checkpoints", "Add or select at least one checkpoint.")
            return
        try:
            report = make_markdown_report(paths)
            assets_dir = DEFAULT_OUTPUTS / "ui_analysis_assets"
            pdf_path = self._write_auto_pdf(
                paths,
                markdown=report,
                assets_dir=assets_dir,
                output_name="ui_analysis_review.pdf",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Analysis Error", str(exc))
            return
        self.current_report_markdown = report
        self.current_pdf_path = pdf_path
        self.current_assets_dir = assets_dir
        self.report_view.setMarkdown(report)
        self._refresh_plot_gallery(assets_dir)
        self._update_results_table()
        self.analysis_tabs.setCurrentWidget(self.report_view)

    def _generate_deep_analysis(self) -> None:
        paths = self._selected_checkpoint_paths()
        if not paths:
            QMessageBox.information(self, "No Checkpoints", "Add or select at least one checkpoint.")
            return
        try:
            assets_dir = DEFAULT_OUTPUTS / "ui_advanced_analysis_assets"
            report = make_advanced_report(paths, output_dir=assets_dir, device="cpu")
            pdf_path = self._write_auto_pdf(
                paths,
                markdown=report,
                assets_dir=assets_dir,
                output_name="ui_advanced_analysis_review.pdf",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Deep Analysis Error", str(exc))
            return
        self.current_report_markdown = report
        self.current_pdf_path = pdf_path
        self.current_assets_dir = assets_dir
        self.report_view.setMarkdown(report)
        self._refresh_plot_gallery(assets_dir)
        self._update_results_table()
        self.analysis_tabs.setCurrentWidget(self.report_view)

    def _write_auto_pdf(
        self,
        paths: list[Path],
        *,
        markdown: str,
        assets_dir: Path,
        output_name: str,
    ) -> Path:
        pdf_path = DEFAULT_OUTPUTS / output_name
        pdf = write_analysis_pdf(
            paths,
            pdf_path,
            output_dir=assets_dir,
            markdown=markdown,
            device="cpu",
            include_deep=False,
        )
        self.pdf_status.setText(f"PDF: {pdf}")
        self.pdf_status.setToolTip(str(pdf))
        return pdf

    def _export_pdf(self) -> None:
        paths = self._selected_checkpoint_paths()
        if not paths:
            QMessageBox.information(self, "No Checkpoints", "Add or select at least one checkpoint.")
            return
        try:
            markdown = self.current_report_markdown or make_markdown_report(paths)
            assets_dir = self.current_assets_dir or (DEFAULT_OUTPUTS / "ui_analysis_assets")
            pdf = self._write_auto_pdf(
                paths,
                markdown=markdown,
                assets_dir=assets_dir,
                output_name="ui_analysis_review.pdf",
            )
        except Exception as exc:
            QMessageBox.critical(self, "PDF Error", str(exc))
            return
        self.current_pdf_path = pdf
        QMessageBox.information(self, "PDF Ready", f"Saved PDF:\n{pdf}")

    def _refresh_plot_gallery(self, assets_dir: Path) -> None:
        self.plot_list.clear()
        paths = sorted(Path(assets_dir).glob("*.png")) if Path(assets_dir).exists() else []
        for path in paths:
            item = QListWidgetItem(path.name)
            item.setToolTip(str(path))
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.plot_list.addItem(item)
        if paths:
            self.plot_list.item(0).setSelected(True)
            self.analysis_tabs.setTabText(2, f"Plots ({len(paths)})")
        else:
            self.plot_preview.setPixmap(QPixmap())
            self.plot_preview.setText("No plots for this analysis. Run Deep Analysis to generate prediction-level plots.")
            self.analysis_tabs.setTabText(2, "Plots")

    def _preview_selected_plot(self) -> None:
        items = self.plot_list.selectedItems()
        if not items:
            return
        path = Path(items[0].data(Qt.ItemDataRole.UserRole))
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.plot_preview.setText(f"Could not load plot:\n{path}")
            return
        target = self.plot_preview.size()
        scaled = pixmap.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.plot_preview.setText("")
        self.plot_preview.setPixmap(scaled)

    def _save_report(self) -> None:
        paths = self._selected_checkpoint_paths()
        if not paths:
            QMessageBox.information(self, "No Checkpoints", "Add or select at least one checkpoint.")
            return
        output, _ = QFileDialog.getSaveFileName(
            self,
            "Save Analysis Report",
            str(DEFAULT_OUTPUTS / "analysis_report.md"),
            "Markdown files (*.md);;All files (*)",
        )
        if not output:
            return
        try:
            if self.current_report_markdown:
                Path(output).write_text(self.current_report_markdown, encoding="utf-8")
            else:
                write_markdown_report(paths, output)
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
            return
        QMessageBox.information(self, "Report Saved", f"Saved report:\n{output}")

    def _update_results_table(self) -> None:
        paths = self._selected_checkpoint_paths()
        rows = []
        for path in paths:
            try:
                summary = load_checkpoint_summary(path)
                interpreted = interpret_experiment(summary)
                rows.append(self._table_row_payload(path, summary, interpreted))
            except Exception:
                continue

        if hasattr(self, "results_table"):
            self._populate_run_table(self.results_table, rows)
        if hasattr(self, "analysis_table"):
            self._populate_run_table(self.analysis_table, rows)

    def _populate_run_table(self, table: QTableWidget, rows: list[dict[str, str]]) -> None:
        table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            values = [
                row["name"],
                row["model"],
                row["data"],
                row["target_norm"],
                row["potential_scale"],
                row["acceleration_scale"],
                row["acceleration_rmse"],
                row["relative_acceleration_rmse"],
                row["angle_deg_p95"],
                row["top_5pct_source_contribution"],
                row["manifest"],
                row["path"],
            ]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col_idx in {4, 5, 6, 7, 8, 9}:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(row_idx, col_idx, item)
        table.resizeRowsToContents()

    def _run_dir_for_checkpoint(self, checkpoint_path: Path, config: dict) -> Path:
        if checkpoint_path.name == "sigma.pt":
            return checkpoint_path.parent
        output = config.get("output", {}) if isinstance(config, dict) else {}
        output_dir = Path(str(output.get("output_dir", DEFAULT_OUTPUTS)))
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        run_name = str(output.get("run_name", checkpoint_path.stem))
        return output_dir / run_name

    def _load_target_scales_for_run(self, run_dir: Path) -> dict:
        path = run_dir / "target_scales.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _table_row_payload(self, path: Path, summary: dict, interpreted: dict) -> dict[str, str]:
        config = summary.get("config", {})
        metrics = summary.get("metrics", {})
        diagnostics = metrics.get("diagnostics") or {}
        model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
        data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
        loss_cfg = config.get("loss", {}) if isinstance(config, dict) else {}
        run_dir = self._run_dir_for_checkpoint(path, config)
        target_scales = self._load_target_scales_for_run(run_dir)

        shells = ", ".join(f"{v:.2f}" for v in interpreted["shell_radii"]) or "-"
        model_type = model_cfg.get("type", "-")
        model_text = f"{model_type} [{shells}] / {interpreted['n_sources']}"
        data_path = data_cfg.get("path")
        data_text = Path(str(data_path)).name if data_path else str(data_cfg.get("type", "synthetic"))
        target_norm = "on" if bool(loss_cfg.get("normalize_targets", False)) else "off"
        manifest_status = "ok" if (run_dir / "run_manifest.json").exists() else "-"

        return {
            "name": interpreted["name"],
            "model": model_text,
            "data": data_text,
            "target_norm": target_norm,
            "potential_scale": self._format_metric(target_scales.get("potential_scale", loss_cfg.get("resolved_potential_scale"))),
            "acceleration_scale": self._format_metric(target_scales.get("acceleration_scale", loss_cfg.get("resolved_acceleration_scale"))),
            "acceleration_rmse": self._format_metric(metrics.get("acceleration_rmse")),
            "relative_acceleration_rmse": self._format_metric(metrics.get("relative_acceleration_rmse")),
            "angle_deg_p95": self._format_metric(metrics.get("angle_deg_p95")),
            "top_5pct_source_contribution": self._format_metric(diagnostics.get("top_5pct_source_contribution")),
            "manifest": manifest_status,
            "path": str(path),
        }

    def _format_metric(self, value) -> str:
        if value is None:
            return "-"
        try:
            return f"{float(value):.3e}"
        except (TypeError, ValueError):
            return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="PyQt6 workbench for discrete VESP experiments.")
    parser.add_argument("--check", action="store_true", help="Build the window offscreen and exit.")
    args = parser.parse_args()
    app = QApplication([sys.argv[0]])
    window = VespWorkbench()
    if args.check:
        print(window.windowTitle())
        window.close()
        return
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
