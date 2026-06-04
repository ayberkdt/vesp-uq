"""PyQt6 workbench for running and analyzing discrete VESP experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt6.QtCore import QProcess, Qt
from PyQt6.QtGui import QAction, QFont, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SINGLE_CONFIG = PROJECT_ROOT / "experimental_vesp" / "configs" / "discrete_single_shell.yaml"
DEFAULT_MULTI_CONFIG = PROJECT_ROOT / "experimental_vesp" / "configs" / "discrete_multishell.yaml"
DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"

try:
    from .analysis import interpret_experiment, load_checkpoint_summary, make_markdown_report, write_markdown_report
    from .advanced_analysis import make_advanced_report
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT))
    from experimental_vesp.analysis import interpret_experiment, load_checkpoint_summary, make_markdown_report, write_markdown_report
    from experimental_vesp.advanced_analysis import make_advanced_report


class VespWorkbench(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VESP Workbench")
        self.resize(1320, 820)
        self.process: QProcess | None = None
        self.checkpoint_paths: list[Path] = []
        self.current_report_markdown = ""

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

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)

        file_menu = self.menuBar().addMenu("File")
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
        layout.setContentsMargins(22, 16, 22, 14)
        layout.setSpacing(18)

        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        title = QLabel("MaxEnt-VESP Workbench")
        title.setObjectName("HeaderTitle")
        subtitle = QLabel("Discrete equivalent-source experiments")
        subtitle.setObjectName("HeaderSubtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)

        status = QLabel("Ridge: augmented least-squares  |  Analysis: prediction-level diagnostics")
        status.setObjectName("HeaderStatus")
        status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout.addLayout(title_block, stretch=1)
        layout.addWidget(status, stretch=1)
        return header

    def _build_run_tab(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)

        controls = QGroupBox("Experiment")
        grid = QGridLayout(controls)

        self.config_path = QLineEdit(str(DEFAULT_SINGLE_CONFIG))
        browse_config = QPushButton("Browse")
        browse_config.clicked.connect(self._browse_config)

        load_single = QPushButton("Single Shell")
        load_single.clicked.connect(lambda: self._load_config(DEFAULT_SINGLE_CONFIG))
        load_multi = QPushButton("Multi Shell")
        load_multi.clicked.connect(lambda: self._load_config(DEFAULT_MULTI_CONFIG))
        save_config = QPushButton("Save Config")
        save_config.clicked.connect(self._save_config)

        run_discrete = QPushButton("Run Discrete")
        run_discrete.clicked.connect(lambda: self._run_training("experimental_vesp.train_discrete"))
        run_multi = QPushButton("Run Multi-Shell")
        run_multi.clicked.connect(lambda: self._run_training("experimental_vesp.train_multishell"))
        stop = QPushButton("Stop")
        stop.clicked.connect(self._stop_process)

        grid.addWidget(QLabel("Config"), 0, 0)
        grid.addWidget(self.config_path, 0, 1, 1, 5)
        grid.addWidget(browse_config, 0, 6)
        grid.addWidget(load_single, 1, 1)
        grid.addWidget(load_multi, 1, 2)
        grid.addWidget(save_config, 1, 3)
        grid.addWidget(run_discrete, 1, 4)
        grid.addWidget(run_multi, 1, 5)
        grid.addWidget(stop, 1, 6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.config_editor = QPlainTextEdit()
        self.config_editor.setFont(QFont("Consolas", 10))
        self.config_editor.setPlaceholderText("YAML config will appear here.")

        self.run_log = QPlainTextEdit()
        self.run_log.setReadOnly(True)
        self.run_log.setFont(QFont("Consolas", 10))
        self.run_log.setPlaceholderText("Training output will stream here.")

        splitter.addWidget(self.config_editor)
        splitter.addWidget(self.run_log)
        splitter.setSizes([560, 700])

        layout.addWidget(controls)
        layout.addWidget(splitter, stretch=1)
        return root

    def _build_analysis_tab(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)

        controls = QGroupBox("Checkpoints")
        row = QHBoxLayout(controls)
        add_files = QPushButton("Add Files")
        add_files.clicked.connect(self._browse_checkpoints)
        add_dir = QPushButton("Add Folder")
        add_dir.clicked.connect(lambda: self._choose_checkpoint_dir())
        remove = QPushButton("Remove Selected")
        remove.clicked.connect(self._remove_selected_checkpoints)
        refresh = QPushButton("Refresh Outputs")
        refresh.clicked.connect(self._refresh_output_checkpoints)
        analyze = QPushButton("Generate Analysis")
        analyze.clicked.connect(self._generate_analysis)
        deep_analyze = QPushButton("Deep Analysis")
        deep_analyze.clicked.connect(self._generate_deep_analysis)
        save = QPushButton("Save Report")
        save.clicked.connect(self._save_report)

        for widget in (add_files, add_dir, remove, refresh, analyze, deep_analyze, save):
            row.addWidget(widget)
        row.addStretch(1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.checkpoint_list = QListWidget()
        self.checkpoint_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.checkpoint_list.itemSelectionChanged.connect(self._update_results_table)

        self.report_view = QTextBrowser()
        self.report_view.setOpenExternalLinks(True)
        self.report_view.setPlaceholderText("Analysis report will appear here.")

        splitter.addWidget(self.checkpoint_list)
        splitter.addWidget(self.report_view)
        splitter.setSizes([420, 880])

        layout.addWidget(controls)
        layout.addWidget(splitter, stretch=1)
        return root

    def _build_results_tab(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        self.results_table = QTableWidget(0, 8)
        self.results_table.setHorizontalHeaderLabels(
            [
                "Run",
                "Shells",
                "Sources",
                "Acc RMSE",
                "Pot RMSE",
                "Low/High",
                "Acc Status",
                "Pot Status",
            ]
        )
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.results_table)
        return root

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #eef2f5;
            }
            QMenuBar {
                background: #101923;
                color: #dfe8ef;
                padding: 4px 8px;
            }
            QMenuBar::item {
                background: transparent;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QMenuBar::item:selected {
                background: #223242;
            }
            QMenu {
                background: #ffffff;
                color: #17212b;
                border: 1px solid #cbd3dc;
            }
            QMenu::item:selected {
                background: #d7e8ef;
            }
            #TopHeader {
                background: #111a24;
                border-bottom: 1px solid #273646;
            }
            #HeaderTitle {
                color: #f5f9fc;
                font-size: 20px;
                font-weight: 700;
            }
            #HeaderSubtitle {
                color: #9db4c5;
                font-size: 12px;
            }
            #HeaderStatus {
                color: #c7d7e2;
                font-size: 12px;
            }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #d2d8df;
                border-radius: 6px;
                margin-top: 12px;
                padding: 10px;
                background: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                background: #1f5f76;
                color: white;
                border: 0;
                border-radius: 5px;
                padding: 7px 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #287892;
            }
            QPushButton:pressed {
                background: #174859;
            }
            QLineEdit, QPlainTextEdit, QTextBrowser, QListWidget, QTableWidget {
                background: #ffffff;
                color: #17212b;
                border: 1px solid #cbd3dc;
                border-radius: 5px;
                selection-background-color: #c7e5d9;
            }
            QTabWidget::pane {
                border: 0;
                background: #eef2f5;
            }
            QTabBar::tab {
                background: #202d3a;
                color: #b9c9d5;
                padding: 9px 18px;
                margin-right: 1px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #102132;
                font-weight: 600;
            }
            QTabBar::tab:hover:!selected {
                background: #2a3a4a;
                color: #ffffff;
            }
            """
        )

    def _browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Config",
            str(PROJECT_ROOT / "experimental_vesp" / "configs"),
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
        self.config_editor.setPlainText(text)

    def _save_config(self) -> None:
        path = Path(self.config_path.text())
        try:
            path.write_text(self.config_editor.toPlainText(), encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
            return
        self._append_log(f"Saved config: {path}\n")

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
        self._add_checkpoint_paths(sorted(path.glob("*.pt")))

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
            item = QListWidgetItem(resolved.name)
            item.setToolTip(str(resolved))
            item.setData(Qt.ItemDataRole.UserRole, str(resolved))
            self.checkpoint_list.addItem(item)
            changed = True
        if changed:
            self._select_all_checkpoints()
            self._update_results_table()

    def _select_all_checkpoints(self) -> None:
        for i in range(self.checkpoint_list.count()):
            self.checkpoint_list.item(i).setSelected(True)

    def _selected_checkpoint_paths(self) -> list[Path]:
        selected = self.checkpoint_list.selectedItems()
        if not selected:
            selected = [self.checkpoint_list.item(i) for i in range(self.checkpoint_list.count())]
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
        except Exception as exc:
            QMessageBox.critical(self, "Analysis Error", str(exc))
            return
        self.current_report_markdown = report
        self.report_view.setMarkdown(report)
        self._update_results_table()

    def _generate_deep_analysis(self) -> None:
        paths = self._selected_checkpoint_paths()
        if not paths:
            QMessageBox.information(self, "No Checkpoints", "Add or select at least one checkpoint.")
            return
        try:
            assets_dir = DEFAULT_OUTPUTS / "ui_advanced_analysis_assets"
            report = make_advanced_report(paths, output_dir=assets_dir, device="cpu")
        except Exception as exc:
            QMessageBox.critical(self, "Deep Analysis Error", str(exc))
            return
        self.current_report_markdown = report
        self.report_view.setMarkdown(report)
        self._update_results_table()

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
                rows.append(interpret_experiment(load_checkpoint_summary(path)))
            except Exception:
                continue

        self.results_table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            shells = ", ".join(f"{v:.2f}" for v in row["shell_radii"]) or "-"
            ratio = row["low_high_altitude_ratio"]
            values = [
                row["name"],
                shells,
                str(row["n_sources"]),
                f"{row['acceleration_rmse']:.6e}",
                f"{row['potential_rmse']:.6e}",
                "-" if ratio is None else f"{ratio:.2f}",
                row["acceleration_status"],
                row["potential_status"],
            ]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col_idx >= 2:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.results_table.setItem(row_idx, col_idx, item)
        self.results_table.resizeColumnsToContents()


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
