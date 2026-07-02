"""Screen page: serve-side risk screening with a persisted model (``python -m vesp.uq.screen``).

Mirrors the serve CLI one-to-one: model artifact + trajectory source (external CSV or the
config's generated ensemble) + optional decision-policy overrides. Results are read back from
``screening_report.json`` / ``trajectory_scores.csv``.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QProgressBar,
    QRadioButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vesp.ui.helpers import SCORING_MODES as CANONICAL_SCORING_MODES
from vesp.ui.helpers import safe_read_json, scoring_hint
from vesp.ui.jobs import ProcessJob
from vesp.ui.paths import OUTPUTS_DIR, list_configs, list_models
from vesp.ui.widgets import (
    Card,
    KpiTile,
    LogConsole,
    ModelArtifactPicker,
    PageHeader,
    PathPicker,
    RunOutputActions,
    StatusChip,
    make_button,
)

MODEL_DEFAULT = "(model policy)"
# Full canonical scoring registry (shared, drift-guarded) + the "use the packaged policy" sentinel.
SCORING_MODES = (MODEL_DEFAULT, *CANONICAL_SCORING_MODES)
UNIT_CHOICES = ("model_normalized_accel", "m/s^2", "km/s^2", "mm/s^2", "um/s^2")
SCORE_COLUMNS = ("trajectory_id", "risk_score", "max_expected_error", "min_radius", "max_domain_risk", "flagged_for_rerun")


class ScreenPage(QWidget):
    """Configure and launch a serve-mode screening; browse the flagged set."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.job = ProcessJob(self)
        self.job.line.connect(lambda line: self.console.append_line(line))
        self.job.finished.connect(self._on_finished)
        self._out_dir: Path | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(14)
        header = PageHeader(
            "Screen an ensemble",
            "Load a persisted layer and risk-screen new trajectories -- the packaged decision "
            "policy applies unless overridden. No refitting; force-risk / OOD only.",
        )
        self.status = StatusChip("idle")
        header.add_action(self.status)
        root.addWidget(header)

        split = QSplitter()
        split.setChildrenCollapsible(False)
        root.addWidget(split, 1)

        # ---------------- left: form ----------------
        left = QWidget()
        col = QVBoxLayout(left)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(12)

        model_card = Card("Model artifact")
        self.model_selector = ModelArtifactPicker()
        self.model_combo = self.model_selector.combo
        self.model_picker = self.model_selector.custom_picker
        model_card.add(self.model_selector)
        col.addWidget(model_card)

        source_card = Card("Trajectory source")
        self.source_generated = QRadioButton("Generated ensemble from config")
        self.source_csv = QRadioButton("External surrogate CSV")
        self.source_generated.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.source_generated)
        group.addButton(self.source_csv)
        source_card.add(self.source_generated)
        self.config_combo = QComboBox()
        for path in list_configs():
            self.config_combo.addItem(path.name, str(path))
        source_card.add(self.config_combo)
        source_card.add(self.source_csv)
        self.csv_picker = PathPicker("trajectory CSV (Format A/B)", name_filter="CSV (*.csv)")
        source_card.add(self.csv_picker)
        units_row = QFormLayout()
        self.units_combo = QComboBox()
        self.units_combo.addItems(UNIT_CHOICES)
        units_row.addRow("CSV accel units", self.units_combo)
        source_card.add_layout(units_row)
        for w in (self.csv_picker, self.units_combo):
            w.setEnabled(False)
        self.source_csv.toggled.connect(self._on_source_toggle)
        col.addWidget(source_card)

        policy_card = Card("Decision policy (overrides)")
        form = QFormLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(9)
        self.scoring = QComboBox()
        self.scoring.addItems(SCORING_MODES)
        for i, mode in enumerate(SCORING_MODES):
            hint = scoring_hint(mode)
            if hint:
                self.scoring.setItemData(i, hint, Qt.ItemDataRole.ToolTipRole)
        form.addRow("Scoring", self.scoring)
        self.threshold_edit = QLineEdit()
        self.threshold_edit.setPlaceholderText("absolute risk budget, e.g. 2.5e-3 (optional)")
        form.addRow("Threshold", self.threshold_edit)
        self.fraction = QDoubleSpinBox()
        self.fraction.setRange(0.0, 1.0)
        self.fraction.setSingleStep(0.05)
        self.fraction.setDecimals(2)
        self.fraction.setValue(0.0)
        self.fraction.setSpecialValueText(MODEL_DEFAULT)
        form.addRow("Rerun fraction", self.fraction)
        self.time_weighting = QComboBox()
        self.time_weighting.addItems((MODEL_DEFAULT, "none", "kepler_r2"))
        form.addRow("Time weighting", self.time_weighting)
        policy_card.add_layout(form)
        col.addWidget(policy_card)

        run_row = QHBoxLayout()
        self.run_button = make_button("Run screening", variant="primary", on_click=self._run)
        self.cancel_button = make_button("Cancel", variant="danger", on_click=self.job.cancel)
        self.cancel_button.setEnabled(False)
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.cancel_button)
        run_row.addStretch(1)
        col.addLayout(run_row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        col.addWidget(self.progress)
        col.addStretch(1)

        # ---------------- right: results ----------------
        right = QWidget()
        rcol = QVBoxLayout(right)
        rcol.setContentsMargins(0, 0, 0, 0)
        rcol.setSpacing(12)

        result_card = Card("Outcome")
        tiles = QHBoxLayout()
        self.kpi_flagged = KpiTile("Flagged")
        self.kpi_policy = KpiTile("Policy")
        self.kpi_speed = KpiTile("Throughput")
        for tile in (self.kpi_flagged, self.kpi_policy, self.kpi_speed):
            tiles.addWidget(tile)
        result_card.add_layout(tiles)
        self.output_actions = RunOutputActions(report_name="screening_report.md")
        self.open_dir = self.output_actions.open_dir
        self.open_report = self.output_actions.open_report
        result_card.add(self.output_actions)
        rcol.addWidget(result_card)

        table_card = Card("Trajectory scores (top risk first)")
        self.table = QTableWidget(0, len(SCORE_COLUMNS))
        self.table.setHorizontalHeaderLabels(SCORE_COLUMNS)
        vertical_header = self.table.verticalHeader()
        if vertical_header is not None:
            vertical_header.setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        horizontal_header = self.table.horizontalHeader()
        if horizontal_header is not None:
            horizontal_header.setStretchLastSection(True)
        table_card.add(self.table)
        rcol.addWidget(table_card, 1)

        log_card = Card("Live log")
        self.console = LogConsole()
        self.console.setMaximumHeight(140)
        log_card.add(self.console)
        rcol.addWidget(log_card)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)

    # ------------------------------------------------------------------ helpers
    def _on_source_toggle(self, csv_mode: bool) -> None:
        self.csv_picker.setEnabled(csv_mode)
        self.units_combo.setEnabled(csv_mode)
        self.config_combo.setEnabled(not csv_mode)

    def refresh_models(self) -> None:
        self.model_selector.refresh(list_models())

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh_models()

    def _selected_model(self) -> Path | None:
        return self.model_selector.selected_path()

    # ------------------------------------------------------------------ launch
    def _run(self) -> None:
        model = self._selected_model()
        if model is None or not model.is_file():
            self.status.set_state("pick a model", "warn")
            return
        args = ["--model", str(model)]
        if self.source_csv.isChecked():
            csv_path = self.csv_picker.path()
            if csv_path is None or not csv_path.is_file():
                self.status.set_state("pick a CSV", "warn")
                return
            args += ["--trajectories", str(csv_path), "--trajectory-units", self.units_combo.currentText()]
        else:
            config = self.config_combo.currentData()
            if not config:
                self.status.set_state("pick a config", "warn")
                return
            args += ["--config", str(config)]

        if self.scoring.currentText() != MODEL_DEFAULT:
            args += ["--scoring", self.scoring.currentText()]
        threshold_text = self.threshold_edit.text().strip()
        if threshold_text:
            try:
                args += ["--threshold", str(float(threshold_text))]
            except ValueError:
                self.status.set_state("bad threshold", "danger")
                return
        if self.fraction.value() > 0.0:
            args += ["--rerun-fraction", f"{self.fraction.value():.4f}"]
        if self.time_weighting.currentText() != MODEL_DEFAULT:
            args += ["--time-weighting", self.time_weighting.currentText()]

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._out_dir = OUTPUTS_DIR / f"ui_screen_{stamp}"
        self.output_actions.set_output(self._out_dir)
        args += ["--out", str(self._out_dir)]

        self.console.clear()
        self.console.append_line("[ui] python -m vesp.uq.screen " + " ".join(args))
        self.status.set_state("running", "accent")
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.output_actions.set_actions_enabled(False)
        self.job.start_module("vesp.uq.screen", args)

    # ------------------------------------------------------------------ results
    def _on_finished(self, code: int) -> None:
        self.progress.setRange(0, 1)
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if code != 0:
            self.status.set_state(f"failed (exit {code})", "danger")
            return
        self._load_results()

    def _load_results(self) -> None:
        if self._out_dir is None:
            return
        report_path = self._out_dir / "screening_report.json"
        report, error = safe_read_json(report_path)
        if report is None:
            self.status.set_state("report missing", "danger")
            self.console.append_line(f"[ui] failed to read {report_path}: {error}")
            return

        sc = report.get("screening", {})
        screen = sc.get("screen", {})
        runtime = report.get("runtime", {})
        n_flagged = screen.get("n_flagged", 0)
        n_total = sc.get("n_trajectories", 0)
        zero_alarm = n_flagged == 0 and screen.get("selection_mode", "fraction") != "fraction"
        self.status.set_state("zero alarms" if zero_alarm else "completed", "ok")
        self.kpi_flagged.set(f"{n_flagged}/{n_total}", "zero alarms" if zero_alarm else "sent to high-fidelity rerun")
        self.kpi_policy.set(
            str(sc.get("scoring", "--")),
            f"selection {screen.get('selection_mode', '--')} (origin {sc.get('selection_origin', '--')})",
        )
        us = runtime.get("score_us_per_output_point")
        self.kpi_speed.set(f"{us:.1f} us/pt" if isinstance(us, (int, float)) else "--", "no refit; scoring only")

        self._fill_table()
        self.output_actions.set_actions_enabled(True)

    def _fill_table(self) -> None:
        self.table.setRowCount(0)
        scores_path = self._out_dir / "trajectory_scores.csv" if self._out_dir else None
        if scores_path is None or not scores_path.is_file():
            return
        with open(scores_path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.sort(key=lambda r: float(r.get("risk_score", "nan") or "nan"), reverse=True)
        accent = QColor(77, 163, 255, 38)
        for record in rows[:200]:
            row = self.table.rowCount()
            self.table.insertRow(row)
            flagged = record.get("flagged_for_rerun") == "1"
            for col, key in enumerate(SCORE_COLUMNS):
                value = record.get(key, "")
                try:
                    text = f"{float(value):.4g}" if key != "trajectory_id" else value
                except (TypeError, ValueError):
                    text = value
                item = QTableWidgetItem(text)
                if flagged:
                    item.setBackground(accent)
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()
