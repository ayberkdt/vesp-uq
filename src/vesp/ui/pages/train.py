"""Train page: fit + calibrate + screen via ``python -m vesp.uq.run``, with live logs.

Overrides chosen in the form are merged into the selected YAML and written to a temporary
config under the run's output directory, so the executed command stays a plain, reproducible
``python -m vesp.uq.run --config <file>`` and the manifest snapshots exactly what ran.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vesp.ui.helpers import SCORING_MODES as CANONICAL_SCORING_MODES
from vesp.ui.helpers import fmt, safe_read_json, scoring_hint
from vesp.ui.jobs import ProcessJob
from vesp.ui.paths import OUTPUTS_DIR, list_configs
from vesp.ui.widgets import (
    Card,
    KpiTile,
    LogConsole,
    PageHeader,
    PathPicker,
    RunOutputActions,
    StatusChip,
    make_button,
)

CONFIG_DEFAULT = "(config default)"
SCORING_MODES = (CONFIG_DEFAULT, *CANONICAL_SCORING_MODES)
COVARIANCE_MODES = (CONFIG_DEFAULT, "exact", "diagonal", "lowrank")
NOISE_MODES = (CONFIG_DEFAULT, "heteroscedastic", "altitude_binned", "homoscedastic")
TRISTATE = (CONFIG_DEFAULT, "on", "off")

CAL_BAND_COLUMNS = ("band", "n", "z_std", "picp_90", "ellipsoid_picp_90", "nll")


class TrainPage(QWidget):
    """Configure and launch a training run; surface calibration + screening results."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.job = ProcessJob(self)
        self.job.line.connect(self._on_line)
        self.job.finished.connect(self._on_finished)
        self._run_dir: Path | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(14)
        header = PageHeader(
            "Train a calibration layer",
            "Fit the equivalent-source error posterior, calibrate per-band uncertainty, screen the "
            "demo ensemble, and package the model with its decision policy + model card.",
        )
        self.status = StatusChip("idle")
        header.add_action(self.status)
        root.addWidget(header)

        split = QSplitter()
        split.setChildrenCollapsible(False)
        root.addWidget(split, 1)

        # ---------------- left: launch form ----------------
        left = QWidget()
        left_col = QVBoxLayout(left)
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(12)

        form_card = Card("Run configuration")
        form = QFormLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(9)

        self.config_combo = QComboBox()
        self._configs = list_configs()
        for path in self._configs:
            self.config_combo.addItem(path.name, str(path))
        self.config_combo.addItem("Browse...", "")
        self.config_picker = PathPicker("custom config (.yaml)", name_filter="YAML (*.yaml *.yml)")
        self.config_combo.currentIndexChanged.connect(
            lambda _i: self.config_picker.setVisible(self.config_combo.currentData() == "")
        )
        self.config_picker.setVisible(False)
        form.addRow("Config", self.config_combo)
        form.addRow("", self.config_picker)

        self.run_name = QLineEdit()
        self.run_name.setPlaceholderText("override run_name (optional)")
        form.addRow("Run name", self.run_name)

        self.seed = QSpinBox()
        self.seed.setRange(-1, 10_000_000)
        self.seed.setValue(-1)
        self.seed.setSpecialValueText(CONFIG_DEFAULT)
        form.addRow("Seed", self.seed)

        self.scoring = QComboBox()
        self.scoring.addItems(SCORING_MODES)
        for i, mode in enumerate(SCORING_MODES):
            hint = scoring_hint(mode)
            if hint:
                self.scoring.setItemData(i, hint, Qt.ItemDataRole.ToolTipRole)
        form.addRow("Risk scoring", self.scoring)

        self.covariance = QComboBox()
        self.covariance.addItems(COVARIANCE_MODES)
        form.addRow("Covariance", self.covariance)

        self.noise = QComboBox()
        self.noise.addItems(NOISE_MODES)
        form.addRow("Noise model", self.noise)

        self.domain = QComboBox()
        self.domain.addItems(TRISTATE)
        form.addRow("Domain support", self.domain)

        self.conformal = QComboBox()
        self.conformal.addItems(TRISTATE)
        form.addRow("Conformal prediction", self.conformal)

        self.save_model = QCheckBox("Package the fitted model (vespuq_plugin.pt + model card)")
        self.save_model.setChecked(True)
        form_card.add_layout(form)
        form_card.add(self.save_model)
        left_col.addWidget(form_card)

        run_row = QHBoxLayout()
        self.run_button = make_button("Run training", variant="primary", on_click=self._run)
        self.cancel_button = make_button("Cancel", variant="danger", on_click=self.job.cancel)
        self.cancel_button.setEnabled(False)
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.cancel_button)
        run_row.addStretch(1)
        left_col.addLayout(run_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        left_col.addWidget(self.progress)

        results = Card("Result")
        tiles = QHBoxLayout()
        tiles.setSpacing(10)
        self.kpi_picp = KpiTile("low-band PICP90")
        self.kpi_ratio = KpiTile("low/high epistemic")
        self.kpi_flagged = KpiTile("flagged")
        self.kpi_conformal = KpiTile("conformal")
        self.kpi_speed = KpiTile("scoring")
        for tile in (self.kpi_picp, self.kpi_ratio, self.kpi_flagged, self.kpi_conformal, self.kpi_speed):
            tiles.addWidget(tile)
        results.add_layout(tiles)

        self.cal_table = QTableWidget(0, len(CAL_BAND_COLUMNS))
        self.cal_table.setHorizontalHeaderLabels(CAL_BAND_COLUMNS)
        vertical_header = self.cal_table.verticalHeader()
        if vertical_header is not None:
            vertical_header.setVisible(False)
        self.cal_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        horizontal_header = self.cal_table.horizontalHeader()
        if horizontal_header is not None:
            horizontal_header.setStretchLastSection(True)
        self.cal_table.setMaximumHeight(170)
        results.add(self.cal_table)

        self.output_actions = RunOutputActions(report_name="vespuq_report.md")
        self.open_dir = self.output_actions.open_dir
        self.open_report = self.output_actions.open_report
        results.add(self.output_actions)
        left_col.addWidget(results)
        left_col.addStretch(1)

        # ---------------- right: live log ----------------
        right = QWidget()
        right_col = QVBoxLayout(right)
        right_col.setContentsMargins(0, 0, 0, 0)
        log_card = Card("Live log")
        self.console = LogConsole()
        log_card.add(self.console)
        right_col.addWidget(log_card)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 4)

        note = QLabel(
            "Overrides are merged into a temporary copy of the YAML and executed as a plain "
            "`python -m vesp.uq.run --config ...` -- the run manifest snapshots the exact config, "
            "so every UI run is reproducible from the CLI."
        )
        note.setObjectName("KpiHint")
        note.setWordWrap(True)
        root.addWidget(note)

    # ------------------------------------------------------------------ launch
    def _selected_config(self) -> Path | None:
        data = self.config_combo.currentData()
        if data:
            return Path(data)
        return self.config_picker.path()

    def _merged_config(self, base: Path) -> dict:
        import yaml

        config = yaml.safe_load(base.read_text(encoding="utf-8")) or {}
        return apply_training_overrides(
            config,
            run_name=self.run_name.text().strip(),
            seed=int(self.seed.value()),
            scoring=self.scoring.currentText(),
            covariance=self.covariance.currentText(),
            noise=self.noise.currentText(),
            domain=self.domain.currentText(),
            conformal=self.conformal.currentText(),
            save_model=bool(self.save_model.isChecked()),
        )

    def _run(self) -> None:
        base = self._selected_config()
        if base is None or not base.is_file():
            self.status.set_state("pick a config", "warn")
            return
        try:
            import yaml

            config = self._merged_config(base)
        except Exception as exc:  # malformed YAML -> tell the user, do not crash
            self.status.set_state("config error", "danger")
            self.console.append_line(f"[ui] failed to read config: {exc}")
            return

        output_dir = Path(config.get("output", {}).get("output_dir", str(OUTPUTS_DIR)))
        run_name = str(config.get("output", {}).get("run_name", "vespuq"))
        self._run_dir = output_dir / run_name
        self.output_actions.set_output(self._run_dir)

        fd, tmp_name = tempfile.mkstemp(prefix=f"ui_{base.stem}_", suffix=".yaml")
        tmp = Path(tmp_name)
        with open(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

        self.console.clear()
        self.console.append_line(f"[ui] python -m vesp.uq.run --config {tmp}")
        self.status.set_state("running", "accent")
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.output_actions.set_actions_enabled(False)
        self.job.start_module("vesp.uq.run", ["--config", str(tmp)])

    # ------------------------------------------------------------------ job events
    def _on_line(self, line: str) -> None:
        self.console.append_line(line)

    def _on_finished(self, code: int) -> None:
        self.progress.setRange(0, 1)
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if code == 0:
            self.status.set_state("completed", "ok")
            self._load_results()
        else:
            self.status.set_state(f"failed (exit {code})", "danger")

    # ------------------------------------------------------------------ results
    def _load_results(self) -> None:
        if self._run_dir is None:
            return
        report_path = self._run_dir / "vespuq_report.json"
        if not report_path.is_file():
            self.console.append_line(f"[ui] report not found: {report_path}")
            return
        report, error = safe_read_json(report_path)
        if report is None:
            self.console.append_line(f"[ui] failed to parse report: {error}")
            return

        cal = report.get("experiment_1_calibration", {})
        summary = report.get("summary", {})
        screen = report.get("experiment_3_screening", {}).get("screen", {})
        runtime = report.get("runtime", {})

        low = cal.get("low", {})
        self.kpi_picp.set(fmt(low.get("picp_90"), digits=3), "target 0.90 in the low band")
        self.kpi_ratio.set(
            fmt(cal.get("low_high_epistemic_std_ratio"), digits=3), "should be > 1: uncertainty grows when low"
        )
        self.kpi_flagged.set(
            f"{screen.get('n_flagged', '--')}/{screen.get('n_trajectories', '--')}",
            f"capture rate {fmt(summary.get('capture_rate'), digits=3)}",
        )
        conformal = report.get("conformal_calibration") or {}
        global_cal = conformal.get("global") or {}
        self.kpi_conformal.set(
            fmt(global_cal.get("scale"), digits=3) if conformal.get("enabled") else "off",
            str(conformal.get("scope", "")) if conformal.get("enabled") else "config default",
        )
        self.kpi_speed.set(
            f"{fmt(runtime.get('score_us_per_output_point'), digits=3)} us/pt",
            f"fit {fmt(runtime.get('fit_seconds'), digits=3)} s",
        )

        self.cal_table.setRowCount(0)
        for band in ("all", "low", "mid", "high"):
            metrics = cal.get(band)
            if not isinstance(metrics, dict):
                continue
            row = self.cal_table.rowCount()
            self.cal_table.insertRow(row)
            values = (
                band,
                str(metrics.get("n", "")),
                fmt(metrics.get("z_std"), digits=3),
                fmt(metrics.get("picp_90"), digits=3),
                fmt(metrics.get("ellipsoid_picp_90"), digits=3),
                fmt(metrics.get("nll"), digits=3),
            )
            for col, text in enumerate(values):
                self.cal_table.setItem(row, col, QTableWidgetItem(text))
        self.cal_table.resizeColumnsToContents()
        self.output_actions.set_actions_enabled(True)


def apply_training_overrides(
    config: dict,
    *,
    run_name: str = "",
    seed: int = -1,
    scoring: str = CONFIG_DEFAULT,
    covariance: str = CONFIG_DEFAULT,
    noise: str = CONFIG_DEFAULT,
    domain: str = CONFIG_DEFAULT,
    conformal: str = CONFIG_DEFAULT,
    save_model: bool = True,
) -> dict:
    """Return a config copy with UI overrides applied.

    Keeping this pure makes the Train page easier to test without constructing Qt widgets.
    """

    out = copy.deepcopy(config)
    if run_name:
        out.setdefault("output", {})["run_name"] = run_name
    if seed >= 0:
        out["seed"] = int(seed)
    uq = out.setdefault("uq", {})
    if scoring != CONFIG_DEFAULT:
        uq.setdefault("risk", {})["scoring"] = scoring
    if covariance != CONFIG_DEFAULT:
        uq["covariance_mode"] = covariance
    if noise != CONFIG_DEFAULT:
        uq["noise_model"] = noise
    if domain != CONFIG_DEFAULT:
        uq.setdefault("risk", {})["domain_support"] = domain == "on"
    if conformal != CONFIG_DEFAULT:
        uq.setdefault("conformal", {})["apply"] = conformal == "on"
    out.setdefault("output", {})["save_model"] = bool(save_model)
    return out
