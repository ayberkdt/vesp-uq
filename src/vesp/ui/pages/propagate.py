"""Propagate page: drive the MC / STM force-error covariance propagation scripts.

Wraps ``scripts/run_linear_propagation.py`` (deterministic STM ``6x6`` covariance; writes CSV
artifacts) and ``scripts/run_propagation.py`` (Monte Carlo orbit-dispersion cross-check; log
output only) as subprocesses. The exploratory-not-validated framing from
``docs/VESP_UQ_LIMITATIONS.md`` is shown in-page: these map the *force-error posterior* into an
orbit-level spread -- never an operational orbit-determination or covariance-realism product.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QRadioButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from vesp.ui.jobs import ProcessJob, open_in_file_manager
from vesp.ui.paths import OUTPUTS_DIR, ROOT, list_configs
from vesp.ui.theme import TOKENS
from vesp.ui.widgets import Card, KpiTile, LogConsole, PageHeader, StatusChip, make_button


class PropagatePage(QWidget):
    """Configure and launch a covariance-propagation run; plot the sigma growth (STM)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.job = ProcessJob(self)
        self.job.line.connect(lambda line: self.console.append_line(line))
        self.job.finished.connect(self._on_finished)
        self._out_dir: Path | None = None
        self._canvas = None

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(14)
        header = PageHeader(
            "Covariance propagation",
            "Map the fitted force-error posterior into an orbit-level dispersion: deterministic "
            "linearized STM covariance, or the Monte Carlo sampler as a cross-check.",
        )
        self.status = StatusChip("idle")
        header.actions.addWidget(self.status)
        root.addWidget(header)

        caveat = QLabel(
            "Exploratory, NOT validated orbit determination: this propagates the local "
            "force-model error posterior only -- no measurement processing, no process noise, no "
            "dynamic mismodelling beyond the fitted residual, and no position-error claim "
            "(docs/VESP_UQ_LIMITATIONS.md)."
        )
        caveat.setProperty("chip", "warn")
        caveat.setWordWrap(True)
        root.addWidget(caveat)

        split = QSplitter()
        split.setChildrenCollapsible(False)
        root.addWidget(split, 1)

        # ---------------- left: form ----------------
        left = QWidget()
        col = QVBoxLayout(left)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(12)

        form_card = Card("Run configuration")
        form = QFormLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(9)
        self.config_combo = QComboBox()
        for path in list_configs():
            self.config_combo.addItem(path.name, str(path))
        form.addRow("Config", self.config_combo)
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.5, 500.0)
        self.duration.setDecimals(1)
        self.duration.setValue(7.0)
        self.duration.setSuffix("  time units")
        form.addRow("Duration", self.duration)
        form_card.add_layout(form)

        self.mode_stm = QRadioButton("Linearized STM (deterministic 6x6 covariance, writes CSV)")
        self.mode_mc = QRadioButton("Monte Carlo sampler (cross-check; log output only)")
        self.mode_stm.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.mode_stm)
        group.addButton(self.mode_mc)
        form_card.add(self.mode_stm)
        form_card.add(self.mode_mc)
        col.addWidget(form_card)

        run_row = QHBoxLayout()
        self.run_button = make_button("Propagate", variant="primary", on_click=self._run)
        self.cancel_button = make_button("Cancel", variant="danger", on_click=self.job.cancel)
        self.cancel_button.setEnabled(False)
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.cancel_button)
        run_row.addStretch(1)
        col.addLayout(run_row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        col.addWidget(self.progress)

        result_card = Card("Result (STM)")
        tiles = QHBoxLayout()
        self.kpi_final = KpiTile("final position sigma")
        self.kpi_max = KpiTile("max position sigma")
        self.kpi_points = KpiTile("output points")
        for tile in (self.kpi_final, self.kpi_max, self.kpi_points):
            tiles.addWidget(tile)
        result_card.add_layout(tiles)
        self.open_dir = make_button("Open run folder", variant="ghost", on_click=self._open_dir)
        self.open_dir.setEnabled(False)
        result_card.add(self.open_dir)
        col.addWidget(result_card)
        col.addStretch(1)

        # ---------------- right: plot + log ----------------
        right = QWidget()
        rcol = QVBoxLayout(right)
        rcol.setContentsMargins(0, 0, 0, 0)
        rcol.setSpacing(12)
        plot_card = Card("Position-sigma growth")
        self.plot_host = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_host)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_placeholder = QLabel("Run an STM propagation to plot sigma(t) from the emitted CSV.")
        self.plot_placeholder.setObjectName("KpiHint")
        self.plot_layout.addWidget(self.plot_placeholder)
        plot_card.add(self.plot_host)
        rcol.addWidget(plot_card, 2)

        log_card = Card("Live log")
        self.console = LogConsole()
        log_card.add(self.console)
        rcol.addWidget(log_card, 1)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)

    # ------------------------------------------------------------------ launch
    def _run(self) -> None:
        config = self.config_combo.currentData()
        if not config:
            self.status.set_state("pick a config", "warn")
            return
        stm = self.mode_stm.isChecked()
        if stm:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._out_dir = OUTPUTS_DIR / f"ui_propagation_{stamp}"
            script = ROOT / "scripts" / "run_linear_propagation.py"
            args = [
                sys.executable, "-u", str(script),
                "--config", str(config),
                "--duration", f"{self.duration.value():.3f}",
                "--out-dir", str(self._out_dir),
            ]
        else:
            self._out_dir = None
            script = ROOT / "scripts" / "run_propagation.py"
            args = [sys.executable, "-u", str(script), "--config", str(config)]

        self.console.clear()
        self.console.append_line("[ui] " + " ".join(args[2:]))
        if not stm:
            self.console.append_line("[ui] MC sampler reports through its log; it writes no files.")
        self.status.set_state("running", "accent")
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.open_dir.setEnabled(False)
        self.job.start(args)

    # ------------------------------------------------------------------ results
    def _on_finished(self, code: int) -> None:
        self.progress.setRange(0, 1)
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if code != 0:
            self.status.set_state(f"failed (exit {code})", "danger")
            return
        self.status.set_state("completed", "ok")
        if self._out_dir is not None:
            self._load_states_csv()

    def _load_states_csv(self) -> None:
        path = self._out_dir / "linear_propagation_states.csv"
        if not path.is_file():
            self.console.append_line(f"[ui] states CSV not found: {path}")
            return
        times: list[float] = []
        pos_sigma: list[float] = []
        vel_sigma: list[float] = []
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                times.append(float(row["time"]))
                pos_sigma.append(float(row["position_sigma"]))
                vel_sigma.append(float(row["velocity_sigma"]))
        if not times:
            return
        self.kpi_final.set(f"{pos_sigma[-1]:.3e}", "model-normalized length units")
        self.kpi_max.set(f"{max(pos_sigma):.3e}", "1-sigma 3D dispersion along the orbit")
        self.kpi_points.set(str(len(times)), f"t in [0, {times[-1]:.1f}]")
        self.open_dir.setEnabled(True)
        self._draw(times, pos_sigma, vel_sigma)

    def _draw(self, times: list[float], pos_sigma: list[float], vel_sigma: list[float]) -> None:
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except Exception:
            return
        self.plot_placeholder.setVisible(False)
        if self._canvas is not None:
            self.plot_layout.removeWidget(self._canvas)
            self._canvas.deleteLater()
            self._canvas = None

        fig = Figure(figsize=(5.4, 3.4), facecolor=TOKENS["card"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(TOKENS["surface"])
        ax.plot(times, pos_sigma, color=TOKENS["accent"], lw=2.0, label="position sigma")
        ax.plot(times, vel_sigma, color=TOKENS["ok"], lw=1.4, ls="--", label="velocity sigma")
        positive = [s for s in pos_sigma + vel_sigma if s > 0.0]
        if positive:
            ax.set_yscale("log")
        ax.set_xlabel("time  [time units]", color=TOKENS["text_muted"])
        ax.set_ylabel("1-sigma dispersion  [model units]", color=TOKENS["text_muted"])
        ax.tick_params(colors=TOKENS["text_muted"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(TOKENS["border"])
        ax.grid(True, color=TOKENS["border_soft"], lw=0.6, alpha=0.6)
        legend = ax.legend(loc="lower right", fontsize=8, facecolor=TOKENS["card"], edgecolor=TOKENS["border"])
        for text in legend.get_texts():
            text.set_color(TOKENS["text"])
        fig.tight_layout()
        self._canvas = FigureCanvasQTAgg(fig)
        self.plot_layout.addWidget(self._canvas)

    def _open_dir(self) -> None:
        if self._out_dir is not None and self._out_dir.exists():
            open_in_file_manager(self._out_dir)
