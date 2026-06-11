"""VESP-UQ Mission Console main window: navigation rail + stacked pages.

Run with ``python ui/app_vespuq.py`` (repo root) -- the launcher bootstraps ``sys.path`` and
calls :func:`main`.
"""

from __future__ import annotations

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from vesp.ui.pages.dashboard import DashboardPage
from vesp.ui.pages.model import ModelPage
from vesp.ui.pages.propagate import PropagatePage
from vesp.ui.pages.runs import RunsPage
from vesp.ui.pages.screen import ScreenPage
from vesp.ui.pages.train import TrainPage
from vesp.ui.pages.update import UpdatePage
from vesp.ui.paths import ROOT
from vesp.ui.theme import build_qss

NAV_ITEMS = (
    ("dashboard", "⌂  Dashboard"),
    ("train", "⚙  Train"),
    ("screen", "\U0001f6f0  Screen"),
    ("propagate", "\U0001f4c8  Propagate"),
    ("model", "\U0001f4e6  Model"),
    ("update", "↻  Update"),
    ("runs", "\U0001f5c2  Runs"),
)


class MissionConsole(QMainWindow):
    """Main window: left navigation rail, stacked pages, status bar."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VESP-UQ Mission Console")
        self.resize(1380, 860)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setCentralWidget(central)

        # ---------------- navigation rail ----------------
        rail = QWidget()
        rail.setObjectName("NavRail")
        rail.setFixedWidth(210)
        rail_col = QVBoxLayout(rail)
        rail_col.setContentsMargins(0, 0, 0, 0)
        rail_col.setSpacing(0)
        brand = QLabel("VESP-UQ")
        brand.setObjectName("NavBrand")
        brand_sub = QLabel("MISSION CONSOLE")
        brand_sub.setObjectName("NavBrandSub")
        rail_col.addWidget(brand)
        rail_col.addWidget(brand_sub)

        self.stack = QStackedWidget()
        self.pages: dict[str, QWidget] = {
            "dashboard": DashboardPage(self.navigate),
            "train": TrainPage(),
            "screen": ScreenPage(),
            "propagate": PropagatePage(),
            "model": ModelPage(),
            "update": UpdatePage(),
            "runs": RunsPage(),
        }

        self._nav_buttons: dict[str, QPushButton] = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for key, label in NAV_ITEMS:
            button = QPushButton(label)
            button.setProperty("nav", "true")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, k=key: self.navigate(k))
            group.addButton(button)
            rail_col.addWidget(button)
            self._nav_buttons[key] = button
            self.stack.addWidget(self.pages[key])

        rail_col.addStretch(1)
        footer = QLabel(f"root: {ROOT.name}\nforce-risk / OOD only -- never position error")
        footer.setObjectName("NavFooter")
        footer.setWordWrap(True)
        rail_col.addWidget(footer)

        layout.addWidget(rail)
        layout.addWidget(self.stack, 1)

        status_bar = self.statusBar()
        if status_bar is not None:
            status_bar.showMessage(
                "Train fits and packages a model; Screen serves it over new ensembles without refitting."
            )
        self.navigate("dashboard")

    def navigate(self, key: str) -> None:
        page = self.pages.get(key)
        if page is None:
            return
        self.stack.setCurrentWidget(page)
        button = self._nav_buttons.get(key)
        if button is not None:
            button.setChecked(True)


def main(argv: list[str] | None = None) -> int:
    """Application entry point (used by ``ui/app_vespuq.py``)."""

    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("VESP-UQ Mission Console")
    app.setStyleSheet(build_qss())
    window = MissionConsole()
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover - manual launch path
    raise SystemExit(main())
