"""
ST-LRPS Studio.

PyQt6 dashboard for the lunar scalar-potential surrogate codebase. This module is
a thin launcher that wires together the structured UI components.
"""

import os
import sys

from lunaris.common.paths import project_root_from_file

from vesp.adapters.st_lrps.ui.studio_parts.common_widgets import _NoWheelOnSpinFilter
from vesp.adapters.st_lrps.ui.studio_parts.main_window import MainWindow
from vesp.adapters.st_lrps.ui.studio_parts.qt_common import (
    QApplication,
    QGuiApplication,
    Qt,
    apply_premium_dark_theme,
)


def main() -> None:
    try:
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.chdir(str(project_root_from_file(__file__)))
    app = QApplication(sys.argv)
    apply_premium_dark_theme(app)
    _wheel_guard = _NoWheelOnSpinFilter(app)
    app.installEventFilter(_wheel_guard)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
