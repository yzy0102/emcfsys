from __future__ import annotations

import os
import sys


def _force_pyqt5():
    os.environ.setdefault("QT_API", "pyqt5")
    os.environ.setdefault("PYTEST_QT_API", "pyqt5")

    for name in list(sys.modules):
        if name == "PyQt6" or name.startswith("PyQt6."):
            sys.modules.pop(name, None)


_force_pyqt5()


def pytest_configure():
    _force_pyqt5()
