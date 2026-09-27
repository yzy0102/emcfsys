#!/usr/bin/env python3
"""Run the FRC batch processor directly from a source checkout.

This launcher only makes the repository's ``src`` directory importable and
then delegates to ``frc.batch.main``. It does not alter any FRC calculation
code.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from frc.batch import main


if __name__ == "__main__":
    main()
