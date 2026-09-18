"""Deep Vision Studio 로컬 웹 앱."""

from pathlib import Path
import sys

ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
for directory in (ROOT / "gui", ROOT / "python"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
